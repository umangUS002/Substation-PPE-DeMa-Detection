"""
lightweight_backbone.py
------------------------
A fast, drop-in replacement for the DINOv2 ViT-S/14 feature extractor,
built via knowledge distillation *from* DINOv2 rather than by throwing
DINOv2's knowledge away. This is the single biggest lever for closing
the FPS gap: on a GPU with no tensor cores (GTX 1650 / Turing TU117),
a ViT backbone is disproportionately expensive per image compared to a
similarly-accurate CNN, because attention layers don't map onto plain
CUDA cores nearly as efficiently as convolutions do.

Two backbones are provided behind one interface (`PPEFeatureBackbone`):

  mode="distilled" (default, recommended for the 25-camera deployment)
      A MobileNetV3-Small trunk (~2.5M params vs DINOv2 ViT-S/14's 21M),
      trained offline so its embedding space mimics DINOv2's, then
      fine-tuned end-to-end with the PPE classifier head. In practice
      this is 10-30x fewer FLOPs per crop than the ViT, which is what
      actually matters for a non-tensor-core laptop GPU.

  mode="dino"
      The original DINOv2 ViT-S/14, kept as an opt-in high-accuracy
      fallback -- e.g. for offline re-scoring of borderline/contested
      alerts, or for a subset of high-risk cameras if the 25-feed
      budget allows sacrificing FPS on a few streams.

Distillation recipe (train_distillation, below) is provided as a
ready-to-run script skeleton, not something this session executed --
it needs your labeled PPE crop dataset and a GPU with DINOv2 available
(this sandbox has neither: no internet access to torch.hub and no
CUDA device). Run it once, offline, on your training machine; the
resulting `student.pt` weights are what `PPEFeatureBackbone` loads at
inference time on the CCTV server.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPEClassifierHead(nn.Module):
    """Same head as the original code: linear layer over pooled features."""

    def __init__(self, embed_dim, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


class MobileNetV3StudentBackbone(nn.Module):
    """
    Distillation student: MobileNetV3-Small trunk -> pooled embedding.
    Output embedding dim is 576 (MobileNetV3-Small's `features` output
    channel count before its own classifier) projected down to match
    DINOv2 vits14's 384-dim space, so the classifier head is a drop-in
    swap and the distillation loss can compare like-for-like vectors.
    """

    def __init__(self, embed_dim=384, pretrained=True):
        super().__init__()
        import torchvision

        weights = (
            torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # conv trunk only, drop the ImageNet classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        trunk_out_channels = backbone.classifier[0].in_features  # 576 for mobilenet_v3_small
        self.project = nn.Linear(trunk_out_channels, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.project(x)
        return x


class PPEFeatureBackbone(nn.Module):
    """
    Unified interface used by optimized_adapter.py. Hides whether the
    underlying feature extractor is the distilled CNN or the original
    DINOv2 ViT, so the rest of the pipeline doesn't care.
    """

    def __init__(self, mode="distilled", student_weights_path=None, device="cpu"):
        super().__init__()
        assert mode in ("distilled", "dino")
        self.mode = mode
        self.device = device
        self.embed_dim = 384

        if mode == "distilled":
            self.backbone = MobileNetV3StudentBackbone(embed_dim=self.embed_dim)
            if student_weights_path:
                state = torch.load(student_weights_path, map_location=device)
                self.backbone.load_state_dict(state)
            # MobileNetV3 expects 224x224 ImageNet-style normalized RGB, same as DINOv2's input contract.
            self.input_size = 224
        else:
            # Requires internet access to torch.hub on first load, and is
            # ~15-40x the FLOPs of the distilled path per crop -- see
            # docs/scaling_bottleneck_analysis.docx for the throughput math.
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.input_size = 224

        self.backbone.to(device)
        self.backbone.eval()

    @torch.inference_mode()
    def forward(self, x):
        return self.backbone(x)


def train_distillation(
    teacher_model,           # frozen DINOv2 vits14, eval mode, on GPU
    student_backbone,        # MobileNetV3StudentBackbone, trainable
    unlabeled_crop_loader,   # DataLoader yielding batches of 224x224 person crops (no labels needed for this stage)
    optimizer,
    device,
    epochs=10,
):
    """
    Stage 1 of the recipe: feature distillation.

    Train the student so its embedding matches DINOv2's for the same
    input crop (cosine similarity loss on the two embeddings). This
    stage needs *no PPE labels at all* -- any large pile of person
    crops pulled from your CCTV archive works, since we're only
    matching feature geometry, not classifying anything yet.

    This is a skeleton meant to be dropped into your training
    environment (needs internet for torch.hub + a GPU); it is not
    executed as part of this deliverable.
    """
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    student_backbone.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for crops in unlabeled_crop_loader:
            crops = crops.to(device)

            with torch.no_grad():
                teacher_feat = F.normalize(teacher_model(crops), dim=-1)

            student_feat = F.normalize(student_backbone(crops), dim=-1)

            # 1 - cosine similarity, averaged over the batch.
            loss = (1.0 - (teacher_feat * student_feat).sum(dim=-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        print(f"[distill] epoch {epoch+1}/{epochs} avg cosine-distance loss: {running_loss / max(1, n_batches):.4f}")

    return student_backbone


def finetune_classifier(
    student_backbone,        # output of train_distillation, still trainable (or freeze for linear-probe only)
    classifier_head,         # PPEClassifierHead(embed_dim=384, num_classes=3)
    labeled_loader,          # DataLoader yielding (crop, multi_label_targets)
    optimizer,
    device,
    epochs=15,
):
    """
    Stage 2: fine-tune on your labeled PPE crops (Helmet / Gloves /
    Footwear multi-label targets), same loss shape as the original
    classifier head expects (BCEWithLogitsLoss).
    """
    criterion = nn.BCEWithLogitsLoss()
    student_backbone.train()
    classifier_head.train()

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for crops, targets in labeled_loader:
            crops, targets = crops.to(device), targets.to(device)

            features = student_backbone(crops)
            logits = classifier_head(features)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        print(f"[finetune] epoch {epoch+1}/{epochs} avg BCE loss: {running_loss / max(1, n_batches):.4f}")

    return student_backbone, classifier_head
