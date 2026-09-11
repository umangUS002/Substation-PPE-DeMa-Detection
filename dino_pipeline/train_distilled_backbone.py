"""
train_distilled_backbone.py
----------------------------
Two-stage training script to distill DINOv2 into MobileNetV3-Small
and then fine-tune the PPE classifier on top.

Run this on Kaggle with GPU enabled and your dino_crops dataset attached.

Stage 1: Feature Distillation (no labels needed)
  - Teaches MobileNetV3-Small to produce the same 384-dim features as DINOv2
  - Uses cosine similarity loss between teacher and student embeddings
  - Uses ALL your person crops (no PPE labels required)

Stage 2: Classifier Fine-tuning (uses metadata.json labels)
  - Trains the PPE classifier head on the student's features
  - BCEWithLogitsLoss for multi-label (Helmet, Gloves, Footwear)

Output:
  - /kaggle/working/student_backbone.pt   (distilled MobileNetV3 weights)
  - /kaggle/working/dino_classifier_distilled.pt (classifier trained on student features)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TF
import cv2
from tqdm import tqdm


# ─── Model Definitions ──────────────────────────────────────────────────────────

class MobileNetV3StudentBackbone(nn.Module):
    """MobileNetV3-Small trunk → 384-dim embedding to match DINOv2's output space."""
    def __init__(self, embed_dim=384, pretrained=True):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        trunk_out_channels = backbone.classifier[0].in_features  # 576
        self.project = nn.Linear(trunk_out_channels, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.project(x)
        return x


class PPEClassifierHead(nn.Module):
    """Same classifier head used by both original and optimized adapters."""
    def __init__(self, embed_dim=384, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


# ─── Datasets ────────────────────────────────────────────────────────────────────

class UnlabeledCropDataset(Dataset):
    """Stage 1: Just loads crop images, no labels needed for distillation."""
    def __init__(self, data_dir):
        self.image_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        print(f"[INFO] Found {len(self.image_paths)} crops for distillation")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.image_paths[idx])
        if img_bgr is None:
            # Return a blank tensor if image is corrupted
            return torch.zeros(3, 224, 224)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        tensor = TF.to_tensor(img_resized)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor


class LabeledCropDataset(Dataset):
    """Stage 2: Loads crops + PPE labels from metadata.json."""
    def __init__(self, data_dir, metadata_file):
        self.data_dir = data_dir
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        print(f"[INFO] Found {len(self.metadata)} labeled crops for classifier training")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.data_dir, item["image"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return torch.zeros(3, 224, 224), torch.zeros(3)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        tensor = TF.to_tensor(img_resized)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        return tensor, labels


# ─── Stage 1: Feature Distillation ───────────────────────────────────────────────

def train_distillation(data_dir, device, epochs=10, batch_size=32, lr=1e-3):
    """
    Train MobileNetV3-Small to mimic DINOv2's feature embeddings.
    No PPE labels needed — just person crops.
    """
    print("\n" + "=" * 60)
    print("  STAGE 1: Feature Distillation (DINOv2 → MobileNetV3)")
    print("=" * 60)

    # Load teacher (DINOv2 - frozen)
    print("[INFO] Loading DINOv2 teacher model...")
    teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Load student (MobileNetV3-Small - trainable)
    student = MobileNetV3StudentBackbone(embed_dim=384).to(device)

    # Dataset & Loader
    dataset = UnlabeledCropDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Distill Epoch {epoch+1}/{epochs}")
        for crops in pbar:
            crops = crops.to(device)

            # Teacher features (frozen, no grad)
            with torch.no_grad():
                teacher_feat = Fn.normalize(teacher(crops), dim=-1)

            # Student features (trainable)
            student_feat = Fn.normalize(student(crops), dim=-1)

            # Cosine similarity loss: want student to match teacher
            loss = (1.0 - (teacher_feat * student_feat).sum(dim=-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'cos_dist': f'{loss.item():.4f}'})

        avg_loss = running_loss / max(1, n_batches)
        scheduler.step()
        print(f"  Epoch {epoch+1}/{epochs} — Avg Cosine Distance: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), "/kaggle/working/student_backbone.pt")
            print(f"  ✅ Saved best student weights (loss={best_loss:.4f})")

    # Clean up teacher from GPU memory
    del teacher
    torch.cuda.empty_cache()

    print(f"\n[DONE] Stage 1 complete. Best cosine distance: {best_loss:.4f}")
    return student


# ─── Stage 2: Classifier Fine-tuning ────────────────────────────────────────────

def train_classifier(data_dir, metadata_file, student, device, epochs=15, batch_size=16, lr=1e-3):
    """
    Fine-tune the PPE classifier head on top of the distilled student backbone.
    Uses labeled crops from metadata.json.
    """
    print("\n" + "=" * 60)
    print("  STAGE 2: PPE Classifier Fine-tuning on Student Features")
    print("=" * 60)

    dataset = LabeledCropDataset(data_dir, metadata_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    classifier = PPEClassifierHead(embed_dim=384, num_classes=3).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Fine-tune both the student backbone and the classifier together
    all_params = list(student.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(all_params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    student.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        student.train()
        classifier.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Finetune Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            features = student(imgs)
            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'bce_loss': f'{loss.item():.4f}'})

        avg_loss = running_loss / len(loader)
        scheduler.step()
        print(f"  Epoch {epoch+1}/{epochs} — Avg BCE Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), "/kaggle/working/student_backbone.pt")
            torch.save(classifier.state_dict(), "/kaggle/working/dino_classifier_distilled.pt")
            print(f"  ✅ Saved best weights (loss={best_loss:.4f})")

    print(f"\n[DONE] Stage 2 complete. Best BCE loss: {best_loss:.4f}")


# ─── Main Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ╔══════════════════════════════════════════════════════════╗
    # ║  KAGGLE CONFIGURATION — Change these paths as needed!   ║
    # ╚══════════════════════════════════════════════════════════╝
    DATA_DIR = "/kaggle/input/dino-ppe-crops/dino_crops"
    METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

    # Training hyperparameters
    DISTILL_EPOCHS = 10    # Stage 1: feature distillation
    FINETUNE_EPOCHS = 15   # Stage 2: classifier fine-tuning
    BATCH_SIZE = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        print("[HINT] Make sure your dino_crops dataset is attached in Kaggle!")
        exit(1)

    # Stage 1: Distill DINOv2 → MobileNetV3
    student = train_distillation(DATA_DIR, device, epochs=DISTILL_EPOCHS, batch_size=BATCH_SIZE)

    # Stage 2: Fine-tune classifier on student features
    if os.path.exists(METADATA_FILE):
        train_classifier(DATA_DIR, METADATA_FILE, student, device, epochs=FINETUNE_EPOCHS, batch_size=BATCH_SIZE)
    else:
        print(f"\n[WARNING] No metadata.json found at {METADATA_FILE}")
        print("[INFO] Stage 2 skipped. You can still use the distilled backbone with the original classifier.")

    print("\n" + "=" * 60)
    print("  🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("  Download these files from /kaggle/working/:")
    print("    1. student_backbone.pt       → put in models/")
    print("    2. dino_classifier_distilled.pt → rename to dino_classifier.pt in models/")
    print()
    print("  Then in detect_video.py, change backbone_mode to 'distilled':")
    print('    backbone_mode="distilled",')
    print('    student_weights_path="models/student_backbone.pt"')
    print("=" * 60)
