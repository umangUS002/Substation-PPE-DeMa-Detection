"""
Worker ReID (Employee ID Identification) Training Pipeline.
Trains a Deep Metric Feature Extractor (ResNet-50 backbone) using Triplet Loss + Cross-Entropy Loss
to map worker crop bounding boxes to 512-dimensional embedding vectors for plant employee matching.
"""

import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Triplet Loss Function for Metric Learning
# ----------------------------------------------------
class TripletLoss(nn.Module):
    """
    Triplet Loss: Enforces distance d(anchor, positive) + margin < d(anchor, negative).
    Minimizes distance between crops of the same worker, maximizes distance between different workers.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# ----------------------------------------------------
# 2. ResNet-50 Worker ReID Feature Extractor Model
# ----------------------------------------------------
class WorkerReIDNet(nn.Module):
    def __init__(self, num_classes=6, feature_dim=512):
        super(WorkerReIDNet, self).__init__()
        base = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1]) # Remove fc
        self.embedding = nn.Linear(2048, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        emb = self.embedding(features)
        emb_norm = F.normalize(self.bn(emb), p=2, dim=1) # L2 Normalization for Cosine Distance
        out = self.classifier(emb_norm)
        return emb_norm, out

# ----------------------------------------------------
# 3. Worker ReID Training Loop
# ----------------------------------------------------
def train_reid_model(epochs=10, batch_size=4, lr=0.0003, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Worker ReID Model Training on: {device}", flush=True)
    print(f"[INFO] Config: epochs={epochs}, batch_size={batch_size}, lr={lr}", flush=True)

    model = WorkerReIDNet(num_classes=6, feature_dim=512).to(device)
    triplet_criterion = TripletLoss(margin=0.3)
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    weights_path = os.path.join(save_dir, "worker_reid_model.pt")

    print("[INFO] Simulating Worker ReID Feature Embedding Optimization...", flush=True)
    history_losses = []

    model.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Simulate anchor, positive, negative image crops
        anchor = torch.randn(batch_size, 3, 224, 224).to(device)
        positive = anchor + torch.randn(batch_size, 3, 224, 224).to(device) * 0.1 # Same worker
        negative = torch.randn(batch_size, 3, 224, 224).to(device) # Different worker
        labels = torch.randint(0, 6, (batch_size,)).to(device)

        optimizer.zero_grad()

        emb_a, out_a = model(anchor)
        emb_p, _ = model(positive)
        emb_n, _ = model(negative)

        loss_t = triplet_criterion(emb_a, emb_p, emb_n)
        loss_c = cross_entropy(out_a, labels)
        total_loss = loss_t + loss_c

        total_loss.backward()
        optimizer.step()

        history_losses.append(total_loss.item())
        print(f"Epoch [{epoch:02d}/{epochs}] - ReID Triplet Loss: {loss_t.item():.4f} | Class Loss: {loss_c.item():.4f} | Total Loss: {total_loss.item():.4f}", flush=True)

        torch.save(model.state_dict(), weights_path)

    elapsed = time.time() - start_time
    print(f"\n[OK] Worker ReID Training complete in {elapsed:.2f} seconds.")
    print(f"[SAVE] Saved worker ReID model weights to '{weights_path}'")

    # Plot Loss Curve
    plt.figure(figsize=(7, 4))
    plt.plot(history_losses, color='cyan', label='ReID Loss (Triplet + CrossEntropy)')
    plt.title("Worker Employee ID ReID Model Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "reid_training_loss.png"))
    plt.close()
    print(f"[SAVE] Loss plot saved to 'results/reid_training_loss.png'")

    return history_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Worker ReID (Employee ID Identification) Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")

    args = parser.parse_args()
    train_reid_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
