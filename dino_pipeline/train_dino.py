import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import argparse
from torchvision.transforms import functional as F
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dino_pipeline.dino_adapter import PPEClassifierHead

class DINOCropDataset(Dataset):
    def __init__(self, data_dir, metadata_file):
        self.data_dir = data_dir
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
            
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.data_dir, item["image"])
        
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        tensor = F.to_tensor(img_resized)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        return tensor, labels

def train_classifier(data_dir, epochs=20, batch_size=16, lr=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Training DINO PPE Classifier on {device}")
    
    metadata_file = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"[ERROR] Could not find metadata.json in {data_dir}. Run prepare_dino_crops.py first.")
        return
        
    dataset = DINOCropDataset(data_dir, metadata_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 1. Load Frozen Feature Extractor
    print("[INFO] Loading Meta DINOv2...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino.to(device)
    dino.eval() # Completely frozen
    
    # 2. Initialize Classifier Head
    classifier = PPEClassifierHead(embed_dim=384, num_classes=3)
    classifier.to(device)
    
    # BCEWithLogitsLoss for Multi-Label Classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    print("[INFO] Starting Training...")
    
    # Optional: Pre-extract features to memory if dataset is small
    # Since dataset is ~50 per class, it easily fits in memory and makes training take 1 second.
    # But for safety, we'll do standard dynamic forward passes.
    
    best_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Extract features without gradients
            with torch.no_grad():
                features = dino(imgs)
                
            # Train the linear head
            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(classifier.state_dict(), "models/dino_classifier.pt")
            
    print(f"[INFO] Training Complete! Best weights saved to models/dino_classifier.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dino_crops", help="Directory containing crops and metadata.json")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    train_classifier(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
