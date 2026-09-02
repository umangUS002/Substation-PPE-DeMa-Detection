import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.transforms import functional as F
from tqdm import tqdm

# --- 1. CLASSIFIER ARCHITECTURE ---
class PPEClassifierHead(nn.Module):
    """Lightweight Multi-Label Classifier to run on top of DINOv2 features."""
    def __init__(self, embed_dim=384, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

# --- 2. DATASET LOADER ---
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

# --- 3. TRAINING LOOP ---
def train_kaggle():
    # ---------- KAGGLE CONFIGURATION ----------
    # CHANGE THIS TO YOUR KAGGLE DATASET PATH!
    # Usually it looks like: /kaggle/input/your-dataset-name/dino_crops
    DATA_DIR = "/kaggle/input/dino-ppe-crops/dino_crops" 
    EPOCHS = 20
    BATCH_SIZE = 16
    LR = 0.001
    # ------------------------------------------

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Training DINO PPE Classifier on {device}")
    
    metadata_file = os.path.join(DATA_DIR, "metadata.json")
    if not os.path.exists(metadata_file):
        print(f"[ERROR] Could not find metadata.json at {metadata_file}.")
        return
        
    dataset = DINOCropDataset(DATA_DIR, metadata_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("[INFO] Loading Meta DINOv2 (Freezing parameters)...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino.to(device)
    dino.eval() 
    
    classifier = PPEClassifierHead(embed_dim=384, num_classes=3)
    classifier.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=LR)
    
    print("[INFO] Starting Training...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        classifier.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                features = dino(imgs)
                
            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save weights to Kaggle's working directory so you can download them later
            torch.save(classifier.state_dict(), "/kaggle/working/dino_classifier.pt")
            
    print("[INFO] Training Complete! Best weights saved to /kaggle/working/dino_classifier.pt")

if __name__ == "__main__":
    train_kaggle()
