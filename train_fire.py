import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.sequence_loader import SequenceBufferDataset
from models.fire_ash_detector import Combustion3DClassifier

def train_fire_model(data_dir="data/videos", epochs=20, batch_size=8, learning_rate=1e-4):
    """
    Training script for the 3D CNN Temporal Path (Combustion3DClassifier).
    Expects videos to be in data_dir/combustion and data_dir/background.
    """
    print("Initializing Fire & Ash 3D CNN Training Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Dataset & DataLoader
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found. Please create it and add videos.")
        return

    try:
        dataset = SequenceBufferDataset(data_dir=data_dir, sequence_length=16)
        if len(dataset) == 0:
            print(f"[ERROR] No videos found in '{data_dir}'. Ensure subfolders exist (e.g., combustion/).")
            return
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        print(f"Loaded {len(dataset)} video sequences across {len(dataset.classes)} classes: {dataset.classes}")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # 2. Initialize Model, Loss, Optimizer
    model = Combustion3DClassifier(in_channels=3, num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting Training Loop...")
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (sequences, labels) in enumerate(dataloader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"--- Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% ---")
        
        # Save checkpoint
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/fire_3dcnn_epoch_{epoch+1}.pt")

    print("\nTraining Complete! Best model saved to 'models/' directory.")

if __name__ == "__main__":
    train_fire_model()
