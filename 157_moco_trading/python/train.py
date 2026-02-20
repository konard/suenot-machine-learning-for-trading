import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN1DEncoder, MoCo
# Re-using augmentations logic from Chapter 156 if compatible, or defining here
import sys
import os

# To avoid complexity in this demo, let's define a simple augmentation 
def simple_augmentation(x):
    # Just add some jitter for demo purposes
    return x + torch.randn_like(x) * 0.05

def train_moco():
    print("Starting MoCo Self-Supervised Training Loop...")
    
    # 1. Initialize Encoders and MoCo
    q_enc = CNN1DEncoder()
    k_enc = CNN1DEncoder()
    model = MoCo(q_enc, k_enc, K=1024, m=0.999, T=0.07)
    
    # 2. Mock Market Data (Batch, Channels, SeqLen)
    data = torch.randn(2048, 1, 128)
    
    optimizer = optim.Adam(model.encoder_q.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 5
    batch_size = 64
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < batch_size: continue
            
            # Create two augmented views
            im_q = simple_augmentation(batch)
            im_k = simple_augmentation(batch)
            
            # Forward pass
            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / (len(data)//batch_size):.4f}")

    print("MoCo Pre-training completed. Momentum-stable features are ready.")

if __name__ == "__main__":
    train_moco()
