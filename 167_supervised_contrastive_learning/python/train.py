import torch
import torch.optim as optim
import math
import numpy as np
from model import SupConEncoder
from supcon_loss import SupConLoss

def synthesize_labeled_batch(batch_size=64, window_size=64):
    """
    Generates time-series windows with explicit labels:
    0: Trending Up
    1: Trending Down
    2: Mean Reverting
    """
    batch_x = torch.zeros(batch_size, 1, window_size)
    batch_y = torch.zeros(batch_size, dtype=torch.long)
    
    t = torch.linspace(0, 1, window_size)
    
    for i in range(batch_size):
        label = i % 3
        noise = torch.randn(window_size) * 0.1
        
        if label == 0: # Up
            window = t + noise
        elif label == 1: # Down
            window = -t + noise
        else: # Reverting
            window = torch.sin(t * 10) * 0.5 + noise
            
        # Normalize
        window = (window - window.mean()) / (window.std() + 1e-6)
        batch_x[i, 0, :] = window
        batch_y[i] = label
        
    return batch_x, batch_y

def train_supcon():
    # Parameters
    BATCH_SIZE = 128
    WINDOW_SIZE = 64
    FEATURE_DIM = 64
    PROJECTION_DIM = 32
    EPOCHS = 20
    STEPS_PER_EPOCH = 30
    
    print(f"Initializing Supervised Contrastive Learning (SupCon)...")
    
    model = SupConEncoder(feature_dim=FEATURE_DIM, projection_dim=PROJECTION_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = SupConLoss(temperature=0.07)
    
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        
        for step in range(STEPS_PER_EPOCH):
            # 1. Generate Labeled Data
            x, y = synthesize_labeled_batch(batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
            
            optimizer.zero_grad()
            
            # 2. Forward Pass: Get latents and normalized projections
            _, z = model(x)
            
            # 3. SupCon Loss
            loss = criterion(z, y)
            
            # 4. Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / STEPS_PER_EPOCH
        print(f"Epoch {epoch}/{EPOCHS} | SupCon Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), "supcon_trading_model.pth")
    print("Training complete. Model saved to supcon_trading_model.pth")

if __name__ == "__main__":
    train_supcon()
