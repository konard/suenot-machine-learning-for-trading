import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import CNN1DEncoder, SimCLR
from augmentations import apply_simclr_augmentations

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    """
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0) # (2*N, proj_dim)
    
    sim_matrix = torch.matmul(z, z.T) / temperature # (2*N, 2*N)
    
    # Positive pairs are (i, i+N) and (i+N, i)
    mask = torch.eye(batch_size, dtype=torch.bool).to(z.device)
    mask = torch.cat([
        torch.cat([torch.zeros_like(mask), mask], dim=1),
        torch.cat([mask, torch.zeros_like(mask)], dim=1)
    ], dim=0)

    # Self-similarity mask to ignore diagonal
    self_mask = ~torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    
    positives = sim_matrix[mask].view(2 * batch_size, 1)
    negatives = sim_matrix[self_mask].view(2 * batch_size, -1)
    
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z.device)
    
    return F.cross_entropy(logits, labels)

def train_simclr():
    print("Starting Self-Supervised SimCLR Pre-training for Stocks...")
    
    # 1. Mock Stock Data (Batch, Channels, SeqLen)
    # In reality, this would be a large dataset of historical windows
    data = torch.randn(1024, 1, 128) 
    
    encoder = CNN1DEncoder()
    model = SimCLR(encoder)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    batch_size = 64
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < batch_size: continue
            
            # Create two augmented views
            x_i = apply_simclr_augmentations(batch)
            x_j = apply_simclr_augmentations(batch)
            
            optimizer.zero_grad()
            z_i = model(x_i)
            z_j = model(x_j)
            
            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / (len(data)//batch_size):.4f}")

    print("Pre-training completed. Encoder is now ready for downstream tasks.")
    # torch.save(encoder.state_dict(), "encoder.pth")

if __name__ == "__main__":
    train_simclr()
