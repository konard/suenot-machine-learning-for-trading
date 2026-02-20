import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN1DEncoder, TripletNet

def synthesize_financial_triplets(batch_size=128, seq_len=128):
    """
    Generates synthetic triplets (Anchor, Positive, Negative) simulating financial regimes.
    - Anchor: A base random walk.
    - Positive: Anchor + Jitter + Scaling (representing the same regime).
    - Negative: Reversed trend (representing a completely different regime).
    """
    # 1. Generate Base Anchor (Random Walk with momentum)
    momentum = torch.randn(batch_size, 1, 1).sign() * (torch.rand(batch_size, 1, 1) * 0.5 + 0.5)
    noise = torch.randn(batch_size, 1, seq_len) * 0.1
    anchor_diffs = momentum * 0.05 + noise
    anchor = torch.cumsum(anchor_diffs, dim=2)
    
    # 2. Generate Positive (Same Underlying Trend, modified)
    jitter = torch.randn_like(anchor) * 0.05
    scale = (torch.rand(batch_size, 1, 1) * 0.4) + 0.8 # 0.8x to 1.2x scaling
    positive = (anchor + jitter) * scale
    
    # 3. Generate Negative (Hard Negative: Inverted Trend)
    # If anchor went up, negative goes down heavily.
    negative = anchor * -1.5 + (torch.randn_like(anchor) * 0.05)
    
    # Normalize each window independently to mean 0, std 1
    def normalize(t):
        means = t.mean(dim=2, keepdim=True)
        stds = t.std(dim=2, keepdim=True) + 1e-6
        return (t - means) / stds

    return normalize(anchor), normalize(positive), normalize(negative)

def train_triplet_learning():
    """
    Training script for Triplet Learning.
    Uses PyTorch's native TripletMarginLoss.
    """
    print("Starting Triplet Learning Training Loop...")
    
    epochs = 40
    batch_size = 256
    lr = 1e-3
    margin = 1.0 # The required distance gap between (A,P) and (A,N)

    encoder = CNN1DEncoder(hidden_dim=128)
    model = TripletNet(encoder)
    
    # Triplet Margin Loss: max(0, d(a,p) - d(a,n) + margin)
    criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # 100 steps per epoch
        for _ in range(100):
            anchor, positive, negative = synthesize_financial_triplets(batch_size, seq_len=128)
            
            optimizer.zero_grad()
            
            z_a, z_p, z_n = model(anchor, positive, negative)
            
            loss = criterion(z_a, z_p, z_n)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Triplet Loss: {epoch_loss / 100:.4f}")

    print("Triplet Learning Pre-training completed.")
    torch.save(encoder.state_dict(), "triplet_encoder.pth")

if __name__ == "__main__":
    train_triplet_learning()
