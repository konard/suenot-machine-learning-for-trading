import torch
import torch.optim as optim
from model import BarlowTwins

def simple_augmentation(x):
    """
    Creates a 'noisy view' of the stock chart (Jittering for demo).
    """
    return x + torch.randn_like(x) * 0.05

def train_barlow_twins():
    print("Starting Barlow Twins Self-Supervised Training Loop...")
    
    # 1. Initialize Network (Symmetric, identical branches)
    # The paper uses lambda = 0.0051 for dimensionality of 8192, 
    # but for smaller dimensions like 128, a larger lambda might be required.
    # Let's scale up lambda slightly to enforce decorrelation powerfully in small dimensions.
    model = BarlowTwins(lambda_param=0.0051)
    
    # 2. Mock Market Data (Batch, Channels, SeqLen)
    # Using larger batches helps the Cross-Correlation matrix become statistically robust
    data = torch.randn(4096, 1, 128)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    batch_size = 128
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < batch_size: continue
            
            # Create two augmented views
            v1 = simple_augmentation(batch)
            v2 = simple_augmentation(batch)
            
            # Barlow Twins symmetric forward and loss calculation
            loss = model(v1, v2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / (len(data)//batch_size):.4f}")

    print("Barlow Twins Pre-training completed. Orthogonal, non-redundant features extracted.")

if __name__ == "__main__":
    train_barlow_twins()
