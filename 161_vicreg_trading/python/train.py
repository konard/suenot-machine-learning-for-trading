import torch
import torch.optim as optim
from model import VICReg, VICRegLoss

def train_vicreg():
    """
    Training script for VICReg using mock financial data.
    """
    print("Starting VICReg (Variance-Invariance-Covariance) Training Loop...")
    
    # Hyperparameters
    base_dim = 64
    proj_dim = 256
    batch_size = 256
    epochs = 50
    lr = 5e-4

    model = VICReg(base_encoder_dim=base_dim, projection_dim=proj_dim)
    criterion = VICRegLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Mock Data: 1024 windows of shape (1 channel, 128 ticks)
    # We create two "augmented views" by adding different noise to the same base signal
    base_data = torch.randn(1024, 1, 128)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Shuffle for batching
        indices = torch.randperm(1024)
        
        for i in range(0, 1024, batch_size):
            batch_indices = indices[i : i + batch_size]
            x = base_data[batch_indices]
            
            # Augmentations: x1 and x2
            # For this MVP, we use simple Gaussian Noise and Jittering
            x1 = x + torch.randn_like(x) * 0.1
            x2 = x + torch.randn_like(x) * 0.1
            
            optimizer.zero_grad()
            
            z1 = model(x1)
            z2 = model(x2)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / (1024//batch_size):.4f}")

    print("VICReg Pre-training completed. Features regularized for high variance and low covariance.")
    torch.save(model.state_dict(), "vicreg_model.pth")

if __name__ == "__main__":
    train_vicreg()
