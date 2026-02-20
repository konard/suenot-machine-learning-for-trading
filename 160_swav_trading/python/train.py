import torch
import torch.optim as optim
from model import SwAV

def simple_augmentation(x):
    """
    Creates a 'noisy view' of the stock chart.
    """
    return x + torch.randn_like(x) * 0.05

def train_swav():
    print("Starting SwAV (Swapping Assignments between Views) Training Loop...")
    
    # Initialize Network
    # Using 10 prototypes (clusters) to force the model to categorize the charts into 10 generic regimes
    model = SwAV(n_prototypes=10)
    
    # Mock Market Data (Batch, Channels, SeqLen)
    # SwAV requires batch_size >= n_prototypes, preferably much larger, to make the Equipartition
    # constraint computationally stable.
    data = torch.randn(2048, 1, 128)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 5
    batch_size = 256 # Ample size to feed the Sinkhorn transport map
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < batch_size: continue
            
            # Create two augmented views
            v1 = simple_augmentation(batch)
            v2 = simple_augmentation(batch)
            
            # Swapped Assignment forward and loss calculation
            loss = model(v1, v2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / (len(data)//batch_size):.4f}")

    print("SwAV Pre-training completed. Continuous features matched to discrete Prototypes.")
    torch.save(model.state_dict(), "swav_model.pth")

if __name__ == "__main__":
    train_swav()
