import torch
import torch.optim as optim
from model import BYOL

def simple_augmentation(x):
    """
    Creates a 'noisy view' of the stock chart (Jittering for demo).
    """
    return x + torch.randn_like(x) * 0.05

def train_byol():
    print("Starting BYOL Self-Supervised Training Loop (No Negative Pairs)...")
    
    # 1. Initialize BYOL Network
    model = BYOL()
    
    # In practice: target network EMA momentum m usually increases from 0.99 to 1.0
    # over the course of training.
    
    # 2. Mock Market Data (Batch, Channels, SeqLen)
    data = torch.randn(2048, 1, 128)
    
    # Optimizer only updates the online network
    optimizer = optim.Adam(list(model.online_encoder.parameters()) + 
                           list(model.online_projector.parameters()) + 
                           list(model.online_predictor.parameters()), lr=1e-3)
    
    epochs = 5
    batch_size = 64
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < batch_size: continue
            
            # Create two augmented views (no need for negative samples in the batch!)
            v1 = simple_augmentation(batch)
            v2 = simple_augmentation(batch)
            
            # BYOL forward pass (calculates symmetrized target prediction loss)
            loss = model(v1, v2)
            
            # Backward and optimize online network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update the target network via exponential moving average
            model.update_target_network()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / (len(data)//batch_size):.4f}")

    print("BYOL Pre-training completed. Robust features extracted without negatives.")

if __name__ == "__main__":
    train_byol()
