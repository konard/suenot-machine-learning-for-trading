import torch
import torch.optim as optim
import math
from model import TemporalContrastiveModel, TemporalInfoNCELoss

def generate_continuous_market_stream(total_length=10000):
    """
    Generates a continuous synthetic price stream with distinct 'regimes'
    to prevent representation collapse during Temporal Contrastive Learning.
    """
    # 1. Base random walk (high frequency noise)
    steps = torch.randn(total_length) * 0.5
    
    # 2. Add distinct regime shifts (low frequency structural changes)
    # Using multiple sine waves of different frequencies creates complex, shifting patterns
    regime_1 = torch.sin(torch.linspace(0, 50 * math.pi, total_length)) * 2.0
    regime_2 = torch.cos(torch.linspace(0, 10 * math.pi, total_length)) * 3.0
    
    # 3. Add sharp structural breaks
    breaks = torch.zeros(total_length)
    breaks[::1000] = torch.randn(total_length // 1000) * 10.0
    
    adjusted_steps = steps + regime_1 + regime_2 + breaks
    price_stream = torch.cumsum(adjusted_steps, dim=0)
    
    return price_stream

def sample_temporal_pairs(stream, batch_size=256, window_size=128, lag=10):
    """
    Samples valid temporal pairs (x_t, x_{t+lag}) from the continuous stream.
    `lag` dictates the temporal neighborhood distance.
    Returns: (batch_t, batch_t_plus_lag)
    """
    max_start_idx = len(stream) - window_size - lag - 1
    
    # Random starting points for the anchor windows
    start_indices = torch.randint(0, max_start_idx, (batch_size,))
    
    batch_t = torch.zeros((batch_size, 1, window_size))
    batch_t_plus = torch.zeros((batch_size, 1, window_size))
    
    for i, idx in enumerate(start_indices):
        w_t = stream[idx : idx + window_size]
        w_t_plus = stream[idx + lag : idx + lag + window_size]
        
        # Independent Z-score normalization for each window to focus on local shapes,
        # not absolute price levels.
        batch_t[i, 0, :] = (w_t - w_t.mean()) / (w_t.std() + 1e-6)
        batch_t_plus[i, 0, :] = (w_t_plus - w_t_plus.mean()) / (w_t_plus.std() + 1e-6)
        
    return batch_t, batch_t_plus

def train_tcl():
    """
    Training script for Temporal Contrastive Learning.
    """
    print("Starting Temporal Contrastive Learning (TCL) Training Loop...")
    
    epochs = 60
    batch_size = 512
    lr = 1e-3
    window_size = 128
    temporal_lag = 15 # The positive pair is 15 ticks "in the future"

    # Generate the global synthetic dataset history with more complexity
    stream = generate_continuous_market_stream(total_length=30000)

    model = TemporalContrastiveModel(base_encoder_dim=64, projection_dim=128)
    criterion = TemporalInfoNCELoss(temperature=0.07) # Lower temp for sharper contrast
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # 50 iterations per epoch
        for _ in range(50):
            x_t, x_t_plus = sample_temporal_pairs(stream, batch_size, window_size, temporal_lag)
            
            optimizer.zero_grad()
            
            # Forward pass provides (embedding, projected_view)
            _, v1 = model(x_t)
            _, v2 = model(x_t_plus)
            
            # InfoNCE pulls v1 and v2 together while pushing batched negatives away
            loss = criterion(v1, v2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Temporal InfoNCE Loss: {epoch_loss / 50:.4f}")

    print("TCL Pre-training completed. Model learned to group temporally adjacent regimes.")
    torch.save(model.state_dict(), "tcl_model.pth")

if __name__ == "__main__":
    train_tcl()
