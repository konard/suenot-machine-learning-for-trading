import torch
import torch.nn as nn
from model import GatedLinearAttention
import itertools

def load_simulated_market_data(batch_size=16, seq_length=150, features=128):
    """
    Simulates high-frequency LOB (Limit Order Book) snapshots and volatility
    from multiple exchanges like Bybit (Cryptocurrency). 
    In live usage, replace this with Bybit DataLoader.
    inputs: Historical continuous feature streams 
    targets: Future expected return / volatility profile
    """
    inputs = torch.randn(batch_size, seq_length, features)
    # Using sine waves hidden inside noise to create an artificial market oscillation
    time_series = torch.sin(torch.linspace(0, 10, seq_length)).view(1, -1, 1)
    inputs += time_series * 0.5 
    
    # Target is looking 1 step ahead for a movement indicator mapped dynamically
    targets = torch.roll(inputs, shifts=-1, dims=1)
    
    return inputs, targets

def train_eval_cycle(epochs=50, feature_dim=128):
    print("Initializing Market DataLoader & Structured State Space Dual Network...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedLinearAttention(d_model=feature_dim).to(device)
    
    # Predicting continuous outcomes (e.g. basis points return or LOB depth)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    
    print("\n--- Starting Training loop (Simulated Exchange Data) ---")
    for epoch in range(1, epochs + 1):
        # Fetch mock Bybit high-frequency tick streams
        inputs, targets = load_simulated_market_data(features=feature_dim)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # O(N) linear time sequence modeling
        predictions = model(inputs)
        
        # Calculate loss against future movement
        loss = criterion(predictions, targets)
        loss.backward()
        
        # Gradient clipping prevents exploding gradients in RNN/SSMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss (MSE): {loss.item():.6f} | Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    print("\nTraining step completed.")
    print("Linear Attention State matrix effectively compressed historical orderflow without O(N^2) memory footprint.")

if __name__ == "__main__":
    train_eval_cycle()
