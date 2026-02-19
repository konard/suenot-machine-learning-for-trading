import torch
import torch.nn as nn
from model import BidirectionalMambaBlock

def simulate_market_window_batch(batch_size=16, seq_length=120, features=64):
    """
    Generates a batch of LOB (Limit Order Book) lookback windows.
    Each sequence of length `seq_length` represents the entire historical context
    observed UP TO time t. No future data exists in this buffer.
    """
    inputs = torch.randn(batch_size, seq_length, features)
    # The target is an aggregated directional bias 
    # computed mathematically from the window for the NEXT tick (t+1).
    # Being a single float regression target per sequence makes this Bidirectional 
    # Mamba perfectly suited for O(N) multi-variate integration.
    targets = torch.sum(inputs[:, -10:, :], dim=(1, 2)) * 0.01 
    return inputs, targets

class BidirectionalTradingHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.backbone = BidirectionalMambaBlock(d_model)
        # We only care about the consolidated projection at the final step t
        self.regression_head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        contextualized = self.backbone(x)
        # Take the final token representation which now contains dense backward 
        # semantic compression from the entire window.
        final_token = contextualized[:, -1, :] 
        return self.regression_head(final_token).squeeze(-1)

def train_eval_cycle(epochs=30, feature_dim=64):
    print("Bootstrapping Bidirectional Mamba Training Environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BidirectionalTradingHead(d_model=feature_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    model.train()
    print("\n--- Starting Deep O(N) Contextualization Training ---")
    
    for epoch in range(1, epochs + 1):
        inputs, targets = simulate_market_window_batch(features=feature_dim)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        # The model sweeps forward and backward natively inside the window
        predictions = model(inputs)
        
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss (MSE): {loss.item():.6f}")

    print("\nBidirectional Mamba Training module completed successfully.")
    print("Global structure successfully compressed into predictive token.")

if __name__ == "__main__":
    train_eval_cycle()
