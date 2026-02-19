import torch
from model import LinearAttentionSSM

def train_eval_cycle():
    print("Loading financial data (Bybit crypto data & Stocks)...")
    model = LinearAttentionSSM(128)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data for sequence regression
    inputs = torch.randn(16, 20, 128)
    targets = torch.randn(16, 20, 128)
    
    # Train step
    model.train()
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed. Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_eval_cycle()
