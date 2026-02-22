import torch
import torch.nn as nn
import numpy as np
from model import TradingNN
from dp_core import DPSGDManager

def generate_market_data(num_samples=1000, input_dim=20):
    x = torch.randn(num_samples, input_dim)
    # Simple strategy: sum of features > 0 is target
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    return x, y

def train_with_privacy(noise_level=0.1, epochs=10):
    print(f"\n--- Training with DP (Noise Multiplier: {noise_level}) ---")
    
    input_dim = 20
    x, y = generate_market_data()
    model = TradingNN(input_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    dp_manager = DPSGDManager(l2_norm_clip=1.0, noise_multiplier=noise_level)
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Apply DP: Clip and Noise
        dp_manager.apply_dp_step(optimizer, model.parameters())
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")
            
    return model

def evaluate(model, data, labels):
    model.eval()
    with torch.no_grad():
        preds = model(data)
        mse = torch.mean((preds - labels)**2)
    return mse.item()

def run_experiment():
    print("Differential Privacy Comparison for Trading Models")
    
    test_x, test_y = generate_market_data(num_samples=200)
    
    # 1. Baseline (No Privacy)
    model_baseline = train_with_privacy(noise_level=0.0)
    mse_baseline = evaluate(model_baseline, test_x, test_y)
    
    # 2. Strong Privacy (High Noise)
    model_private = train_with_privacy(noise_level=0.5)
    mse_private = evaluate(model_private, test_x, test_y)
    
    print("\n--- Summary Results ---")
    print(f"Baseline MSE (Noise=0.0): {mse_baseline:.6f}")
    print(f"Private MSE  (Noise=0.5): {mse_private:.6f}")
    print(f"Privacy Utility Loss: {((mse_private - mse_baseline)/mse_baseline)*100:.2f}%")

if __name__ == "__main__":
    run_experiment()
