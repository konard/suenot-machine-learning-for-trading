import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import SirenNet

def generate_mock_vol_surface(n_points=1000):
    """
    Generates a synthetic implied volatility surface with a typical 
    volatility smile and term structure skew.
    """
    moneyness = np.linspace(-0.5, 0.5, 50) # Log-moneyness
    maturity = np.linspace(0.1, 2.0, 20)  # Years to expiry
    
    M, T = np.meshgrid(moneyness, maturity)
    
    # Simple model: IV = base + smile_effect + term_structure_effect
    # Smile is a parabola, term structure decays with time
    IV = 0.2 + 0.4 * M**2 + 0.05 / T
    
    # Flatten for training
    coords = np.stack([M.ravel(), T.ravel()], axis=-1)
    targets = IV.ravel()[:, None]
    
    return torch.tensor(coords, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def train_siren_surface():
    print("Starting SIREN calibration on Implied Volatility Surface...")
    coords, targets = generate_mock_vol_surface()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords, targets = coords.to(device), targets.to(device)
    
    model = SirenNet(in_features=2, hidden_features=128, hidden_layers=3, out_features=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(coords)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.8f}")
            
    print(f"Final Calibration Loss: {loss.item():.8f}")
    
    # Quick visual check (optional, can be saved to a file)
    model.eval()
    with torch.no_grad():
        preds = model(coords).cpu().numpy()
    
    print("Surface calibration completed.")
    return model

if __name__ == "__main__":
    train_siren_surface()
