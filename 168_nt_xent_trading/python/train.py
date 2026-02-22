import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import CNNEncoder
from nt_xent_loss import NTXentLoss

def generate_augmented_data(batch_size=64, window_size=64):
    """
    Simulates two augmented views of the same market situation.
    Augmentation 1: Jittering (Noise)
    Augmentation 2: Scaling
    """
    # Base signal (Random but structured)
    base = torch.cumsum(torch.randn(batch_size, 1, window_size) * 0.1, dim=2)
    
    # View 1: Base + Jitter
    view1 = base + torch.randn_like(base) * 0.05
    
    # View 2: Base * Random Scale + Jitter
    scale = (torch.rand(batch_size, 1, 1) * 0.4 + 0.8)
    view2 = (base * scale) + torch.randn_like(base) * 0.05
    
    # Normalization
    view1 = (view1 - view1.mean(dim=2, keepdim=True)) / (view1.std(dim=2, keepdim=True) + 1e-6)
    view2 = (view2 - view2.mean(dim=2, keepdim=True)) / (view2.std(dim=2, keepdim=True) + 1e-6)
    
    return view1, view2

def run_experiment(temperature):
    print(f"\n--- Running Experiment with Temperature: {temperature} ---")
    
    BATCH_SIZE = 128
    WINDOW_SIZE = 64
    EPOCHS = 10
    STEPS_PER_EPOCH = 20
    
    model = CNNEncoder(projection_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = NTXentLoss(temperature=temperature)
    
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for step in range(STEPS_PER_EPOCH):
            v1, v2 = generate_augmented_data(BATCH_SIZE, WINDOW_SIZE)
            
            optimizer.zero_grad()
            
            z1 = model(v1)
            z2 = model(v2)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss/STEPS_PER_EPOCH:.4f}")
    
    return total_loss/STEPS_PER_EPOCH

def main():
    temperatures = [0.07, 0.5, 1.0]
    final_losses = []
    
    for t in temperatures:
        loss = run_experiment(t)
        final_losses.append(loss)
        
    print("\nSummary of Temperature Sweep:")
    for t, l in zip(temperatures, final_losses):
        print(f"Temp {t:.2f} -> Final Loss: {l:.4f}")

if __name__ == "__main__":
    main()
