import torch
import numpy as np
from model import VICReg

def evaluate_embeddings():
    """
    Evaluates the properties of the learned embedding space to verify successful 
    VICReg regularization (High Variance, Low Covariance).
    """
    print("Evaluating VICReg Embedding Space...")
    
    model = VICReg()
    import os
    if os.path.exists("vicreg_model.pth"):
        model.load_state_dict(torch.load("vicreg_model.pth"))
        print("Loaded trained VICReg weights.")
    else:
        print("Warning: Using untrained random weights.")
    
    model.eval()
    
    # Generate diverse test data
    test_data = torch.randn(512, 1, 128)
    
    with torch.no_grad():
        z = model(test_data)
        
    # Check 1: Variance
    # Every dimension should have a standard deviation ideally > 1.0 (or close to gamma)
    stds = torch.sqrt(z.var(dim=0) + 1e-04)
    avg_std = stds.mean().item()
    min_std = stds.min().item()
    
    print(f"Average Dimension Std: {avg_std:.4f} (Target approx 1.0)")
    print(f"Minimum Dimension Std: {min_std:.4f} (If 0, point collapse occurred)")

    # Check 2: Covariance
    # We want off-diagonal elements of the corr matrix to be close to 0
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
    
    # Normalize to correlation matrix for better interpretability
    std_diag = torch.sqrt(torch.diag(cov) + 1e-08)
    corr = cov / (std_diag.unsqueeze(1) @ std_diag.unsqueeze(0))
    
    # Sum of absolute off-diagonal correlations
    off_diag_sum = (torch.abs(corr).sum() - torch.trace(torch.abs(corr))) / (z.size(1)**2 - z.size(1))
    
    print(f"Average Off-diagonal Absolute Correlation: {off_diag_sum:.4f} (Lower is better)")
    
    if avg_std > 0.5 and off_diag_sum < 0.2:
        print("RESULT: SUCCESS - The embedding space is diverse and dimensions are decorrelated.")
    else:
        print("RESULT: WARNING - Potential representation collapse or high redundancy detected.")

if __name__ == "__main__":
    evaluate_embeddings()
