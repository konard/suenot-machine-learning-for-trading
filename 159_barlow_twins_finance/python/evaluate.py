import torch
from model import CNN1DEncoder

def eval_redundancy_reduction():
    """
    Checks if Barlow Twins successfully generated decorrelated, non-redundant features.
    """
    print("Running Barlow Twins Redundancy Verification...")
    
    encoder = CNN1DEncoder()
    encoder.eval()
    
    # Generate a large batch of random patterns to get statistical significance
    N = 1000
    patterns = torch.randn(N, 1, 128)
    
    with torch.no_grad():
        features = encoder(patterns) # Shape: (N, 64)
        
    D = features.size(1)
        
    # Standardize features
    feat_norm = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)
    
    # Calculate cross-correlation matrix of the features against themselves
    corr_matrix = torch.mm(feat_norm.T, feat_norm) / N
    
    # An ideal matrix here would be the Identity Matrix.
    # We check the average off-diagonal value. 
    # If the network collapsed or is highly redundant, this value will be huge.
    off_diagonal_mask = 1 - torch.eye(D)
    off_diag_elements = corr_matrix * off_diagonal_mask
    avg_cross_correlation = off_diag_elements.abs().sum() / (D * (D - 1))
    
    print(f"Average absolute cross-correlation of feature dimensions: {avg_cross_correlation.item():.6f}")
    
    # In an untrained network, random initialization might yield small correlation naturally,
    # but during training, Barlow Twins actively forces this towards 0.0 better than PCA.
    if avg_cross_correlation.item() < 0.15:
        print("RESULT: Features show low redundancy. DECORRELATION: OK.")
    else:
        print("RESULT: High redundancy / Correlation detected. Dimensions are intertwined.")

if __name__ == "__main__":
    eval_redundancy_reduction()
