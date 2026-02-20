import torch
import torch.nn.functional as F
import os
import math
from model import TemporalContrastiveModel
from train import generate_continuous_market_stream, sample_temporal_pairs

def evaluate_smoothness():
    """
    Evaluates the temporal smoothness of the learned embeddings.
    A successful TCL model will show that embeddings from nearby time windows
    are much closer (higher cosine similarity) than embeddings from distant time windows.
    """
    print("Evaluating Temporal Contrastive Embeddings (Smoothness Test)...")
    
    model = TemporalContrastiveModel(base_encoder_dim=64, projection_dim=128)
    if os.path.exists("tcl_model.pth"):
        model.load_state_dict(torch.load("tcl_model.pth"))
        print("Loaded trained TCL weights.")
    else:
        print("Warning: Using untrained random weights.")
    
    model.eval()
    
    # Generate a long stream to test varied temporal distances
    stream = generate_continuous_market_stream(total_length=5000)
    batch_size = 512
    window_size = 128
    
    # Test 1: Immediate Neighbors (Lag = 15)
    lag_near = 15
    x_t, x_t_near = sample_temporal_pairs(stream, batch_size, window_size, lag_near)
    
    # Test 2: Distant Neighbors (Lag = 500)
    lag_far = 500
    _, x_t_far = sample_temporal_pairs(stream, batch_size, window_size, lag_far)
    
    with torch.no_grad():
        # Get raw embeddings (z) not the projected ones (v)
        z_t, _ = model(x_t)
        z_t_near, _ = model(x_t_near)
        z_t_far, _ = model(x_t_far)
        
    # Calculate Euclidean Distances
    dist_near = torch.norm(z_t - z_t_near, dim=1).mean().item()
    dist_far = torch.norm(z_t - z_t_far, dim=1).mean().item()
    
    print(f"Average Euclidean Distance (Near Neighbors, Lag=15): {dist_near:.4f} (Target: small)")
    print(f"Average Euclidean Distance (Distant Random, Lag=500): {dist_far:.4f} (Target: large)")
    
    diff = dist_far - dist_near
    print(f"Distance Separation Gap: {diff:.4f}")
    
    if diff > 1.0 and dist_near < 5.0:
        print("RESULT: SUCCESS - The network successfully maps temporal continuity to the embedding space.")
    else:
        print("RESULT: WARNING - Temporal smoothing is insufficient. The network failed to learn the time arrow.")

if __name__ == "__main__":
    evaluate_smoothness()
