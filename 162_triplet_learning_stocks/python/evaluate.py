import torch
import torch.nn.functional as F
import os
from model import CNN1DEncoder
from train import synthesize_financial_triplets

def evaluate_triplets():
    """
    Evaluates the learned embedding distances for Anchors, Positives, and Negatives.
    Verifies that d(A, P) + Margin < d(A, N).
    """
    print("Evaluating Triplet Embedding Distances...")
    
    encoder = CNN1DEncoder(hidden_dim=128)
    if os.path.exists("triplet_encoder.pth"):
        encoder.load_state_dict(torch.load("triplet_encoder.pth"))
        print("Loaded trained Triplet Encoder weights.")
    else:
        print("Warning: Using untrained random weights.")
    
    encoder.eval()
    
    # Generate test triplets
    anchor, positive, negative = synthesize_financial_triplets(batch_size=1024, seq_len=128)
    margin = 1.0
    
    with torch.no_grad():
        z_a = encoder(anchor)
        z_p = encoder(positive)
        z_n = encoder(negative)
        
    # Calculate Euclidean distances
    dist_ap = F.pairwise_distance(z_a, z_p, p=2)
    dist_an = F.pairwise_distance(z_a, z_n, p=2)
    
    mean_dist_ap = dist_ap.mean().item()
    mean_dist_an = dist_an.mean().item()
    
    print(f"Average d(Anchor, Positive): {mean_dist_ap:.4f} (Should be small)")
    print(f"Average d(Anchor, Negative): {mean_dist_an:.4f} (Should be large)")
    
    diff = mean_dist_an - mean_dist_ap
    print(f"Distance Gap: {diff:.4f} (Target > Margin = {margin})")
    
    # Accuracy: Percentage of triplets where the negative is further than the positive by at least the margin
    correct_triplets = ((dist_an - dist_ap) > margin).float().mean().item()
    print(f"Hard Triplet Accuracy: {correct_triplets * 100:.2f}%")
    
    if diff > (margin * 0.8) and correct_triplets > 0.8:
        print("RESULT: SUCCESS - The network successfully separates disparate market regimes.")
    else:
        print("RESULT: WARNING - Separation margin is not sufficient or accuracy is low.")

if __name__ == "__main__":
    evaluate_triplets()
