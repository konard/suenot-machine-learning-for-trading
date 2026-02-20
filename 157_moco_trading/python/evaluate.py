import torch
from model import CNN1DEncoder

def check_feature_stability():
    """
    Verfies that the encoder produces consistent features for similar patterns.
    """
    print("Running MoCo Feature Stability Verification...")
    
    encoder = CNN1DEncoder()
    encoder.eval()
    
    # Create a base pattern
    base_pattern = torch.randn(1, 1, 128)
    
    # Create a noisy version
    noisy_pattern = base_pattern + torch.randn_like(base_pattern) * 0.01
    
    with torch.no_grad():
        feat_base = encoder(base_pattern)
        feat_noisy = encoder(noisy_pattern)
        
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(feat_base, feat_noisy)
    
    print(f"Base Pattern Feature Tail (first 5 dims): {feat_base[0, :5].tolist()}")
    print(f"Cosine Similarity (Base vs Noisy): {similarity.item():.6f}")
    
    if similarity.item() > 0.95:
        print("RESULT: High feature stability detected. REPRESENTATION: OK.")
    else:
        print("RESULT: Features are sensitive to noise. Training required.")

if __name__ == "__main__":
    check_feature_stability()
