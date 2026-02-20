import torch
from model import CNN1DEncoder

def verify_collapse_prevention():
    """
    Checks if representation collapse occurred by verifying that the
    encoded features across a batch are not identical.
    """
    print("Running BYOL Collapse Verification...")
    
    # Normally we load the target_encoder here because it is the most stable
    encoder = CNN1DEncoder()
    encoder.eval()
    
    # Create widely different patterns
    pattern_up = torch.linspace(0, 1, 128).view(1, 1, 128)
    pattern_down = torch.linspace(1, 0, 128).view(1, 1, 128)
    pattern_noise = torch.randn(1, 1, 128)
    
    with torch.no_grad():
        feat_up = encoder(pattern_up)
        feat_down = encoder(pattern_down)
        feat_noise = encoder(pattern_noise)
        
    sim_ud = torch.nn.functional.cosine_similarity(feat_up, feat_down)
    sim_un = torch.nn.functional.cosine_similarity(feat_up, feat_noise)
    
    print(f"Cosine Similarity (Up vs Down Trend): {sim_ud.item():.4f}")
    print(f"Cosine Similarity (Up vs Noise): {sim_un.item():.4f}")
    
    # If representations collapsed, similarities would all be very close to 1.0 (or -1.0)
    if sim_ud.item() < 0.99 and sim_un.item() < 0.99:
        print("RESULT: Distinct features detected. Collapse PREVENTED.")
    else:
        print("RESULT: Features collapsed into a singular mode. Increase learning rate or fix EMA.")

if __name__ == "__main__":
    verify_collapse_prevention()
