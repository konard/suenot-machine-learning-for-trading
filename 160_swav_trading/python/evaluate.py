import torch
from model import SwAV

def eval_cluster_assignments():
    """
    Checks if SwAV avoids representation collapse by assigning distinct structural
    patterns into separate cluster Prototypes.
    """
    print("Running SwAV Clustering Stability Verification...")
    
    # We initialize an untrained SwAV model. In a real test, you'd load trained weights.
    # To simulate trained distinct continuous features, we will create heavily divergent
    # price action patterns.
    model = SwAV(n_prototypes=10)
    import os
    if os.path.exists("swav_model.pth"):
        model.load_state_dict(torch.load("swav_model.pth"))
        print("Loaded trained SwAV weights.")
    else:
        print("Warning: Running with UNTRAINED random weights.")
    model.eval()
    
    # Generate completely opposed market scenarios
    # Scenario 1: Straight upwards trend
    trend_up = torch.linspace(0, 5, 128).view(1, 1, 128).repeat(5, 1, 1)
    
    # Scenario 2: High frequency noise (Sideways chop)
    chop = torch.randn(5, 1, 128)
    
    with torch.no_grad():
        # Get continuous embeddings
        z_up = model.projector(model.encoder(trend_up))
        z_chop = model.projector(model.encoder(chop))
        
        # Normalize
        z_up = torch.nn.functional.normalize(z_up, dim=1, p=2)
        z_chop = torch.nn.functional.normalize(z_chop, dim=1, p=2)
        
        # Normalize Prototypes
        w = model.prototypes.weight.data
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        
        # Assign to nearest Prototype directly via Argmax
        # (Sinkhorn is only used during training for equal scaling!)
        scores_up = torch.mm(z_up, w.t())
        scores_chop = torch.mm(z_chop, w.t())
        
        clusters_up = torch.argmax(scores_up, dim=1)
        clusters_chop = torch.argmax(scores_chop, dim=1)
        
    print(f"Trend UP cluster assignments: {clusters_up.tolist()}")
    print(f"Chop cluster assignments: {clusters_chop.tolist()}")
    
    # If the network collapsed, both the Up Trend and Chop would fall into the exact same bin.
    # If they are distinct, SwAV successfully separated the market regimes.
    if clusters_up[0].item() != clusters_chop[0].item():
         print("RESULT: Distinct representations naturally fall into distinct Prototypes. COLLAPSE PREVENTED.")
    else:
         print("RESULT: Warning: Inputs were assigned to the exact same prototype. Potential representation collapse.")

if __name__ == "__main__":
    eval_cluster_assignments()
