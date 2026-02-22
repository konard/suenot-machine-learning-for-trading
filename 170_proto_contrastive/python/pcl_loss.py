import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNCELoss(nn.Module):
    """
    ProtoNCE Loss: Contrastive loss against prototypes.
    """
    def __init__(self, temperature=0.1):
        super(ProtoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, labels, concentrations):
        """
        Args:
            features: [batch_size, dim] - Current sample embeddings
            prototypes: [num_clusters, dim] - Cluster centroids
            labels: [batch_size] - Indices of prototypes for each sample
            concentrations: [num_clusters] - Density estimate (phi) for each cluster
        """
        # 1. Similarity to all prototypes
        # (batch, clusters)
        logits = torch.mm(features, prototypes.t())
        
        # 2. Scale by concentration/temperature
        # Note: In PCL, Each cluster can have a different temperature (concentration)
        # We broadcast concentrations to match logits
        temp = concentrations.unsqueeze(0) # (1, clusters)
        logits = logits / temp
        
        # 3. Cross Entropy against the correct prototype label
        loss = F.cross_entropy(logits, labels)
        
        return loss

def info_nce_loss(z_i, z_j, temperature=0.07):
    """Instance-level contrastive loss (InfoNCE/NT-Xent)"""
    batch_size = z_i.shape[0]
    device = z_i.device
    
    # Combined representations: [2*B, D]
    features = torch.cat([z_i, z_j], dim=0)
    
    # Similarity matrix: [2*B, 2*B]
    logits = torch.matmul(features, features.T) / temperature
    
    # Mask out self-contrast
    mask = torch.eye(2 * batch_size, device=device).bool()
    logits = logits.masked_fill(mask, -1e9)
    
    # Targets
    targets = torch.arange(2 * batch_size, device=device)
    targets = (targets + batch_size) % (2 * batch_size)
    
    return F.cross_entropy(logits, targets)
