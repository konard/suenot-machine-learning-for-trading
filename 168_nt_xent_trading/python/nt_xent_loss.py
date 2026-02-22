import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    Standard loss for SimCLR and modern contrastive learning.
    Reference: https://arxiv.org/abs/2002.05709
    """
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-8

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: [batch_size, dim] - first augmented view
            z_j: [batch_size, dim] - second augmented view
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # 1. Normalize embeddings to unit hypersphere
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        # 2. Combined representations
        # Shape: [2 * batch_size, dim]
        features = torch.cat([z_i, z_j], dim=0)
        
        # 3. Similarity matrix
        # Shape: [2*batch_size, 2*batch_size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 4. Mask self-similarities
        mask = torch.eye(2 * batch_size, device=device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
        
        # 5. Targets (positives)
        # For sample i, the positive is at i + batch_size
        # For sample i + batch_size, the positive is at i
        targets = torch.arange(2 * batch_size, device=device)
        targets = (targets + batch_size) % (2 * batch_size)
        
        # 6. Standard Cross Entropy Loss
        loss = F.cross_entropy(similarity_matrix, targets)
        
        return loss
