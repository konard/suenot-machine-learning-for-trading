import torch
import torch.nn.functional as F

class HardNegativeMiner:
    """
    Utility to identify 'hard' negative samples within a batch based on cosine similarity.
    """
    def __init__(self, top_k=5):
        self.top_k = top_k

    def find_hard_negatives(self, v_anchor, v_candidates):
        """
        For each anchor in v_anchor, find the Top-K most similar items in v_candidates 
        that are NOT the positive match (assuming identity diagonal).
        
        Args:
            v_anchor: Tensor of shape (batch, dim)
            v_candidates: Tensor of shape (batch, dim)
            
        Returns:
            Indices of the hardest negatives for each anchor: (batch, top_k)
        """
        # Ensure normalization for cosine similarity
        v_anchor = F.normalize(v_anchor, p=2, dim=1)
        v_candidates = F.normalize(v_candidates, p=2, dim=1)
        
        # Sim matrix: (batch, batch)
        sim_matrix = v_anchor @ v_candidates.T
        
        batch_size = v_anchor.size(0)
        
        # Mask out the diagonal (positive pairs)
        # We fill the diagonal with a very small number so they aren't picked as 'hard negatives'
        mask = torch.eye(batch_size, device=v_anchor.device).bool()
        sim_matrix.masked_fill_(mask, -1.0)
        
        # Find Top-K similarities
        # we want the HIGHEST similarity negatives (the 'hardest')
        _, hard_indices = torch.topk(sim_matrix, k=min(self.top_k, batch_size - 1), dim=1, largest=True)
        
        return hard_indices

def mining_loss(v1, v2, miner, logit_scale):
    """
    Contrastive Loss using only the identified Hard Negatives.
    """
    v1 = F.normalize(v1, p=2, dim=1)
    v2 = F.normalize(v2, p=2, dim=1)
    
    batch_size = v1.size(0)
    device = v1.device
    
    # Identify hard negatives for each sample in v1 from v2
    hard_indices_v2 = miner.find_hard_negatives(v1, v2) # (batch, k)
    
    # Logits for positive pairs (diagonal)
    pos_logits = (v1 * v2).sum(dim=1) * logit_scale.exp() # (batch,)
    
    # Logits for hard negative pairs
    # For each i, we take the k hard negatives from v2
    # v1: (batch, dim) -> (batch, 1, dim)
    # v2_hard: (batch, k, dim)
    v2_hard = v2[hard_indices_v2]
    neg_logits = torch.bmm(v1.unsqueeze(1), v2_hard.transpose(1, 2)).squeeze(1) * logit_scale.exp() # (batch, k)
    
    # Combined logits: [pos, neg1, neg2, ..., negK]
    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1) # (batch, k+1)
    
    # Target is always index 0 (the positive pair)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    loss = F.cross_entropy(logits, labels)
    return loss
