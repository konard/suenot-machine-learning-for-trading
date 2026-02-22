import torch
import numpy as np

class SecureMasker:
    """
    Handles additive masking and unmasking for Secure Aggregation.
    """
    def __init__(self, client_id):
        self.client_id = client_id
        # In a real system, these would be derived from Diffie-Hellman key exchanges
        self.shared_seeds = {} 

    def set_shared_seed(self, other_client_id, seed):
        self.shared_seeds[other_client_id] = seed

    def mask_weights(self, weights):
        """
        Apply pairwise masks to the state_dict.
        For each pair (i, j), client i adds noise and client j subtracts it.
        """
        masked_weights = {k: v.clone() for k, v in weights.items()}
        
        for other_id, seed in self.shared_seeds.items():
            # Deterministic RNG based on shared seed
            torch.manual_seed(seed)
            for k, v in masked_weights.items():
                noise = torch.randn(v.shape) * 10.0 # High noise to prove blinding
                if self.client_id < other_id:
                    # We are the "adder"
                    masked_weights[k] += noise
                else:
                    # We are the "subtractor"
                    masked_weights[k] -= noise
                    
        return masked_weights

class SecureAggregator:
    """
    Server-side component that sums blinded updates.
    """
    def aggregate(self, blinded_weights_list):
        """
        Sums the weights. If all clients participated, masks cancel out.
        """
        num_clients = len(blinded_weights_list)
        summed_weights = {}
        
        for k in blinded_weights_list[0].keys():
            # Sum blinded parameters
            layer_sum = torch.stack([w[k] for w in blinded_weights_list], dim=0).sum(dim=0)
            # Average
            summed_weights[k] = layer_sum / num_clients
            
        return summed_weights
