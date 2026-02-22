import torch

class TernaryCompressor:
    """
    Implements Ternary Gradient Compression with Adaptive Thresholding.
    Weights are quantized into {-1, 0, +1}.
    """
    def __init__(self, initial_k=0.7):
        self.k = initial_k # Multiplier for standard deviation

    def compress(self, tensor):
        """
        Quantizes tensor into ternary values using std-based threshold.
        """
        std = tensor.std()
        gamma = self.k * std
        
        # Determine scaling factor (mean of values exceeding threshold)
        mask_pos = tensor > gamma
        mask_neg = tensor < -gamma
        
        # Calculate mean magnitude of non-zero elements
        non_zero_vals = tensor[mask_pos | mask_neg]
        if non_zero_vals.numel() > 0:
            scale = non_zero_vals.abs().mean()
        else:
            scale = tensor.abs().mean()
            
        compressed = torch.zeros_like(tensor)
        compressed[mask_pos] = 1.0
        compressed[mask_neg] = -1.0
        
        return compressed, scale, gamma

class AdaptiveThresholdManager:
    """
    Dynamically adjusts the k-parameter for TGC based on bandwidth targets
    or convergence signals.
    """
    def __init__(self, target_sparsity=0.9):
        self.target_sparsity = target_sparsity
        self.current_k = 0.5

    def adjust_k(self, current_sparsity):
        """
        Simple feedback loop to stay near target sparsity.
        If sparsity is too low (too much data), increase k.
        """
        if current_sparsity < self.target_sparsity:
            self.current_k += 0.05
        else:
            self.current_k -= 0.05
            
        self.current_k = max(0.1, min(5.0, self.current_k))
        return self.current_k
