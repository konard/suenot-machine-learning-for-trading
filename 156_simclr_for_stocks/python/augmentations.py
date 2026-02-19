import torch
import numpy as np

def jitter(x, sigma=0.03):
    """
    Adds Gaussian noise to the time series.
    """
    return x + torch.normal(mean=0, std=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    """
    Multiplies the sequence by a random scalar. 
    Simulates changes in volatility/magnitude.
    """
    factor = torch.randn(x.shape[0], 1, 1) * sigma + 1.0
    return x * factor

def permutation(x, max_segments=5):
    """
    Randomly permutes segments of the time series.
    """
    batch_size, channels, seq_len = x.shape
    x_new = torch.zeros_like(x)
    num_segments = np.random.randint(2, max_segments + 1)
    
    seg_size = seq_len // num_segments
    for i in range(batch_size):
        idx = np.random.permutation(num_segments)
        for j, pos in enumerate(idx):
            x_new[i, :, j*seg_size:(j+1)*seg_size] = x[i, :, pos*seg_size:(pos+1)*seg_size]
    return x_new

def masking(x, mask_ratio=0.15):
    """
    Randomly zeros out segments of the time series.
    """
    batch_size, channels, seq_len = x.shape
    mask = torch.rand(batch_size, 1, seq_len) > mask_ratio
    return x * mask.float()

def apply_simclr_augmentations(x):
    """
    Applies a random combination of augmentations to create 
    a "view" of the data.
    """
    # Pick randomly between different transformation types
    choice = np.random.rand()
    if choice < 0.25:
        return jitter(x)
    elif choice < 0.5:
        return scaling(x)
    elif choice < 0.75:
        return permutation(x)
    else:
        return masking(x)

if __name__ == "__main__":
    dummy_x = torch.randn(8, 1, 128) # (Batch, Features, Time)
    aug1 = apply_simclr_augmentations(dummy_x)
    aug2 = apply_simclr_augmentations(dummy_x)
    print(f"Original shape: {dummy_x.shape}")
    print(f"Augmentation 1 shape: {aug1.shape}")
    print(f"Augmentation 2 shape: {aug2.shape}")
