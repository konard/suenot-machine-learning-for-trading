import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(z_all, c_all, model):
    """
    Computes InfoNCE loss for Contrastive Predictive Coding.
    For each c_t and each future step k, the model predicts z_{t+k}.
    
    Args:
        z_all: Actual latent vectors (batch, seq_len, latent_dim)
        c_all: Context vectors (batch, seq_len, context_dim)
        model: CPCModel instance
    """
    batch_size, seq_len, latent_dim = z_all.shape
    predict_steps = model.predict_steps
    device = z_all.device

    total_loss = 0
    count = 0

    # We iterate through time steps t where we have enough 'future' to compare
    # t ranges from 0 to (seq_len - predict_steps - 1)
    for k in range(1, predict_steps + 1):
        # Current contexts c_t
        # c_t: (batch, seq_len - k, context_dim)
        c_t = c_all[:, :-k, :]
        
        # Predicted latents z_hat_{t+k}: (batch, S, latent_dim)
        z_hat = model.predict_future(c_t, k)
        
        # Actual latents z_{t+k} (Positives): (batch, S, latent_dim)
        z_pos = z_all[:, k:, :]
        
        # We need to compare z_hat[b, s] with z_pos[b', s] for all b'
        # To do this efficiently, we permute to (S, batch, latent_dim)
        z_hat = z_hat.transpose(0, 1) # (S, batch, latent_dim)
        z_pos = z_pos.transpose(0, 1) # (S, batch, latent_dim)
        
        # Logits: (S, batch, batch)
        # For each time step s, we get a similarity matrix (batch, batch)
        logits = torch.bmm(z_hat, z_pos.transpose(1, 2)) # (S, batch, batch)
        
        # Flatten to (S * batch, batch)
        logits = logits.view(-1, batch_size)
        
        # Labels are 0, 1, ..., batch_size-1 for each time step
        # Labels shape: (S * batch,)
        labels = torch.arange(batch_size, device=device).repeat(c_t.size(1))
        
        loss = F.cross_entropy(logits, labels)
        total_loss += loss
        count += 1

    return total_loss / count
