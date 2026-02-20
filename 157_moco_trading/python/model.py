import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DEncoder(nn.Module):
    """
    Base 1D-CNN Encoder for stock price windows.
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = self.conv_block(x)
        h = self.adaptive_pool(h).squeeze(-1)
        return h

class MoCo(nn.Module):
    """
    MoCo architecture: Query Encoder, Momentum Encoder, and a FIFO Queue.
    """
    def __init__(self, encoder_q, encoder_k, dim=64, K=4096, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 64)
        K: queue size; number of negative keys (default: 4096)
        m: momentum coefficient (default: 0.999)
        T: temperature (default: 0.07)
        """
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        # Create the encoders
        self.encoder_q = encoder_q # Query encoder
        self.encoder_k = encoder_k # Key encoder (Momentum)

        # Initialize the momentum encoder weights from the query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False # Not updated by backprop

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder: th_k = m*th_k + (1-m)*th_q
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the FIFO queue with new keys from the current batch.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr (handle wrap-around)
        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[:batch_size]

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K # Move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query patterns
            im_k: a batch of positive key patterns (augmented views)
        Output:
            logits, labels
        """
        # 1. Compute query features
        q = self.encoder_q(im_q) # (N, C)
        q = F.normalize(q, dim=1)

        # 2. Compute key features (momentum update)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k) # (N, C)
            k = F.normalize(k, dim=1)

        # 3. Compute logits
        # Einstein sum is 20x-100x faster than traditional matmul for this
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # (N, 1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) # (N, K)

        # Logits: (N, 1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T # Scale by temperature

        # Labels: positive samples are always at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        # 4. Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

if __name__ == "__main__":
    print("Initializing MoCo Engine for Financial Sequence Contrast...")
    q_enc = CNN1DEncoder()
    k_enc = CNN1DEncoder()
    model = MoCo(q_enc, k_enc, K=128) # Small queue for test
    
    im_q = torch.randn(8, 1, 128)
    im_k = torch.randn(8, 1, 128)
    
    logits, labels = model(im_q, im_k)
    print(f"Logits shape: {logits.shape} (N, 1+K)")
    print(f"Labels shape: {labels.shape}")
    print("MoCo architecture verified.")
