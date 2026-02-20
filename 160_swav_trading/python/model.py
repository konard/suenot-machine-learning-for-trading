import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DEncoder(nn.Module):
    """
    Base 1D-CNN Encoder for stock price windows.
    Returns representation y.
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

class MLPProjector(nn.Module):
    """
    Projection head mapping representations to the sphere for prototype matching.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def sinkhorn_knopp(scores, epsilon=0.05, n_iters=3):
    """
    Runs the Sinkhorn-Knopp algorithm for Online Optimal Transport over a batch.
    Args:
        scores: (N, K) matrix, N=batch_size, K=number_of_prototypes
        epsilon: Entropy regularization (temperature)
        n_iters: Number of iterations (usually 3 is natively enough)
    Returns:
        Q: (N, K) The discrete/soft cluster assignments with enforced equipartition constraint
    """
    # For numerical stability, subtract the max
    scores = scores - torch.max(scores, dim=0, keepdim=True)[0]
    Q = torch.exp(scores / epsilon).t() # (K, N)
    B = Q.shape[1]
    K = Q.shape[0]

    # Normalize matrix to make it a joint probability distribution
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(n_iters):
        # 1. Normalize Rows (Prototypes constraint: Each cluster should have equal mass B/K)
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # 2. Normalize Columns (Batch constraint: Each sample sum to 1.0/B)
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= sum_of_cols
        Q /= B

    Q *= B # Returns to assignment probabilities per sample
    return Q.t() # (N, K)

class SwAVLoss(nn.Module):
    def __init__(self, temperature=0.1, sinkhorn_epsilon=0.05, sinkhorn_iters=3):
        super().__init__()
        self.tau = temperature
        self.epsilon = sinkhorn_epsilon
        self.iters = sinkhorn_iters

    def forward(self, z1, z2, prototypes):
        """
        Swapped Prediction Loss.
        z1, z2: (Batch, Dim) L2 normalized embeddings
        prototypes: (Dim, K) L2 normalized weight matrix
        """
        # 1. Calculate similarities (Dot products on L2 normalized vectors = Cosine Similarity)
        scores1 = torch.mm(z1, prototypes) # (N, K)
        scores2 = torch.mm(z2, prototypes) # (N, K)
        
        # 2. Compute Sinkhorn-Knopp cluster assignments (q targets)
        q1 = sinkhorn_knopp(scores1, epsilon=self.epsilon, n_iters=self.iters)
        q2 = sinkhorn_knopp(scores2, epsilon=self.epsilon, n_iters=self.iters)
        
        # 3. Swap predictability - View 1 predicts View 2's clusters, and vice versa.
        # Softmax is applied via temperature parameter tau over continuous scores.
        # Loss is cross entropy over the soft labels q.
        
        p1 = F.log_softmax(scores1 / self.tau, dim=1)
        p2 = F.log_softmax(scores2 / self.tau, dim=1)
        
        loss_swap1 = -torch.sum(q2 * p1, dim=1).mean()
        loss_swap2 = -torch.sum(q1 * p2, dim=1).mean()
        
        return (loss_swap1 + loss_swap2) / 2.0

class SwAV(nn.Module):
    def __init__(self, base_encoder_dim=64, projection_dim=128, hidden_dim=256, n_prototypes=10, temperature=0.1):
        super().__init__()
        self.encoder = CNN1DEncoder(hidden_dim=base_encoder_dim)
        self.projector = MLPProjector(base_encoder_dim, hidden_dim, projection_dim)
        
        # Prototypes matrix (columns are unit vectors representing cluster centers)
        self.prototypes = nn.Linear(projection_dim, n_prototypes, bias=False)
        self.criterion = SwAVLoss(temperature=temperature)
        
    def forward(self, v1, v2):
        """
        Forward pass providing the representations and calculating the Swapped prediction loss.
        """
        # Get projected continuous embeddings
        z1 = self.projector(self.encoder(v1))
        z2 = self.projector(self.encoder(v2))
        
        # L2 Normalize the continuous embeddings before passing to Prototypes
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        
        # L2 Normalize the Prototypes weights
        with torch.no_grad():
            w = self.prototypes.weight.data
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # In SwAV standard implementation, you extract the active weights as (D, K)
        proto_w = self.prototypes.weight.t()
        
        loss = self.criterion(z1, z2, proto_w)
        return loss

if __name__ == "__main__":
    print("Testing SwAV Architecture and Sinkhorn-Knopp Optimal Transport...")
    model = SwAV(n_prototypes=10) # Testing with 10 cluster bins ("Postal sorter bins")
    x1 = torch.randn(128, 1, 128) # Batch size >= Prototypes is highly recommended for Sinkhorn
    x2 = torch.randn(128, 1, 128)
    loss = model(x1, x2)
    print(f"Total Loss forward pass: {loss.item():.4f}")
    assert loss.requires_grad == True
    print("SwAV initialized correctly. Sinkhorn-Knopp Equipartition operational.")
