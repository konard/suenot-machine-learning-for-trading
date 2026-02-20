import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DEncoder(nn.Module):
    """
    Standard 1D-CNN Encoder for capturing temporal patterns.
    Outputs a continuous representation vector z_t.
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

class TemporalContrastiveModel(nn.Module):
    """
    Combines the Encoder and a Projection Head for TCL.
    Projector maps embeddings to a space where InfoNCE is applied.
    """
    def __init__(self, base_encoder_dim=64, projection_dim=64):
        super().__init__()
        self.encoder = CNN1DEncoder(hidden_dim=base_encoder_dim)
        
        # 2-layer MLP Projector (Removed BatchNorm to prevent mode collapse across batch)
        self.projector = nn.Sequential(
            nn.Linear(base_encoder_dim, base_encoder_dim),
            nn.ReLU(),
            nn.Linear(base_encoder_dim, projection_dim)
        )

    def forward(self, x):
        # L2 Normalize the raw embedding (z) as well for better downstream distance metrics
        z_raw = self.encoder(x)
        z = F.normalize(z_raw, p=2, dim=1)
        
        # Project and then L2 normalize for InfoNCE Cosine Similarity
        # Removing L2 normalization on Z to give the encoder more representational freedom,
        # but KEEPING it on V for the InfoNCE cosine distance.
        v_raw = self.projector(z_raw)
        v = F.normalize(v_raw, p=2, dim=1)
        
        return z_raw, v

class TemporalInfoNCELoss(nn.Module):
    """
    InfoNCE Loss adapted for Temporal Contrastive Learning.
    Assumes the batch is structured as: [Batch_Window_T, Batch_Window_T_plus_1]
    (i.e., v1 and v2 are pairs of temporally adjacent windows).
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, v1, v2):
        batch_size = v1.size(0)
        
        # Concatenate all views: shape (2*N, D)
        views = torch.cat([v1, v2], dim=0)
        
        # Cosine similarity matrix: (2*N, 2*N)
        sim_matrix = torch.matmul(views, views.T) / self.temperature
        
        # Mask to remove self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=v1.device)
        sim_matrix.masked_fill_(mask, -9e15)

        # Create labels: 
        # For view i in [0, N-1], its positive is i+N
        # For view i+N in [N, 2N-1], its positive is i
        labels = torch.arange(batch_size, device=v1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Standard Cross-Entropy over the similarity matrix
        # sim_matrix is (2N, 2N)
        # However, for each view i, we only want to pull it to its positive i+N (or i-N)
        # And push away all OTHER views.
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
