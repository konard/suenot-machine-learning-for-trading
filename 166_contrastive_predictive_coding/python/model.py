import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder g_enc: 1D-CNN to extract local latent features from raw time-series windows.
    Each window x_t (e.g., 64 steps) is mapped to a latent vector z_t.
    """
    def __init__(self, input_dim=1, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim, window_size)
        z = self.conv(x)
        return z.squeeze(-1) # (batch_size, latent_dim)

class CPCModel(nn.Module):
    """
    CPC Model combining g_enc (Encoder) and g_ar (Autoregressive model).
    """
    def __init__(self, input_dim=1, latent_dim=128, context_dim=128, predict_steps=4):
        super(CPCModel, self).__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.predict_steps = predict_steps

        # g_enc: Frame encoder
        self.encoder = Encoder(input_dim, latent_dim)

        # g_ar: Autoregressive context model (GRU)
        self.gru = nn.GRU(latent_dim, context_dim, batch_first=True)

        # Predictors: W_k for each step k=1...predict_steps
        # maps c_t -> z_{t+k}
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, latent_dim) for _ in range(predict_steps)
        ])

    def forward(self, x_seq):
        """
        Args:
            x_seq: (batch_size, seq_len, input_dim, window_size)
        Returns:
            z_all: (batch_size, seq_len, latent_dim) - actual latents
            c_all: (batch_size, seq_len, context_dim) - context vectors
        """
        batch_size, seq_len, in_dim, win_size = x_seq.shape
        
        # Reshape to encode all windows at once
        x_flat = x_seq.view(batch_size * seq_len, in_dim, win_size)
        z_flat = self.encoder(x_flat)
        z_all = z_flat.view(batch_size, seq_len, self.latent_dim)

        # Pass latents through GRU to get context vectors
        c_all, _ = self.gru(z_all)

        return z_all, c_all

    def predict_future(self, c_t, step_k):
        """
        Predicts z_{t+k} from c_t using the k-th predictor.
        """
        return self.predictors[step_k - 1](c_t)
