import torch
import torch.nn as nn

class LinearAttentionSSM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # A simple linear attention/SSM approximation block
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        x: shape (batch_size, sequence_length, d_model)
        Returns output of same shape
        """
        return self.proj(x)

if __name__ == "__main__":
    model = LinearAttentionSSM(64)
    dummy_input = torch.randn(2, 10, 64)
    output = model(dummy_input)
    print("LinearAttentionSSM Model Output Shape:", output.shape)
