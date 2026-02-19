import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttentionSSMCell(nn.Module):
    """
    The core connection between Transformers and State Space Models (SSM).
    This cell demonstrates the theoretical concept of Structured State Space Duality (SSD)
    derived from the 2024 paper "Transformers are SSMs".
    
    Instead of calculating O(N^2) Softmax attention, Neural Networks can compute 
    S_t = A * S_{t-1} + K_t^T V_t (where A is a decay/forgetting gate).
    """

    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Projections to Query, Key, Value vectors similar to a standard Transformer
        self.q_proj = nn.Linear(d_model, d_state)
        self.k_proj = nn.Linear(d_model, d_state)
        self.v_proj = nn.Linear(d_model, d_state)
        self.o_proj = nn.Linear(d_state, d_model)
        
        # Log-decay parameters representing the 'A' matrix in continuous SSMs
        # Controls the forgetting mechanism of the recurrent state
        self.log_decay = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)
        
    def forward(self, x, state=None):
        """
        Recurrent RNN-style forward pass for O(1) inference matching high-frequency trading.
        x: (batch, seq_len, d_model)
        state: Initial hidden matrix (batch, d_state, d_state)
        """
        batch_size, seq_len, _ = x.shape
        
        q = F.elu(self.q_proj(x)) + 1.0 # Positive feature mapping phi(Q)
        k = F.elu(self.k_proj(x)) + 1.0 # Positive feature mapping phi(K)
        v = self.v_proj(x)
        
        # Exponential moving average decay parameter from state space theory
        decay = torch.exp(self.log_decay).unsqueeze(0).unsqueeze(-1)
        
        if state is None:
            # The "summary notebook" or State Matrix S_t
            state = torch.zeros(batch_size, self.d_state, self.d_state, device=x.device)
            
        outputs = []
        
        # Unroll over sequence length
        for t in range(seq_len):
            qt = q[:, t, :]  # (batch, d_state)
            kt = k[:, t, :]  # (batch, d_state)
            vt = v[:, t, :]  # (batch, d_state)
            
            # 1. Update the state matrix (SSM continuous time update equivalent)
            # S_t = S_{t-1} * decay + K_t^T * V_t
            # where decay corresponds to e^(A * delta_t) in SSMs
            state = state * decay + torch.bmm(kt.unsqueeze(2), vt.unsqueeze(1))
            
            # 2. Extract output
            # O_t = Q_t * S_t
            ot = torch.bmm(qt.unsqueeze(1), state).squeeze(1)
            outputs.append(ot)
            
        # Stack the sequence outputs and project back to dimensions
        out = torch.stack(outputs, dim=1)
        return self.o_proj(out), state

class GatedLinearAttention(nn.Module):
    """
    High-level module incorporating LayerNorms, residual connections, and 
    Feed Forward networks wrapped around LinearAttentionSSMCell.
    Suitable for forecasting on multidimensional TimeSeries sets.
    """
    def __init__(self, d_model, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': LinearAttentionSSMCell(d_model),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model*4),
                    nn.SiLU(),
                    nn.Linear(d_model*4, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(layers)
        ])

    def forward(self, x):
        h = x
        for layer in self.layers:
            attn_out, _ = layer['attn'](layer['norm1'](h))
            h = h + attn_out
            ffn_out = layer['ffn'](layer['norm2'](h))
            h = h + ffn_out
        return h

if __name__ == "__main__":
    print("Initializing Linear Attention SSM (Transformers are SSMs equivalent)...")
    model = GatedLinearAttention(d_model=128)
    
    # Simulating a batch size of 32 incoming 100-step sequences 
    # capturing 128 features (e.g. Limit order book snapshots)
    dummy_input = torch.randn(32, 100, 128)
    print(f"Feeding sequential LOB tensor of dimensions {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Generated Output shape: {output.shape} (Matches input dimension!)")
    print("Linear Attention computed in O(N) sequence time with hidden continuous decay!")
