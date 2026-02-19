import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSelectiveScan(nn.Module):
    """
    A simplified single-direction Selective Scan mechanism mimicking 
    the core of the Mamba architecture.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state
        self.proj_a = nn.Linear(d_model, d_state)
        self.proj_x = nn.Linear(d_model, d_state)
        self.proj_out = nn.Linear(d_state, d_model)
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        states = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            # Simulating data-dependent selection mechanism for 'A' (forget)
            A_t = torch.sigmoid(self.proj_a(xt)) 
            # Bx is the input projection
            B_xt = self.proj_x(xt)
            
            # Selective continuous-time update h_t = A*h_{t-1} + B*x_t
            states = states * A_t + B_xt
            
            outputs.append(self.proj_out(states))
            
        return torch.stack(outputs, dim=1)

class BidirectionalMambaBlock(nn.Module):
    """
    Vision Mamba inspired Bidirectional SSM Block.
    Processes sequences Forward and Backward, capturing intricate correlations
    without any data leakage if bounded strictly inside a historical window.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        # Two entirely distinct selective scan paths
        self.forward_scan = SimpleSelectiveScan(d_model, d_state)
        self.backward_scan = SimpleSelectiveScan(d_model, d_state)
        
        self.norm = nn.LayerNorm(d_model)
        # Fusion projection combining both semantic directions
        self.out_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x):
        # 1. Forward Pass
        fwd_out = self.forward_scan(x)
        
        # 2. Backward Pass
        # Flip sequence dimension (dim=1) to process from end to start
        x_reversed = torch.flip(x, dims=[1])
        bwd_out_reversed = self.backward_scan(x_reversed)
        # Flip back to chronologically align with the forward pass
        bwd_out = torch.flip(bwd_out_reversed, dims=[1])
        
        # 3. Fusion
        # The hidden representations contain context from the entire sequence
        fused = torch.cat([fwd_out, bwd_out], dim=-1)
        out = self.out_proj(fused)
        
        return self.norm(x + out) # Residual connection

if __name__ == "__main__":
    print("Initializing Bidirectional Mamba Architecture...")
    model = BidirectionalMambaBlock(d_model=64)
    print("Model Architecture Built Successfully.")
    
    # Batch size 32, Sequence Length (Lookback Window) 100, Features 64
    dummy_input = torch.randn(32, 100, 64)
    print(f"Feeding sequential LOB window (batch, seq, features): {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Fused Output shape: {output.shape} (Processed bidirectionally in O(N))")
