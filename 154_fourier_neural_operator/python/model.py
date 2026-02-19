import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    """
    1D Fourier Layer. It computes the 1D Discrete Fourier Transform, filters 
    high-frequency market noise by keeping only the top `modes`, multiplies 
    a learnable weight matrix to the complex spectrum, and returns back 
    via Inverse Fourier Transform.
    """
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # Number of Fourier modes to multiply, at most length//2 + 1

        self.scale = (1 / (in_channels * out_channels))
        # Learnable complex weights for the frequency spectrum. 
        # Resolves structural PDE macros in the continuous domain.
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def complex_mult(self, input_spectrum, weights):
        """
        Multiply frequency components by our parametric weights.
        Complex multiplication via Einstein summation.
        input: (batch, in_channel, frequency_modes)
        weights: (in_channel, out_channel, frequency_modes)
        output: (batch, out_channel, frequency_modes)
        """
        return torch.einsum("bix,iox->box", input_spectrum, weights)

    def forward(self, x):
        batchsize, in_channels, seq_len = x.shape
        
        # 1. Forward 1D Fast Fourier Transform to the frequency domain (R -> C)
        # Using real-fft because financial time series are solely real numbers.
        x_ft = torch.fft.rfft(x)

        # Initialize the output frequency tensor with zeros
        out_ft = torch.zeros(
            batchsize, self.out_channels, x_ft.size(-1), 
            device=x.device, dtype=torch.cfloat
        )
        
        # 2. Extract lower modes, apply parametric complex weights
        # TRUNCATION TRICK: High frequencies (index > self.modes) remain completely zeroed out!
        # This completely erases unpredictable highly volatile micro-noise from the AI tracking.
        out_ft[:, :, :self.modes] = self.complex_mult(
            x_ft[:, :, :self.modes], 
            self.weights
        )

        # 3. Inverse 1D Fast Fourier Transform back to time domain (C -> R)
        x_filtered = torch.fft.irfft(out_ft, n=seq_len)
        return x_filtered

class FNO1dBlock(nn.Module):
    """
    Single Fourier Neural Operator Block combining spectral convolution, 
    local linear spatial transformation, and non-linear activation.
    """
    def __init__(self, d_model, modes):
        super(FNO1dBlock, self).__init__()
        self.spectral_conv = SpectralConv1d(d_model, d_model, modes)
        self.w_local = nn.Conv1d(d_model, d_model, 1) # Standard 1x1 map for local topology

    def forward(self, x):
        # x is (batch, channels, time)
        
        # Global Fourier mapping (extracts macro structures across entire time range)
        x_global = self.spectral_conv(x)
        
        # Local non-frequency mapping
        x_local = self.w_local(x)
        
        # Fusion
        return torch.nn.functional.gelu(x_global + x_local)

class FNOModel(nn.Module):
    """
    Complete Fourier Neural Operator architecture for 
    Financial Sequence mapping and PDE solving.
    """
    def __init__(self, in_features, d_model, out_features, modes, layers=3):
        super(FNOModel, self).__init__()
        self.p_map = nn.Linear(in_features, d_model)
        
        self.fno_blocks = nn.ModuleList([
            FNO1dBlock(d_model, modes) for _ in range(layers)
        ])
        
        self.q_map = nn.Linear(d_model, out_features)

    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        
        # Project channel dimensions
        x = self.p_map(x)
        
        # Transpose to (batch, channels, time) for 1D Convs and FFT
        x = x.permute(0, 2, 1)
        
        for block in self.fno_blocks:
            x = block(x)
            
        # Transpose back to (batch, time, channels)
        x = x.permute(0, 2, 1)
        
        # Final prediction projection (e.g. Next target volatility)
        x = self.q_map(x)
        return x

if __name__ == "__main__":
    print("Initializing FNO (Fourier Neural Operator) Model...")
    # Analyzing sequence length of 100 with 10 features, 
    # capturing the fundamental 16 modes (low frequencies)
    model = FNOModel(in_features=10, d_model=32, out_features=1, modes=16)
    
    # Simulating Batch, Sequence Length, Features
    dummy_input = torch.randn(64, 100, 10)
    print(f"Executing spectral pass on Time-Series Tensor: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Fourier output mapped successfully: {output.shape}")
    print("Model resolved Global Context without O(N^2) Softmax Attention!")
