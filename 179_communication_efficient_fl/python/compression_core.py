import torch

class GradientCompressor:
    """
    Implements quantization and sparsification for federated updates.
    """
    @staticmethod
    def sign_quantize(tensor):
        """
        1-bit quantization (SignSGD).
        Returns the signs (+1 or -1) and the mean absolute value for scaling.
        """
        magnitude = tensor.abs().mean()
        compressed = torch.sign(tensor)
        return compressed, magnitude

    @staticmethod
    def top_k_sparsify(tensor, k_ratio=0.1):
        """
        Sparsifies the tensor by keeping only the Top-K largest elements.
        """
        k = max(1, int(tensor.numel() * k_ratio))
        values, indices = torch.topk(tensor.view(-1).abs(), k)
        
        # Create sparse mask
        mask = torch.zeros_like(tensor.view(-1))
        mask[indices] = tensor.view(-1)[indices]
        
        return mask.view_as(tensor)

    @staticmethod
    def decompress_sign(compressed, magnitude):
        """
        Reconstructs the tensor from 1-bit signs.
        """
        return compressed * magnitude

class ErrorFeedbackManager:
    """
    Manages residual error accumulation to prevent information loss during compression.
    """
    def __init__(self, model):
        self.residuals = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
        }

    def compensate_and_compress(self, name, gradient, k_ratio=0.1):
        """
        Adds previous error to current gradient, compresses, and updates error.
        """
        # 1. Add residual error
        v = gradient + self.residuals[name]
        
        # 2. Compress (using Top-K for this demo)
        compressed = GradientCompressor.top_k_sparsify(v, k_ratio=k_ratio)
        
        # 3. Update residual
        self.residuals[name] = v - compressed
        
        return compressed
