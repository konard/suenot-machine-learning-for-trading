import torch

class DPSGDManager:
    """
    Implements core DP-SGD mechanisms: Gradient Clipping and Noise Injection.
    """
    def __init__(self, l2_norm_clip=1.0, noise_multiplier=0.1):
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier

    def clip_gradients(self, parameters):
        """
        Clips gradients of the parameters to the maximum L2 norm.
        In a real DP setup, this should be done per-sample, but here we 
        demonstrate the principle on the aggregated gradient for simplicity.
        """
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.l2_norm_clip)

    def add_noise(self, parameters):
        """
        Adds Gaussian noise to gradients to provide DP.
        Noise scale is proportional to (l2_norm_clip * noise_multiplier).
        """
        for param in parameters:
            if param.grad is not None:
                # Add noise: ~ N(0, (sigma * C)^2)
                noise = torch.randn_like(param.grad) * (self.l2_norm_clip * self.noise_multiplier)
                param.grad += noise

    def apply_dp_step(self, optimizer, model_parameters):
        """
        Full DP step: Clip -> Noise -> Step
        """
        self.clip_gradients(model_parameters)
        self.add_noise(model_parameters)
        optimizer.step()
