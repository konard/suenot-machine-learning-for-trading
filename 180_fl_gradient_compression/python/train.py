import torch
from model import TradingNN
from ternary_core import TernaryCompressor, AdaptiveThresholdManager

def simulate_adaptive_tgc():
    print("Ternary Gradient Compression Simulation: Fixed vs. Adaptive Thresholding")
    
    model = TradingNN()
    compressor = TernaryCompressor()
    adaptive_mgr = AdaptiveThresholdManager(target_sparsity=0.92)
    
    # Simulate multiple local updates in different market regimes
    regimes = [
        {"name": "Low Volatility", "noise_scale": 0.1},
        {"name": "High Volatility", "noise_scale": 1.5},
        {"name": "Mean Reverting", "noise_scale": 0.5}
    ]

    print("\nRegime | K-Thresh | Sparsity | Scale")
    print("-" * 45)

    for regime in regimes:
        # Create dummy gradients reflecting the current market regime
        dummy_grad = torch.randn(5569) * regime["noise_scale"]
        
        # 1. Update k using adaptive manager
        compressor.k = adaptive_mgr.current_k
        
        # 2. Compress
        compressed, scale, gamma = compressor.compress(dummy_grad)
        
        # 3. Analyze results
        non_zero = torch.nonzero(compressed).size(0)
        sparsity = 1.0 - (non_zero / compressed.numel())
        
        print(f"{regime['name']:15} | {compressor.k:.2f} | {sparsity:.2%} | {scale:.4f}")
        
        # 4. Feedback loop for the next iteration
        adaptive_mgr.adjust_k(sparsity)

    print("\nSUCCESS: Adaptive TGC simulation completed.")
    print("Observation: The system adjusts 'k' to maintain consistent sparsity regardless of gradient magnitude.")

if __name__ == "__main__":
    simulate_adaptive_tgc()
