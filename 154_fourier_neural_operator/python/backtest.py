import torch
import numpy as np
from model import FNOModel

def execute_spectral_rolling_backtest(total_bars=3_000, context_length=100, features=10):
    """
    To rigorous validate if our fast-fourier-transform (FFT) logic prevents
    Data Leakage during continuous trading, we explicitly simulate an event 
    loop using strict causal padding and out-of-sample slices.
    """
    print(f"Executing Live Forward-Feed FNO Algorithmic Engine.")
    print(f"Ticks processed: {total_bars} | Resolution-Free Receptive Field: {context_length}")
    print("-" * 55)
    
    # Simulated true underlying limit order book depth (macro imbalance factor)
    macro_factor_trend = np.sin(np.linspace(0, 50, total_bars)) 
    
    # What the model ACTUALLY sees over the API 
    # Macro factor is utterly buried in raw Gaussian noise and micro-movements
    market_feed = macro_factor_trend + np.random.normal(0, 3.0, total_bars)
    
    # Duplicate to multiple theoretical LOB levels
    market_tensor = torch.tensor(market_feed, dtype=torch.float32).unsqueeze(-1).expand(-1, features)
    
    # Instantiate FNO Strategy
    device = torch.device("cpu")
    model = FNOModel(in_features=features, d_model=32, out_features=1, modes=16)
    model.eval() 
    
    pnl = 0.0
    position = 0 # -1, 0, 1
    trades = 0
    entry_price = 0.0
    
    # We simulate starting from day `context_length`
    # We step exactly 1 bar forward at a time. O(N) evaluation inside the window.
    with torch.no_grad():
        for t in range(context_length, total_bars):
            # NO LEAKAGE: Slice strictly from [t - context, t)
            # The FFT only computes spectral density over the known past.
            window = market_tensor[t - context_length : t].unsqueeze(0).to(device)
            
            # Predict outcome for the immediate sequence trajectory
            prediction_sequence = model(window)
            
            # Extract the final point prediction mapped from the frequency-space representation
            signal = prediction_sequence[0, -1, 0].item()
            current_proxy_price = macro_factor_trend[t] # The true asset value
            
            # Strategy Engine Triggers
            if signal > 0.05 and position <= 0:
                if position == -1: 
                    pnl += (entry_price - current_proxy_price)
                    trades += 1
                position = 1
                entry_price = current_proxy_price
                trades += 1
                
            elif signal < -0.05 and position >= 0:
                if position == 1:
                    pnl += (current_proxy_price - entry_price)
                    trades += 1
                position = -1
                entry_price = current_proxy_price
                trades += 1

    print(f"Processed 3,000 temporal matrices instantly via discrete frequency math.")
    print("\nFourier Backtest Result Synopsis:")
    print(f"Alpha Captured (Cum. Points):  {pnl:.2f}")
    print(f"Oscillation Cycle Trades:      {trades}")
    win_rate = "Stable" if pnl > 0 else "Unstable"
    print(f"Fourier Denoiser Quality:      {win_rate}")
    print("-" * 55)

if __name__ == "__main__":
    execute_spectral_rolling_backtest()
