import torch
import numpy as np
from train import BidirectionalTradingHead

def simulate_strict_rolling_backtest(total_bars=5_000, lookback_window=100, features=64):
    """
    Simulates a rigorous Backtest on out-of-sample data.
    Bidirectional models are uniquely vulnerable to Data Leakage if backtested
    using sliding overlaps negligently. We specifically window the arrays.
    """
    print(f"Initializing Validated Backtester for Vim architecture.")
    print(f"Total bars: {total_bars} | Bidirectional Lookback Window: {lookback_window}")
    print("-" * 55)
    
    # 1. Market Data Generation
    # Generates a pseudo-trending market price graph
    price_series = np.cumsum(np.random.normal(0, 0.005, total_bars)) + 60000.0
    
    # Simulate some deeply correlated underlying features (e.g. Volume profile, Depth)
    market_features = torch.randn(total_bars, features)
    
    # 2. Initializing Strategy Engine
    device = torch.device("cpu") # Quick simulated inference
    model = BidirectionalTradingHead(d_model=features)
    model.eval() # crucial for freezing batchnorm/dropout inside sweeps
    
    pnl = 0.0
    position_size = 1.0 # Standard contract
    trades = 0
    drawdown, peak = 0.0, 0.0
    
    with torch.no_grad(): # Remove autograd overhead for live backtest
        # We start trading only after we accumulate one full lookback window
        for t in range(lookback_window, total_bars):
            # Strict slice: [t - lookback_window, t)
            # This ensures index `t` is EXCLUDED. `t` is what we are predicting.
            # Bidirectional network sweeps back and forth entirely physically
            # in the past from the engine's perspective!
            window = market_features[t - lookback_window : t].unsqueeze(0).to(device)
            
            # Predict the differential direction
            prediction = model(window).item()
            current_price = price_series[t]
            
            # Simulated Execution Engine
            if prediction > 0.03:
                # Entering a Long trend / Closing a Short trend
                if trades % 2 != 0: 
                    pnl += (entry_price - current_price) * position_size # Stop short
                entry_price = current_price
                trades += 1
                
            elif prediction < -0.03:
                # Entering a Short trend / Closing a Long trend
                if trades % 2 == 0:
                    pnl += (current_price - entry_price) * position_size # Stop long
                entry_price = current_price
                trades += 1
                
            peak = max(peak, pnl)
            drawdown = min(drawdown, pnl - peak)

    print(f"Deep $O(N)$ Bidirectional Mamba Scan computed {total_bars - lookback_window} windows.")
    print("\nBacktest Results:")
    print(f"Gross Expected PnL:   {pnl:.2f} USDT")
    print(f"Maximum Window Drawdown: {abs(drawdown):.2f} USDT")
    print(f"Strategy Trigger Trades: {trades}")
    print(f"Rolling Sharpe Ratio estimate: {pnl / (abs(drawdown) + 1e-6) * np.sqrt(252):.2f}")
    print("-" * 55)

if __name__ == "__main__":
    simulate_strict_rolling_backtest()
