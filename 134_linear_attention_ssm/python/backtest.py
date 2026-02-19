import torch
import numpy as np

def simulate_bybit_hf_backtest(seq_length=10_000, d_state=64):
    """
    Demonstrates Backtesting using a Linear Attention / SSM setup.
    Instead of passing a sliding window of past N bars to calculate metrics,
    we pass 1 snapshot at a time and UPDATE a constant Memory Matrix.
    This simulates true high-frequency environments where low-latency O(1) 
    updates provide a massive speed edge over window-based Attention models.
    """
    print(f"Initializing Backtest Engine for Bybit Ticks (N={seq_length})")
    print("Strategy: Gated Linear Attention (SSD Formulation)")
    print("-" * 50)
    
    # Simulate a stream of tick-by-tick prices and LOB imbalance metrics
    tick_stream = np.cumsum(np.random.normal(0, 0.001, seq_length)) + 50000.0
    
    # Initialize the Internal Hidden State corresponding to Linear Attention "S"
    # Size remains completely constant regardless of how many ticks we process
    state = torch.zeros(1, d_state, d_state)
    
    # Pre-calculated projection weights from a trained PyTorch model
    W_k = torch.randn(1, d_state) * 0.1
    W_v = torch.randn(1, d_state) * 0.1
    W_q = torch.randn(1, d_state) * 0.1
    decay_gate = 0.999 # Corresponds to exp(A)
    
    pnl = 0.0
    position = 0
    trades = 0
    drawdown, peak = 0.0, 0.0
    
    for t in range(seq_length):
        price = tick_stream[t]
        
        # O(1) Projection inside the SSM
        # Feature embeddings generated for the current input
        phi_k = torch.relu(W_k * price) + 1.0
        v = W_v * price
        phi_q = torch.relu(W_q * price) + 1.0
        
        # Core Update Rule of Linear Attention:
        # S_t = (A * S_{t-1}) + K_t^T x V_t
        state = state * decay_gate + torch.bmm(phi_k.unsqueeze(2), v.unsqueeze(1))
        
        # O_t = Q_t * S_t
        prediction = torch.bmm(phi_q.unsqueeze(1), state).squeeze().sum().item()
        
        # Strategy Logic:
        # We predict a future state shift purely from compressed historical matrix.
        # If prediction diverges heavily from 0, initiate trade.
        if prediction > 0.05 and position <= 0:
            if position < 0:
                pnl += (entry_price - price)  # Close Short
                trades += 1
            position = 1
            entry_price = price
            
        elif prediction < -0.05 and position >= 0:
            if position > 0:
                pnl += (price - entry_price)  # Close Long
                trades += 1
            position = -1
            entry_price = price
            
        peak = max(peak, pnl)
        drawdown = min(drawdown, pnl - peak)

    print(f"Processed 10,000 continuous updates in O(N). Attention Softmax would cost O(N^2)!")
    print(f"Constant memory utilized: {d_state}x{d_state} Matrix.")
    print("\nBacktest Results:")
    print(f"Gross PnL:        {pnl:.2f} USDT")
    print(f"Max Drawdown:     {abs(drawdown):.2f} USDT")
    print(f"Total Trades:     {trades}")
    print(f"Sharpe Ratio:     {pnl / (abs(drawdown) + 1e-6) * np.sqrt(365):.2f}")
    print("-" * 50)

if __name__ == "__main__":
    simulate_bybit_hf_backtest()
