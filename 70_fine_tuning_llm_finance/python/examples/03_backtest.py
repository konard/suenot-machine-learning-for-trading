#!/usr/bin/env python3
"""
Example 03: Backtesting LLM Trading Signals

This example demonstrates:
1. Loading historical price data
2. Generating trading signals from sentiment predictions
3. Running a backtest simulation
4. Analyzing performance metrics
5. Visualizing results

Usage:
    python 03_backtest.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from model import FinancialSentimentLoRA
from backtest import LLMBacktester, BacktestResult
from signals import TradingSignalGenerator
from evaluate import evaluate_trading, generate_evaluation_report


def create_mock_price_data(
    ticker: str = "AAPL",
    days: int = 252,  # 1 trading year
    start_price: float = 150.0
) -> pd.DataFrame:
    """
    Create mock stock price data for demonstration.

    In production, use YahooFinanceLoader for real data.

    Args:
        ticker: Stock ticker symbol
        days: Number of trading days
        start_price: Starting price

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Generate dates (trading days only)
    dates = pd.date_range(
        end=datetime.now(),
        periods=days,
        freq='B'  # Business days
    )

    # Generate returns with realistic stock characteristics
    # Mean return ~8% annually, volatility ~20% annually
    daily_mean = 0.08 / 252
    daily_vol = 0.20 / np.sqrt(252)

    returns = np.random.randn(days) * daily_vol + daily_mean

    # Add some momentum and mean reversion
    momentum = np.zeros(days)
    for i in range(1, days):
        momentum[i] = 0.1 * returns[i-1]  # Slight momentum
        if i > 20:
            mean_reversion = -0.05 * (np.sum(returns[i-20:i]) / 20)
            returns[i] += mean_reversion

    returns += momentum

    # Calculate prices
    prices = start_price * np.cumprod(1 + returns)

    # Generate OHLCV
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Intraday volatility
        intraday_vol = abs(np.random.randn()) * 0.01 + 0.005

        high = close * (1 + intraday_vol)
        low = close * (1 - intraday_vol)
        open_price = close * (1 + np.random.randn() * 0.005)

        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume (higher on volatile days)
        base_volume = 50_000_000
        volume = base_volume * (1 + 2 * intraday_vol) * (0.5 + np.random.rand())

        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': int(volume),
            'ticker': ticker,
        })

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    return df


def generate_mock_sentiments(
    df: pd.DataFrame,
    model: FinancialSentimentLoRA,
    device: torch.device
) -> List[Dict]:
    """
    Generate mock sentiment predictions for backtesting.

    In production, you would use actual news/social media data.

    Args:
        df: DataFrame with price data
        model: Trained sentiment model
        device: Computation device

    Returns:
        List of sentiment dictionaries with date, sentiment, and confidence
    """
    sentiments = []
    label_names = ['Bullish', 'Neutral', 'Bearish']

    # Calculate returns for signal generation
    returns = df['close'].pct_change()

    for i, (date, row) in enumerate(df.iterrows()):
        if i < 5:  # Skip first few days
            sentiments.append({
                'date': date,
                'sentiment': 'Neutral',
                'confidence': 0.5,
            })
            continue

        # Create feature vector based on recent price action
        features = np.zeros(768, dtype=np.float32)

        # Recent returns
        recent_returns = returns.iloc[max(0, i-5):i+1].values
        avg_return = np.mean(recent_returns) if len(recent_returns) > 0 else 0

        # Price momentum
        if i >= 20:
            momentum = (row['close'] / df['close'].iloc[i-20] - 1)
        else:
            momentum = 0

        # Encode into features
        features[0:100] = avg_return * 50
        features[100:200] = momentum * 10
        features[200:300] = np.random.randn(100) * 0.1

        # Get model prediction
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(x)
            probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
            pred = np.argmax(probs)

        sentiments.append({
            'date': date,
            'sentiment': label_names[pred],
            'confidence': float(probs[pred]),
        })

    return sentiments


def run_backtest(
    df: pd.DataFrame,
    sentiments: List[Dict],
    initial_capital: float = 100_000.0
) -> BacktestResult:
    """
    Run a backtest using LLM sentiment signals.

    Args:
        df: DataFrame with price data
        sentiments: List of sentiment predictions
        initial_capital: Starting capital

    Returns:
        BacktestResult with performance metrics
    """
    # Create backtester
    backtester = LLMBacktester(
        initial_capital=initial_capital,
        position_size=0.2,  # 20% of capital per trade
        stop_loss=0.05,     # 5% stop loss
        take_profit=0.15,   # 15% take profit
        transaction_cost=0.001,  # 0.1% transaction cost
    )

    # Create signal generator
    signal_generator = TradingSignalGenerator(
        bullish_threshold=0.6,
        bearish_threshold=0.6,
        confidence_threshold=0.5,
    )

    # Convert sentiments to signals
    signals = []
    for sent in sentiments:
        signal = signal_generator.generate_signal(
            sentiment=sent['sentiment'],
            confidence=sent['confidence']
        )
        signals.append({
            'date': sent['date'],
            'signal': signal['signal'],
            'confidence': signal['confidence'],
        })

    # Run backtest
    result = backtester.run(
        prices=df['close'].values,
        signals=[s['signal'] for s in signals],
        dates=df.index.tolist(),
    )

    return result


def print_backtest_report(result: BacktestResult, ticker: str):
    """Print a formatted backtest report."""

    print(f"\n{'='*60}")
    print(f"Backtest Report: {ticker}")
    print(f"{'='*60}")

    print(f"\n[Performance Summary]")
    print(f"  Initial Capital: ${result.initial_capital:,.2f}")
    print(f"  Final Capital: ${result.final_capital:,.2f}")
    print(f"  Total Return: {result.total_return*100:+.2f}%")
    print(f"  Annualized Return: {result.annualized_return*100:+.2f}%")

    print(f"\n[Risk Metrics]")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown*100:.2f}%")
    print(f"  Volatility (Ann.): {result.volatility*100:.2f}%")

    print(f"\n[Trade Statistics]")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Winning Trades: {result.winning_trades}")
    print(f"  Losing Trades: {result.losing_trades}")
    print(f"  Win Rate: {result.win_rate*100:.1f}%")
    print(f"  Avg Win: {result.avg_win*100:+.2f}%")
    print(f"  Avg Loss: {result.avg_loss*100:.2f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")

    print(f"\n[Benchmark Comparison]")
    if hasattr(result, 'benchmark_return'):
        print(f"  Benchmark Return: {result.benchmark_return*100:+.2f}%")
        print(f"  Alpha: {(result.total_return - result.benchmark_return)*100:+.2f}%")
    else:
        print("  (Benchmark data not available)")

    print(f"\n{'='*60}")


def visualize_results(
    df: pd.DataFrame,
    result: BacktestResult,
    ticker: str,
    save_path: Optional[str] = None
):
    """
    Create visualization of backtest results.

    Args:
        df: DataFrame with price data
        result: BacktestResult object
        ticker: Stock ticker
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nNote: matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Price and signals
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{ticker} - Price and Trading Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Portfolio value
    ax2 = axes[1]
    if hasattr(result, 'equity_curve') and result.equity_curve is not None:
        ax2.plot(df.index[:len(result.equity_curve)], result.equity_curve,
                label='Portfolio Value', color='green')
    ax2.axhline(y=result.initial_capital, color='gray', linestyle='--',
                label='Initial Capital')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title('Portfolio Equity Curve')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Drawdown
    ax3 = axes[2]
    if hasattr(result, 'drawdown_series') and result.drawdown_series is not None:
        ax3.fill_between(df.index[:len(result.drawdown_series)],
                        result.drawdown_series * 100,
                        0, color='red', alpha=0.3, label='Drawdown')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_xlabel('Date')
    ax3.set_title('Drawdown Over Time')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main function to run the backtesting example."""

    print("=" * 60)
    print("Backtesting LLM Trading Signals")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Configuration
    ticker = "AAPL"
    initial_capital = 100_000.0

    # 1. Load price data
    print("\n[1/5] Loading historical price data...")

    # Note: In production, use YahooFinanceLoader:
    # from data_loader import YahooFinanceLoader
    # loader = YahooFinanceLoader()
    # df = loader.get_daily(ticker, period="2y")

    # For demonstration, use mock data
    df = create_mock_price_data(ticker=ticker, days=252)

    print(f"  Ticker: {ticker}")
    print(f"  Data points: {len(df)}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # 2. Load sentiment model
    print("\n[2/5] Loading sentiment model...")
    model = FinancialSentimentLoRA(
        base_dim=768,
        num_heads=12,
        lora_rank=8,
        lora_alpha=16.0,
        num_classes=3,
        dropout=0.1
    )
    model.to(device)
    model.eval()
    print("  Model loaded successfully")

    # 3. Generate sentiment predictions
    print("\n[3/5] Generating sentiment predictions...")
    sentiments = generate_mock_sentiments(df, model, device)

    # Count sentiments
    sentiment_counts = {}
    for s in sentiments:
        sentiment_counts[s['sentiment']] = sentiment_counts.get(s['sentiment'], 0) + 1

    print(f"  Total predictions: {len(sentiments)}")
    for sentiment, count in sentiment_counts.items():
        print(f"    {sentiment}: {count} ({100*count/len(sentiments):.1f}%)")

    # 4. Run backtest
    print("\n[4/5] Running backtest simulation...")
    result = run_backtest(df, sentiments, initial_capital)
    print("  Backtest completed")

    # 5. Analyze results
    print("\n[5/5] Analyzing results...")
    print_backtest_report(result, ticker)

    # Generate evaluation metrics
    if hasattr(result, 'daily_returns') and result.daily_returns is not None:
        eval_metrics = evaluate_trading(
            returns=result.daily_returns,
            benchmark_returns=df['close'].pct_change().dropna().values
        )

        print("\n[Additional Evaluation Metrics]")
        print(f"  Information Ratio: {eval_metrics.get('information_ratio', 0):.2f}")
        print(f"  Calmar Ratio: {eval_metrics.get('calmar_ratio', 0):.2f}")
        print(f"  Beta: {eval_metrics.get('beta', 1):.2f}")
        print(f"  Alpha (Ann.): {eval_metrics.get('alpha', 0)*100:.2f}%")

    # Try to visualize (if matplotlib available)
    print("\n[Visualization]")
    try:
        visualize_results(df, result, ticker, save_path="backtest_results.png")
    except Exception as e:
        print(f"  Visualization skipped: {e}")

    # Compare with buy-and-hold
    print("\n[Strategy Comparison]")
    buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    print(f"  LLM Strategy Return: {result.total_return*100:+.2f}%")
    print(f"  Buy & Hold Return: {buy_hold_return*100:+.2f}%")
    print(f"  Outperformance: {(result.total_return - buy_hold_return)*100:+.2f}%")

    print("\n" + "=" * 60)
    print("Backtesting example completed successfully!")
    print("=" * 60)

    # Print usage notes
    print("\n[Usage Notes]")
    print("""
    To use with real data:

    1. Load real price data:
       from data_loader import YahooFinanceLoader
       loader = YahooFinanceLoader()
       df = loader.get_daily("AAPL", period="2y")

    2. Train the sentiment model on financial news data

    3. Generate predictions using actual news/social media

    4. Run backtest with real signals

    5. Analyze and optimize strategy parameters
    """)


if __name__ == "__main__":
    main()
