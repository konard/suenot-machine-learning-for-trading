#!/usr/bin/env python3
"""
Example 02: Cryptocurrency Analysis with Bybit Data

This example demonstrates:
1. Loading cryptocurrency data from Bybit exchange
2. Computing technical features
3. Generating trading signals using the LLM model
4. Analyzing crypto-specific metrics (funding rate, etc.)

Usage:
    python 02_crypto_analysis.py
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
from data_loader import BybitDataLoader
from signals import CryptoSignalGenerator


def create_mock_crypto_data(
    symbol: str = "BTCUSDT",
    days: int = 30,
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Create mock cryptocurrency OHLCV data for demonstration.

    In production, use BybitDataLoader.get_klines() for real data.

    Args:
        symbol: Trading pair symbol
        days: Number of days of data
        interval: Candlestick interval

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Calculate number of candles
    if interval == "1h":
        candles_per_day = 24
    elif interval == "4h":
        candles_per_day = 6
    elif interval == "1d":
        candles_per_day = 1
    else:
        candles_per_day = 24

    n_candles = days * candles_per_day

    # Generate timestamps
    end_time = datetime.now()
    timestamps = [end_time - timedelta(hours=i * (24 // candles_per_day))
                  for i in range(n_candles)]
    timestamps.reverse()

    # Generate price data with realistic crypto volatility
    base_price = 45000  # BTC starting price
    returns = np.random.randn(n_candles) * 0.02  # 2% hourly volatility

    # Add some trend
    trend = np.linspace(0, 0.1, n_candles)
    returns += trend / n_candles

    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        volatility = abs(np.random.randn()) * 0.01 + 0.005

        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = close * (1 + np.random.randn() * 0.005)

        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = np.random.exponential(100) * 1e6  # Volume in USD

        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df


def compute_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features for crypto analysis.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Volume features
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(24 * 365)  # Annualized

    # Price momentum
    df['momentum_7'] = df['close'] / df['close'].shift(7) - 1
    df['momentum_30'] = df['close'] / df['close'].shift(30) - 1

    return df.dropna()


def simulate_sentiment_predictions(
    df: pd.DataFrame,
    model: FinancialSentimentLoRA,
    device: torch.device
) -> np.ndarray:
    """
    Simulate sentiment predictions based on technical indicators.

    In production, you would feed actual news/social media data to the model.

    Args:
        df: DataFrame with computed features
        model: Trained sentiment model
        device: Computation device

    Returns:
        Array of sentiment predictions (0=Bullish, 1=Neutral, 2=Bearish)
    """
    predictions = []

    for idx, row in df.iterrows():
        # Create a feature vector from technical indicators
        features = np.zeros(768, dtype=np.float32)

        # Encode technical signals into features
        # RSI signal
        if row['rsi'] < 30:
            features[0:50] = 0.5  # Oversold -> potentially bullish
        elif row['rsi'] > 70:
            features[0:50] = -0.5  # Overbought -> potentially bearish

        # MACD signal
        if row['macd_hist'] > 0:
            features[50:100] = 0.3
        else:
            features[50:100] = -0.3

        # Trend signal
        if row['close'] > row['sma_50']:
            features[100:150] = 0.4
        else:
            features[100:150] = -0.4

        # Momentum signal
        features[150:200] = np.clip(row['momentum_7'] * 10, -1, 1)

        # Add some noise
        features += np.random.randn(768) * 0.1

        # Get model prediction
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(x)
            pred = torch.argmax(output, dim=-1).item()

        predictions.append(pred)

    return np.array(predictions)


def generate_trading_report(
    df: pd.DataFrame,
    signals: List[Dict],
    symbol: str
) -> str:
    """
    Generate a trading analysis report.

    Args:
        df: DataFrame with price data
        signals: List of trading signals
        symbol: Trading pair symbol

    Returns:
        Formatted report string
    """
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"Crypto Trading Analysis Report: {symbol}")
    report.append(f"{'='*60}")

    # Price summary
    current_price = df['close'].iloc[-1]
    price_change_24h = (df['close'].iloc[-1] / df['close'].iloc[-24] - 1) * 100
    price_change_7d = (df['close'].iloc[-1] / df['close'].iloc[-168] - 1) * 100 if len(df) >= 168 else 0

    report.append(f"\n[Price Summary]")
    report.append(f"  Current Price: ${current_price:,.2f}")
    report.append(f"  24h Change: {price_change_24h:+.2f}%")
    report.append(f"  7d Change: {price_change_7d:+.2f}%")

    # Technical indicators
    report.append(f"\n[Technical Indicators]")
    report.append(f"  RSI (14): {df['rsi'].iloc[-1]:.1f}")
    report.append(f"  MACD: {df['macd'].iloc[-1]:.2f}")
    report.append(f"  MACD Signal: {df['macd_signal'].iloc[-1]:.2f}")
    report.append(f"  20-day Volatility: {df['volatility_20'].iloc[-1]*100:.1f}%")

    # Signal summary
    if signals:
        buy_signals = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in signals if s['signal'] == 'SELL')
        hold_signals = sum(1 for s in signals if s['signal'] == 'HOLD')

        report.append(f"\n[Signal Summary (Last {len(signals)} periods)]")
        report.append(f"  BUY signals: {buy_signals} ({100*buy_signals/len(signals):.1f}%)")
        report.append(f"  SELL signals: {sell_signals} ({100*sell_signals/len(signals):.1f}%)")
        report.append(f"  HOLD signals: {hold_signals} ({100*hold_signals/len(signals):.1f}%)")

        # Recent signals
        report.append(f"\n[Recent Signals]")
        for signal in signals[-5:]:
            report.append(f"  {signal['timestamp']}: {signal['signal']} "
                         f"(confidence: {signal['confidence']:.2f})")

    # Trading recommendation
    if signals:
        recent_signals = [s['signal'] for s in signals[-10:]]
        buy_ratio = recent_signals.count('BUY') / len(recent_signals)
        sell_ratio = recent_signals.count('SELL') / len(recent_signals)

        report.append(f"\n[Trading Recommendation]")
        if buy_ratio > 0.6:
            report.append("  Overall: BULLISH - Consider long positions")
        elif sell_ratio > 0.6:
            report.append("  Overall: BEARISH - Consider short positions or exit")
        else:
            report.append("  Overall: NEUTRAL - Hold current positions")

    report.append(f"\n{'='*60}")

    return '\n'.join(report)


def main():
    """Main function to run the crypto analysis example."""

    print("=" * 60)
    print("Cryptocurrency Analysis with Bybit Data")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 1. Load crypto data
    print("\n[1/5] Loading cryptocurrency data...")

    # Note: In production, use BybitDataLoader for real data:
    # loader = BybitDataLoader()
    # df = loader.get_klines("BTCUSDT", interval="60", limit=720)

    # For demonstration, use mock data
    symbol = "BTCUSDT"
    df = create_mock_crypto_data(symbol=symbol, days=30, interval="1h")

    print(f"  Symbol: {symbol}")
    print(f"  Data points: {len(df)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Latest price: ${df['close'].iloc[-1]:,.2f}")

    # 2. Compute features
    print("\n[2/5] Computing technical features...")
    df = compute_crypto_features(df)
    print(f"  Features computed: {len(df.columns)}")

    # 3. Load/create sentiment model
    print("\n[3/5] Loading sentiment model...")
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

    # 4. Generate signals
    print("\n[4/5] Generating trading signals...")

    # Get sentiment predictions
    predictions = simulate_sentiment_predictions(df, model, device)

    # Create signal generator
    signal_generator = CryptoSignalGenerator(
        bullish_threshold=0.6,
        bearish_threshold=0.6,
        funding_rate_impact=0.1,
        volatility_adjustment=True,
    )

    # Generate signals for each time point
    signals = []
    label_names = ['Bullish', 'Neutral', 'Bearish']

    for i, (idx, row) in enumerate(df.iterrows()):
        pred = predictions[i]

        # Create mock probabilities based on prediction
        probs = np.zeros(3)
        probs[pred] = 0.7
        probs[(pred + 1) % 3] = 0.2
        probs[(pred + 2) % 3] = 0.1

        # Get mock funding rate (slightly negative on average)
        funding_rate = np.random.randn() * 0.0005 - 0.0001

        # Generate signal
        signal = signal_generator.generate_signal(
            sentiment=label_names[pred],
            confidence=probs[pred],
            funding_rate=funding_rate,
            volatility=row['volatility_20'],
        )

        signals.append({
            'timestamp': idx,
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'funding_impact': signal.get('funding_impact', 0),
        })

    print(f"  Generated {len(signals)} signals")

    # 5. Generate report
    print("\n[5/5] Generating analysis report...")
    report = generate_trading_report(df, signals, symbol)
    print(report)

    # Additional: Show Bybit API usage example
    print("\n" + "=" * 60)
    print("Bybit API Usage Example (for reference)")
    print("=" * 60)
    print("""
    # To use real Bybit data:

    from data_loader import BybitDataLoader

    loader = BybitDataLoader()

    # Get recent klines
    df = loader.get_klines(
        symbol="BTCUSDT",
        interval="60",  # 1 hour
        limit=720       # 30 days
    )

    # Get orderbook
    orderbook = loader.get_orderbook("BTCUSDT", limit=50)

    # Get recent trades
    trades = loader.get_recent_trades("BTCUSDT", limit=1000)

    # Get ticker info
    ticker = loader.get_ticker("BTCUSDT")
    """)

    print("\n" + "=" * 60)
    print("Crypto analysis example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
