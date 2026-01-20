#!/usr/bin/env python3
"""
Full Pipeline Demo

Demonstrates the complete prompt engineering trading pipeline:
1. Load market data
2. Detect market regime
3. Analyze sentiment from news
4. Generate trading signals
5. Backtest the strategy
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import MockLLMClient
from data_loader import MockDataLoader, prepare_market_data_for_prompt, MarketSnapshot
from regime_detection import MarketRegimeDetector, MarketRegime
from sentiment_analysis import FinancialSentimentAnalyzer, Sentiment
from signal_generator import PromptBasedSignalGenerator, SignalDirection
from backtest import LLMSignalBacktester, BacktestConfig


async def run_full_pipeline():
    """Run the complete trading pipeline."""
    print("\n" + "#" * 70)
    print("  PROMPT ENGINEERING FOR TRADING - FULL PIPELINE DEMO")
    print("#" * 70)

    # =========================================================================
    # STEP 1: Initialize Components
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Initializing Components")
    print("=" * 70)

    # Create mock LLM responses for the pipeline
    mock_responses = {
        # Regime detection response
        "regime": json.dumps({
            "regime": "RISK_ON_TRENDING",
            "confidence": 75,
            "key_drivers": [
                "Strong equity momentum",
                "Low VIX",
                "Positive fund flows"
            ],
            "recommended_strategies": ["trend_following", "buy_dips"],
            "avoid_strategies": ["short_selling"],
            "position_sizing": "MAINTAIN",
            "next_regime_probabilities": {
                "RISK_ON_TRENDING": 0.60,
                "RISK_ON_VOLATILE": 0.25,
                "TRANSITIONAL": 0.15
            },
            "reasoning": "Market in steady uptrend with controlled volatility"
        }),

        # Sentiment responses
        "sentiment_positive": json.dumps({
            "sentiment": "POSITIVE",
            "confidence": 82,
            "market_impact": "BULLISH",
            "time_horizon": "SHORT_TERM",
            "key_factors": ["earnings beat", "strong guidance"],
            "reasoning": "Solid results exceed expectations"
        }),
        "sentiment_neutral": json.dumps({
            "sentiment": "NEUTRAL",
            "confidence": 60,
            "market_impact": "NEUTRAL",
            "time_horizon": "SHORT_TERM",
            "key_factors": ["inline results"],
            "reasoning": "Results met but did not exceed expectations"
        }),

        # Signal responses
        "signal_buy": json.dumps({
            "direction": "BUY",
            "confidence": 78,
            "entry_price": 185.00,
            "stop_loss": 178.00,
            "take_profit": 198.00,
            "position_size_pct": 5,
            "timeframe": "1d",
            "reasoning": "Strong momentum with positive sentiment",
            "key_factors": ["bullish regime", "positive news", "technical support"]
        }),
        "signal_hold": json.dumps({
            "direction": "HOLD",
            "confidence": 55,
            "entry_price": 380.00,
            "stop_loss": 365.00,
            "take_profit": 400.00,
            "position_size_pct": 0,
            "timeframe": "1d",
            "reasoning": "Mixed signals, wait for confirmation"
        })
    }

    llm_client = MockLLMClient(responses=mock_responses)
    data_loader = MockDataLoader()

    print("Components initialized:")
    print("  - LLM Client (Mock)")
    print("  - Data Loader (Mock)")
    print("  - Regime Detector")
    print("  - Sentiment Analyzer")
    print("  - Signal Generator")
    print("  - Backtester")

    # =========================================================================
    # STEP 2: Load Market Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Loading Market Data")
    print("=" * 70)

    symbols = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    market_data = {}
    for symbol in symbols:
        bars = await data_loader.get_ohlcv(symbol, "1d", start_date, end_date)
        market_data[symbol] = bars
        print(f"  Loaded {len(bars)} bars for {symbol}")

    # Create market snapshot for AAPL
    aapl_bars = market_data["AAPL"]
    aapl_snapshot = MarketSnapshot(
        symbol="AAPL",
        price=aapl_bars[-1].close,
        change_pct=((aapl_bars[-1].close - aapl_bars[-2].close) / aapl_bars[-2].close * 100),
        volume=aapl_bars[-1].volume,
        high_52w=max(bar.high for bar in aapl_bars),
        low_52w=min(bar.low for bar in aapl_bars),
        avg_volume=sum(bar.volume for bar in aapl_bars) / len(aapl_bars)
    )

    prompt_data = prepare_market_data_for_prompt(aapl_snapshot, aapl_bars)
    print(f"\n  AAPL Current Price: ${aapl_snapshot.price:.2f}")
    print(f"  52-Week Range: ${aapl_snapshot.low_52w:.2f} - ${aapl_snapshot.high_52w:.2f}")
    if "rsi_14" in prompt_data:
        print(f"  RSI(14): {prompt_data['rsi_14']:.1f}")

    # =========================================================================
    # STEP 3: Detect Market Regime
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Detecting Market Regime")
    print("=" * 70)

    regime_detector = MarketRegimeDetector(llm_client)

    # Prepare regime analysis data
    regime_data = {
        "sp500_price": 4785.50, "sp500_change": 1.2, "sp500_ytd": 8.5,
        "nasdaq_price": 15050.25, "nasdaq_change": 1.8,
        "russell_price": 2025.30, "russell_change": 0.9,
        "vix": 14.5, "vix_avg": 16.0, "vix_term_structure": "Contango",
        "realized_vol": 12.0,
        "treasury_10y": 4.20, "treasury_2y": 4.60, "yield_curve": -40,
        "credit_spread": 105,
        "dxy": 104.0, "dxy_change": 0.2,
        "gold": 2050, "gold_change": 0.5,
        "oil": 78.00, "oil_change": -1.0,
        "adv_dec_ratio": 1.6, "pct_above_200": 62, "high_low_diff": 120,
        "fund_flows": "$12B inflow", "put_call": 0.88, "aaii_spread": 10
    }

    regime_analysis = await regime_detector.detect_regime(regime_data)

    print(f"\n  Detected Regime: {regime_analysis.regime.value}")
    print(f"  Confidence: {regime_analysis.confidence}%")
    print(f"  Key Drivers:")
    for driver in regime_analysis.key_drivers[:3]:
        print(f"    - {driver}")
    print(f"  Recommended: {', '.join(regime_analysis.recommended_strategies)}")
    print(f"  Avoid: {', '.join(regime_analysis.avoid_strategies)}")

    # Get regime-specific strategy parameters
    strategy_params = regime_detector.get_strategy_params(regime_analysis.regime)
    print(f"\n  Strategy Parameters for {regime_analysis.regime.value}:")
    print(f"    Position Multiplier: {strategy_params['position_multiplier']}x")
    print(f"    Preferred Sectors: {', '.join(strategy_params['sectors'])}")

    # =========================================================================
    # STEP 4: Analyze News Sentiment
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Analyzing News Sentiment")
    print("=" * 70)

    sentiment_analyzer = FinancialSentimentAnalyzer(llm_client)

    # Simulated news headlines
    news_items = [
        ("AAPL", "Apple Reports Strong iPhone Sales, Beats Q4 Estimates by 8%"),
        ("MSFT", "Microsoft Cloud Revenue Grows in Line with Expectations"),
        ("GOOGL", "Google Announces Major AI Investment at Developer Conference")
    ]

    sentiment_results = {}
    for symbol, headline in news_items:
        print(f"\n  Analyzing: {headline[:50]}...")
        result = await sentiment_analyzer.analyze(headline, symbol)
        sentiment_results[symbol] = result

        emoji = "+" if result.sentiment == Sentiment.POSITIVE else \
                ("-" if result.sentiment == Sentiment.NEGATIVE else "=")
        print(f"    [{emoji}] {result.sentiment.value} ({result.confidence}%)")
        print(f"    Impact: {result.market_impact}")

    # =========================================================================
    # STEP 5: Generate Trading Signals
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Generating Trading Signals")
    print("=" * 70)

    signal_generator = PromptBasedSignalGenerator(llm_client)

    signals = []
    base_time = datetime.now() - timedelta(days=30)

    for i, (symbol, bars) in enumerate(market_data.items()):
        # Prepare technical data
        closes = [bar.close for bar in bars]

        tech_data = {
            "symbol": symbol,
            "current_price": closes[-1],
            "sma_20": sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1],
            "sma_50": sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1],
            "sma_200": closes[-1],  # Simplified
            "rsi": 55,  # Simplified
            "macd": 0.5,
            "macd_signal": 0.3,
            "volume": bars[-1].volume,
            "avg_volume": sum(bar.volume for bar in bars) / len(bars),
            "support": min(bar.low for bar in bars[-10:]),
            "resistance": max(bar.high for bar in bars[-10:]),
            "atr": 3.0,
            "trend": "BULLISH" if closes[-1] > closes[-20] else "BEARISH"
        }

        # Factor in sentiment
        sentiment = sentiment_results.get(symbol)
        if sentiment:
            tech_data["sentiment"] = sentiment.sentiment.value
            tech_data["sentiment_confidence"] = sentiment.confidence

        # Factor in regime
        tech_data["regime"] = regime_analysis.regime.value
        tech_data["regime_confidence"] = regime_analysis.confidence

        print(f"\n  Generating signal for {symbol}...")
        signal = await signal_generator.generate_signal(tech_data)
        signal.timestamp = base_time + timedelta(days=i * 3)
        signals.append(signal)

        print(f"    Direction: {signal.direction.value}")
        print(f"    Confidence: {signal.confidence}%")
        if signal.direction != SignalDirection.HOLD:
            print(f"    Entry: ${signal.entry_price:.2f}")
            print(f"    Stop: ${signal.stop_loss:.2f}")
            print(f"    Target: ${signal.take_profit:.2f}")

    # =========================================================================
    # STEP 6: Backtest the Strategy
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Backtesting the Strategy")
    print("=" * 70)

    # Prepare price data for backtester
    price_data = {}
    for symbol, bars in market_data.items():
        price_data[symbol] = [bar.to_dict() for bar in bars]

    # Configure backtest based on regime
    position_mult = strategy_params["position_multiplier"]
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.1 * position_mult,  # Adjust for regime
        max_positions=3,
        commission_pct=0.001,
        slippage_pct=0.0005,
        use_stop_loss=True,
        use_take_profit=True
    )

    print(f"\n  Backtest Configuration:")
    print(f"    Initial Capital: ${config.initial_capital:,.2f}")
    print(f"    Position Size: {config.position_size_pct*100:.1f}% (regime-adjusted)")
    print(f"    Max Positions: {config.max_positions}")

    backtester = LLMSignalBacktester(config)
    result = await backtester.run(signals, price_data)

    # =========================================================================
    # STEP 7: Report Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Pipeline Results Summary")
    print("=" * 70)

    print(f"\n  MARKET CONTEXT:")
    print(f"    Regime: {regime_analysis.regime.value}")
    print(f"    Regime Confidence: {regime_analysis.confidence}%")

    print(f"\n  SENTIMENT SUMMARY:")
    positive = sum(1 for s in sentiment_results.values() if s.sentiment == Sentiment.POSITIVE)
    neutral = sum(1 for s in sentiment_results.values() if s.sentiment == Sentiment.NEUTRAL)
    negative = sum(1 for s in sentiment_results.values() if s.sentiment == Sentiment.NEGATIVE)
    print(f"    Positive: {positive}, Neutral: {neutral}, Negative: {negative}")

    print(f"\n  SIGNALS GENERATED:")
    buys = sum(1 for s in signals if s.direction == SignalDirection.BUY)
    sells = sum(1 for s in signals if s.direction == SignalDirection.SELL)
    holds = sum(1 for s in signals if s.direction == SignalDirection.HOLD)
    print(f"    Buy: {buys}, Sell: {sells}, Hold: {holds}")
    avg_confidence = sum(s.confidence for s in signals) / len(signals) if signals else 0
    print(f"    Average Confidence: {avg_confidence:.1f}%")

    print(f"\n  BACKTEST PERFORMANCE:")
    print(f"    Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
    print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"    Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"    Win Rate: {result.win_rate:.1f}%")
    print(f"    Profit Factor: {result.profit_factor:.2f}")
    print(f"    Total Trades: {result.total_trades}")

    print(f"\n  LLM SIGNAL QUALITY:")
    print(f"    Confidence-Return Correlation: {result.confidence_correlation:.3f}")

    # =========================================================================
    # Conclusion
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("""
This demo showed the complete prompt engineering trading pipeline:

1. Data Loading - Retrieved market data for analysis
2. Regime Detection - Identified market environment (RISK_ON_TRENDING)
3. Sentiment Analysis - Processed news for market sentiment
4. Signal Generation - Created trading signals based on all inputs
5. Backtesting - Validated strategy performance historically

Key Takeaways:
- Prompt engineering enables structured LLM analysis for trading
- Multiple inputs (technical, sentiment, regime) improve signal quality
- Backtesting validates whether LLM signals have predictive value
- Confidence correlation helps assess LLM reliability

DISCLAIMER: This is for educational purposes only.
Always validate strategies thoroughly before live trading.
    """)


async def main():
    """Run the full pipeline demo."""
    await run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
