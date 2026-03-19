"""
XLNet-based Financial Sentiment Analyzer and Trading Signal Generator.

This module provides a complete pipeline for:
1. Sentiment analysis of financial texts using XLNet (via Hugging Face)
2. Trading signal generation from sentiment scores
3. Backtesting sentiment-based trading strategies

Usage:
    python sentiment_analyzer.py

Note: For the full XLNet model, install: pip install transformers torch
      The module also works standalone with a simple rule-based fallback.
"""

import math
import random
from typing import Dict, List, Optional, Tuple


# ─── Financial Lexicon Sentiment Scorer ───────────────────────────────

# Loughran-McDonald inspired financial sentiment lexicon
POSITIVE_WORDS = {
    "bullish", "surge", "rally", "growth", "profit", "gain", "upgrade",
    "beat", "strong", "positive", "optimistic", "record", "breakout",
    "momentum", "outperform", "buy", "accumulate", "upside", "recovery",
    "expansion", "revenue", "earnings", "dividend", "innovation",
    "surpass", "exceed", "boost", "improve", "advance", "accelerate",
}

NEGATIVE_WORDS = {
    "bearish", "crash", "decline", "loss", "drop", "downgrade",
    "miss", "weak", "negative", "pessimistic", "correction", "selloff",
    "underperform", "sell", "risk", "downside", "recession",
    "contraction", "debt", "default", "warning", "concern", "fear",
    "volatile", "plunge", "slump", "deteriorate", "worsen", "collapse",
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "hardly", "barely"}


def lexicon_sentiment(text: str) -> float:
    """Score sentiment using a financial lexicon approach.

    Returns a score in [-1, 1].
    """
    words = text.lower().split()
    score = 0.0
    negate = False

    for word in words:
        clean = "".join(c for c in word if c.isalnum())
        if clean in NEGATION_WORDS:
            negate = True
            continue

        if clean in POSITIVE_WORDS:
            score += -1.0 if negate else 1.0
            negate = False
        elif clean in NEGATIVE_WORDS:
            score += 1.0 if negate else -1.0
            negate = False
        else:
            negate = False

    # Normalize to [-1, 1]
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / max(abs(score), 3.0)))


# ─── XLNet Sentiment Analyzer ────────────────────────────────────────

class FinancialSentimentAnalyzer:
    """XLNet-based sentiment analyzer for financial texts.

    Falls back to lexicon-based scoring if transformers is not available.
    """

    def __init__(self, model_name: str = "xlnet-base-cased", num_labels: int = 3):
        self.label_map = {0: "bearish", 1: "neutral", 2: "bullish"}
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        try:
            from transformers import XLNetTokenizer, XLNetForSequenceClassification
            import torch

            self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
            self.model = XLNetForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.model.eval()
            self._use_xlnet = True
            print(f"Loaded XLNet model: {model_name}")
        except ImportError:
            self._use_xlnet = False
            print("transformers not available, using lexicon-based fallback")

    def analyze(self, text: str, max_length: int = 512) -> Dict:
        """Analyze sentiment of a financial text."""
        if self._use_xlnet:
            return self._analyze_xlnet(text, max_length)
        return self._analyze_lexicon(text)

    def _analyze_xlnet(self, text: str, max_length: int) -> Dict:
        import torch
        import numpy as np

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

        predicted = int(np.argmax(probs))
        return {
            "label": self.label_map[predicted],
            "confidence": float(probs[predicted]),
            "scores": {self.label_map[i]: float(probs[i]) for i in range(len(probs))},
            "raw_score": float(probs[2] - probs[0]),  # bullish - bearish
        }

    def _analyze_lexicon(self, text: str) -> Dict:
        score = lexicon_sentiment(text)
        if score > 0.15:
            label = "bullish"
        elif score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        confidence = abs(score) * 0.5 + 0.5  # map to [0.5, 1.0]
        return {
            "label": label,
            "confidence": confidence,
            "scores": {
                "bearish": max(0, -score),
                "neutral": 1.0 - abs(score),
                "bullish": max(0, score),
            },
            "raw_score": score,
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for a batch of texts."""
        return [self.analyze(text) for text in texts]


# ─── Trading Signal Generator ────────────────────────────────────────

class SentimentSignalGenerator:
    """Generate trading signals from sentiment scores using exponential decay."""

    def __init__(self, decay: float = 0.9, aggressiveness: float = 2.0):
        self.decay = decay
        self.aggressiveness = aggressiveness
        self.history: List[float] = []

    def update(self, sentiment_score: float) -> float:
        """Add a new sentiment score and compute the aggregated signal."""
        self.history.append(sentiment_score)
        signal = sum(
            self.decay ** i * self.history[-(i + 1)]
            for i in range(len(self.history))
        )
        return signal

    def position(self, signal: float) -> float:
        """Convert aggregated signal to position size in [-1, 1]."""
        return 2.0 * (1.0 / (1.0 + math.exp(-self.aggressiveness * signal))) - 1.0

    def generate(self, sentiment_score: float) -> Dict:
        """Update and return position recommendation."""
        signal = self.update(sentiment_score)
        pos = self.position(signal)
        if pos > 0.3:
            action = "BUY"
        elif pos < -0.3:
            action = "SELL"
        else:
            action = "HOLD"
        return {"signal": signal, "position": pos, "action": action}

    def reset(self):
        """Clear signal history."""
        self.history.clear()


# ─── Backtester ──────────────────────────────────────────────────────

class SentimentBacktester:
    """Backtest a sentiment-based trading strategy."""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0.0
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]

    def step(self, price: float, sentiment_position: float) -> float:
        """Execute one step of the backtest."""
        target_units = sentiment_position * self.capital / price
        delta = target_units - self.position

        if abs(delta) > 0.001:
            self.trades.append({
                "price": price,
                "units": delta,
                "direction": "buy" if delta > 0 else "sell",
            })

        self.position = target_units
        equity = self.capital + self.position * price
        self.equity_curve.append(equity)
        return equity

    def metrics(self) -> Dict:
        """Compute performance metrics."""
        returns = [
            (self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
            for i in range(1, len(self.equity_curve))
            if self.equity_curve[i - 1] != 0
        ]
        if not returns:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "num_trades": 0,
            }

        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0.0

        max_equity = self.equity_curve[0]
        max_drawdown = 0.0
        for eq in self.equity_curve:
            max_equity = max(max_equity, eq)
            drawdown = (max_equity - eq) / max_equity
            max_drawdown = max(max_drawdown, drawdown)

        downside = [r for r in returns if r < 0]
        downside_std = (
            (sum(r ** 2 for r in downside) / len(downside)) ** 0.5
            if downside else 0.0
        )
        sortino = (avg_return / downside_std * (252 ** 0.5)) if downside_std > 0 else 0.0

        return {
            "total_return": (self.equity_curve[-1] - self.initial_capital) / self.initial_capital,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "num_trades": len(self.trades),
        }


# ─── Demo ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("XLNet Finance - Sentiment Trading Demo")
    print("=" * 60)

    # Step 1: Initialize analyzer
    print("\n[1] Initializing Sentiment Analyzer...\n")
    analyzer = FinancialSentimentAnalyzer()

    # Step 2: Analyze sample financial texts
    print("\n[2] Analyzing Financial Texts...\n")
    sample_texts = [
        "Bitcoin surges past $100,000 on massive institutional buying momentum",
        "Market crashes as regulatory crackdown causes panic selling across crypto",
        "Trading volume remains stable with neutral market outlook",
        "Strong quarterly earnings beat analyst estimates driving stock rally",
        "Recession fears mount as economic decline accelerates globally",
        "Ethereum upgrade boosts network innovation and reduces transaction fees",
        "Company issues profit warning amid weak demand and rising debt",
        "Analysts upgrade price target on positive growth forecast",
    ]

    for text in sample_texts:
        result = analyzer.analyze(text)
        print(f"  [{result['label']:>7}] ({result['raw_score']:+.4f}) \"{text[:70]}...\"")

    # Step 3: Generate trading signals
    print("\n[3] Generating Trading Signals...\n")
    signal_gen = SentimentSignalGenerator(decay=0.9, aggressiveness=2.0)

    for text in sample_texts:
        result = analyzer.analyze(text)
        signal = signal_gen.generate(result["raw_score"])
        print(
            f"  Sentiment={result['raw_score']:+.4f} | "
            f"Signal={signal['signal']:+.4f} | "
            f"Position={signal['position']:+.4f} | "
            f"Action={signal['action']}"
        )

    # Step 4: Backtest
    print("\n[4] Backtesting Sentiment Strategy...\n")
    random.seed(42)
    backtester = SentimentBacktester(initial_capital=10000.0)
    signal_gen_bt = SentimentSignalGenerator(decay=0.9, aggressiveness=2.0)

    # Generate synthetic prices
    prices = []
    price = 95000.0
    for _ in range(50):
        price *= 1.0 + random.gauss(0, 0.02)
        prices.append(price)

    # Run backtest
    news_cycle = sample_texts * 7  # repeat to cover 50 bars
    for i, (p, text) in enumerate(zip(prices, news_cycle)):
        result = analyzer.analyze(text)
        signal = signal_gen_bt.generate(result["raw_score"])
        equity = backtester.step(p, signal["position"])

        if i % 10 == 0 or i == len(prices) - 1:
            print(
                f"  Bar {i:>3}: Price={p:.2f}, "
                f"Position={signal['position']:+.4f}, Equity={equity:.2f}"
            )

    # Step 5: Performance metrics
    print("\n[5] Performance Metrics:\n")
    metrics = backtester.metrics()
    print(f"  Total Return:     {metrics['total_return']:+.2%}")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:    {metrics['sortino_ratio']:.4f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"  Number of Trades: {metrics['num_trades']}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
