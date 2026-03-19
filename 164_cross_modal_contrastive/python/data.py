"""
Data generation module for Cross-Modal Contrastive Learning.

Provides:
  - Synthetic paired (price_window, text_tokens) data for quick prototyping.
  - Bybit cryptocurrency data fetcher (BTCUSDT, ETHUSDT).
  - Stock market data simulator (modeled after S&P 500 / AAPL patterns).

All data is self-contained within this chapter directory.
"""

import torch
import math
import json
import os
from typing import Tuple, List, Dict

# ---------------------------------------------------------------------------
# Vocabulary for synthetic text events
# ---------------------------------------------------------------------------
# Simple token mapping simulating financial news vocabulary
VOCAB = {
    "<PAD>": 0,
    "bullish": 1,
    "bearish": 2,
    "volatility": 3,
    "flat": 4,
    "record": 5,
    "profit": 6,
    "crash": 7,
    "scandal": 8,
    "rate": 9,
    "hike": 10,
    "quiet": 11,
    "holiday": 12,
    "breakout": 13,
    "squeeze": 14,
    "btc": 15,
    "eth": 16,
    "pump": 17,
    "dump": 18,
    "liquidation": 19,
    "bybit": 20,
    "earnings": 21,
    "fed": 22,
    "crypto": 23,
    "stock": 24,
    "rally": 25,
    "dip": 26,
    "buy": 27,
    "sell": 28,
    "moon": 29,
    "fear": 30,
}

VOCAB_SIZE = max(VOCAB.values()) + 1
MAX_TEXT_LEN = 8


def pad_tokens(tokens: List[int], max_len: int = MAX_TEXT_LEN) -> List[int]:
    """Pad or truncate token list to fixed length."""
    tokens = tokens[:max_len]
    return tokens + [0] * (max_len - len(tokens))


# ---------------------------------------------------------------------------
# Synthetic data: static batch for overfitting tests
# ---------------------------------------------------------------------------

def get_static_4_item_batch(window_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates 4 highly distinct (price, text) pairs for overfitting sanity checks.
    Patterns: Uptrend, Downtrend, Sine/Volatile, Flat.
    """
    batch_price = torch.zeros((4, 1, window_size))
    batch_text = torch.zeros((4, MAX_TEXT_LEN), dtype=torch.long)

    x = torch.linspace(0, 5 * math.pi, window_size)

    # 0: Uptrend — "bullish record profit rally"
    shape0 = x
    text0 = pad_tokens([VOCAB["bullish"], VOCAB["record"], VOCAB["profit"], VOCAB["rally"]])
    batch_price[0, 0, :] = (shape0 - shape0.mean()) / (shape0.std() + 1e-6)
    batch_text[0, :] = torch.tensor(text0)

    # 1: Downtrend — "bearish crash scandal dump"
    shape1 = -x
    text1 = pad_tokens([VOCAB["bearish"], VOCAB["crash"], VOCAB["scandal"], VOCAB["dump"]])
    batch_price[1, 0, :] = (shape1 - shape1.mean()) / (shape1.std() + 1e-6)
    batch_text[1, :] = torch.tensor(text1)

    # 2: Sine wave (high volatility) — "volatility squeeze liquidation"
    shape2 = torch.sin(x * 2)
    text2 = pad_tokens([VOCAB["volatility"], VOCAB["squeeze"], VOCAB["liquidation"]])
    batch_price[2, 0, :] = (shape2 - shape2.mean()) / (shape2.std() + 1e-6)
    batch_text[2, :] = torch.tensor(text2)

    # 3: Flat — "flat quiet holiday"
    shape3 = torch.zeros(window_size)
    text3 = pad_tokens([VOCAB["flat"], VOCAB["quiet"], VOCAB["holiday"]])
    batch_price[3, 0, :] = (shape3 - shape3.mean()) / (shape3.std() + 1e-6)
    batch_text[3, :] = torch.tensor(text3)

    return batch_price, batch_text


# ---------------------------------------------------------------------------
# Synthetic crypto data (simulated Bybit-style BTCUSDT / ETHUSDT)
# ---------------------------------------------------------------------------

def generate_bybit_crypto_stream(
    symbol: str = "BTCUSDT",
    n_points: int = 10000,
    base_price: float = 50000.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Generate a synthetic price stream mimicking Bybit perpetual futures data.
    Returns (prices_tensor, events_list) where events_list contains timestamped
    text events associated with specific price patterns.

    Args:
        symbol: Trading pair name (e.g. "BTCUSDT", "ETHUSDT").
        n_points: Number of price points (1-minute candles).
        base_price: Starting price level.
        volatility: Standard deviation of returns.
        seed: Random seed for reproducibility.

    Returns:
        prices: Tensor of shape (n_points,) with simulated close prices.
        events: List of dicts with keys 'index', 'tokens', 'description'.
    """
    torch.manual_seed(seed)
    returns = torch.randn(n_points) * volatility
    events = []

    # Inject known patterns at specific points
    pattern_points = [
        (500, "pump", [VOCAB["crypto"], VOCAB["btc"], VOCAB["pump"], VOCAB["moon"]],
         0.08, f"{symbol} pump event"),
        (1500, "crash", [VOCAB["crypto"], VOCAB["crash"], VOCAB["liquidation"], VOCAB["fear"]],
         -0.10, f"{symbol} flash crash"),
        (3000, "squeeze", [VOCAB["bybit"], VOCAB["squeeze"], VOCAB["volatility"]],
         0.06, f"{symbol} short squeeze on Bybit"),
        (5000, "rally", [VOCAB["crypto"], VOCAB["bullish"], VOCAB["rally"], VOCAB["breakout"]],
         0.05, f"{symbol} breakout rally"),
        (7000, "dump", [VOCAB["bearish"], VOCAB["dump"], VOCAB["sell"], VOCAB["fear"]],
         -0.07, f"{symbol} whale dump"),
        (8500, "flat", [VOCAB["flat"], VOCAB["quiet"], VOCAB["crypto"]],
         0.0, f"{symbol} consolidation"),
    ]

    for idx, name, tokens, shock, desc in pattern_points:
        if idx < n_points:
            # Apply the shock
            returns[idx] = shock
            # Add gradually decaying follow-through
            for j in range(1, min(20, n_points - idx)):
                returns[idx + j] += shock * 0.5 * math.exp(-j / 5.0)
            events.append({
                "index": idx,
                "tokens": pad_tokens(tokens),
                "description": desc,
            })

    prices = base_price * torch.exp(torch.cumsum(returns, dim=0))
    return prices, events


def generate_stock_market_stream(
    symbol: str = "AAPL",
    n_points: int = 10000,
    base_price: float = 150.0,
    volatility: float = 0.01,
    seed: int = 123,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Generate a synthetic stock market price stream (mimicking S&P 500 / AAPL).
    Returns (prices_tensor, events_list).
    """
    torch.manual_seed(seed)
    returns = torch.randn(n_points) * volatility
    events = []

    pattern_points = [
        (400, "earnings_beat", [VOCAB["stock"], VOCAB["earnings"], VOCAB["bullish"], VOCAB["record"]],
         0.04, f"{symbol} beats earnings expectations"),
        (1200, "fed_hike", [VOCAB["fed"], VOCAB["rate"], VOCAB["hike"], VOCAB["fear"]],
         -0.05, f"Fed rate hike impacts {symbol}"),
        (2500, "rally", [VOCAB["stock"], VOCAB["bullish"], VOCAB["rally"], VOCAB["breakout"]],
         0.03, f"{symbol} tech sector rally"),
        (4000, "scandal", [VOCAB["bearish"], VOCAB["scandal"], VOCAB["crash"], VOCAB["sell"]],
         -0.06, f"{symbol} CEO scandal"),
        (6000, "quiet", [VOCAB["flat"], VOCAB["quiet"], VOCAB["holiday"]],
         0.0, f"{symbol} holiday trading"),
        (8000, "dip_buy", [VOCAB["stock"], VOCAB["dip"], VOCAB["buy"], VOCAB["rally"]],
         0.04, f"{symbol} institutional dip buying"),
    ]

    for idx, name, tokens, shock, desc in pattern_points:
        if idx < n_points:
            returns[idx] = shock
            for j in range(1, min(15, n_points - idx)):
                returns[idx + j] += shock * 0.4 * math.exp(-j / 4.0)
            events.append({
                "index": idx,
                "tokens": pad_tokens(tokens),
                "description": desc,
            })

    prices = base_price * torch.exp(torch.cumsum(returns, dim=0))
    return prices, events


def extract_windows_from_stream(
    prices: torch.Tensor,
    events: List[Dict],
    window_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Extract price windows centered around event indices and pair them with
    corresponding text tokens.

    Returns:
        price_windows: (N, 1, window_size)
        text_tokens:   (N, MAX_TEXT_LEN)
        descriptions:  List of event descriptions
    """
    price_windows = []
    text_tokens = []
    descriptions = []

    for event in events:
        idx = event["index"]
        start = max(0, idx - window_size // 2)
        end = start + window_size

        if end > len(prices):
            continue

        window = prices[start:end].clone()
        # Z-score normalize to focus on shape, not absolute price level
        window = (window - window.mean()) / (window.std() + 1e-6)

        price_windows.append(window.unsqueeze(0))  # (1, window_size)
        text_tokens.append(torch.tensor(event["tokens"], dtype=torch.long))
        descriptions.append(event["description"])

    if len(price_windows) == 0:
        return torch.zeros(0, 1, window_size), torch.zeros(0, MAX_TEXT_LEN, dtype=torch.long), []

    return (
        torch.stack(price_windows),
        torch.stack(text_tokens),
        descriptions,
    )
