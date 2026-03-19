"""
Training script for Cross-Modal Contrastive Learning (CLIP-style).

Supports two training modes:
  1. Static overfit test (4-item batch) — verifies the architecture works.
  2. Market data training — uses synthetic Bybit crypto + stock data streams.
"""

import torch
import torch.optim as optim
import argparse

from model import CrossModalCLIPModel, CLIPLoss, SymmetricCLIPLoss
from data import (
    VOCAB_SIZE,
    get_static_4_item_batch,
    generate_bybit_crypto_stream,
    generate_stock_market_stream,
    extract_windows_from_stream,
)


def train_overfit(epochs: int = 200, lr: float = 1e-3, window_size: int = 128):
    """
    Overfit on a static 4-item batch to verify the architecture can learn
    cross-modal alignment.
    """
    print("=" * 60)
    print("Mode: Static Overfit Test (4 items)")
    print("=" * 60)

    model = CrossModalCLIPModel(vocab_size=VOCAB_SIZE, projection_dim=32)
    criterion = CLIPLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_price, x_text = get_static_4_item_batch(window_size)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for _ in range(20):
            optimizer.zero_grad()

            v_price, v_text = model(x_price, x_text)

            # Negative pairs by rolling the text
            x_text_neg = torch.roll(x_text, shifts=1, dims=0)
            _, v_text_neg = model(x_price, x_text_neg)

            loss = criterion(v_price, v_text, v_text_neg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / 20:.4f}")

    print("Overfit training completed.")
    torch.save(model.state_dict(), "cross_modal_clip.pth")
    print("Model saved to cross_modal_clip.pth")


def train_market_data(
    epochs: int = 100,
    lr: float = 5e-4,
    window_size: int = 128,
    use_symmetric_loss: bool = True,
):
    """
    Train on synthetic Bybit crypto + stock market data streams.
    Uses events from both BTCUSDT (Bybit) and AAPL (stock) as training pairs.
    """
    print("=" * 60)
    print("Mode: Market Data Training (Bybit Crypto + Stock)")
    print("=" * 60)

    # --- Generate data from both modalities ---
    print("Generating Bybit BTCUSDT stream...")
    btc_prices, btc_events = generate_bybit_crypto_stream(
        symbol="BTCUSDT", n_points=10000, base_price=50000.0, seed=42
    )
    btc_windows, btc_text, btc_desc = extract_windows_from_stream(
        btc_prices, btc_events, window_size
    )

    print("Generating Bybit ETHUSDT stream...")
    eth_prices, eth_events = generate_bybit_crypto_stream(
        symbol="ETHUSDT", n_points=10000, base_price=3000.0,
        volatility=0.025, seed=99
    )
    eth_windows, eth_text, eth_desc = extract_windows_from_stream(
        eth_prices, eth_events, window_size
    )

    print("Generating AAPL stock stream...")
    aapl_prices, aapl_events = generate_stock_market_stream(
        symbol="AAPL", n_points=10000, base_price=150.0, seed=123
    )
    aapl_windows, aapl_text, aapl_desc = extract_windows_from_stream(
        aapl_prices, aapl_events, window_size
    )

    # Concatenate all data sources
    all_prices = torch.cat([btc_windows, eth_windows, aapl_windows], dim=0)
    all_text = torch.cat([btc_text, eth_text, aapl_text], dim=0)
    all_desc = btc_desc + eth_desc + aapl_desc

    n_samples = all_prices.size(0)
    print(f"Total training pairs: {n_samples}")
    for desc in all_desc:
        print(f"  - {desc}")

    # --- Initialize model ---
    model = CrossModalCLIPModel(vocab_size=VOCAB_SIZE, projection_dim=32)

    if use_symmetric_loss:
        criterion = SymmetricCLIPLoss(temperature=0.07)
    else:
        criterion = CLIPLoss(margin=0.2)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    for epoch in range(epochs):
        optimizer.zero_grad()

        v_price, v_text = model(all_prices, all_text)

        if use_symmetric_loss:
            loss = criterion(v_price, v_text, model.logit_scale)
        else:
            # Roll text to create negatives
            all_text_neg = torch.roll(all_text, shifts=1, dims=0)
            _, v_text_neg = model(all_prices, all_text_neg)
            loss = criterion(v_price, v_text, v_text_neg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    print("Market data training completed.")
    torch.save(model.state_dict(), "cross_modal_clip_market.pth")
    print("Model saved to cross_modal_clip_market.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Modal CLIP Training")
    parser.add_argument(
        "--mode",
        choices=["overfit", "market"],
        default="overfit",
        help="Training mode: 'overfit' for static 4-item test, 'market' for Bybit+stock data",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    if args.mode == "overfit":
        train_overfit(
            epochs=args.epochs or 200,
            lr=args.lr or 1e-3,
        )
    else:
        train_market_data(
            epochs=args.epochs or 100,
            lr=args.lr or 5e-4,
        )
