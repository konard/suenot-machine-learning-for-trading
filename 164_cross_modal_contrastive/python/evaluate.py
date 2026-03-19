"""
Evaluation script for Cross-Modal Contrastive Learning.

Tests zero-shot retrieval: given a text query, find the matching price chart.
Supports both the static overfit model and the market data model.
"""

import torch
import torch.nn.functional as F
import argparse

from model import CrossModalCLIPModel
from data import (
    VOCAB_SIZE,
    get_static_4_item_batch,
    generate_bybit_crypto_stream,
    generate_stock_market_stream,
    extract_windows_from_stream,
)


def evaluate_overfit():
    """
    Evaluate zero-shot retrieval on the static 4-item batch (overfit model).
    """
    print("=" * 60)
    print("Evaluating Cross-Modal Zero-Shot Retrieval (Overfit Model)")
    print("=" * 60)

    model = CrossModalCLIPModel(vocab_size=VOCAB_SIZE, projection_dim=32)
    model.load_state_dict(torch.load("cross_modal_clip.pth", weights_only=True))
    model.eval()

    gallery_price, gallery_text = get_static_4_item_batch(window_size=128)
    labels = ["Uptrend/Bullish", "Downtrend/Bearish", "Volatile/Squeeze", "Flat/Quiet"]

    with torch.no_grad():
        v_price, v_text = model(gallery_price, gallery_text)

        # Normalize for cosine similarity
        v_price = F.normalize(v_price, p=2, dim=1)
        v_text = F.normalize(v_text, p=2, dim=1)

        # NxN similarity matrix
        sim_matrix = v_text @ v_price.T

        print("\nSimilarity Matrix (Text -> Price):")
        print(f"{'':>20}", end="")
        for label in labels:
            print(f"{label:>20}", end="")
        print()
        for i, label in enumerate(labels):
            print(f"{label:>20}", end="")
            for j in range(4):
                print(f"{sim_matrix[i, j].item():>20.4f}", end="")
            print()

        correct_top1 = 0
        for i in range(4):
            top_idx = torch.argmax(sim_matrix[i]).item()
            if top_idx == i:
                correct_top1 += 1

        recall = correct_top1 / 4 * 100
        print(f"\nText -> Chart Retrieval Recall@1: {recall}%")

        if recall > 80:
            print("RESULT: SUCCESS — Modalities are aligned.")
        else:
            print("RESULT: WARNING — Low retrieval accuracy.")


def evaluate_market():
    """
    Evaluate zero-shot retrieval across Bybit crypto and stock market events.
    """
    print("=" * 60)
    print("Evaluating Cross-Modal Retrieval (Market Data Model)")
    print("=" * 60)

    model = CrossModalCLIPModel(vocab_size=VOCAB_SIZE, projection_dim=32)
    model.load_state_dict(torch.load("cross_modal_clip_market.pth", weights_only=True))
    model.eval()

    # Load all data sources
    btc_prices, btc_events = generate_bybit_crypto_stream("BTCUSDT", seed=42)
    btc_w, btc_t, btc_d = extract_windows_from_stream(btc_prices, btc_events)

    eth_prices, eth_events = generate_bybit_crypto_stream(
        "ETHUSDT", n_points=10000, base_price=3000.0, volatility=0.025, seed=99
    )
    eth_w, eth_t, eth_d = extract_windows_from_stream(eth_prices, eth_events)

    aapl_prices, aapl_events = generate_stock_market_stream("AAPL", seed=123)
    aapl_w, aapl_t, aapl_d = extract_windows_from_stream(aapl_prices, aapl_events)

    all_prices = torch.cat([btc_w, eth_w, aapl_w], dim=0)
    all_text = torch.cat([btc_t, eth_t, aapl_t], dim=0)
    all_desc = btc_d + eth_d + aapl_d

    n = all_prices.size(0)

    with torch.no_grad():
        v_price, v_text = model(all_prices, all_text)

        v_price = F.normalize(v_price, p=2, dim=1)
        v_text = F.normalize(v_text, p=2, dim=1)

        sim_matrix = v_text @ v_price.T

        correct_top1 = 0
        correct_top3 = 0

        print("\nRetrieval Results:")
        for i in range(n):
            sims = sim_matrix[i]
            top_indices = torch.argsort(sims, descending=True)
            rank = (top_indices == i).nonzero(as_tuple=True)[0].item() + 1

            if top_indices[0] == i:
                correct_top1 += 1
            if i in top_indices[:3]:
                correct_top3 += 1

            match_label = "MATCH" if top_indices[0] == i else f"rank={rank}"
            print(f"  Query: '{all_desc[i]}' -> [{match_label}]")

        print(f"\nRecall@1: {correct_top1 / n * 100:.1f}%")
        print(f"Recall@3: {correct_top3 / n * 100:.1f}%")

        if correct_top1 / n > 0.5:
            print("RESULT: SUCCESS — Cross-modal alignment working on mixed market data.")
        else:
            print("RESULT: PARTIAL — Some alignment learned, may need more training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Modal CLIP Evaluation")
    parser.add_argument(
        "--mode",
        choices=["overfit", "market"],
        default="overfit",
        help="Evaluation mode",
    )
    args = parser.parse_args()

    if args.mode == "overfit":
        evaluate_overfit()
    else:
        evaluate_market()
