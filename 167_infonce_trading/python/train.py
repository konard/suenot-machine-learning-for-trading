import torch
import torch.optim as optim
import numpy as np
from model import InfoNCEModel
from infonce_loss import infonce_loss_with_accuracy


def generate_regime_data(batch_size=32, window_size=64, num_regimes=4):
    """
    Generates synthetic market data with distinct regimes:
      0 - Low-volatility uptrend
      1 - High-volatility downtrend
      2 - Mean-reverting sideways
      3 - Momentum breakout

    Returns:
        windows: (batch_size, 1, window_size)
        labels:  (batch_size,) — regime labels (for evaluation only)
    """
    windows = torch.zeros(batch_size, 1, window_size)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i in range(batch_size):
        regime = np.random.randint(0, num_regimes)
        labels[i] = regime
        t = torch.linspace(0, 4 * np.pi, window_size)

        if regime == 0:
            # Low-vol uptrend
            trend = t * 0.05
            noise = torch.randn(window_size) * 0.02
            windows[i, 0] = trend + noise
        elif regime == 1:
            # High-vol downtrend
            trend = -t * 0.08
            noise = torch.randn(window_size) * 0.15
            windows[i, 0] = trend + noise
        elif regime == 2:
            # Mean-reverting
            windows[i, 0] = torch.sin(t * 2) * 0.3 + torch.randn(window_size) * 0.05
        else:
            # Momentum breakout
            breakpoint = np.random.randint(window_size // 3, 2 * window_size // 3)
            flat = torch.randn(window_size) * 0.02
            flat[breakpoint:] += torch.linspace(0, 0.5, window_size - breakpoint)
            windows[i, 0] = flat

        # Normalize each window
        w = windows[i, 0]
        windows[i, 0] = (w - w.mean()) / (w.std() + 1e-6)

    return windows, labels


def generate_bybit_like_data(batch_size=32, window_size=64):
    """
    Generates synthetic crypto-style data mimicking Bybit BTCUSDT patterns:
    high volatility, sudden wicks, and volume spikes.

    Returns:
        windows: (batch_size, 1, window_size)
    """
    windows = torch.zeros(batch_size, 1, window_size)

    for i in range(batch_size):
        t = torch.linspace(0, 1, window_size)
        base = torch.cumsum(torch.randn(window_size) * 0.02, dim=0)

        # Random wick (spike)
        if np.random.random() > 0.5:
            wick_pos = np.random.randint(10, window_size - 10)
            wick_size = (np.random.random() - 0.5) * 1.0
            base[wick_pos] += wick_size

        # Random volatility regime shift
        shift_pos = np.random.randint(window_size // 4, 3 * window_size // 4)
        vol_mult = 1.0 + np.random.random() * 2.0
        base[shift_pos:] *= vol_mult

        windows[i, 0] = (base - base.mean()) / (base.std() + 1e-6)

    return windows


def train():
    BATCH_SIZE = 32
    WINDOW_SIZE = 64
    EMBED_DIM = 64
    TEMPERATURE = 0.1
    EPOCHS = 20
    STEPS_PER_EPOCH = 50
    LR = 3e-4

    print("=== InfoNCE Training for Market Regime Learning ===")
    print(f"Batch Size: {BATCH_SIZE}, Embed Dim: {EMBED_DIM}, Temp: {TEMPERATURE}")

    model = InfoNCEModel(input_dim=1, embed_dim=EMBED_DIM, temperature=TEMPERATURE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        total_acc = 0.0

        for step in range(STEPS_PER_EPOCH):
            # Alternate between stock-like and crypto-like data
            if step % 2 == 0:
                x, _ = generate_regime_data(batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
            else:
                x = generate_bybit_like_data(batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)

            optimizer.zero_grad()

            z_anchor, z_positive = model(x)

            loss, acc = infonce_loss_with_accuracy(
                z_anchor, z_positive, temperature=TEMPERATURE
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        avg_loss = total_loss / STEPS_PER_EPOCH
        avg_acc = total_acc / STEPS_PER_EPOCH
        print(f"Epoch {epoch:2d}/{EPOCHS} | InfoNCE Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%}")

    # Save model
    torch.save(model.state_dict(), "infonce_trading_model.pth")
    print("Training complete. Model saved to infonce_trading_model.pth")

    # Quick evaluation: check embedding quality
    evaluate_embeddings(model)


def evaluate_embeddings(model):
    """
    Evaluates whether learned embeddings separate market regimes.
    """
    model.eval()
    print("\n=== Embedding Evaluation ===")

    with torch.no_grad():
        x, labels = generate_regime_data(batch_size=128, window_size=64, num_regimes=4)
        z = model.encoder(x)
        z = torch.nn.functional.normalize(z, dim=-1)

        regime_names = ["Low-Vol Up", "High-Vol Down", "Mean-Revert", "Breakout"]

        # Compute average intra-regime and inter-regime similarity
        for r in range(4):
            mask = labels == r
            if mask.sum() < 2:
                continue
            z_regime = z[mask]
            intra_sim = torch.mm(z_regime, z_regime.t())
            n = z_regime.size(0)
            # Exclude diagonal
            intra_mean = (intra_sim.sum() - n) / (n * (n - 1))
            print(f"  {regime_names[r]:15s} | Intra-similarity: {intra_mean:.4f} | Count: {n}")

    print("Evaluation complete.")


if __name__ == "__main__":
    train()
