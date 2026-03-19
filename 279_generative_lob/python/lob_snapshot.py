"""
LOB Snapshot Data Structures

Core data structures for representing and manipulating limit order book states,
including normalization and feature vector conversion.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PriceLevel:
    """A single price level in the order book."""
    price: float
    quantity: float


@dataclass
class LobSnapshot:
    """
    A snapshot of the limit order book at a single point in time.

    Attributes:
        bids: List of bid price levels (highest first).
        asks: List of ask price levels (lowest first).
        timestamp: Unix timestamp in milliseconds.
    """
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)
    timestamp: int = 0

    def mid_price(self) -> float:
        """Compute the mid-price."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0].price + self.asks[0].price) / 2.0

    def spread(self) -> float:
        """Compute the bid-ask spread."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price

    def order_imbalance(self, levels: int = 5) -> float:
        """Compute order imbalance at top N levels."""
        bid_vol = sum(
            self.bids[i].quantity for i in range(min(levels, len(self.bids)))
        )
        ask_vol = sum(
            self.asks[i].quantity for i in range(min(levels, len(self.asks)))
        )
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def to_feature_vector(self, num_levels: int) -> np.ndarray:
        """
        Flatten to a normalized feature vector.

        For each level: (relative_bid_price, log_bid_qty, relative_ask_price, log_ask_qty).

        Args:
            num_levels: Number of price levels to include.

        Returns:
            Feature vector of shape (num_levels * 4,).
        """
        mid = self.mid_price()
        features = []

        for i in range(num_levels):
            if i < len(self.bids) and mid > 0:
                features.append((self.bids[i].price - mid) / mid)
                features.append(np.log(1.0 + self.bids[i].quantity))
            else:
                features.extend([0.0, 0.0])

            if i < len(self.asks) and mid > 0:
                features.append((self.asks[i].price - mid) / mid)
                features.append(np.log(1.0 + self.asks[i].quantity))
            else:
                features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    @staticmethod
    def from_feature_vector(
        features: np.ndarray, mid_price: float, num_levels: int
    ) -> "LobSnapshot":
        """
        Reconstruct a LOB snapshot from a feature vector and reference mid-price.

        Args:
            features: Feature vector of shape (num_levels * 4,).
            mid_price: Reference mid-price for denormalization.
            num_levels: Number of price levels.

        Returns:
            Reconstructed LobSnapshot.
        """
        bids = []
        asks = []

        for i in range(num_levels):
            base = i * 4
            if base + 3 < len(features):
                bid_price = features[base] * mid_price + mid_price
                bid_qty = max(0.0, np.exp(features[base + 1]) - 1.0)
                ask_price = features[base + 2] * mid_price + mid_price
                ask_qty = max(0.0, np.exp(features[base + 3]) - 1.0)

                bids.append(PriceLevel(price=float(bid_price), quantity=float(bid_qty)))
                asks.append(PriceLevel(price=float(ask_price), quantity=float(ask_qty)))

        return LobSnapshot(bids=bids, asks=asks, timestamp=0)


def generate_test_snapshots(
    count: int, num_levels: int, base_price: float = 50000.0
) -> List[LobSnapshot]:
    """
    Generate a set of realistic-looking synthetic LOB snapshots for testing.

    Args:
        count: Number of snapshots to generate.
        num_levels: Number of price levels per side.
        base_price: Base mid-price (e.g., 50000 for BTC).

    Returns:
        List of synthetic LobSnapshot objects.
    """
    rng = np.random.default_rng(42)
    snapshots = []

    for i in range(count):
        mid = base_price + rng.uniform(-100, 100)
        spread = rng.uniform(0.5, 5.0)

        bids = []
        asks = []
        for j in range(num_levels):
            bid_price = mid - spread / 2 - j * rng.uniform(0.5, 2.0)
            ask_price = mid + spread / 2 + j * rng.uniform(0.5, 2.0)
            bid_qty = rng.uniform(0.1, 10.0)
            ask_qty = rng.uniform(0.1, 10.0)

            bids.append(PriceLevel(price=bid_price, quantity=bid_qty))
            asks.append(PriceLevel(price=ask_price, quantity=ask_qty))

        snapshots.append(LobSnapshot(bids=bids, asks=asks, timestamp=i))

    return snapshots


if __name__ == "__main__":
    # Quick test
    snapshots = generate_test_snapshots(10, 5, 50000.0)
    snap = snapshots[0]
    print(f"Mid-price: {snap.mid_price():.2f}")
    print(f"Spread: {snap.spread():.4f}")
    print(f"Order imbalance: {snap.order_imbalance():.4f}")

    features = snap.to_feature_vector(5)
    print(f"Feature vector shape: {features.shape}")

    reconstructed = LobSnapshot.from_feature_vector(features, snap.mid_price(), 5)
    print(f"Reconstructed mid-price: {reconstructed.mid_price():.2f}")
    print("LobSnapshot test passed!")
