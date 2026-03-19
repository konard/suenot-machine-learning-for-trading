"""
SSM-GNN Hybrid Model for Trading.

Combines State Space Models (temporal encoding per asset) with
Graph Neural Networks (cross-asset message passing) for multi-asset
trading signal generation.
"""

import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DiagonalSSM:
    """
    Diagonal State Space Model.

    Implements a simplified S4-style SSM with diagonal state matrix
    for efficient sequential processing of per-asset time series.
    """

    def __init__(self, d_input: int, d_model: int, d_state: int):
        """
        Args:
            d_input: Input feature dimension.
            d_model: Model hidden dimension.
            d_state: State dimension (size of latent state vector).
        """
        self.d_input = d_input
        self.d_model = d_model
        self.d_state = d_state

        # Initialize parameters
        rng = np.random.default_rng(42)

        # Input projection
        self.W_in = rng.standard_normal((d_input, d_model)) * 0.01

        # HiPPO-inspired diagonal A initialization
        self.A_diag = -np.arange(1, d_state + 1, dtype=np.float64)

        # B, C, D parameters
        self.B = rng.standard_normal((d_state, d_model)) * 0.01
        self.C = rng.standard_normal((d_model, d_state)) * 0.01
        self.D = np.zeros(d_model)

        # Step size (log-space for positivity)
        self.log_dt = np.log(np.ones(d_model) * 0.01)

    def discretize(self):
        """Discretize continuous parameters using zero-order hold."""
        dt = np.exp(self.log_dt)

        # For diagonal A: A_bar = exp(dt * A_diag)
        # A_diag is (d_state,), dt is (d_model,)
        # We broadcast: A_bar shape = (d_state, d_model)
        A_bar = np.exp(dt[np.newaxis, :] * self.A_diag[:, np.newaxis])

        # B_bar = (A_bar - I) / A_diag * B, simplified for diagonal case
        # For stability, use: B_bar = dt * B (first-order approximation)
        B_bar = dt[np.newaxis, :] * self.B

        return A_bar, B_bar

    def forward(self, u: np.ndarray) -> np.ndarray:
        """
        Forward pass through SSM.

        Args:
            u: Input tensor of shape (seq_len, d_input).

        Returns:
            Output tensor of shape (seq_len, d_model).
        """
        seq_len = u.shape[0]

        # Project input
        u_proj = u @ self.W_in  # (seq_len, d_model)

        # Discretize
        A_bar, B_bar = self.discretize()

        # Initialize state
        x = np.zeros((self.d_state, self.d_model))

        outputs = []
        for t in range(seq_len):
            # State update: x = A_bar * x + B_bar * u_t
            x = A_bar * x + B_bar * u_proj[t:t+1, :]
            # Output: y_m = Σ_s C[m,s] * x[s,m] + D_m * u_m
            y = np.einsum('ms,sm->m', self.C, x) + self.D * u_proj[t]
            outputs.append(y)

        return np.array(outputs)  # (seq_len, d_model)


class GraphAttentionLayer:
    """
    Graph Attention Network (GAT) layer.

    Implements single-head graph attention for cross-asset
    message passing on the correlation/sector graph.
    """

    def __init__(self, d_in: int, d_out: int):
        """
        Args:
            d_in: Input feature dimension per node.
            d_out: Output feature dimension per node.
        """
        self.d_in = d_in
        self.d_out = d_out

        rng = np.random.default_rng(123)
        self.W = rng.standard_normal((d_in, d_out)) * math.sqrt(2.0 / (d_in + d_out))
        self.a_src = rng.standard_normal(d_out) * 0.01
        self.a_dst = rng.standard_normal(d_out) * 0.01

    def forward(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            node_features: (n_nodes, d_in) node feature matrix.
            edge_index: (2, n_edges) source-target edge indices.
            edge_weight: Optional (n_edges,) edge weights.

        Returns:
            (n_nodes, d_out) updated node features.
        """
        n_nodes = node_features.shape[0]

        # Linear transform
        h = node_features @ self.W  # (n_nodes, d_out)

        # Attention coefficients
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        e = np.zeros(len(src_idx))
        for k in range(len(src_idx)):
            i, j = src_idx[k], dst_idx[k]
            score = np.dot(self.a_src, h[i]) + np.dot(self.a_dst, h[j])
            e[k] = max(0.2 * score, score)  # LeakyReLU with negative_slope=0.2

        # Softmax per destination node
        alpha = np.zeros_like(e)
        for node in range(n_nodes):
            mask = dst_idx == node
            if np.any(mask):
                scores = e[mask]
                scores = scores - np.max(scores)  # numerical stability
                exp_scores = np.exp(scores)
                if edge_weight is not None:
                    exp_scores *= edge_weight[mask]
                alpha[mask] = exp_scores / (np.sum(exp_scores) + 1e-10)

        # Aggregate
        out = np.zeros((n_nodes, self.d_out))
        for k in range(len(src_idx)):
            out[dst_idx[k]] += alpha[k] * h[src_idx[k]]

        # Activation (ELU)
        out = np.where(out > 0, out, np.exp(out) - 1)

        return out


class SSMGNNHybrid:
    """
    SSM-GNN Hybrid model for multi-asset trading.

    Architecture:
    1. Per-asset SSM encoder for temporal feature extraction.
    2. GAT layer(s) for cross-asset message passing.
    3. Prediction head for signal generation.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        d_state: int = 16,
        n_gnn_layers: int = 2,
        n_heads: int = 4,
        n_classes: int = 3,
    ):
        """
        Args:
            n_features: Number of input features per asset per time step.
            d_model: Hidden dimension.
            d_state: SSM state dimension.
            n_gnn_layers: Number of GNN layers.
            n_heads: Number of attention heads (used for multi-head by repeating).
            n_classes: Number of output classes (e.g., 3 = up/flat/down).
        """
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_gnn_layers = n_gnn_layers
        self.n_heads = n_heads

        # SSM encoder (shared across assets)
        self.ssm = DiagonalSSM(n_features, d_model, d_state)

        # GNN layers
        self.gnn_layers = []
        for _ in range(n_gnn_layers):
            self.gnn_layers.append(GraphAttentionLayer(d_model, d_model))

        # Prediction head
        rng = np.random.default_rng(456)
        self.W_pred = rng.standard_normal((d_model, n_classes)) * 0.01
        self.b_pred = np.zeros(n_classes)

    def forward(
        self,
        features: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            features: (n_assets, seq_len, n_features) input feature tensor.
            edge_index: (2, n_edges) graph edge indices.
            edge_weight: Optional (n_edges,) edge weights.

        Returns:
            (n_assets, n_classes) prediction logits.
        """
        n_assets = features.shape[0]

        logger.debug("SSM-GNN forward: %d assets, seq_len=%d, features=%d",
                      n_assets, features.shape[1], features.shape[2])

        # Step 1: SSM encoding per asset (take last time step output)
        node_embeddings = np.zeros((n_assets, self.d_model))
        for i in range(n_assets):
            ssm_output = self.ssm.forward(features[i])  # (seq_len, d_model)
            node_embeddings[i] = ssm_output[-1]  # last time step

        # Step 2: GNN message passing
        h = node_embeddings
        for gnn_layer in self.gnn_layers:
            h_new = gnn_layer.forward(h, edge_index, edge_weight)
            h = h + h_new  # residual connection

        # Step 3: Prediction head
        logits = h @ self.W_pred + self.b_pred  # (n_assets, n_classes)

        return logits

    def predict_signals(
        self,
        features: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Generate trading signals from model predictions.

        Args:
            features: (n_assets, seq_len, n_features) input features.
            edge_index: (2, n_edges) graph edges.
            edge_weight: Optional edge weights.

        Returns:
            Tuple of (signals, confidences) where:
                signals: (n_assets,) array with values in {-1, 0, 1}
                confidences: (n_assets,) array of confidence scores [0, 1]
        """
        logits = self.forward(features, edge_index, edge_weight)

        # Softmax for probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Classes: 0=down, 1=flat, 2=up
        predicted_class = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        # Map to signals: 0 -> -1, 1 -> 0, 2 -> +1
        signals = predicted_class - 1

        return signals.astype(int), confidence


def compute_technical_features(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Compute technical indicators from price series.

    Args:
        prices: (seq_len,) price array.
        window: Lookback window for indicators.

    Returns:
        (seq_len, n_features) feature array with columns:
        [returns, volatility, rsi, sma_ratio, volume_ma_ratio]
    """
    seq_len = len(prices)
    features = np.zeros((seq_len, 5))

    # Returns
    features[1:, 0] = np.diff(prices) / (prices[:-1] + 1e-10)

    # Rolling volatility
    for t in range(window, seq_len):
        features[t, 1] = np.std(features[t-window:t, 0])

    # RSI
    for t in range(window, seq_len):
        returns = features[t-window:t, 0]
        gains = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        losses = -np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
        rs = gains / (losses + 1e-10)
        features[t, 2] = 100 - 100 / (1 + rs)

    # SMA ratio (price / SMA)
    for t in range(window, seq_len):
        sma = np.mean(prices[t-window:t])
        features[t, 3] = prices[t] / (sma + 1e-10) - 1.0

    # Momentum (rate of change)
    for t in range(window, seq_len):
        features[t, 4] = (prices[t] - prices[t-window]) / (prices[t-window] + 1e-10)

    return features
