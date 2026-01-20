"""
Anomaly detection algorithms for financial data.

This module provides multiple approaches to anomaly detection:
1. Statistical methods (Z-score, Mahalanobis distance)
2. Zero-shot LLM classification
3. Embedding-based detection (similarity to normal patterns)
4. Time series encoder with reconstruction loss
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import logging

import numpy as np
import pandas as pd
from scipy import stats

# Optional sklearn imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    IsolationForest = None
    LocalOutlierFactor = None
    EllipticEnvelope = None

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of financial anomalies."""
    PRICE_SPIKE = "price_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    PATTERN_BREAK = "pattern_break"
    PUMP_AND_DUMP = "pump_and_dump"
    FLASH_CRASH = "flash_crash"
    UNUSUAL_CORRELATION = "unusual_correlation"
    NEWS_MISMATCH = "news_mismatch"
    UNKNOWN = "unknown"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    score: float  # 0-1, higher = more anomalous
    anomaly_type: AnomalyType
    confidence: float  # 0-1
    explanation: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "score": self.score,
            "anomaly_type": self.anomaly_type.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "details": self.details,
        }


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseAnomalyDetector":
        """Train detector on normal data."""
        pass

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies in data."""
        pass

    @abstractmethod
    def detect_single(self, row: pd.Series) -> AnomalyResult:
        """Detect anomaly in a single data point."""
        pass


class StatisticalAnomalyDetector(BaseAnomalyDetector):
    """
    Statistical anomaly detection using multiple methods:
    - Z-score
    - Mahalanobis distance
    - Isolation Forest
    - Local Outlier Factor
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        contamination: float = 0.05,
        methods: Optional[List[str]] = None,
    ):
        """
        Initialize statistical detector.

        Args:
            z_threshold: Z-score threshold for anomaly
            contamination: Expected proportion of anomalies
            methods: Which methods to use (default: all)
        """
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.methods = methods or ["zscore", "isolation_forest", "lof"]

        # Fitted parameters
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._isolation_forest: Optional[IsolationForest] = None
        self._lof: Optional[LocalOutlierFactor] = None

        self._feature_cols = ["returns", "volume_ratio", "range_ratio", "returns_zscore"]

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract and prepare features for detection."""
        available_cols = [c for c in self._feature_cols if c in data.columns]

        if not available_cols:
            # Compute basic features if not present
            if "close" in data.columns:
                features = pd.DataFrame()
                features["returns"] = data["close"].pct_change()
                if "volume" in data.columns:
                    vol_ma = data["volume"].rolling(20).mean()
                    features["volume_ratio"] = data["volume"] / vol_ma
                if "high" in data.columns and "low" in data.columns:
                    price_range = (data["high"] - data["low"]) / data["close"]
                    range_ma = price_range.rolling(20).mean()
                    features["range_ratio"] = price_range / range_ma
                return features.dropna().values
            else:
                raise ValueError("Data must have 'close' column or pre-computed features")

        return data[available_cols].dropna().values

    def fit(self, data: pd.DataFrame) -> "StatisticalAnomalyDetector":
        """
        Fit detector on historical normal data.

        Args:
            data: DataFrame with OHLCV or pre-computed features
        """
        X = self._prepare_features(data)

        if len(X) < 10:
            raise ValueError("Need at least 10 samples to fit detector")

        # Z-score parameters
        self._means = np.nanmean(X, axis=0)
        self._stds = np.nanstd(X, axis=0)
        self._stds = np.where(self._stds > 0, self._stds, 1)

        # Mahalanobis distance (covariance inverse)
        try:
            cov = np.cov(X.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            self._cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            self._cov_inv = None
            logger.warning("Could not compute covariance inverse")

        # Isolation Forest
        if "isolation_forest" in self.methods and SKLEARN_AVAILABLE:
            self._isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
            self._isolation_forest.fit(X)
        elif "isolation_forest" in self.methods:
            logger.warning("sklearn not available, skipping Isolation Forest")

        # Local Outlier Factor (for novelty detection)
        if "lof" in self.methods and SKLEARN_AVAILABLE:
            self._lof = LocalOutlierFactor(
                n_neighbors=min(20, len(X) - 1),
                contamination=self.contamination,
                novelty=True,
            )
            self._lof.fit(X)
        elif "lof" in self.methods:
            logger.warning("sklearn not available, skipping LOF")

        logger.info(f"Fitted detector on {len(X)} samples")
        return self

    def _compute_zscore(self, x: np.ndarray) -> Tuple[float, int]:
        """Compute max Z-score and index of most anomalous feature."""
        if self._means is None:
            return 0.0, 0

        z = np.abs((x - self._means) / self._stds)
        max_idx = np.argmax(z)
        return float(z[max_idx]), int(max_idx)

    def _compute_mahalanobis(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance."""
        if self._cov_inv is None or self._means is None:
            return 0.0

        diff = x - self._means
        return float(np.sqrt(diff @ self._cov_inv @ diff.T))

    def detect_single(self, row: pd.Series) -> AnomalyResult:
        """
        Detect anomaly in a single data point.

        Args:
            row: Series with feature values

        Returns:
            AnomalyResult with detection details
        """
        # Prepare features
        available_cols = [c for c in self._feature_cols if c in row.index]
        if not available_cols:
            return AnomalyResult(
                is_anomaly=False,
                score=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
                confidence=0.0,
                explanation="Insufficient features for detection",
                details={},
            )

        x = row[available_cols].values.astype(float)

        # Skip if any NaN
        if np.any(np.isnan(x)):
            return AnomalyResult(
                is_anomaly=False,
                score=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
                confidence=0.0,
                explanation="Missing values in features",
                details={},
            )

        scores = {}
        anomaly_flags = []

        # Z-score
        if "zscore" in self.methods:
            z_max, z_idx = self._compute_zscore(x)
            scores["zscore"] = z_max
            if z_max > self.z_threshold:
                anomaly_flags.append(f"Z-score {z_max:.2f} exceeds threshold")

        # Mahalanobis
        if "mahalanobis" in self.methods and self._cov_inv is not None:
            maha = self._compute_mahalanobis(x)
            scores["mahalanobis"] = maha
            # Chi-squared threshold (95% for n features)
            threshold = stats.chi2.ppf(0.95, len(x))
            if maha > threshold:
                anomaly_flags.append(f"Mahalanobis distance {maha:.2f} exceeds threshold")

        # Isolation Forest
        if "isolation_forest" in self.methods and self._isolation_forest is not None:
            if_score = -self._isolation_forest.score_samples(x.reshape(1, -1))[0]
            scores["isolation_forest"] = if_score
            if if_score > 0.5:
                anomaly_flags.append("Isolation Forest flags as outlier")

        # Local Outlier Factor
        if "lof" in self.methods and self._lof is not None:
            lof_score = -self._lof.score_samples(x.reshape(1, -1))[0]
            scores["lof"] = lof_score
            if lof_score > 1.5:
                anomaly_flags.append("LOF flags as outlier")

        # Aggregate scores
        if scores:
            # Normalize and combine scores
            combined_score = np.mean([
                min(scores.get("zscore", 0) / self.z_threshold, 2) / 2,
                min(scores.get("isolation_forest", 0), 1),
                min(scores.get("lof", 0) / 2, 1),
            ])
        else:
            combined_score = 0.0

        is_anomaly = len(anomaly_flags) > 0
        confidence = min(combined_score, 1.0) if is_anomaly else 1 - combined_score

        # Determine anomaly type based on which feature is most anomalous
        anomaly_type = AnomalyType.UNKNOWN
        if is_anomaly and "zscore" in scores:
            feature_names = available_cols
            _, max_idx = self._compute_zscore(x)
            if max_idx < len(feature_names):
                feature_name = feature_names[max_idx]
                if "volume" in feature_name:
                    anomaly_type = AnomalyType.VOLUME_ANOMALY
                elif "return" in feature_name:
                    anomaly_type = AnomalyType.PRICE_SPIKE
                elif "range" in feature_name:
                    anomaly_type = AnomalyType.PATTERN_BREAK

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=combined_score,
            anomaly_type=anomaly_type,
            confidence=confidence,
            explanation="; ".join(anomaly_flags) if anomaly_flags else "Normal behavior",
            details={"scores": scores, "features": dict(zip(available_cols, x))},
        )

    def detect(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """
        Detect anomalies in all rows of data.

        Args:
            data: DataFrame with features

        Returns:
            List of AnomalyResult for each row
        """
        results = []
        for idx, row in data.iterrows():
            result = self.detect_single(row)
            results.append(result)
        return results


class ZeroShotAnomalyDetector(BaseAnomalyDetector):
    """
    Zero-shot anomaly detection using LLM classification.

    Uses an LLM to classify market conditions as normal or anomalous
    based on a natural language description.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ):
        """
        Initialize zero-shot detector.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key
            custom_prompt: Custom system prompt for classification
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        import os
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

        self.system_prompt = custom_prompt or """You are a financial market anomaly detector.
Analyze the market data provided and determine if there are any anomalies.

Consider these anomaly types:
- PRICE_SPIKE: Unusual price movement without clear catalyst
- VOLUME_ANOMALY: Trading volume significantly different from normal
- PUMP_AND_DUMP: Signs of coordinated price manipulation
- FLASH_CRASH: Sudden severe price drop
- PATTERN_BREAK: Break from established price patterns

Respond in JSON format:
{
    "is_anomaly": true/false,
    "score": 0.0-1.0,
    "type": "anomaly_type",
    "confidence": 0.0-1.0,
    "explanation": "detailed explanation"
}"""

    def fit(self, data: pd.DataFrame) -> "ZeroShotAnomalyDetector":
        """Zero-shot detector doesn't need training."""
        return self

    def _format_market_data(self, data: pd.DataFrame) -> str:
        """Format market data as natural language description."""
        if len(data) == 0:
            return "No data available"

        latest = data.iloc[-1]

        # Compute summary statistics
        if len(data) > 1:
            returns = data["close"].pct_change().dropna()
            avg_return = returns.mean() * 100
            latest_return = returns.iloc[-1] * 100 if len(returns) > 0 else 0

            if "volume" in data.columns:
                vol_avg = data["volume"].mean()
                vol_latest = data["volume"].iloc[-1]
                vol_ratio = vol_latest / vol_avg if vol_avg > 0 else 1
            else:
                vol_ratio = 1
        else:
            avg_return = 0
            latest_return = 0
            vol_ratio = 1

        description = f"""Market Data Summary:
- Latest price: ${latest.get('close', 0):.2f}
- Latest return: {latest_return:.2f}%
- Average return: {avg_return:.4f}%
- Volume ratio (vs average): {vol_ratio:.2f}x
- Price range: ${latest.get('low', 0):.2f} - ${latest.get('high', 0):.2f}
"""

        # Add context about recent trend
        if len(data) >= 5:
            recent_returns = data["close"].pct_change().tail(5)
            trend = "up" if recent_returns.sum() > 0 else "down"
            description += f"- Recent 5-period trend: {trend} ({recent_returns.sum()*100:.2f}%)\n"

        return description

    def detect_single(self, row: pd.Series) -> AnomalyResult:
        """
        Detect anomaly using LLM classification.

        Args:
            row: Series with market data

        Returns:
            AnomalyResult from LLM analysis
        """
        # Format single row as DataFrame
        data = pd.DataFrame([row])
        description = self._format_market_data(data)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": description},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            import json
            result = json.loads(response.choices[0].message.content)

            anomaly_type = AnomalyType.UNKNOWN
            try:
                anomaly_type = AnomalyType(result.get("type", "unknown").lower())
            except ValueError:
                pass

            return AnomalyResult(
                is_anomaly=result.get("is_anomaly", False),
                score=result.get("score", 0.0),
                anomaly_type=anomaly_type,
                confidence=result.get("confidence", 0.5),
                explanation=result.get("explanation", "No explanation provided"),
                details={"llm_response": result},
            )

        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            return AnomalyResult(
                is_anomaly=False,
                score=0.0,
                anomaly_type=AnomalyType.UNKNOWN,
                confidence=0.0,
                explanation=f"Detection failed: {e}",
                details={},
            )

    def detect(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies using LLM (rate-limited)."""
        results = []
        for idx, row in data.iterrows():
            result = self.detect_single(row)
            results.append(result)
        return results


class EmbeddingAnomalyDetector(BaseAnomalyDetector):
    """
    Anomaly detection using embedding similarity.

    Normal patterns are encoded as embeddings, and anomalies are detected
    when new data is far from the normal embedding cluster.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        threshold_percentile: float = 95,
        n_reference: int = 100,
    ):
        """
        Initialize embedding detector.

        Args:
            embedding_dim: Dimension of embeddings
            threshold_percentile: Percentile for anomaly threshold
            n_reference: Number of reference embeddings to keep
        """
        try:
            from .embeddings import FinancialDataEmbedder
        except ImportError:
            from embeddings import FinancialDataEmbedder

        self.embedder = FinancialDataEmbedder(embedding_dim=embedding_dim)
        self.threshold_percentile = threshold_percentile
        self.n_reference = n_reference

        self._reference_embeddings: Optional[np.ndarray] = None
        self._threshold: float = 0.0

    def fit(self, data: pd.DataFrame, window_size: int = 20) -> "EmbeddingAnomalyDetector":
        """
        Fit detector on normal data.

        Args:
            data: OHLCV DataFrame
            window_size: Window size for embedding generation
        """
        self.embedder.fit(data)

        # Generate reference embeddings from sliding windows
        embeddings = []
        for i in range(window_size, len(data)):
            window = data.iloc[i-window_size:i]
            emb = self.embedder.embed(window)
            embeddings.append(emb)

        if len(embeddings) > self.n_reference:
            # Sample reference embeddings
            indices = np.random.choice(len(embeddings), self.n_reference, replace=False)
            embeddings = [embeddings[i] for i in indices]

        self._reference_embeddings = np.vstack(embeddings)

        # Compute threshold from distances to nearest reference
        distances = []
        for emb in embeddings:
            dists = np.linalg.norm(self._reference_embeddings - emb, axis=1)
            distances.append(np.min(dists))

        self._threshold = np.percentile(distances, self.threshold_percentile)
        logger.info(f"Set anomaly threshold to {self._threshold:.4f}")

        return self

    def detect_single(self, row: pd.Series) -> AnomalyResult:
        """Detect anomaly based on embedding distance."""
        # Need context window for embedding
        return AnomalyResult(
            is_anomaly=False,
            score=0.0,
            anomaly_type=AnomalyType.UNKNOWN,
            confidence=0.0,
            explanation="Single-row detection not supported for embedding detector",
            details={},
        )

    def detect_window(self, window: pd.DataFrame) -> AnomalyResult:
        """
        Detect anomaly in a data window.

        Args:
            window: DataFrame window for embedding

        Returns:
            AnomalyResult
        """
        if self._reference_embeddings is None:
            raise ValueError("Detector not fitted")

        # Generate embedding for window
        emb = self.embedder.embed(window)

        # Compute distance to nearest reference
        distances = np.linalg.norm(self._reference_embeddings - emb, axis=1)
        min_distance = np.min(distances)

        # Score based on distance
        score = min(min_distance / self._threshold, 2.0) / 2.0
        is_anomaly = min_distance > self._threshold

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            anomaly_type=AnomalyType.PATTERN_BREAK if is_anomaly else AnomalyType.UNKNOWN,
            confidence=min(score, 1.0) if is_anomaly else 1 - score,
            explanation=f"Distance {min_distance:.4f} {'exceeds' if is_anomaly else 'within'} threshold {self._threshold:.4f}",
            details={"distance": min_distance, "threshold": self._threshold},
        )

    def detect(self, data: pd.DataFrame, window_size: int = 20) -> List[AnomalyResult]:
        """Detect anomalies using sliding windows."""
        results = []
        for i in range(window_size, len(data)):
            window = data.iloc[i-window_size:i]
            result = self.detect_window(window)
            results.append(result)
        return results


class TimeSeriesAnomalyDetector(BaseAnomalyDetector):
    """
    Time series anomaly detection using autoencoder reconstruction.

    Anomalies are detected when the reconstruction error is high,
    indicating the pattern doesn't match learned normal patterns.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        sequence_length: int = 20,
        threshold_percentile: float = 95,
    ):
        """
        Initialize time series detector.

        Args:
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            sequence_length: Input sequence length
            threshold_percentile: Percentile for anomaly threshold
        """
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.threshold_percentile = threshold_percentile

        self._model = None
        self._threshold: float = 0.0
        self._feature_cols = ["returns", "volume_ratio"]

    def _build_model(self, input_dim: int):
        """Build autoencoder model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim, seq_len):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim * seq_len, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim * seq_len),
                )
                self.input_dim = input_dim
                self.seq_len = seq_len

            def forward(self, x):
                # Flatten sequence
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)
                z = self.encoder(x_flat)
                x_recon = self.decoder(z)
                return x_recon.view(batch_size, self.seq_len, self.input_dim)

        return Autoencoder(input_dim, self.hidden_dim, self.latent_dim, self.sequence_length)

    def fit(self, data: pd.DataFrame, epochs: int = 50, lr: float = 0.001) -> "TimeSeriesAnomalyDetector":
        """
        Train autoencoder on normal data.

        Args:
            data: OHLCV DataFrame
            epochs: Training epochs
            lr: Learning rate
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        # Prepare features
        available_cols = [c for c in self._feature_cols if c in data.columns]
        if not available_cols:
            # Compute basic features
            features = pd.DataFrame()
            features["returns"] = data["close"].pct_change()
            if "volume" in data.columns:
                vol_ma = data["volume"].rolling(20).mean()
                features["volume_ratio"] = data["volume"] / vol_ma
            data = pd.concat([data, features], axis=1)
            available_cols = [c for c in self._feature_cols if c in data.columns]

        # Create sequences
        sequences = []
        feature_data = data[available_cols].dropna().values

        for i in range(len(feature_data) - self.sequence_length):
            seq = feature_data[i:i+self.sequence_length]
            sequences.append(seq)

        if len(sequences) < 10:
            raise ValueError("Not enough data for training")

        X = np.array(sequences, dtype=np.float32)

        # Normalize
        self._mean = X.mean(axis=(0, 1))
        self._std = X.std(axis=(0, 1))
        self._std = np.where(self._std > 0, self._std, 1)
        X = (X - self._mean) / self._std

        # Build model
        self._model = self._build_model(X.shape[2])

        # Train
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                x_recon = self._model(x)
                loss = criterion(x_recon, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

        # Compute threshold from reconstruction errors
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X)
            X_recon = self._model(X_tensor).numpy()
            errors = np.mean((X - X_recon) ** 2, axis=(1, 2))
            self._threshold = np.percentile(errors, self.threshold_percentile)

        logger.info(f"Set anomaly threshold to {self._threshold:.6f}")
        return self

    def detect_single(self, row: pd.Series) -> AnomalyResult:
        """Single-row detection not supported."""
        return AnomalyResult(
            is_anomaly=False,
            score=0.0,
            anomaly_type=AnomalyType.UNKNOWN,
            confidence=0.0,
            explanation="Single-row detection not supported",
            details={},
        )

    def detect_sequence(self, sequence: np.ndarray) -> AnomalyResult:
        """
        Detect anomaly in a sequence.

        Args:
            sequence: Array of shape (sequence_length, n_features)

        Returns:
            AnomalyResult
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required")

        if self._model is None:
            raise ValueError("Model not trained")

        # Normalize
        x = (sequence - self._mean) / self._std
        x = x.astype(np.float32)

        # Get reconstruction
        self._model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).unsqueeze(0)
            x_recon = self._model(x_tensor).numpy()[0]

        # Compute error
        error = np.mean((x - x_recon) ** 2)
        score = min(error / self._threshold, 2.0) / 2.0
        is_anomaly = error > self._threshold

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            anomaly_type=AnomalyType.PATTERN_BREAK if is_anomaly else AnomalyType.UNKNOWN,
            confidence=min(score, 1.0) if is_anomaly else 1 - score,
            explanation=f"Reconstruction error {error:.6f} {'exceeds' if is_anomaly else 'within'} threshold",
            details={"error": error, "threshold": self._threshold},
        )

    def detect(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies in all sequences."""
        # Prepare features
        available_cols = [c for c in self._feature_cols if c in data.columns]
        if not available_cols:
            features = pd.DataFrame()
            features["returns"] = data["close"].pct_change()
            if "volume" in data.columns:
                vol_ma = data["volume"].rolling(20).mean()
                features["volume_ratio"] = data["volume"] / vol_ma
            data = pd.concat([data, features], axis=1)
            available_cols = [c for c in self._feature_cols if c in data.columns]

        feature_data = data[available_cols].dropna().values

        results = []
        for i in range(len(feature_data) - self.sequence_length):
            seq = feature_data[i:i+self.sequence_length]
            result = self.detect_sequence(seq)
            results.append(result)

        return results


class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """
    Ensemble of multiple anomaly detectors.

    Combines results from multiple detectors for more robust detection.
    """

    def __init__(
        self,
        detectors: Optional[List[BaseAnomalyDetector]] = None,
        voting: str = "soft",
        threshold: float = 0.5,
    ):
        """
        Initialize ensemble detector.

        Args:
            detectors: List of detectors to use
            voting: "soft" (average scores) or "hard" (majority vote)
            threshold: Threshold for final anomaly decision
        """
        self.detectors = detectors or []
        self.voting = voting
        self.threshold = threshold

    def add_detector(self, detector: BaseAnomalyDetector) -> None:
        """Add a detector to the ensemble."""
        self.detectors.append(detector)

    def fit(self, data: pd.DataFrame) -> "EnsembleAnomalyDetector":
        """Fit all detectors."""
        for detector in self.detectors:
            detector.fit(data)
        return self

    def detect_single(self, row: pd.Series) -> AnomalyResult:
        """Combine results from all detectors."""
        if not self.detectors:
            raise ValueError("No detectors in ensemble")

        results = [d.detect_single(row) for d in self.detectors]

        if self.voting == "soft":
            avg_score = np.mean([r.score for r in results])
            is_anomaly = avg_score > self.threshold
        else:
            votes = sum(1 for r in results if r.is_anomaly)
            is_anomaly = votes > len(results) / 2
            avg_score = votes / len(results)

        # Get most common anomaly type
        type_counts: Dict[AnomalyType, int] = {}
        for r in results:
            if r.is_anomaly:
                type_counts[r.anomaly_type] = type_counts.get(r.anomaly_type, 0) + 1

        anomaly_type = max(type_counts, key=type_counts.get) if type_counts else AnomalyType.UNKNOWN

        explanations = [r.explanation for r in results if r.is_anomaly]

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=avg_score,
            anomaly_type=anomaly_type,
            confidence=np.mean([r.confidence for r in results]),
            explanation=" | ".join(explanations) if explanations else "Normal behavior",
            details={"individual_results": [r.to_dict() for r in results]},
        )

    def detect(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies using ensemble."""
        return [self.detect_single(row) for _, row in data.iterrows()]


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data

    print("Loading sample data...")
    data = load_sample_data(source="bybit")

    if not data.empty:
        print(f"Loaded {len(data)} rows")

        print("\nTesting StatisticalAnomalyDetector...")
        detector = StatisticalAnomalyDetector()
        detector.fit(data.iloc[:-10])  # Train on all but last 10

        # Test on last 10 rows
        for _, row in data.tail(10).iterrows():
            result = detector.detect_single(row)
            if result.is_anomaly:
                print(f"ANOMALY: {result.explanation}")
            else:
                print(f"Normal: score={result.score:.3f}")
    else:
        print("Could not load sample data")
