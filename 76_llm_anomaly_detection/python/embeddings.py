"""
Embedding generation for LLM-based anomaly detection.

This module provides tools to convert financial data (text and numerical)
into vector embeddings suitable for anomaly detection.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedding generators."""

    @abstractmethod
    def embed(self, data: Any) -> np.ndarray:
        """Generate embedding for input data."""
        pass

    @abstractmethod
    def embed_batch(self, data: List[Any]) -> np.ndarray:
        """Generate embeddings for a batch of inputs."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Generate text embeddings using Sentence Transformers.

    This is useful for:
    - News headlines and articles
    - Social media sentiment
    - Company filings and reports
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: Sentence transformer model name
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded {model_name} with {self.embedding_dim}-dim embeddings")

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )


class OpenAIEmbedder(BaseEmbedder):
    """
    Generate embeddings using OpenAI's API.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name
            api_key: API key (uses env var if not provided)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )

        import os
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model

        # Embedding dimensions by model
        self.embedding_dim = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(model, 1536)

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return np.array(response.data[0].embedding)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        embeddings = [d.embedding for d in response.data]
        return np.array(embeddings)


class FinancialDataEmbedder(BaseEmbedder):
    """
    Convert numerical financial data into embeddings.

    Approaches:
    1. Direct feature encoding
    2. Statistical encoding
    3. Time series encoding with transformers
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        window_size: int = 20,
        normalize: bool = True,
    ):
        """
        Initialize financial data embedder.

        Args:
            embedding_dim: Output embedding dimension
            window_size: Lookback window for feature computation
            normalize: Whether to normalize features
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.normalize = normalize

        # Feature statistics for normalization
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def _compute_features(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """Extract features from OHLCV data."""
        features = []

        # Price features
        close = ohlcv["close"].values
        high = ohlcv["high"].values
        low = ohlcv["low"].values
        volume = ohlcv["volume"].values

        # Returns
        returns = np.diff(close) / close[:-1]
        log_returns = np.log(close[1:] / close[:-1])

        # Volatility (rolling std)
        if len(returns) >= self.window_size:
            vol = pd.Series(returns).rolling(self.window_size).std().values
        else:
            vol = np.std(returns) * np.ones(len(returns))

        # Volume features
        if len(volume) > 1:
            vol_ma = pd.Series(volume[1:]).rolling(min(self.window_size, len(volume)-1)).mean().values
            vol_ratio = volume[1:] / np.where(vol_ma > 0, vol_ma, 1)
        else:
            vol_ratio = np.array([1.0])

        # Price range
        if len(high) > 1:
            price_range = (high[1:] - low[1:]) / close[1:]
        else:
            price_range = np.array([0.0])

        # Combine features
        min_len = min(len(returns), len(vol_ratio), len(price_range))
        if min_len > 0:
            features = np.column_stack([
                returns[-min_len:],
                vol[-min_len:] if len(vol) >= min_len else np.zeros(min_len),
                vol_ratio[-min_len:],
                price_range[-min_len:],
            ])
        else:
            features = np.zeros((1, 4))

        return features

    def fit(self, ohlcv_data: pd.DataFrame) -> "FinancialDataEmbedder":
        """
        Fit normalizer on training data.

        Args:
            ohlcv_data: DataFrame with OHLCV columns
        """
        features = self._compute_features(ohlcv_data)
        self._feature_means = np.nanmean(features, axis=0)
        self._feature_stds = np.nanstd(features, axis=0)
        self._feature_stds = np.where(self._feature_stds > 0, self._feature_stds, 1)
        return self

    def embed(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """
        Generate embedding for OHLCV data.

        Args:
            ohlcv: DataFrame with OHLCV columns

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        features = self._compute_features(ohlcv)

        if self.normalize and self._feature_means is not None:
            features = (features - self._feature_means) / self._feature_stds

        # Aggregate to fixed size embedding
        if len(features) >= self.window_size:
            # Use last window_size rows
            window = features[-self.window_size:]
        else:
            # Pad with zeros
            pad_size = self.window_size - len(features)
            window = np.vstack([np.zeros((pad_size, features.shape[1])), features])

        # Flatten and project to embedding_dim
        flat = window.flatten()

        # Simple linear projection (in practice, use learned projection)
        if len(flat) >= self.embedding_dim:
            embedding = flat[:self.embedding_dim]
        else:
            embedding = np.pad(flat, (0, self.embedding_dim - len(flat)))

        return embedding

    def embed_batch(self, ohlcv_list: List[pd.DataFrame]) -> np.ndarray:
        """Generate embeddings for multiple OHLCV DataFrames."""
        embeddings = [self.embed(df) for df in ohlcv_list]
        return np.vstack(embeddings)


class MultiModalEmbedder:
    """
    Combine embeddings from multiple sources (text + numerical).

    This is useful for comprehensive anomaly detection using:
    - Price and volume data
    - News headlines
    - Social media sentiment
    """

    def __init__(
        self,
        text_embedder: Optional[BaseEmbedder] = None,
        financial_embedder: Optional[FinancialDataEmbedder] = None,
        fusion_method: str = "concat",
    ):
        """
        Initialize multi-modal embedder.

        Args:
            text_embedder: Embedder for text data
            financial_embedder: Embedder for financial data
            fusion_method: How to combine embeddings ("concat", "mean", "weighted")
        """
        self.text_embedder = text_embedder
        self.financial_embedder = financial_embedder
        self.fusion_method = fusion_method

    def embed(
        self,
        text: Optional[str] = None,
        ohlcv: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Generate combined embedding.

        Args:
            text: Text data (news, sentiment, etc.)
            ohlcv: Financial OHLCV data
            weights: Weights for each modality

        Returns:
            Combined embedding vector
        """
        embeddings = []

        if text is not None and self.text_embedder is not None:
            text_emb = self.text_embedder.embed(text)
            embeddings.append(("text", text_emb))

        if ohlcv is not None and self.financial_embedder is not None:
            fin_emb = self.financial_embedder.embed(ohlcv)
            embeddings.append(("financial", fin_emb))

        if not embeddings:
            raise ValueError("No data provided for embedding")

        if len(embeddings) == 1:
            return embeddings[0][1]

        if self.fusion_method == "concat":
            return np.concatenate([e[1] for e in embeddings])

        elif self.fusion_method == "mean":
            # Normalize to same dimension first
            max_dim = max(e[1].shape[0] for e in embeddings)
            normalized = []
            for name, emb in embeddings:
                if len(emb) < max_dim:
                    emb = np.pad(emb, (0, max_dim - len(emb)))
                normalized.append(emb)
            return np.mean(normalized, axis=0)

        elif self.fusion_method == "weighted":
            weights = weights or {"text": 0.5, "financial": 0.5}
            max_dim = max(e[1].shape[0] for e in embeddings)
            result = np.zeros(max_dim)
            for name, emb in embeddings:
                w = weights.get(name, 0.5)
                if len(emb) < max_dim:
                    emb = np.pad(emb, (0, max_dim - len(emb)))
                result += w * emb
            return result

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


def create_market_context_embedding(
    ohlcv: pd.DataFrame,
    news: Optional[List[str]] = None,
    text_model: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Create a comprehensive embedding of market context.

    This combines price/volume patterns with news sentiment
    for contextual anomaly detection.

    Args:
        ohlcv: OHLCV DataFrame
        news: List of recent news headlines
        text_model: Sentence transformer model name

    Returns:
        Combined embedding vector
    """
    # Financial embedding
    fin_embedder = FinancialDataEmbedder(embedding_dim=64)
    fin_emb = fin_embedder.embed(ohlcv)

    # Text embedding (if news provided)
    if news and len(news) > 0:
        try:
            text_embedder = SentenceTransformerEmbedder(model_name=text_model)
            # Combine all news into single embedding
            news_embeddings = text_embedder.embed_batch(news)
            news_emb = np.mean(news_embeddings, axis=0)

            # Concatenate
            return np.concatenate([fin_emb, news_emb])
        except ImportError:
            logger.warning("sentence-transformers not available, using only financial embedding")
            return fin_emb

    return fin_emb


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": price * (1 + np.random.randn(100) * 0.001),
        "high": price * (1 + abs(np.random.randn(100) * 0.005)),
        "low": price * (1 - abs(np.random.randn(100) * 0.005)),
        "close": price,
        "volume": 1000000 + np.random.randn(100) * 100000,
    })

    # Test financial embedder
    print("Testing FinancialDataEmbedder...")
    fin_embedder = FinancialDataEmbedder(embedding_dim=32)
    fin_embedder.fit(df)
    embedding = fin_embedder.embed(df)
    print(f"Financial embedding shape: {embedding.shape}")

    # Test text embedder (if available)
    try:
        print("\nTesting SentenceTransformerEmbedder...")
        text_embedder = SentenceTransformerEmbedder()
        text_emb = text_embedder.embed("Bitcoin price surges on ETF approval news")
        print(f"Text embedding shape: {text_emb.shape}")
    except ImportError:
        print("sentence-transformers not installed, skipping text embedder test")

    # Test multi-modal embedder
    print("\nTesting MultiModalEmbedder...")
    multi_embedder = MultiModalEmbedder(
        financial_embedder=fin_embedder,
    )
    combined_emb = multi_embedder.embed(ohlcv=df)
    print(f"Combined embedding shape: {combined_emb.shape}")
