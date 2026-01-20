# Chapter 76: LLM Anomaly Detection in Financial Markets

This chapter explores **Large Language Model (LLM)-based anomaly detection** for financial data analysis. We demonstrate how LLMs can identify unusual patterns, suspicious transactions, market manipulation, and other anomalies in both traditional stock markets and cryptocurrency exchanges like Bybit.

<p align="center">
<img src="https://i.imgur.com/Zx8KQPL.png" width="70%">
</p>

## Contents

1. [Introduction to LLM Anomaly Detection](#introduction-to-llm-anomaly-detection)
    * [Why Use LLMs for Anomaly Detection?](#why-use-llms-for-anomaly-detection)
    * [Traditional vs LLM-based Approaches](#traditional-vs-llm-based-approaches)
    * [Applications in Finance](#applications-in-finance)
2. [Theoretical Foundation](#theoretical-foundation)
    * [Anomaly Types in Financial Data](#anomaly-types-in-financial-data)
    * [LLM Representation Learning](#llm-representation-learning)
    * [Zero-shot and Few-shot Detection](#zero-shot-and-few-shot-detection)
3. [Detection Methods](#detection-methods)
    * [Text-based Anomaly Detection](#text-based-anomaly-detection)
    * [Time Series Embedding Anomalies](#time-series-embedding-anomalies)
    * [Multi-modal Anomaly Detection](#multi-modal-anomaly-detection)
4. [Practical Examples](#practical-examples)
    * [01: Detecting Unusual Trading Patterns](#01-detecting-unusual-trading-patterns)
    * [02: News-based Market Anomaly Detection](#02-news-based-market-anomaly-detection)
    * [03: Crypto Market Manipulation Detection (Bybit)](#03-crypto-market-manipulation-detection-bybit)
5. [Rust Implementation](#rust-implementation)
6. [Python Implementation](#python-implementation)
7. [Backtesting Framework](#backtesting-framework)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to LLM Anomaly Detection

Anomaly detection is crucial in financial markets for identifying fraud, market manipulation, unusual trading patterns, and other irregularities. Traditional statistical methods often fail to capture the complex, multi-dimensional nature of financial anomalies. LLMs offer a powerful alternative by leveraging their deep understanding of patterns and context.

### Why Use LLMs for Anomaly Detection?

```
LLM ADVANTAGES FOR FINANCIAL ANOMALY DETECTION:
======================================================================

+------------------------------------------------------------------+
|  1. CONTEXTUAL UNDERSTANDING                                      |
|     Traditional: Z-score flags "price up 5%" as anomaly           |
|     LLM: Considers context - "5% up after earnings beat = normal" |
|          vs "5% up with no news = suspicious"                     |
+------------------------------------------------------------------+
|  2. MULTI-MODAL ANALYSIS                                          |
|     Traditional: Analyzes price OR text separately                |
|     LLM: Combines price action + news + sentiment + order flow    |
|          for holistic anomaly assessment                          |
+------------------------------------------------------------------+
|  3. ZERO-SHOT CAPABILITY                                          |
|     Traditional: Requires labeled anomaly data for each type      |
|     LLM: Can detect novel anomaly types never seen before         |
|          by understanding what "normal" looks like                |
+------------------------------------------------------------------+
|  4. EXPLANATION GENERATION                                        |
|     Traditional: Flags anomaly with numeric score                 |
|     LLM: "This pattern resembles pump-and-dump: sudden volume     |
|          spike with coordinated social media activity"            |
+------------------------------------------------------------------+
```

### Traditional vs LLM-based Approaches

| Aspect | Traditional Methods | LLM-based Detection |
|--------|---------------------|---------------------|
| Data Types | Numerical only | Text, numerical, multi-modal |
| Training Data | Large labeled datasets | Few-shot or zero-shot |
| Novel Anomalies | Poor detection | Strong generalization |
| Explainability | Limited (scores only) | Natural language explanations |
| Context Awareness | Rule-based | Semantic understanding |
| Adaptation | Requires retraining | Prompt-based adaptation |
| Computation | Lightweight | More resource-intensive |

### Applications in Finance

```
FINANCIAL ANOMALY DETECTION USE CASES:
======================================================================

MARKET SURVEILLANCE
+------------------------------------------------------------------+
| - Pump-and-dump scheme detection                                  |
| - Wash trading identification                                     |
| - Front-running pattern recognition                               |
| - Spoofing and layering detection                                 |
+------------------------------------------------------------------+

RISK MANAGEMENT
+------------------------------------------------------------------+
| - Flash crash early warning                                       |
| - Liquidity crisis detection                                      |
| - Correlation breakdown alerts                                    |
| - Volatility regime anomalies                                     |
+------------------------------------------------------------------+

FRAUD DETECTION
+------------------------------------------------------------------+
| - Insider trading pattern recognition                             |
| - Account takeover attempts                                       |
| - Unusual transaction patterns                                    |
| - Fake news and market manipulation                               |
+------------------------------------------------------------------+

CRYPTO-SPECIFIC (Bybit, etc.)
+------------------------------------------------------------------+
| - Whale movement tracking                                         |
| - Exchange flow anomalies                                         |
| - DeFi exploit detection                                          |
| - Rug pull warning signals                                        |
+------------------------------------------------------------------+
```

## Theoretical Foundation

### Anomaly Types in Financial Data

Financial anomalies can be categorized into several types:

```
ANOMALY TAXONOMY:
======================================================================

1. POINT ANOMALIES (Single Instance)
   +---------------------------------------------------------------+
   |  - Individual data point significantly different from others   |
   |  - Example: Single large trade in illiquid market              |
   |  - Detection: Embedding distance from cluster centroid         |
   +---------------------------------------------------------------+

2. CONTEXTUAL ANOMALIES (Context-Dependent)
   +---------------------------------------------------------------+
   |  - Normal in one context, anomalous in another                 |
   |  - Example: High volume normal during earnings, suspicious     |
   |             on random Tuesday                                  |
   |  - Detection: LLM context understanding + conditional scoring  |
   +---------------------------------------------------------------+

3. COLLECTIVE ANOMALIES (Pattern-Based)
   +---------------------------------------------------------------+
   |  - Sequence of events that together indicate anomaly           |
   |  - Example: Series of small trades followed by large move      |
   |  - Detection: Sequence modeling with attention mechanisms      |
   +---------------------------------------------------------------+

4. SEMANTIC ANOMALIES (Meaning-Based)
   +---------------------------------------------------------------+
   |  - Anomalies in textual/semantic content                       |
   |  - Example: Press release with unusual language patterns       |
   |  - Detection: LLM semantic analysis and deviation scoring      |
   +---------------------------------------------------------------+
```

### LLM Representation Learning

LLMs create rich representations that capture semantic meaning:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class FinancialAnomalyEncoder(nn.Module):
    """
    Encode financial data (text + numerical) into anomaly-detection
    friendly representations using LLM backbone.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        numerical_features: int = 20,
        embedding_dim: int = 256
    ):
        super().__init__()

        # Text encoder (LLM backbone)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_dim = self.text_encoder.config.hidden_size

        # Numerical encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)
        )

        # Fusion and projection to anomaly space
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + 128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Anomaly scoring head
        self.anomaly_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(
        self,
        texts: list,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode multi-modal inputs into embeddings.

        Args:
            texts: List of text descriptions
            numerical_features: Tensor of shape (batch, num_features)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        # Tokenize and encode text
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            text_outputs = self.text_encoder(**inputs)
            # Use CLS token representation
            text_embeddings = text_outputs.last_hidden_state[:, 0]

        # Encode numerical features
        num_embeddings = self.numerical_encoder(numerical_features)

        # Fuse modalities
        combined = torch.cat([text_embeddings, num_embeddings], dim=-1)
        embeddings = self.fusion(combined)

        return embeddings

    def compute_anomaly_score(
        self,
        embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute anomaly score for embeddings.

        Args:
            embeddings: Query embeddings (batch, embedding_dim)
            reference_embeddings: Normal reference distribution

        Returns:
            Anomaly scores in [0, 1]
        """
        if reference_embeddings is not None:
            # Distance-based scoring
            centroid = reference_embeddings.mean(dim=0, keepdim=True)
            distances = torch.norm(embeddings - centroid, dim=-1)
            ref_distances = torch.norm(reference_embeddings - centroid, dim=-1)

            # Normalize by reference distribution
            mean_dist = ref_distances.mean()
            std_dist = ref_distances.std()
            z_scores = (distances - mean_dist) / (std_dist + 1e-8)

            # Convert to probability
            scores = torch.sigmoid(z_scores)
        else:
            # Use learned anomaly head
            scores = self.anomaly_head(embeddings).squeeze(-1)

        return scores


# Example usage
encoder = FinancialAnomalyEncoder()

# Sample financial events
texts = [
    "Apple reports quarterly earnings beating analyst expectations",
    "Unknown company sees 500% volume spike with no news",
    "Fed announces interest rate decision as expected"
]

numerical = torch.randn(3, 20)  # Mock numerical features

embeddings = encoder.encode(texts, numerical)
scores = encoder.compute_anomaly_score(embeddings)

print(f"Anomaly scores: {scores.tolist()}")
```

### Zero-shot and Few-shot Detection

LLMs excel at detecting anomalies without extensive labeled data:

```python
class ZeroShotAnomalyDetector:
    """
    Detect anomalies using LLM's zero-shot capabilities.

    Uses prompt engineering to leverage LLM's understanding
    of what constitutes "normal" financial behavior.
    """

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name

        self.system_prompt = """You are a financial anomaly detection expert.
Analyze the following market data and determine if it represents
anomalous behavior. Consider:
- Historical context and typical patterns
- Market conditions and news
- Statistical likelihood
- Potential manipulation indicators

Respond with:
1. ANOMALY_SCORE: A score from 0.0 (normal) to 1.0 (highly anomalous)
2. ANOMALY_TYPE: Category of anomaly (if any)
3. EXPLANATION: Brief reasoning for your assessment
4. CONFIDENCE: Your confidence level (low/medium/high)
"""

    def analyze(self, market_data: dict) -> dict:
        """
        Analyze market data for anomalies.

        Args:
            market_data: Dictionary containing:
                - symbol: Trading symbol
                - price_change: Recent price change %
                - volume_ratio: Volume vs average
                - news: Recent news headlines
                - context: Additional context

        Returns:
            Dictionary with anomaly assessment
        """
        prompt = self._format_prompt(market_data)

        # In production, call actual LLM API
        # response = openai.ChatCompletion.create(...)

        # Mock response for demonstration
        response = self._mock_analysis(market_data)

        return response

    def _format_prompt(self, data: dict) -> str:
        """Format market data as analysis prompt."""
        return f"""
MARKET DATA ANALYSIS REQUEST
============================

Symbol: {data.get('symbol', 'UNKNOWN')}
Timestamp: {data.get('timestamp', 'N/A')}

PRICE ACTION:
- Current Price: ${data.get('price', 0):.2f}
- Price Change (24h): {data.get('price_change', 0):.2f}%
- Volume Ratio (vs 20-day avg): {data.get('volume_ratio', 1):.2f}x

MARKET CONTEXT:
- Overall Market: {data.get('market_trend', 'N/A')}
- Sector Performance: {data.get('sector_trend', 'N/A')}
- VIX Level: {data.get('vix', 'N/A')}

RECENT NEWS:
{data.get('news', 'No recent news')}

ORDER FLOW:
- Bid/Ask Imbalance: {data.get('order_imbalance', 0):.2f}
- Large Trade Count: {data.get('large_trades', 0)}

Please analyze this data for potential anomalies.
"""

    def _mock_analysis(self, data: dict) -> dict:
        """Mock analysis for demonstration."""
        volume_ratio = data.get('volume_ratio', 1)
        price_change = abs(data.get('price_change', 0))

        # Simple heuristic for demonstration
        if volume_ratio > 5 and price_change > 10:
            return {
                'anomaly_score': 0.85,
                'anomaly_type': 'UNUSUAL_ACTIVITY',
                'explanation': 'Significant volume spike with large price '
                              'movement without corresponding news catalyst. '
                              'Pattern suggests potential manipulation or '
                              'undisclosed material information.',
                'confidence': 'high'
            }
        elif volume_ratio > 3:
            return {
                'anomaly_score': 0.5,
                'anomaly_type': 'ELEVATED_VOLUME',
                'explanation': 'Volume elevated but within bounds that could '
                              'be explained by normal market activity.',
                'confidence': 'medium'
            }
        else:
            return {
                'anomaly_score': 0.1,
                'anomaly_type': 'NORMAL',
                'explanation': 'Activity appears within normal parameters.',
                'confidence': 'high'
            }


# Example usage
detector = ZeroShotAnomalyDetector()

# Normal case
normal_data = {
    'symbol': 'AAPL',
    'price': 175.50,
    'price_change': 2.1,
    'volume_ratio': 1.2,
    'news': 'Apple announces new product launch event',
    'market_trend': 'Bullish',
    'vix': 15.2
}

# Suspicious case
suspicious_data = {
    'symbol': 'XYZ',
    'price': 3.50,
    'price_change': 45.0,
    'volume_ratio': 12.5,
    'news': 'No recent news',
    'market_trend': 'Neutral',
    'vix': 15.2,
    'large_trades': 50
}

print("Normal case:", detector.analyze(normal_data))
print("Suspicious case:", detector.analyze(suspicious_data))
```

## Detection Methods

### Text-based Anomaly Detection

Detect anomalies in financial text such as press releases, social media, and news:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.covariance import EllipticEnvelope
from typing import List, Tuple

class TextAnomalyDetector:
    """
    Detect anomalous text patterns in financial communications.

    Uses LLM embeddings to create a representation space where
    anomalous texts can be identified by their distance from
    the normal distribution.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        contamination: float = 0.05
    ):
        """
        Initialize the detector.

        Args:
            model_name: Pre-trained model for text encoding
            contamination: Expected proportion of anomalies
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.contamination = contamination
        self.detector = None
        self.reference_mean = None
        self.reference_std = None

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding.numpy().flatten())

        return np.array(embeddings)

    def fit(self, normal_texts: List[str]):
        """
        Fit the detector on normal texts.

        Args:
            normal_texts: List of known normal financial texts
        """
        embeddings = self._encode_texts(normal_texts)

        # Fit outlier detector
        self.detector = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        self.detector.fit(embeddings)

        # Store statistics for scoring
        self.reference_mean = embeddings.mean(axis=0)
        self.reference_std = embeddings.std(axis=0) + 1e-8

    def detect(self, texts: List[str]) -> List[dict]:
        """
        Detect anomalies in new texts.

        Args:
            texts: Texts to analyze

        Returns:
            List of detection results
        """
        if self.detector is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        embeddings = self._encode_texts(texts)

        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.detector.predict(embeddings)

        # Get anomaly scores (Mahalanobis distance-based)
        scores = -self.detector.score_samples(embeddings)

        # Normalize scores to [0, 1]
        scores_normalized = 1 / (1 + np.exp(-scores))

        results = []
        for i, (text, pred, score) in enumerate(zip(texts, predictions, scores_normalized)):
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'is_anomaly': pred == -1,
                'anomaly_score': float(score),
                'z_scores': self._compute_feature_zscore(embeddings[i])
            })

        return results

    def _compute_feature_zscore(self, embedding: np.ndarray) -> float:
        """Compute aggregate z-score for embedding."""
        z_scores = (embedding - self.reference_mean) / self.reference_std
        return float(np.abs(z_scores).mean())


# Example usage
detector = TextAnomalyDetector()

# Normal financial texts
normal_texts = [
    "Company reports quarterly revenue of $5.2 billion",
    "Board announces 10% dividend increase",
    "CEO discusses expansion plans in earnings call",
    "Analyst upgrades stock to buy with $150 target",
    "Company acquires competitor for $2 billion",
    "Q3 earnings beat consensus by 5 cents",
    "Management reaffirms full-year guidance",
    "New product launch drives strong demand"
]

# Fit on normal texts
detector.fit(normal_texts)

# Test texts (mix of normal and anomalous)
test_texts = [
    "Quarterly results in line with expectations",  # Normal
    "!!! HUGE NEWS - STOCK WILL 10X GUARANTEED - BUY NOW !!!",  # Pump scheme
    "Company files for Chapter 11 bankruptcy protection",  # News (unusual but not manipulation)
    "insider info: massive deal coming, get in before moon",  # Manipulation
    "Strong performance driven by core business growth"  # Normal
]

results = detector.detect(test_texts)
for r in results:
    print(f"Anomaly: {r['is_anomaly']}, Score: {r['anomaly_score']:.3f} - {r['text']}")
```

### Time Series Embedding Anomalies

Detect anomalies in price and volume time series using LLM-inspired embeddings:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class TimeSeriesAnomalyEncoder(nn.Module):
    """
    Transformer-based encoder for financial time series anomaly detection.

    Converts OHLCV data into embeddings that can be used for
    anomaly detection via distance metrics or reconstruction error.
    """

    def __init__(
        self,
        input_dim: int = 5,  # OHLCV
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        seq_length: int = 60,
        dropout: float = 0.1
    ):
        super().__init__()

        self.seq_length = seq_length
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_length, d_model) * 0.1
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projections
        self.embedding_proj = nn.Linear(d_model, d_model)
        self.reconstruction_head = nn.Linear(d_model, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series into embedding.

        Args:
            x: Input tensor of shape (batch, seq_length, input_dim)

        Returns:
            Embeddings of shape (batch, d_model)
        """
        # Project to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1)]

        # Transformer encoding
        encoded = self.transformer(x)

        # Global pooling
        embedding = encoded.mean(dim=1)
        embedding = self.embedding_proj(embedding)

        return embedding

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input through autoencoder.

        Args:
            x: Input tensor

        Returns:
            Reconstructed tensor
        """
        # Encode
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1)]
        encoded = self.transformer(x)

        # Decode
        reconstructed = self.reconstruction_head(encoded)

        return reconstructed

    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        reference_embeddings: Optional[torch.Tensor] = None,
        method: str = 'reconstruction'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly scores.

        Args:
            x: Input time series
            reference_embeddings: Normal reference embeddings for distance method
            method: 'reconstruction' or 'distance'

        Returns:
            Tuple of (anomaly_scores, embeddings)
        """
        embeddings = self.encode(x)

        if method == 'reconstruction':
            reconstructed = self.reconstruct(x)
            # Reconstruction error as anomaly score
            mse = ((x - reconstructed) ** 2).mean(dim=(1, 2))
            scores = mse

        elif method == 'distance' and reference_embeddings is not None:
            # Distance to normal cluster centroid
            centroid = reference_embeddings.mean(dim=0, keepdim=True)
            distances = torch.norm(embeddings - centroid, dim=-1)

            # Z-score normalization
            ref_distances = torch.norm(reference_embeddings - centroid, dim=-1)
            mean_dist = ref_distances.mean()
            std_dist = ref_distances.std() + 1e-8
            scores = (distances - mean_dist) / std_dist

        else:
            raise ValueError(f"Unknown method: {method}")

        return scores, embeddings


class TimeSeriesAnomalyDetector:
    """
    Complete anomaly detection system for financial time series.
    """

    def __init__(
        self,
        seq_length: int = 60,
        threshold_percentile: float = 95
    ):
        self.seq_length = seq_length
        self.threshold_percentile = threshold_percentile

        self.encoder = TimeSeriesAnomalyEncoder(seq_length=seq_length)
        self.threshold = None
        self.reference_embeddings = None

    def fit(self, normal_data: np.ndarray, epochs: int = 50):
        """
        Train the detector on normal data.

        Args:
            normal_data: Array of shape (num_samples, seq_length, features)
            epochs: Training epochs
        """
        self.encoder.train()
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)

        tensor_data = torch.FloatTensor(normal_data)

        for epoch in range(epochs):
            optimizer.zero_grad()

            reconstructed = self.encoder.reconstruct(tensor_data)
            loss = nn.MSELoss()(reconstructed, tensor_data)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        # Compute reference embeddings and threshold
        self.encoder.eval()
        with torch.no_grad():
            scores, embeddings = self.encoder.compute_anomaly_score(
                tensor_data, method='reconstruction'
            )
            self.reference_embeddings = embeddings
            self.threshold = np.percentile(
                scores.numpy(),
                self.threshold_percentile
            )

        print(f"Threshold set at {self.threshold_percentile}th percentile: {self.threshold:.6f}")

    def detect(
        self,
        data: np.ndarray
    ) -> list:
        """
        Detect anomalies in new data.

        Args:
            data: Array of shape (num_samples, seq_length, features)

        Returns:
            List of detection results
        """
        self.encoder.eval()
        tensor_data = torch.FloatTensor(data)

        with torch.no_grad():
            scores, embeddings = self.encoder.compute_anomaly_score(
                tensor_data,
                reference_embeddings=self.reference_embeddings,
                method='reconstruction'
            )

        results = []
        for i, score in enumerate(scores.numpy()):
            results.append({
                'index': i,
                'anomaly_score': float(score),
                'is_anomaly': score > self.threshold,
                'threshold': self.threshold
            })

        return results


# Example usage
def generate_sample_data(n_samples: int, seq_length: int, anomaly_ratio: float = 0.1):
    """Generate synthetic OHLCV data with injected anomalies."""
    np.random.seed(42)

    # Normal data (random walk)
    data = np.zeros((n_samples, seq_length, 5))

    for i in range(n_samples):
        # Generate price series
        returns = np.random.randn(seq_length) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))

        # OHLCV
        data[i, :, 0] = prices * (1 + np.random.randn(seq_length) * 0.005)  # Open
        data[i, :, 1] = prices * (1 + np.abs(np.random.randn(seq_length) * 0.01))  # High
        data[i, :, 2] = prices * (1 - np.abs(np.random.randn(seq_length) * 0.01))  # Low
        data[i, :, 3] = prices  # Close
        data[i, :, 4] = np.random.exponential(1000, seq_length)  # Volume

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    for idx in anomaly_indices:
        # Inject spike
        spike_pos = np.random.randint(seq_length // 2, seq_length)
        data[idx, spike_pos:, 3] *= 1.5  # Price spike
        data[idx, spike_pos:, 4] *= 10   # Volume spike

    return data, anomaly_indices


# Generate data
data, anomaly_indices = generate_sample_data(100, 60, 0.1)
print(f"Generated {len(data)} samples, {len(anomaly_indices)} anomalies")

# Split into train (normal only) and test
normal_mask = ~np.isin(np.arange(len(data)), anomaly_indices)
train_data = data[normal_mask][:70]
test_data = data

# Train detector
detector = TimeSeriesAnomalyDetector(seq_length=60)
detector.fit(train_data, epochs=30)

# Detect anomalies
results = detector.detect(test_data)

# Evaluate
detected_anomalies = [r['index'] for r in results if r['is_anomaly']]
true_positives = len(set(detected_anomalies) & set(anomaly_indices))
false_positives = len(detected_anomalies) - true_positives
precision = true_positives / max(len(detected_anomalies), 1)
recall = true_positives / max(len(anomaly_indices), 1)

print(f"\nResults:")
print(f"True Anomalies: {len(anomaly_indices)}")
print(f"Detected: {len(detected_anomalies)}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
```

### Multi-modal Anomaly Detection

Combine multiple data sources for comprehensive anomaly detection:

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

class MultiModalAnomalyDetector(nn.Module):
    """
    Multi-modal anomaly detector combining:
    - Price/volume time series
    - News/text sentiment
    - Order flow data
    - Social media signals
    """

    def __init__(
        self,
        price_dim: int = 5,
        text_dim: int = 768,
        orderflow_dim: int = 10,
        social_dim: int = 20,
        hidden_dim: int = 128,
        output_dim: int = 64
    ):
        super().__init__()

        # Modality encoders
        self.price_encoder = nn.Sequential(
            nn.Linear(price_dim * 60, hidden_dim),  # Flattened time series
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        self.orderflow_encoder = nn.Sequential(
            nn.Linear(orderflow_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.social_encoder = nn.Sequential(
            nn.Linear(social_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Fusion and anomaly scoring
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        self.anomaly_scorer = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Modality-specific anomaly heads
        self.price_anomaly = nn.Linear(output_dim, 1)
        self.text_anomaly = nn.Linear(output_dim, 1)
        self.orderflow_anomaly = nn.Linear(output_dim, 1)
        self.social_anomaly = nn.Linear(output_dim, 1)

    def forward(
        self,
        price_data: torch.Tensor,
        text_embedding: torch.Tensor,
        orderflow_data: torch.Tensor,
        social_data: torch.Tensor,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for anomaly detection.

        Args:
            price_data: (batch, seq_len, price_dim)
            text_embedding: (batch, text_dim)
            orderflow_data: (batch, orderflow_dim)
            social_data: (batch, social_dim)
            return_details: Whether to return per-modality scores

        Returns:
            Dictionary with anomaly scores
        """
        batch_size = price_data.size(0)

        # Encode each modality
        price_flat = price_data.reshape(batch_size, -1)
        price_emb = self.price_encoder(price_flat)
        text_emb = self.text_encoder(text_embedding)
        orderflow_emb = self.orderflow_encoder(orderflow_data)
        social_emb = self.social_encoder(social_data)

        # Stack for attention
        modal_embeddings = torch.stack([
            price_emb, text_emb, orderflow_emb, social_emb
        ], dim=1)  # (batch, 4, output_dim)

        # Cross-modal attention
        attended, _ = self.cross_attention(
            modal_embeddings,
            modal_embeddings,
            modal_embeddings
        )

        # Concatenate and fuse
        fused = self.fusion(attended.reshape(batch_size, -1))

        # Compute overall anomaly score
        overall_score = self.anomaly_scorer(fused)

        result = {
            'overall_anomaly_score': overall_score.squeeze(-1),
            'fused_embedding': fused
        }

        if return_details:
            result['price_anomaly'] = torch.sigmoid(
                self.price_anomaly(price_emb)
            ).squeeze(-1)
            result['text_anomaly'] = torch.sigmoid(
                self.text_anomaly(text_emb)
            ).squeeze(-1)
            result['orderflow_anomaly'] = torch.sigmoid(
                self.orderflow_anomaly(orderflow_emb)
            ).squeeze(-1)
            result['social_anomaly'] = torch.sigmoid(
                self.social_anomaly(social_emb)
            ).squeeze(-1)

        return result


class AnomalyAnalyzer:
    """
    High-level analyzer that combines multi-modal detection
    with explanation generation.
    """

    def __init__(self):
        self.detector = MultiModalAnomalyDetector()
        self.threshold = 0.7

    def analyze(
        self,
        market_event: Dict
    ) -> Dict:
        """
        Analyze a market event for anomalies.

        Args:
            market_event: Dictionary containing multi-modal data

        Returns:
            Analysis results with explanations
        """
        # Prepare tensors (mock implementation)
        price_data = torch.randn(1, 60, 5)
        text_emb = torch.randn(1, 768)
        orderflow = torch.randn(1, 10)
        social = torch.randn(1, 20)

        # Get scores
        with torch.no_grad():
            results = self.detector(
                price_data, text_emb, orderflow, social,
                return_details=True
            )

        overall_score = results['overall_anomaly_score'].item()

        # Generate explanation
        explanation = self._generate_explanation(results, market_event)

        return {
            'symbol': market_event.get('symbol', 'UNKNOWN'),
            'timestamp': market_event.get('timestamp'),
            'is_anomaly': overall_score > self.threshold,
            'overall_score': overall_score,
            'component_scores': {
                'price': results['price_anomaly'].item(),
                'text': results['text_anomaly'].item(),
                'orderflow': results['orderflow_anomaly'].item(),
                'social': results['social_anomaly'].item()
            },
            'explanation': explanation,
            'recommended_action': self._recommend_action(overall_score)
        }

    def _generate_explanation(
        self,
        results: Dict,
        event: Dict
    ) -> str:
        """Generate human-readable explanation."""
        components = []

        if results['price_anomaly'].item() > 0.6:
            components.append("unusual price movement pattern detected")

        if results['text_anomaly'].item() > 0.6:
            components.append("suspicious text/news content identified")

        if results['orderflow_anomaly'].item() > 0.6:
            components.append("abnormal order flow patterns observed")

        if results['social_anomaly'].item() > 0.6:
            components.append("unusual social media activity detected")

        if not components:
            return "No significant anomalies detected across all modalities."

        return f"Anomaly detected: {', '.join(components)}."

    def _recommend_action(self, score: float) -> str:
        """Recommend action based on score."""
        if score > 0.9:
            return "IMMEDIATE_REVIEW"
        elif score > 0.7:
            return "ELEVATED_MONITORING"
        elif score > 0.5:
            return "STANDARD_MONITORING"
        else:
            return "NO_ACTION_REQUIRED"


# Example usage
analyzer = AnomalyAnalyzer()

event = {
    'symbol': 'BTCUSDT',
    'timestamp': '2024-01-15T14:30:00Z',
    'price_change': 5.2,
    'volume_ratio': 3.5,
    'news': 'Bitcoin sees sudden surge with no apparent catalyst'
}

result = analyzer.analyze(event)
print(f"Symbol: {result['symbol']}")
print(f"Is Anomaly: {result['is_anomaly']}")
print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Component Scores: {result['component_scores']}")
print(f"Explanation: {result['explanation']}")
print(f"Action: {result['recommended_action']}")
```

## Practical Examples

### 01: Detecting Unusual Trading Patterns

See `python/examples/01_unusual_trading.py` for complete implementation.

```python
# Quick start: Detect unusual trading patterns
from python.detector import TradingPatternDetector
from python.data_loader import YahooFinanceLoader

# Load market data
loader = YahooFinanceLoader()
spy_data = loader.get_daily("SPY", period="1y")

# Initialize detector
detector = TradingPatternDetector(
    lookback_window=60,
    threshold_percentile=95
)

# Fit on historical data
detector.fit(spy_data)

# Detect anomalies in recent data
anomalies = detector.detect(spy_data[-30:])

print("Detected Anomalies:")
for a in anomalies:
    if a['is_anomaly']:
        print(f"  Date: {a['date']}, Score: {a['score']:.3f}")
        print(f"  Reason: {a['explanation']}")
```

### 02: News-based Market Anomaly Detection

See `python/examples/02_news_anomaly.py` for complete implementation.

```python
# News-based anomaly detection
from python.detector import NewsAnomalyDetector

detector = NewsAnomalyDetector()

# Analyze news headlines
headlines = [
    "Apple reports strong Q4 earnings, stock rises 3%",
    "!!URGENT!! This penny stock will 100x tomorrow - insider info",
    "Fed maintains interest rates as expected",
    "Company X CEO arrested for fraud, shares halted"
]

results = detector.analyze_batch(headlines)

for headline, result in zip(headlines, results):
    print(f"\n{headline[:50]}...")
    print(f"  Anomaly Score: {result['score']:.3f}")
    print(f"  Type: {result['anomaly_type']}")
    print(f"  Action: {result['recommended_action']}")
```

### 03: Crypto Market Manipulation Detection (Bybit)

See `python/examples/03_crypto_manipulation.py` for complete implementation.

```python
# Crypto manipulation detection on Bybit
from python.data_loader import BybitDataLoader
from python.detector import CryptoManipulationDetector

# Initialize Bybit loader
bybit = BybitDataLoader()

# Get recent BTC data
btc_data = bybit.get_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    days=30
)

# Initialize manipulation detector
detector = CryptoManipulationDetector(
    volume_spike_threshold=5.0,
    price_spike_threshold=3.0
)

# Fit on data
detector.fit(btc_data)

# Real-time monitoring (simulation)
for i in range(-10, 0):
    window = btc_data.iloc[i-60:i]
    result = detector.analyze_window(window)

    if result['is_manipulation_suspected']:
        print(f"\n!!! ALERT at {window.index[-1]} !!!")
        print(f"  Manipulation Type: {result['manipulation_type']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Indicators: {result['indicators']}")
```

## Rust Implementation

The Rust implementation provides high-performance anomaly detection for production environments. See `rust/` directory for complete code.

```rust
//! LLM Anomaly Detection - Rust Implementation
//!
//! High-performance anomaly detection for financial data,
//! designed for low-latency production environments.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub timestamp: i64,
    pub symbol: String,
    pub anomaly_score: f64,
    pub is_anomaly: bool,
    pub anomaly_type: AnomalyType,
    pub confidence: f64,
    pub explanation: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    Normal,
    VolumeSurge,
    PriceSpike,
    PatternAnomaly,
    ManipulationSuspected,
    Unknown,
}

/// Statistical anomaly detector using z-scores and rolling statistics
pub struct StatisticalAnomalyDetector {
    lookback_window: usize,
    volume_threshold: f64,
    price_threshold: f64,
    price_history: Vec<f64>,
    volume_history: Vec<f64>,
}

impl StatisticalAnomalyDetector {
    pub fn new(lookback_window: usize, volume_threshold: f64, price_threshold: f64) -> Self {
        Self {
            lookback_window,
            volume_threshold,
            price_threshold,
            price_history: Vec::with_capacity(lookback_window),
            volume_history: Vec::with_capacity(lookback_window),
        }
    }

    pub fn update(&mut self, price: f64, volume: f64) {
        self.price_history.push(price);
        self.volume_history.push(volume);

        // Keep only lookback window
        if self.price_history.len() > self.lookback_window {
            self.price_history.remove(0);
            self.volume_history.remove(0);
        }
    }

    pub fn detect(&self, price: f64, volume: f64) -> AnomalyResult {
        let price_z = self.compute_z_score(price, &self.price_history);
        let volume_z = self.compute_z_score(volume, &self.volume_history);

        let mut anomaly_type = AnomalyType::Normal;
        let mut score = 0.0;

        if volume_z.abs() > self.volume_threshold {
            anomaly_type = AnomalyType::VolumeSurge;
            score = score.max(volume_z.abs() / 10.0);
        }

        if price_z.abs() > self.price_threshold {
            anomaly_type = if anomaly_type == AnomalyType::VolumeSurge {
                AnomalyType::ManipulationSuspected
            } else {
                AnomalyType::PriceSpike
            };
            score = score.max(price_z.abs() / 10.0);
        }

        score = score.min(1.0);

        AnomalyResult {
            timestamp: chrono::Utc::now().timestamp(),
            symbol: String::new(),
            anomaly_score: score,
            is_anomaly: anomaly_type != AnomalyType::Normal,
            anomaly_type,
            confidence: self.compute_confidence(price_z, volume_z),
            explanation: self.generate_explanation(price_z, volume_z, anomaly_type),
        }
    }

    fn compute_z_score(&self, value: f64, history: &[f64]) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        let mean: f64 = history.iter().sum::<f64>() / history.len() as f64;
        let variance: f64 = history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / history.len() as f64;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        (value - mean) / std
    }

    fn compute_confidence(&self, price_z: f64, volume_z: f64) -> f64 {
        let max_z = price_z.abs().max(volume_z.abs());
        1.0 - (-max_z * 0.5).exp()
    }

    fn generate_explanation(&self, price_z: f64, volume_z: f64, anomaly_type: AnomalyType) -> String {
        match anomaly_type {
            AnomalyType::Normal => "No anomaly detected".to_string(),
            AnomalyType::VolumeSurge => {
                format!("Volume surge detected (z-score: {:.2})", volume_z)
            }
            AnomalyType::PriceSpike => {
                format!("Price spike detected (z-score: {:.2})", price_z)
            }
            AnomalyType::ManipulationSuspected => {
                format!(
                    "Potential manipulation: both price (z={:.2}) and volume (z={:.2}) anomalous",
                    price_z, volume_z
                )
            }
            _ => "Unknown anomaly pattern".to_string(),
        }
    }
}

/// Bybit market data for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    pub symbol: String,
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
}

/// Real-time anomaly monitor for cryptocurrency markets
pub struct CryptoAnomalyMonitor {
    detectors: HashMap<String, StatisticalAnomalyDetector>,
    lookback_window: usize,
    volume_threshold: f64,
    price_threshold: f64,
}

impl CryptoAnomalyMonitor {
    pub fn new(lookback_window: usize, volume_threshold: f64, price_threshold: f64) -> Self {
        Self {
            detectors: HashMap::new(),
            lookback_window,
            volume_threshold,
            price_threshold,
        }
    }

    pub fn process_tick(&mut self, tick: &MarketTick) -> AnomalyResult {
        let detector = self
            .detectors
            .entry(tick.symbol.clone())
            .or_insert_with(|| {
                StatisticalAnomalyDetector::new(
                    self.lookback_window,
                    self.volume_threshold,
                    self.price_threshold,
                )
            });

        let mut result = detector.detect(tick.price, tick.volume);
        result.symbol = tick.symbol.clone();
        result.timestamp = tick.timestamp;

        detector.update(tick.price, tick.volume);

        result
    }

    pub fn get_active_symbols(&self) -> Vec<&String> {
        self.detectors.keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_detection() {
        let mut detector = StatisticalAnomalyDetector::new(20, 3.0, 3.0);

        // Add normal data
        for i in 0..20 {
            detector.update(100.0 + (i as f64) * 0.1, 1000.0);
        }

        let result = detector.detect(101.0, 1050.0);
        assert!(!result.is_anomaly);
        assert_eq!(result.anomaly_type, AnomalyType::Normal);
    }

    #[test]
    fn test_volume_spike_detection() {
        let mut detector = StatisticalAnomalyDetector::new(20, 3.0, 3.0);

        // Add normal data
        for _ in 0..20 {
            detector.update(100.0, 1000.0);
        }

        // Test with volume spike
        let result = detector.detect(100.0, 10000.0);
        assert!(result.is_anomaly);
        assert_eq!(result.anomaly_type, AnomalyType::VolumeSurge);
    }

    #[test]
    fn test_manipulation_detection() {
        let mut detector = StatisticalAnomalyDetector::new(20, 3.0, 3.0);

        // Add normal data
        for _ in 0..20 {
            detector.update(100.0, 1000.0);
        }

        // Both price and volume spike = manipulation suspected
        let result = detector.detect(150.0, 10000.0);
        assert!(result.is_anomaly);
        assert_eq!(result.anomaly_type, AnomalyType::ManipulationSuspected);
    }
}
```

## Python Implementation

The Python implementation includes comprehensive modules for research and development. See `python/` directory for full code.

**Key modules:**

| Module | Description |
|--------|-------------|
| `detector.py` | Core anomaly detection algorithms |
| `data_loader.py` | Yahoo Finance and Bybit data loaders |
| `embeddings.py` | LLM-based text and time series embeddings |
| `signals.py` | Trading signal generation from anomaly scores |
| `backtest.py` | Backtesting framework for anomaly-based strategies |
| `evaluate.py` | Evaluation metrics (precision, recall, F1, etc.) |

## Backtesting Framework

Test anomaly detection strategies on historical data:

```python
from python.backtest import AnomalyBacktester
from python.detector import MultiModalAnomalyDetector
from python.data_loader import BybitDataLoader

# Load historical data
bybit = BybitDataLoader()
btc_data = bybit.get_historical_klines("BTCUSDT", "1h", days=90)

# Initialize backtester
backtester = AnomalyBacktester(
    initial_capital=100000,
    anomaly_threshold=0.7,
    position_size=0.1
)

# Run backtest
results = backtester.run(
    data=btc_data,
    detector=MultiModalAnomalyDetector(),
    strategy="avoid_anomalies"  # Reduce exposure during anomalies
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Anomalies Avoided: {results['anomalies_detected']}")
```

## Best Practices

### Detection Guidelines

```
LLM ANOMALY DETECTION BEST PRACTICES:
======================================================================

1. DATA PREPARATION
   +----------------------------------------------------------------+
   | - Normalize features before embedding                           |
   | - Handle missing data explicitly                                |
   | - Preserve temporal order for time series                       |
   | - Separate training data by time (avoid future leakage)        |
   +----------------------------------------------------------------+

2. MODEL SELECTION
   +----------------------------------------------------------------+
   | - Use domain-specific models (FinBERT for finance)             |
   | - Consider compute constraints for real-time detection          |
   | - Balance precision vs recall based on use case                 |
   | - Ensemble multiple detection methods                           |
   +----------------------------------------------------------------+

3. THRESHOLD TUNING
   +----------------------------------------------------------------+
   | - Set thresholds based on acceptable false positive rate        |
   | - Use different thresholds for different anomaly types         |
   | - Adaptive thresholds for changing market conditions           |
   | - Regular recalibration as market dynamics evolve              |
   +----------------------------------------------------------------+

4. EVALUATION
   +----------------------------------------------------------------+
   | - Use time-based train/test splits                              |
   | - Report precision, recall, and F1 at multiple thresholds      |
   | - Evaluate on different market regimes                          |
   | - Consider operational metrics (latency, throughput)           |
   +----------------------------------------------------------------+

5. PRODUCTION DEPLOYMENT
   +----------------------------------------------------------------+
   | - Implement circuit breakers for system stability               |
   | - Log all anomaly detections for audit                          |
   | - Human-in-the-loop for high-confidence anomalies              |
   | - Regular model performance monitoring                          |
   +----------------------------------------------------------------+
```

### Common Pitfalls

```
COMMON MISTAKES TO AVOID:
======================================================================

X Using future data in training/evaluation
  -> Always use strict temporal splits

X Over-fitting to historical anomaly patterns
  -> Test on novel anomaly types

X Ignoring class imbalance (anomalies are rare)
  -> Use appropriate metrics (F1, PR-AUC)

X Single modality detection
  -> Combine multiple data sources

X Static thresholds
  -> Adapt to changing market conditions

X No explanation for detections
  -> Always provide reasoning for alerts

X Delayed detection
  -> Optimize for low-latency in production
```

## Resources

### Papers

1. **Are Large Language Models Anomaly Detectors?** (Chen et al., 2023)
   - https://arxiv.org/abs/2306.04069

2. **Deep Learning for Anomaly Detection: A Review** (Pang et al., 2021)
   - https://arxiv.org/abs/2007.02500

3. **Time-Series Anomaly Detection Service at Microsoft** (Ren et al., 2019)
   - https://arxiv.org/abs/1906.03821

4. **Financial Fraud Detection with Deep Learning** (Zhang et al., 2020)
   - Various financial fraud detection approaches

### Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| KDD Cup 1999 | Network intrusion data | 4.9M samples |
| Credit Card Fraud | Anonymized credit transactions | 284K samples |
| Yahoo S5 | Time series anomaly benchmark | 367 series |
| NAB | Numenta Anomaly Benchmark | 58 files |

### Tools & Libraries

- [PyOD](https://github.com/yzhao062/pyod) - Python outlier detection
- [Alibi Detect](https://github.com/SeldonIO/alibi-detect) - Anomaly detection library
- [ADTK](https://github.com/arundo/adtk) - Anomaly detection toolkit
- [Candle](https://github.com/huggingface/candle) - Rust ML framework

### Directory Structure

```
76_llm_anomaly_detection/
+-- README.md              # This file (English)
+-- README.ru.md           # Russian translation
+-- readme.simple.md       # Beginner-friendly explanation
+-- readme.simple.ru.md    # Beginner-friendly (Russian)
+-- python/
|   +-- __init__.py
|   +-- detector.py        # Core anomaly detection
|   +-- embeddings.py      # LLM embedding generation
|   +-- data_loader.py     # Yahoo Finance & Bybit loaders
|   +-- signals.py         # Trading signal generation
|   +-- backtest.py        # Backtesting framework
|   +-- evaluate.py        # Evaluation metrics
|   +-- requirements.txt   # Python dependencies
|   +-- examples/
|       +-- 01_unusual_trading.py
|       +-- 02_news_anomaly.py
|       +-- 03_crypto_manipulation.py
+-- rust/
    +-- Cargo.toml
    +-- src/
        +-- lib.rs         # Library root
        +-- detector.rs    # Anomaly detection
        +-- embeddings.rs  # Embedding generation
        +-- data_loader.rs # Data loading
        +-- signals.rs     # Signal generation
        +-- backtest.rs    # Backtesting
    +-- examples/
        +-- detect_anomalies.rs
        +-- monitor_crypto.rs
        +-- backtest.rs
```
