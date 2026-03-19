# Chapter 287: CLIP Multimodal Trading

## Introduction: CLIP for Multimodal Trading -- Aligning Text News with Price Data

Contrastive Language-Image Pre-training (CLIP), introduced by Radford et al. (2021), demonstrated a powerful principle: by training a model to align two modalities -- images and text -- in a shared embedding space using contrastive learning, the resulting representations enable remarkable zero-shot generalization. In CLIP's original domain, a model can classify images it has never seen into categories described only by natural language, without any task-specific fine-tuning.

For trading, this paradigm opens a compelling frontier. Instead of aligning images and text, we align **market price data** (OHLCV sequences) and **textual market commentary** (news headlines, analyst reports, social media sentiment). The result is a shared embedding space where price regimes and their textual descriptions live side by side. A bullish breakout pattern in the price encoder's embedding space sits near the text embedding of "Bitcoin surges past resistance on strong institutional inflows." A volatility squeeze in price space aligns with "Markets consolidate as traders await Fed decision."

Why is this alignment valuable? Traditional trading systems treat text data and price data as separate inputs, often combining them through late fusion (independent feature extraction followed by concatenation). CLIP-style contrastive alignment instead forces the two modalities to learn a **shared semantic space** from the ground up. This yields several practical advantages:

1. **Zero-shot regime classification**: Given a new textual description of a market state (e.g., "flash crash with rapid recovery"), the model can identify which price sequences match that description without ever being explicitly trained on that label.

2. **Cross-modal retrieval**: Given a price sequence, retrieve the most relevant news headlines. Given a headline, identify which historical price patterns it most closely resembles.

3. **Transfer learning**: Embeddings trained on one asset class can generalize to others, because the shared text-price space captures abstract market concepts rather than asset-specific patterns.

4. **Robust sentiment analysis**: Instead of training a separate NLP model on labeled sentiment data, the contrastive alignment implicitly learns sentiment by associating bullish/bearish text with corresponding price movements.

In this chapter, we build a CLIP-inspired contrastive learning system in Rust for cryptocurrency trading on Bybit. Our implementation includes text and price encoders, the InfoNCE contrastive loss, cosine similarity computation, and zero-shot classification of market regimes from textual descriptions.

## Mathematical Foundations

### Contrastive Learning and the InfoNCE Loss

The core of CLIP is contrastive learning: given a batch of N paired examples (text_i, price_i), the model learns to maximize agreement between corresponding pairs while minimizing agreement between non-corresponding pairs.

Each modality has its own encoder:

```
f_text: text -> R^d       (text encoder)
f_price: price_seq -> R^d  (price encoder)
```

Both encoders map their respective inputs to a shared d-dimensional embedding space. The encoders are trained jointly so that matched pairs produce similar embeddings while unmatched pairs produce dissimilar embeddings.

For a batch of N pairs, we compute the NxN similarity matrix:

```
S_ij = cos_sim(f_text(text_i), f_price(price_j)) / tau
```

where `tau > 0` is a learned temperature parameter that controls the sharpness of the distribution, and the cosine similarity is defined as:

```
cos_sim(a, b) = (a . b) / (||a|| * ||b||)
```

The InfoNCE (Noise-Contrastive Estimation) loss treats the problem as N-way classification. For the text-to-price direction:

```
L_text = -(1/N) * sum_{i=1}^{N} log( exp(S_ii) / sum_{j=1}^{N} exp(S_ij) )
```

Symmetrically, for the price-to-text direction:

```
L_price = -(1/N) * sum_{i=1}^{N} log( exp(S_ii) / sum_{j=1}^{N} exp(S_ji) )
```

The total CLIP loss is the average of both directions:

```
L_CLIP = (L_text + L_price) / 2
```

This symmetric loss ensures that both modalities contribute equally to the shared embedding space. Each direction acts as a softmax classifier where the correct match should have the highest similarity score among all candidates in the batch.

### Temperature Parameter

The temperature `tau` plays a critical role in contrastive learning. A lower temperature makes the softmax distribution sharper, focusing more on hard negatives. A higher temperature produces softer distributions that are easier to optimize early in training:

```
P(match = j | text_i) = exp(S_ij / tau) / sum_{k=1}^{N} exp(S_ik / tau)
```

In CLIP, the temperature is a learnable log-parameterized scalar `tau = exp(log_tau)` initialized to a value around 0.07. For trading applications, we typically find temperatures in the range [0.05, 0.2] work well, with lower values preferred when the training set contains many similar-looking price regimes.

### Cosine Similarity and Embedding Normalization

Before computing similarities, embeddings are L2-normalized:

```
z = f(x) / ||f(x)||_2
```

This normalization is essential because it constrains all embeddings to lie on a unit hypersphere, making cosine similarity equivalent to a dot product and ensuring that the similarity scores are bounded in [-1, 1]. Without normalization, the model can trivially increase similarity scores by scaling up embedding magnitudes, which destabilizes training.

### Zero-Shot Classification

Once the model is trained, zero-shot classification works by treating candidate text labels as a classifier. Given a set of K text descriptions {label_1, ..., label_K} and a query price sequence q:

```
prediction = argmax_k cos_sim(f_price(q), f_text(label_k))
```

For trading, the labels might be market regime descriptions:

- "Strong bullish trend with increasing volume"
- "Bearish breakdown with panic selling"
- "Range-bound consolidation with low volatility"
- "Volatile whipsaw with no clear direction"

The model assigns the price sequence to whichever textual description produces the highest cosine similarity -- without having been explicitly trained on these specific categories.

### Embedding Space Geometry

The shared embedding space has interesting geometric properties relevant to trading. After training:

- **Clusters** form around common market regimes, with both price and text embeddings intermixed.
- **Directions** in embedding space correspond to semantic concepts: a vector from "bearish" to "bullish" embeddings defines a sentiment axis.
- **Distances** between embeddings correlate with semantic similarity: two different bullish headlines will be closer to each other than to a bearish headline, and similarly for their corresponding price patterns.

These geometric properties enable arithmetic in embedding space. For example, embedding("correction") - embedding("trend") + embedding("breakout price pattern") might yield an embedding near price patterns that show a breakout following a correction -- a form of analogical reasoning over market data.

## Applications in Trading

### News-Price Alignment

The most direct application is aligning news events with price responses. By encoding both news headlines and the corresponding price windows in the shared space, the model learns which types of news drive which types of price action. In production, when a new headline arrives, the model can instantly retrieve the most similar historical price responses, providing a data-driven expectation for how the market might react.

This is particularly valuable for cryptocurrency markets, where news flow from social media, regulatory announcements, and on-chain metrics can move prices rapidly. A CLIP-style model trained on historical news-price pairs can identify the probable market impact of a new headline by finding its nearest neighbors in the shared embedding space.

### Chart Pattern Recognition with Text

Traditional chart pattern recognition relies on manually coded rules or convolutional neural networks trained on labeled chart images. A CLIP-style approach instead describes patterns in text and lets the model learn the alignment:

- "Double bottom with neckline breakout" maps to the corresponding price structure
- "Head and shoulders with declining volume" maps to the distribution pattern
- "Ascending triangle with tightening range" maps to the consolidation structure

This text-supervised approach is more flexible than rigid geometric rules and naturally handles pattern variations. The model learns that a "double bottom" can have different heights, widths, and symmetry levels, because the text description abstracts over these surface-level variations.

### Zero-Shot Sector and Regime Classification

Perhaps the most powerful application is zero-shot classification. Without any labeled training data for specific regimes, the model can classify price sequences into arbitrary categories defined only by text. A portfolio manager can define new regime categories on the fly:

- "Risk-on environment with rotation into growth stocks"
- "Flight to safety with gold and treasury strength"
- "Crypto-specific narrative-driven rally"

The model assigns each price window to the most similar textual description, enabling dynamic regime classification that adapts to the portfolio manager's evolving mental model of the market.

### Cross-Modal Anomaly Detection

When a price sequence has very low maximum similarity to any textual description in the model's vocabulary, it may represent a novel market condition not well captured by historical text-price pairs. This provides a natural anomaly detection mechanism: low alignment scores signal that the current market is behaving in ways the model has not seen described in text, warranting caution.

## Rust Implementation

Our Rust implementation provides a CLIP-inspired contrastive learning framework with the following components:

- **TextEncoder**: Converts text into embedding vectors using character n-gram features. In production, this would be replaced by a transformer-based encoder, but the n-gram approach captures enough structure for demonstration purposes and is fast to compute.

- **PriceEncoder**: Converts OHLCV price sequences into embedding vectors using a simple convolutional approach with learnable filters. The encoder extracts local price patterns at multiple scales and projects them to the shared embedding dimension.

- **CLIPModel**: Combines both encoders with a learnable temperature parameter and implements the full contrastive learning pipeline, including the InfoNCE loss, cosine similarity matrix computation, and zero-shot classification.

- **BybitClient**: Fetches real-time and historical kline data from Bybit's REST API, providing OHLCV sequences for encoding.

The implementation emphasizes numerical correctness and clarity over performance. All matrix operations use the `ndarray` crate, providing numpy-like array semantics in Rust. The encoders use fixed random projections for simplicity, but the architecture is designed so that learned weight updates could be added with an autograd framework.

## Bybit Data Integration

Our system integrates with Bybit's public REST API to fetch real-time kline (candlestick) data. The `BybitClient` fetches OHLCV data for any supported trading pair and interval, converts it to the ndarray format expected by the price encoder, and handles pagination for historical data retrieval.

The data pipeline works as follows:

1. Fetch kline data from `https://api.bybit.com/v5/market/kline` for the desired symbol and interval
2. Parse the JSON response into structured `Kline` records with open, high, low, close, and volume fields
3. Convert kline sequences into 2D arrays with shape (sequence_length, 5) for the price encoder
4. Normalize each feature column (OHLCV) to zero mean and unit variance for stable encoding

For live trading, the client can be called repeatedly to obtain the latest candles, with the price encoder producing updated embeddings that are compared against textual market descriptions for real-time regime classification.

## Key Takeaways

1. **CLIP-style contrastive learning** creates a shared embedding space for text and price data, enabling cross-modal reasoning about markets without modality-specific labels.

2. **The InfoNCE loss** is symmetric and treats alignment as mutual N-way classification, ensuring both modalities contribute equally to the learned representation.

3. **Zero-shot classification** is the killer application: define new market regimes in natural language and classify price sequences into them without any retraining.

4. **Temperature tuning** is critical -- lower temperatures sharpen the similarity distribution and improve discrimination between similar regimes, but can make training unstable.

5. **Cosine similarity with L2 normalization** ensures embeddings are compared by direction rather than magnitude, preventing trivial solutions and keeping similarity scores interpretable.

6. **Cross-modal retrieval** enables novel workflows: given a price pattern, find the most relevant news; given a headline, find the most similar historical price action.

7. **Anomaly detection** emerges naturally from low alignment scores, flagging market conditions that lack historical precedent in the text-price training data.

8. **Rust's performance characteristics** make it well-suited for real-time contrastive inference in trading, where low-latency embedding computation and similarity search are essential for time-sensitive decisions.
