# Chapter 259: Multimodal NLP Trading

## Introduction

Multimodal NLP trading combines natural language processing with other data modalities — images, audio, and structured numerical data — to build richer representations of market state and generate more informed trading signals. Traditional NLP approaches for finance analyze text in isolation: earnings call transcripts, news headlines, or analyst reports are processed as standalone inputs. Multimodal methods break this silo by jointly learning from text alongside charts, financial tables, executive tone of voice, and price time series.

The motivation is straightforward. When a human analyst evaluates a stock, they do not read the earnings call transcript in a vacuum. They look at the accompanying slide deck, listen to the CEO's tone, glance at the price chart, and check the numbers in the financial statements. Each modality carries information that the others miss. A confident verbal tone paired with weak numbers tells a different story than the same words paired with strong numbers. Multimodal models attempt to capture these cross-modal interactions automatically.

This chapter presents a complete framework for multimodal NLP trading. We cover the key fusion architectures, the modalities most relevant to financial markets, and a working Rust implementation that connects to the Bybit cryptocurrency exchange for real-time multimodal signal generation.

## Key Concepts

### Modalities in Financial Markets

Financial markets produce data across several distinct modalities:

- **Text**: News articles, earnings call transcripts, SEC filings, social media posts, analyst reports. Text conveys semantic meaning, sentiment, and forward-looking statements.
- **Images**: Candlestick charts, technical analysis patterns, heatmaps of correlation matrices, order book depth visualizations. Images encode spatial patterns that are difficult to express numerically.
- **Audio**: Earnings call recordings, executive interviews, central bank speeches. Audio carries paralinguistic cues — tone, hesitation, stress — that text transcripts lose.
- **Structured data**: Price time series, volume profiles, order book snapshots, fundamental ratios. Structured data provides the quantitative backbone of any trading system.

Each modality has its own encoder architecture. Text uses transformer-based models (BERT, FinBERT), images use convolutional networks or vision transformers (ViT), audio uses mel-spectrogram encoders or wav2vec, and structured data uses temporal models (LSTM, TCN) or simple MLPs.

### Fusion Strategies

The central challenge in multimodal learning is how to combine representations from different modalities. Three main strategies exist:

#### Early Fusion

Early fusion concatenates raw or lightly processed features from all modalities into a single input vector before feeding them into a shared model:

$$\mathbf{z} = f(\text{concat}(\mathbf{x}_{\text{text}}, \mathbf{x}_{\text{image}}, \mathbf{x}_{\text{audio}}, \mathbf{x}_{\text{num}}))$$

**Advantages**: The model can learn cross-modal interactions from the start.
**Disadvantages**: Different modalities have very different scales, dimensions, and statistical properties. The model must learn to handle this heterogeneity, which can be difficult.

#### Late Fusion

Late fusion processes each modality independently through its own encoder, then combines the final representations:

$$\mathbf{h}_i = f_i(\mathbf{x}_i) \quad \text{for each modality } i$$
$$\hat{y} = g(\mathbf{h}_{\text{text}}, \mathbf{h}_{\text{image}}, \mathbf{h}_{\text{audio}}, \mathbf{h}_{\text{num}})$$

The combination function $g$ can be concatenation followed by a linear layer, element-wise addition, or a learned attention mechanism.

**Advantages**: Each encoder is specialized for its modality. Pre-trained unimodal encoders can be reused.
**Disadvantages**: Cross-modal interactions are only captured at the final stage, limiting expressiveness.

#### Cross-Attention Fusion

Cross-attention fusion allows modalities to attend to each other at intermediate layers, enabling rich cross-modal interactions:

$$\text{CrossAttn}(\mathbf{Q}_i, \mathbf{K}_j, \mathbf{V}_j) = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_j^T}{\sqrt{d_k}}\right) \mathbf{V}_j$$

where queries come from modality $i$ and keys/values come from modality $j$. This allows text tokens to attend to relevant regions of an image, or audio frames to attend to specific words in a transcript.

**Advantages**: Captures fine-grained cross-modal interactions at multiple levels.
**Disadvantages**: Computationally expensive. Quadratic complexity in the combined sequence length.

### Sentiment-Image Alignment

A key insight in multimodal financial analysis is that text sentiment and chart patterns often tell complementary stories. Consider these scenarios:

| Text Signal | Chart Signal | Interpretation |
|---|---|---|
| Positive news | Uptrend confirmed | Strong bullish (aligned) |
| Positive news | Downtrend | Potential reversal or news already priced in |
| Negative news | Downtrend confirmed | Strong bearish (aligned) |
| Negative news | Uptrend | Market disagrees with narrative |

When text and visual signals align, the combined signal is stronger than either alone. When they diverge, the divergence itself is informative — it suggests that one modality contains information not yet reflected in the other.

### CLIP-Style Financial Models

Contrastive Language-Image Pre-training (CLIP) learns a shared embedding space where text and images can be directly compared. In finance, this architecture can be adapted to align:

- News headlines with the corresponding price charts
- Earnings call transcripts with financial statement visualizations
- Social media sentiment with order book depth images

The training objective maximizes the cosine similarity between matching text-image pairs while minimizing similarity for non-matching pairs:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_j) / \tau)} \right]$$

where $\text{sim}(\mathbf{t}, \mathbf{v}) = \frac{\mathbf{t} \cdot \mathbf{v}}{|\mathbf{t}||\mathbf{v}|}$ is cosine similarity and $\tau$ is a temperature parameter.

## ML Approaches

### Multimodal Transformer

The multimodal transformer processes tokens from all modalities through a unified transformer architecture. Each token is tagged with a modality embedding (analogous to segment embeddings in BERT) so the model knows which modality each token belongs to:

$$\mathbf{e}_i = \mathbf{x}_i + \mathbf{p}_i + \mathbf{m}_i$$

where $\mathbf{x}_i$ is the token embedding, $\mathbf{p}_i$ is the positional embedding, and $\mathbf{m}_i$ is the modality embedding. The full sequence is then processed by standard transformer layers with self-attention.

### Gated Multimodal Unit (GMU)

The Gated Multimodal Unit learns to dynamically weight contributions from different modalities:

$$\mathbf{h}_{\text{text}} = \tanh(\mathbf{W}_t \mathbf{x}_{\text{text}})$$
$$\mathbf{h}_{\text{visual}} = \tanh(\mathbf{W}_v \mathbf{x}_{\text{visual}})$$
$$\mathbf{z} = \sigma(\mathbf{W}_z [\mathbf{x}_{\text{text}}; \mathbf{x}_{\text{visual}}])$$
$$\mathbf{h} = \mathbf{z} \odot \mathbf{h}_{\text{text}} + (1 - \mathbf{z}) \odot \mathbf{h}_{\text{visual}}$$

The gate $\mathbf{z}$ learns when to rely more on text versus visual information. In financial contexts, this is valuable: during earnings season, text may dominate; during technical breakouts, visual patterns may dominate.

### Multimodal Sentiment Classifier

For trading applications, a practical approach combines:

1. **Text encoder**: FinBERT or similar model extracts sentiment embeddings from news/social media
2. **Numerical encoder**: An MLP processes price returns, volume, and volatility features
3. **Visual encoder**: A CNN processes candlestick chart images or order book heatmaps
4. **Fusion layer**: Late fusion with learned attention weights combines all modalities
5. **Classification head**: Predicts market direction (up/down/neutral) or outputs a continuous signal

The attention-based fusion learns weights $\alpha_i$ for each modality:

$$\alpha_i = \frac{\exp(\mathbf{w}^T \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}^T \mathbf{h}_j)}$$
$$\mathbf{h}_{\text{fused}} = \sum_i \alpha_i \mathbf{h}_i$$

## Feature Engineering

### Text Features

Text features for multimodal trading include:

- **Sentiment scores**: Polarity (positive/negative/neutral) from FinBERT or domain-specific models
- **Named entity counts**: Number of company mentions, sector references, macro indicators
- **Uncertainty language**: Frequency of hedging words ("might", "could", "uncertain")
- **Forward-looking ratio**: Proportion of sentences containing future tense or forward-looking language
- **Novelty score**: How different the current text is from recent historical text (measured by embedding distance)

### Visual Features

Visual features extracted from financial charts:

- **Trend direction**: Detected from candlestick pattern recognition
- **Support/resistance levels**: Identified from price clustering in chart images
- **Volume profile shape**: Visual distribution of volume across price levels
- **Pattern recognition**: Head-and-shoulders, double tops/bottoms, flags, wedges

### Audio Features

Audio features from earnings calls and speeches:

- **Pitch variation**: Standard deviation of fundamental frequency — higher variation suggests stress or excitement
- **Speech rate**: Words per minute — slower speech may indicate careful hedging
- **Pause frequency**: Number and duration of pauses — more pauses suggest uncertainty
- **Vocal energy**: RMS energy of the audio signal — lower energy may indicate lack of conviction

### Cross-Modal Features

Features derived from the interaction between modalities:

- **Sentiment-price divergence**: Difference between text sentiment score and recent price momentum
- **Text-chart alignment score**: Cosine similarity between text embedding and chart image embedding in a shared space
- **Audio-text consistency**: Whether vocal features (confident tone) match text content (positive language)

## Applications

### Earnings Call Analysis

Earnings calls are inherently multimodal: they contain spoken audio, a written transcript, and often an accompanying slide deck with charts and tables. A multimodal system processes all three simultaneously:

1. The text encoder extracts sentiment and key financial metrics from the transcript
2. The audio encoder detects vocal cues — a CEO who sounds nervous while reporting "strong" results sends a mixed signal
3. The visual encoder processes the slide deck charts for trend patterns

Research by Qin and Yang (2019) showed that adding audio features to text-only models improved earnings surprise prediction by 5-8% in accuracy.

### News-Chart Fusion

When a breaking news headline arrives, a multimodal system can:

1. Encode the headline text for sentiment and entity extraction
2. Simultaneously encode the current price chart as an image
3. Compute the alignment between text sentiment and visual trend
4. Generate a trading signal that accounts for both the news content and the market's current technical state

This prevents common mistakes like going long on positive news when the chart shows a clear downtrend (news already priced in) or shorting on negative news when the chart shows strong support holding.

### Social Media Multimodal Signals

Social media posts in financial communities often combine text with screenshots of charts, positions, or order books. A multimodal system can:

- Parse the text for sentiment and trading intent
- Analyze embedded chart images for technical patterns
- Cross-reference the visual evidence with the textual claims
- Weight the signal by the historical accuracy of the source

## Rust Implementation

Our Rust implementation provides a complete multimodal NLP trading toolkit with the following components:

### TextEncoder

The `TextEncoder` struct implements a simple bag-of-words sentiment analyzer with a financial lexicon. It scores text by matching tokens against a dictionary of positive and negative financial terms, producing a sentiment vector that includes polarity, subjectivity, and word-count-based features. This serves as the text modality input for fusion.

### NumericalEncoder

The `NumericalEncoder` struct processes structured market data (price returns, volume ratios, volatility) through a single-layer neural network with configurable dimensions. It normalizes inputs using z-score standardization and outputs a fixed-dimensional embedding suitable for fusion with other modalities.

### VisualFeatureExtractor

The `VisualFeatureExtractor` struct extracts visual features from candlestick data. It computes trend direction, body-to-shadow ratios, volume profile statistics, and pattern indicators (engulfing patterns, doji detection). These features summarize the visual appearance of a price chart in numerical form.

### MultimodalFusion

The `MultimodalFusion` struct implements attention-based late fusion. It takes embeddings from the text, numerical, and visual encoders, computes learned attention weights for each modality, and produces a fused representation. The attention weights are interpretable, showing which modality the model considers most informative at any given time.

### TradingSignalGenerator

The `TradingSignalGenerator` struct wraps the full multimodal pipeline. It accepts raw inputs (text, market data, candlestick data), runs them through the appropriate encoders, fuses the representations, and outputs a trading signal with direction, confidence, and per-modality contribution scores.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data from the `/v5/market/kline` endpoint and order book snapshots from the `/v5/market/orderbook` endpoint. The client handles response parsing and error handling.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to obtain real-time market data for the numerical and visual modalities:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data at configurable intervals. Used for visual feature extraction and numerical encoding.
- **Order book endpoint** (`/v5/market/orderbook`): Provides a snapshot of the current limit order book. Used for computing depth-based features.

The text modality in a production system would connect to news APIs or social media feeds. In our implementation, we demonstrate the architecture with sample financial texts and simulate the text encoder pipeline.

## References

1. Xu, P., Zhu, X., & Clifton, D. A. (2023). Multimodal learning with transformers: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(10), 12113-12132.
2. Qin, Y., & Yang, Y. (2019). What you say and how you say it matters: Predicting stock volatility using verbal and vocal cues. *Proceedings of the 57th Annual Meeting of the ACL*, 390-401.
3. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of ICML*, 8748-8763.
4. Arevalo, J., Solorio, T., Montes-y-Gomez, M., & Gonzalez, F. A. (2017). Gated multimodal units for information fusion. *arXiv preprint arXiv:1702.01992*.
5. Yang, L., Ng, T. L., Smyth, B., & Dong, R. (2020). HTML: Hierarchical transformer-based multi-task learning for volatility prediction. *Proceedings of The Web Conference*, 1066-1072.
6. Sawhney, R., Agarwal, S., Wadhwa, A., Derr, T., & Shah, R. R. (2022). Stock selection via spatiotemporal hypergraph attention network: A learning to rank approach. *Proceedings of AAAI*, 2128-2135.
