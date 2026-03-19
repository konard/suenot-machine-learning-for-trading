# Chapter 256: Aspect-Based Sentiment Analysis for Finance

## Introduction

Aspect-based sentiment analysis (ABSA) goes beyond traditional document-level sentiment classification by identifying specific aspects (topics, entities, or attributes) within a text and determining the sentiment expressed toward each one independently. In financial applications, this distinction is critical: a single earnings report might express positive sentiment about revenue growth while simultaneously conveying negative sentiment about rising costs. A document-level classifier would average these signals, losing the granular information that traders need.

Traditional sentiment analysis assigns a single polarity (positive, negative, neutral) to an entire document. For financial news, this is inadequate. Consider the sentence: "Apple reported record revenue driven by strong iPhone sales, but margins contracted due to supply chain disruptions." A document-level model might rate this as neutral or mildly positive, but an aspect-based model correctly identifies:

- **Revenue** → positive
- **iPhone sales** → positive
- **Margins** → negative
- **Supply chain** → negative

For algorithmic trading, this decomposition enables more precise signal generation. A portfolio manager can weight positions based on which aspects of a company's performance are improving or deteriorating, rather than relying on a blunt aggregate sentiment score.

This chapter presents a complete framework for aspect-based sentiment analysis in finance. We cover the theoretical foundations, implement a working system in both Python-style pseudocode and Rust, and demonstrate how to integrate ABSA signals into a trading strategy using data from both equity and cryptocurrency markets (Bybit).

## Key Concepts

### Aspect Extraction

Aspect extraction is the first step in the ABSA pipeline. It identifies the entities or topics that are the targets of sentiment expression. In financial text, common aspects include:

- **Financial metrics**: revenue, earnings, margins, cash flow, debt
- **Business segments**: products, services, geographic regions
- **Operational factors**: supply chain, workforce, R&D, capex
- **Market factors**: demand, competition, regulation, macro environment

Aspect extraction can be performed using several approaches:

1. **Rule-based**: Predefined dictionaries of financial terms grouped by aspect category. Fast and interpretable but brittle and domain-specific.
2. **Frequency-based**: Identify frequently occurring noun phrases in a financial corpus, then cluster them into aspect categories.
3. **Sequence labeling**: Train a model (CRF, BiLSTM-CRF, or BERT-based) to tag aspect terms in text using BIO (Beginning, Inside, Outside) tagging.
4. **Attention-based**: Use attention weights from a transformer model to implicitly identify which tokens correspond to aspects.

### Aspect Sentiment Classification

Once aspects are extracted, sentiment classification determines the polarity expressed toward each aspect. The key challenge is that the same sentence can contain multiple aspects with different polarities.

Given a sentence $s$ containing aspect term $a$, the goal is to predict the sentiment polarity $y \in \{\text{positive}, \text{negative}, \text{neutral}\}$.

#### Attention-Based Models

Attention mechanisms allow the model to focus on the words most relevant to a specific aspect when determining its sentiment. Given a sentence representation $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_n]$ and an aspect embedding $\mathbf{a}$, the attention weights are computed as:

$$\alpha_i = \frac{\exp(\mathbf{h}_i^T \mathbf{W}_a \mathbf{a})}{\sum_{j=1}^{n} \exp(\mathbf{h}_j^T \mathbf{W}_a \mathbf{a})}$$

The aspect-specific sentence representation is then:

$$\mathbf{r} = \sum_{i=1}^{n} \alpha_i \mathbf{h}_i$$

This representation $\mathbf{r}$ is passed through a softmax classifier to predict sentiment polarity.

#### BERT-Based Approaches

Modern ABSA systems leverage pre-trained language models. The input is constructed by concatenating the sentence and aspect term with special tokens:

$$\text{[CLS]} \; s_1 \; s_2 \; \ldots \; s_n \; \text{[SEP]} \; a_1 \; a_2 \; \ldots \; a_m \; \text{[SEP]}$$

The [CLS] token representation is fed through a classification head:

$$\hat{y} = \text{softmax}(\mathbf{W}_c \cdot \mathbf{h}_{\text{[CLS]}} + \mathbf{b}_c)$$

Financial-domain models like FinBERT provide better initialization for this task, as they are pre-trained on financial corpora and understand domain-specific language.

### Financial Aspect Categories

In structured financial analysis, aspects are organized into a taxonomy:

| Category | Aspects | Example Phrases |
|----------|---------|----------------|
| Profitability | Revenue, Margins, EPS | "revenue grew 15%", "margin compression" |
| Growth | Sales Growth, User Growth, Market Share | "accelerating growth", "market share gains" |
| Risk | Debt, Leverage, Volatility | "debt levels concerning", "reduced leverage" |
| Operations | Supply Chain, Efficiency, Capex | "supply chain normalized", "capex increase" |
| Valuation | P/E, Price Target, Fair Value | "trading at a premium", "attractive valuation" |
| Macro | Interest Rates, Inflation, GDP | "rate hike impact", "inflation headwinds" |

### Sentiment Scoring

For trading applications, binary or ternary sentiment labels are insufficient. We need continuous sentiment scores that capture intensity. The scoring function maps aspect-level predictions to a real-valued signal:

$$\text{Score}(a) = P(\text{positive} | a) - P(\text{negative} | a)$$

This yields a score in $[-1, 1]$ where:
- $+1$: strongly positive sentiment toward aspect $a$
- $0$: neutral or mixed sentiment
- $-1$: strongly negative sentiment toward aspect $a$

The aggregate sentiment for an entity across all its aspects uses volume-weighted averaging:

$$S_{\text{entity}} = \frac{\sum_{a \in A} w_a \cdot \text{Score}(a)}{\sum_{a \in A} w_a}$$

where $w_a$ is the importance weight of aspect $a$ (e.g., revenue aspects may be weighted more heavily than operational aspects for equity valuation).

## ML Approaches

### Aspect Term Extraction with Sequence Labeling

Aspect extraction is modeled as a sequence labeling task using BIO tagging. Given input tokens $\{x_1, x_2, \ldots, x_n\}$, each token is assigned a label from $\{B\text{-}ASP, I\text{-}ASP, O\}$.

A BiLSTM-CRF model first encodes the sequence:

$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_f(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$
$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}_b(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t ; \overleftarrow{\mathbf{h}}_t]$$

The CRF layer models label dependencies:

$$P(\mathbf{y} | \mathbf{x}) = \frac{\exp\left(\sum_{t=1}^{n} \phi(y_{t-1}, y_t, \mathbf{h}_t)\right)}{\sum_{\mathbf{y}'} \exp\left(\sum_{t=1}^{n} \phi(y'_{t-1}, y'_t, \mathbf{h}_t)\right)}$$

### Multi-Aspect Sentiment with Attention Networks

The Attention-based Aspect Extraction (ABAE) model jointly learns aspect embeddings and sentiment classification. It uses an autoencoder structure where:

1. **Encoder**: Maps sentence to aspect space via attention
2. **Aspect embedding matrix**: $\mathbf{T} \in \mathbb{R}^{K \times d}$ where $K$ is number of aspects
3. **Reconstruction**: Reconstructs the sentence from the weighted aspect representation

The attention weights for selecting the most relevant aspect:

$$p_t = \text{softmax}(\mathbf{W} \cdot [\mathbf{e}_{w_t}; \mathbf{e}_{w_t} \odot \bar{\mathbf{e}}_s])$$

where $\mathbf{e}_{w_t}$ is the word embedding of token $t$ and $\bar{\mathbf{e}}_s$ is the average sentence embedding.

### Trading Signal Generation

The ABSA output is transformed into trading signals through several steps:

1. **Aspect aggregation**: Combine aspect-level scores for each entity
2. **Cross-sectional normalization**: Z-score the entity-level scores across the investment universe
3. **Signal smoothing**: Apply exponential moving average to reduce noise

$$z_{i,t} = \frac{S_{i,t} - \mu_t}{\sigma_t}$$

$$\tilde{z}_{i,t} = \lambda \cdot z_{i,t} + (1 - \lambda) \cdot \tilde{z}_{i,t-1}$$

where $\lambda$ is the smoothing factor (typically 0.1–0.3 for daily signals).

## Feature Engineering

### Text Preprocessing for Financial ABSA

Financial text requires specialized preprocessing:

- **Ticker symbol normalization**: Map $AAPL, AAPL, Apple Inc. to canonical form
- **Number handling**: Preserve financial figures ("$2.3B revenue" → aspect: revenue, magnitude: 2.3B)
- **Negation detection**: "did not meet expectations" should flip polarity for the "expectations" aspect
- **Comparative handling**: "margins improved but below peers" contains both absolute and relative sentiment

### Aspect-Specific Feature Vectors

For each detected aspect, we construct a feature vector:

- **Contextual embedding**: BERT representation of the aspect in context
- **Aspect type**: One-hot encoding of aspect category (profitability, growth, risk, etc.)
- **Modifier features**: Presence of intensifiers ("significantly"), diminishers ("slightly"), negations
- **Temporal markers**: Whether the aspect refers to past, present, or future ("expects revenue to grow")
- **Quantitative extraction**: Numerical values associated with the aspect ("+15% YoY")

### Aspect Sentiment Lexicon

A domain-specific lexicon maps financial terms to sentiment polarities:

| Term | Aspect | Polarity | Intensity |
|------|--------|----------|-----------|
| beat estimates | Earnings | Positive | 0.8 |
| margin compression | Profitability | Negative | -0.7 |
| accelerating growth | Growth | Positive | 0.9 |
| debt covenant breach | Risk | Negative | -0.95 |
| guidance raised | Outlook | Positive | 0.85 |
| headcount reduction | Operations | Negative | -0.6 |

## Applications

### Earnings Call Analysis

Earnings calls contain rich aspect-level sentiment information. The ABSA system processes transcripts to extract sentiment for each discussed topic:

1. **Management prepared remarks**: Typically more positive (promotional bias)
2. **Analyst Q&A section**: More revealing as analysts probe specific aspects
3. **Tone shift detection**: Compare aspect sentiment between consecutive quarters

A trading signal based on earnings call ABSA:

$$\text{Signal}_i = \sum_{a \in A_i} w_a \cdot (\text{Score}_{a,t} - \text{Score}_{a,t-1})$$

This captures the *change* in aspect-level sentiment, which is more predictive than the absolute level.

### News Sentiment Decomposition

Financial news articles are decomposed into aspect-level signals:

- **Company-specific aspects**: Revenue outlook, product launches, management changes
- **Sector aspects**: Industry trends, competitive dynamics
- **Macro aspects**: Regulatory changes, economic indicators

For cryptocurrency markets, relevant aspects include:
- **Technology**: Protocol upgrades, security audits, scalability improvements
- **Adoption**: Institutional interest, retail usage, partnership announcements
- **Regulation**: Government policy, exchange compliance, legal proceedings
- **Tokenomics**: Supply dynamics, staking yields, burn mechanisms

### Portfolio Construction

Aspect-level sentiment enables more sophisticated portfolio strategies:

1. **Aspect momentum**: Go long assets with improving key aspects, short those deteriorating
2. **Aspect dispersion**: Trade on the divergence between aspect sentiments (e.g., revenue positive but margins negative signals potential mean reversion)
3. **Cross-asset aspect signals**: If "supply chain" aspect turns positive across multiple companies, overweight the sector

## Rust Implementation

Our Rust implementation provides a complete aspect-based sentiment analysis toolkit with the following components:

### AspectExtractor

The `AspectExtractor` struct uses a dictionary-based approach to identify financial aspects in text. It maintains a mapping from keywords to aspect categories and supports configurable aspect taxonomies. The extractor tokenizes input text, matches tokens against the aspect dictionary, and returns a list of detected aspects with their positions.

### SentimentScorer

The `SentimentScorer` implements a lexicon-based sentiment scoring system. For each detected aspect, it examines the surrounding context window and scores the sentiment using a financial sentiment lexicon. It handles negation detection (words like "not", "no", "never" that flip polarity) and intensity modifiers ("significantly", "slightly"). The scorer outputs continuous sentiment scores in the range [-1, 1].

### AspectSentimentAnalyzer

The `AspectSentimentAnalyzer` combines the `AspectExtractor` and `SentimentScorer` into a unified pipeline. Given a text string, it extracts aspects, scores sentiment for each, and returns a structured result containing aspect-sentiment pairs. It supports batch processing of multiple documents and aggregation of results across a corpus.

### TradingSignalGenerator

The `TradingSignalGenerator` transforms aspect-level sentiment into actionable trading signals. It implements cross-sectional normalization, exponential smoothing, and threshold-based signal generation. It supports configurable aspect weights, allowing traders to emphasize aspects most relevant to their strategy.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data for backtesting sentiment-based strategies on cryptocurrency pairs. The client handles response parsing, error handling, and rate limiting considerations.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to obtain cryptocurrency market data for backtesting sentiment-driven strategies:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data at configurable intervals. Used for computing returns and evaluating the predictive power of aspect sentiment signals.
- **Ticker endpoint** (`/v5/market/tickers`): Provides current price and volume data for constructing real-time sentiment-adjusted signals.

The backtesting workflow:
1. Collect historical news/social media text about a cryptocurrency (e.g., BTCUSDT)
2. Run ABSA to extract aspect-level sentiment time series
3. Fetch corresponding price data from Bybit
4. Evaluate the correlation between aspect sentiment changes and subsequent returns
5. Simulate a trading strategy based on aspect sentiment signals

## References

1. Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. *Proceedings of SemEval 2014*, 27-35.
2. Wang, Y., Huang, M., Zhu, X., & Zhao, L. (2016). Attention-based LSTM for Aspect-level Sentiment Classification. *Proceedings of EMNLP 2016*, 606-615.
3. He, R., Lee, W. S., Ng, H. T., & Dahlmeier, D. (2017). An Unsupervised Neural Attention Model for Aspect Extraction. *Proceedings of ACL 2017*, 388-397.
4. Sun, C., Huang, L., & Qiu, X. (2019). Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence. *Proceedings of NAACL-HLT 2019*, 380-385.
5. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. *arXiv preprint arXiv:1908.10063*.
6. Xu, H., Liu, B., Shu, L., & Yu, P. S. (2019). BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis. *Proceedings of NAACL-HLT 2019*, 2324-2335.
7. Loughran, T., & McDonald, B. (2011). When is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.
