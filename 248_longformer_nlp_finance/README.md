# Chapter 248: Longformer for Financial NLP

## Introduction

Financial analysis routinely involves processing long documents: annual reports (10-K filings), earnings call transcripts, prospectuses, regulatory filings, and multi-page research notes. Standard transformer models like BERT are limited to 512 tokens, which forces practitioners to truncate or chunk these documents and lose cross-section context. The Longformer, introduced by Beltagy, Peters, and Lo (2020), addresses this limitation by replacing the quadratic self-attention mechanism with a combination of local sliding window attention and task-specific global attention, enabling efficient processing of sequences up to 4,096 tokens (or longer with appropriate configuration).

For financial applications, this extended context window is transformative. A single 10-K filing can span 50,000+ words, and critical information (risk factors, management discussion, financial statements) is distributed throughout the document. An analyst seeking to classify the overall sentiment of a filing, detect material risk disclosures, or extract forward-looking statements needs a model that can attend to distant parts of the document simultaneously. The Longformer's attention pattern makes this tractable without the prohibitive memory cost of full quadratic attention.

This chapter presents a complete framework for applying Longformer to financial NLP tasks. We cover the attention mechanism, the key adaptations needed for financial text, and a working Rust implementation that connects to the Bybit cryptocurrency exchange for real-time sentiment-driven analysis.

## Key Concepts

### Attention Complexity in Transformers

Standard self-attention computes pairwise interactions between all tokens in a sequence. For a sequence of length $n$, this requires $O(n^2)$ time and memory. At 512 tokens this is manageable, but at 4,096 tokens the cost grows by a factor of 64, and at 16,384 tokens by a factor of 1,024. This quadratic scaling is the fundamental bottleneck that prevents standard transformers from processing long documents.

The attention score between query $q_i$ and key $k_j$ in standard self-attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q, K, V \in \mathbb{R}^{n \times d}$ and the $QK^T$ product is $O(n^2)$.

### Longformer Attention Pattern

The Longformer replaces full self-attention with a sparse attention pattern that combines three types of attention:

**1. Sliding Window Attention (Local)**

Each token attends to a fixed window of $w$ neighboring tokens on each side. For a window size of $w = 256$, each token attends to 512 surrounding tokens. This captures local context and syntactic structure with $O(n \times w)$ complexity.

$$\text{SlidingAttn}(i) = \{j : |i - j| \leq w\}$$

**2. Dilated Sliding Window Attention**

To increase the receptive field without increasing computation, dilated attention introduces gaps of size $d$ (dilation factor) in the sliding window pattern. With dilation $d = 2$ and window $w = 256$, the effective receptive field doubles while maintaining the same number of attended positions:

$$\text{DilatedAttn}(i) = \{j : |i - j| \leq w \times d \text{ and } (i - j) \mod d = 0\}$$

**3. Global Attention**

Selected tokens (e.g., the [CLS] token, or tokens corresponding to key entities) attend to all tokens in the sequence, and all tokens attend to them. This enables information flow across the entire document:

$$\text{GlobalAttn}(i) = \{1, 2, \ldots, n\} \quad \text{for designated global tokens}$$

The combined complexity is $O(n \times w + n \times g)$ where $g$ is the number of global attention tokens, which is linear in $n$ for fixed $w$ and $g$.

### Pre-training and Fine-tuning

Longformer is initialized from RoBERTa checkpoints and continues pre-training on long documents using masked language modeling (MLM). The position embeddings are extended from 512 to 4,096 by copying the existing embeddings eight times. During fine-tuning for downstream tasks:

- **Document classification**: The [CLS] token receives global attention and its representation is used for classification.
- **Token classification (NER)**: All tokens use local attention; entity spans are classified based on their contextual representations.
- **Question answering**: Question tokens receive global attention to enable cross-document reasoning.

The fine-tuning objective depends on the task. For classification:

$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

where $C$ is the number of classes and $\hat{y}_{i,c}$ is the predicted probability for class $c$.

### Positional Encoding for Long Sequences

Longformer extends the positional embeddings from BERT/RoBERTa to handle longer sequences. The absolute position embeddings are replicated to cover the extended range:

$$PE_{\text{long}}(pos) = PE_{\text{short}}(pos \mod L_{\text{short}})$$

where $L_{\text{short}}$ is the original maximum position (512). During continued pre-training, these replicated embeddings are fine-tuned to learn position-specific patterns for the extended range.

## ML Approaches

### Document-Level Sentiment Classification

The primary financial NLP task for Longformer is classifying the sentiment of entire documents — earnings call transcripts, analyst reports, or news articles — without truncation.

Given a document tokenized into $\{x_1, x_2, \ldots, x_n\}$ with $n \leq 4096$, the Longformer encodes the sequence using its mixed attention pattern. The [CLS] token representation $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^d$ is fed through a classification head:

$$\hat{y} = \text{softmax}(\mathbf{W}\mathbf{h}_{\text{CLS}} + \mathbf{b})$$

where $\mathbf{W} \in \mathbb{R}^{C \times d}$ and $\mathbf{b} \in \mathbb{R}^C$. Classes typically include positive, negative, and neutral sentiment.

### Named Entity Recognition in Financial Documents

Longformer excels at financial NER because entity context often spans long distances. For example, a company name mentioned on page one of a 10-K may be relevant to a risk factor discussed on page ten. The model uses sliding window attention for local entity detection and global attention on section headers to maintain document-level context.

Each token $x_i$ is classified into BIO tags:

$$\hat{y}_i = \text{softmax}(\mathbf{W}_{\text{ner}}\mathbf{h}_i + \mathbf{b}_{\text{ner}})$$

Entity types relevant to finance include: ORG (organizations), MONEY (monetary amounts), PERCENT (percentages), DATE (dates), PRODUCT (financial instruments), and EVENT (market events).

### Risk Factor Detection

Regulatory filings contain risk disclosures that are material to investment decisions. Longformer can classify each paragraph or section as containing a risk factor or not, using the extended context to understand how risks relate to the company's overall business:

$$P(\text{risk} | \text{paragraph}_i) = \sigma(\mathbf{w}^T \mathbf{h}_i + b)$$

where $\mathbf{h}_i$ is the Longformer representation of the paragraph, obtained by pooling over its token representations.

## Feature Engineering

### Token-Level Features

Financial documents contain domain-specific tokens that require careful handling:

- **Financial numbers**: Monetary values, percentages, ratios. These should be normalized (e.g., "$1.2B" and "$1,200M" should be equivalent).
- **Temporal expressions**: Fiscal quarters, year-over-year references. These anchor the document in time.
- **Legal/regulatory terms**: "Material adverse effect", "going concern", "force majeure". These carry specific legal weight.
- **Sentiment indicators**: "Exceeded expectations", "headwinds", "robust growth". FinBERT-style vocabulary is especially valuable here.

### Document Structure Features

Financial documents have explicit structure that can be leveraged:

- **Section headers**: Map to global attention tokens so the model can route information across sections.
- **Table markers**: Financial tables contain critical quantitative data; mark their boundaries.
- **Footnote references**: Connect footnote content to the main text across potentially thousands of tokens.

### Aggregated Sentiment Features

For trading applications, document-level sentiment must be aggregated into actionable signals:

- **Sentiment score**: Continuous value in $[-1, 1]$ from the classification head
- **Sentiment momentum**: Change in sentiment across consecutive documents $\Delta S_t = S_t - S_{t-1}$
- **Cross-document consensus**: Average sentiment across multiple sources for the same asset
- **Sentiment surprise**: Deviation from expected sentiment $S_t - \mathbb{E}[S_t | S_{t-1}, \ldots, S_{t-k}]$

## Applications

### Earnings Call Analysis

Earnings call transcripts are 5,000-15,000 words long, well beyond BERT's capacity. Longformer can:

1. **Full-transcript sentiment**: Classify the overall tone of the call without truncation, capturing the interplay between prepared remarks and Q&A.
2. **Section-level analysis**: Apply global attention to section markers (CEO remarks, CFO remarks, Q&A) to compare tone across sections.
3. **Forward-looking statement detection**: Identify and classify sentences about future expectations, which are distributed throughout the transcript.

### Regulatory Filing Analysis

10-K and 10-Q filings contain critical information for fundamental analysis:

- **Risk factor extraction**: Identify new or changed risk disclosures compared to prior filings.
- **Management Discussion & Analysis (MD&A)**: Extract management's interpretation of financial results and outlook.
- **Material event detection**: Flag sections that describe events likely to affect stock price.

### Crypto Market News Analysis

For cryptocurrency markets, long-form content includes:

- **Whitepaper analysis**: Evaluate the technical sophistication and feasibility of new projects.
- **Governance proposals**: Assess the likely impact of protocol changes on token value.
- **Thread aggregation**: Combine multiple related news articles or social media threads into a single long-context analysis.

## Rust Implementation

Our Rust implementation provides a simplified Longformer-inspired text analysis toolkit for financial NLP:

### SlidingWindowAttention

The `SlidingWindowAttention` struct implements the core local attention mechanism. It computes attention scores within a fixed window around each token position, using dot-product attention with scaling. The implementation handles edge cases at sequence boundaries where the window is truncated.

### GlobalAttentionMask

The `GlobalAttentionMask` struct manages which token positions receive global attention. It provides methods to designate the [CLS] token, section headers, and entity tokens as global attention positions. Global attention tokens attend to all positions and are attended to by all positions.

### LongformerEncoder

The `LongformerEncoder` combines sliding window and global attention into a complete encoding layer. It processes token embeddings through the mixed attention pattern, applies layer normalization, and produces contextualized representations suitable for downstream classification.

### SentimentClassifier

The `SentimentClassifier` implements logistic regression over Longformer-style features for binary sentiment classification. It accepts feature vectors derived from text analysis (sentiment lexicon scores, structural features, attention-weighted aggregations) and outputs a sentiment prediction with confidence.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data from the `/v5/market/kline` endpoint for market data context alongside NLP-derived signals.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to combine NLP-derived sentiment signals with market data:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data for correlating sentiment signals with price movements.
- **Order book endpoint** (`/v5/market/orderbook`): Provides market microstructure context for understanding how sentiment translates into order flow.

The trading strategy combines NLP signals with price data: when Longformer sentiment analysis detects a strong positive or negative signal in a long financial document, the system checks current market conditions via the Bybit API before generating a trading recommendation.

## References

1. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. *arXiv preprint arXiv:2004.05150*.
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
3. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv preprint arXiv:1907.11692*.
4. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv preprint arXiv:1908.10063*.
5. Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A Large Language Model for Extracting Information from Financial Text. *Contemporary Accounting Research*, 40(2), 806-841.
6. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.
