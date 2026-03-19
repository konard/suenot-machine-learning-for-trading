# Chapter 249: BigBird for Financial NLP

## Introduction

BigBird is a sparse attention mechanism that extends Transformer-based models to handle much longer sequences than standard architectures like BERT. While BERT is limited to 512 tokens due to the quadratic cost of full self-attention, BigBird can process sequences of up to 4,096 tokens (or more) with linear computational complexity. This capability makes BigBird particularly valuable for financial NLP, where documents such as SEC filings, earnings call transcripts, credit agreements, and analyst reports routinely exceed thousands of tokens.

The core insight of BigBird is that the full $n \times n$ attention matrix is unnecessary. By combining three types of sparse attention patterns — random attention, window (local) attention, and global attention — BigBird achieves the representational power of full attention while reducing computational cost from $O(n^2)$ to $O(n)$. Zaheer et al. (2020) proved theoretically that this sparse attention mechanism is a universal approximator of sequence functions and is Turing complete, meaning it loses no expressive power relative to full attention.

In the financial domain, BigBird unlocks the ability to process entire documents in a single forward pass. An analyst report that BERT must truncate or split into overlapping chunks can be consumed whole by BigBird, preserving long-range dependencies between an executive summary and a risk disclosure section that may be thousands of tokens apart. This chapter presents the theory behind BigBird's attention mechanism, its application to financial text analysis, and a complete Rust implementation with Bybit market data integration.

## BigBird Attention Mechanism

### Full Attention Baseline

In a standard Transformer, the attention mechanism computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q, K, V \in \mathbb{R}^{n \times d}$ are the query, key, and value matrices, and $n$ is the sequence length. The $QK^T$ product creates an $n \times n$ attention matrix, making the computation $O(n^2 d)$. For $n = 4096$ this matrix has over 16 million entries per head — prohibitively expensive for many applications.

### Sparse Attention in BigBird

BigBird replaces the dense attention matrix with a sparse one defined by a directed graph $\mathcal{G}$ on the set of tokens $\{1, 2, \ldots, n\}$. Token $i$ attends to token $j$ only if the edge $(i, j) \in \mathcal{G}$. The attention for token $i$ becomes:

$$\text{BigBirdAttn}_i(Q, K, V) = \text{softmax}\left(\frac{Q_i K_{\mathcal{N}(i)}^T}{\sqrt{d_k}}\right) V_{\mathcal{N}(i)}$$

where $\mathcal{N}(i)$ is the set of tokens that token $i$ attends to. BigBird constructs $\mathcal{N}(i)$ as the union of three attention patterns:

### 1. Random Attention

Each token attends to $r$ randomly selected other tokens. Formally, for each token $i$, a set $R(i)$ of $r$ tokens is sampled uniformly without replacement from $\{1, \ldots, n\}$. Random connections create shortcuts across the sequence, enabling information to flow between distant positions. This is inspired by the small-world graph property — in a random graph, any two nodes are connected by a short path with high probability.

### 2. Window (Local) Attention

Each token attends to $w$ tokens on either side, forming a sliding window. The local neighborhood is:

$$W(i) = \{j : |i - j| \leq w\}$$

Window attention captures local context, which is essential for understanding phrases, sentences, and nearby clauses. Most linguistic dependencies are local, so window attention handles the majority of useful attention patterns.

### 3. Global Attention

A small set of $g$ tokens attend to all other tokens and are attended to by all other tokens. These global tokens act as information hubs, aggregating and distributing information across the entire sequence. In BigBird, global tokens can be either:

- **Internal Transformer Construction (ITC)**: Existing tokens from the input are designated as global (e.g., the [CLS] token).
- **Extended Transformer Construction (ETC)**: Additional auxiliary tokens are prepended to the sequence specifically to serve as global aggregators.

### Combined Attention Pattern

The final attention set for each token is:

$$\mathcal{N}(i) = R(i) \cup W(i) \cup G$$

where $G$ is the set of global tokens. The total number of attention edges per token is $O(r + w + g)$, making the overall complexity $O(n(r + w + g)) = O(n)$ since $r$, $w$, and $g$ are constants independent of $n$.

### Theoretical Guarantees

Zaheer et al. (2020) proved two key properties:

1. **Universal Approximation**: BigBird's sparse attention can approximate any continuous sequence-to-sequence function to arbitrary precision, provided sufficient global tokens exist.
2. **Turing Completeness**: BigBird with sparse attention is Turing complete, meaning it can simulate any Turing machine. This is achieved through the global tokens, which act as a shared memory accessible by all positions.

## Financial NLP Applications

### Long Document Classification

Financial documents that benefit from BigBird's extended context include:

- **SEC filings** (10-K, 10-Q, 8-K): Annual and quarterly reports averaging 20,000-60,000 words. BigBird can process substantially larger portions than BERT, capturing relationships between business descriptions and risk factors.
- **Earnings call transcripts**: Typically 5,000-10,000 words. BigBird processes the entire transcript, linking management commentary to analyst Q&A.
- **Credit agreements**: Complex legal documents with cross-references between clauses that may be thousands of words apart.
- **Analyst reports**: Multi-page research notes where the investment thesis in the introduction connects to detailed financial projections at the end.

### Sentiment Analysis on Long Texts

Standard sentiment models truncate long documents, potentially missing crucial negative disclosures buried deep in a filing. BigBird captures sentiment signals across the full document:

Given a document $\mathbf{D} = [t_1, t_2, \ldots, t_n]$ with $n \leq 4096$ tokens, BigBird produces contextualized representations:

$$\mathbf{h}_i = \text{BigBird}(\mathbf{D})_i$$

The [CLS] token representation $\mathbf{h}_0$ is passed through a classification head:

$$P(\text{sentiment} = c \mid \mathbf{D}) = \text{softmax}(\mathbf{W}_c \mathbf{h}_0 + \mathbf{b}_c)$$

where $c \in \{\text{positive}, \text{neutral}, \text{negative}\}$.

### Named Entity Recognition in Financial Documents

BigBird excels at financial NER because entity recognition in long documents often requires understanding context from distant parts of the text. For example, determining whether "Apple" refers to the company or the fruit may require context from a section header hundreds of tokens away.

The NER task assigns a label $y_i \in \{\text{ORG}, \text{MONEY}, \text{DATE}, \text{PERCENT}, \text{O}, \ldots\}$ to each token:

$$P(y_i \mid \mathbf{D}) = \text{softmax}(\mathbf{W}_{\text{ner}} \mathbf{h}_i + \mathbf{b}_{\text{ner}})$$

### Trading Signal Extraction

BigBird can extract trading signals from long financial texts by:

1. **Multi-section analysis**: Processing an entire earnings report to detect contradictions between different sections (e.g., revenue growth claims in the CEO letter vs. declining margins in the financial statements).
2. **Temporal reasoning**: Understanding time references across the document to build a timeline of events and forecasts.
3. **Risk factor scoring**: Scoring risk factors mentioned throughout a filing and weighting them by frequency, location, and surrounding context.

## Feature Engineering for Financial BigBird

### Document Encoding

Financial documents require careful preprocessing:

1. **Section segmentation**: Identify document sections (summary, risk factors, financial statements) and assign section embeddings.
2. **Numerical normalization**: Financial numbers ("$1.2B", "15.3%") are tokenized specially to preserve magnitude information.
3. **Temporal markers**: Dates and time references are encoded with positional information relative to the filing date.

### Attention Pattern Optimization for Finance

The BigBird attention pattern can be customized for financial documents:

- **Section-aware global tokens**: Place global tokens at section boundaries rather than only at the start.
- **Financial entity windows**: Widen the local attention window around detected financial entities (ticker symbols, monetary amounts).
- **Cross-reference attention**: Add explicit attention edges between cross-referenced sections (e.g., "as described in Note 7").

## ML Pipeline

### Training Pipeline

1. **Pre-training**: Start from a pre-trained BigBird checkpoint (e.g., `google/bigbird-roberta-base`).
2. **Domain adaptation**: Continue pre-training on a financial corpus (SEC filings, financial news, analyst reports) using masked language modeling.
3. **Fine-tuning**: Train on the downstream task (classification, NER, sentiment) with task-specific heads.

### Evaluation Metrics

- **Classification**: Accuracy, F1-score (macro and weighted), AUC-ROC
- **NER**: Entity-level F1, precision, recall
- **Sentiment**: Cohen's Kappa, accuracy, directional accuracy for trading signals
- **Trading performance**: Sharpe ratio, Sortino ratio, maximum drawdown, annualized return

## Rust Implementation

Our Rust implementation provides a complete BigBird-based financial NLP toolkit with the following components:

### BigBirdConfig

The `BigBirdConfig` struct holds all hyperparameters for the BigBird attention mechanism: sequence length, number of random tokens, window size, number of global tokens, hidden dimension, and number of attention heads. Default values mirror the original paper's configuration.

### BigBirdAttention

The `BigBirdAttention` struct implements the core sparse attention computation. It generates the combined attention mask from random, window, and global patterns, then applies scaled dot-product attention only on non-zero positions. The implementation uses efficient sparse matrix operations to maintain $O(n)$ complexity.

### SentimentClassifier

The `SentimentClassifier` implements a logistic regression model for three-class sentiment classification. It takes document-level features (derived from BigBird attention patterns and token statistics) and predicts positive, neutral, or negative sentiment. Training uses stochastic gradient descent on cross-entropy loss.

### DocumentProcessor

The `DocumentProcessor` handles tokenization and preprocessing of financial text. It splits documents into tokens, applies section detection, identifies financial entities, and prepares input for the BigBird attention mechanism.

### BybitClient

The `BybitClient` provides async HTTP access to the Bybit V5 API for fetching kline data and order book snapshots. Market data is combined with NLP signals to produce trading decisions.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to combine NLP signals with market data:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data for backtesting NLP-driven strategies.
- **Order book endpoint** (`/v5/market/orderbook`): Real-time order book snapshots for execution decisions informed by sentiment analysis.

The trading workflow processes financial text through BigBird for sentiment scoring, then combines these scores with technical indicators from Bybit market data to generate buy/sell signals. A positive sentiment shift detected in an earnings transcript, confirmed by bullish price action, triggers a long entry.

## References

1. Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020). Big Bird: Transformers for Longer Sequences. *Advances in Neural Information Processing Systems*, 33, 17283-17297.
2. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. *arXiv preprint arXiv:2004.05150*.
3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*, 4171-4186.
4. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. *arXiv preprint arXiv:1908.10063*.
5. Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A Large Language Model for Extracting Information from Financial Text. *Contemporary Accounting Research*, 40(2), 806-841.
