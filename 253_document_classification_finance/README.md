# Chapter 253: Document Classification in Finance

## Introduction

Document classification is the task of automatically assigning a predefined category label to a document based on its content. In finance, the volume of textual information — earnings reports, SEC filings, analyst notes, news articles, regulatory disclosures, and social media posts — far exceeds what any human team can process manually. Automated document classification enables financial institutions to route information to the right teams, trigger alerts on material events, and extract structured signals from unstructured text at scale.

The stakes are high. A 10-K filing that discloses a material weakness in internal controls must be flagged immediately for compliance review. A news article about a CEO resignation needs to reach the event-driven trading desk within seconds. An analyst report upgrading a stock from "hold" to "buy" should be captured by sentiment aggregation systems before the market fully prices in the information. Document classification is the first step in all of these workflows.

This chapter presents a complete framework for financial document classification. We cover the key text representation methods, the machine learning models that map representations to categories, and a working Rust implementation that connects to the Bybit cryptocurrency exchange to classify market-related news and announcements by their expected price impact.

## Key Concepts

### Document Representation

Before a document can be classified, it must be converted into a numerical representation that a machine learning model can process. The choice of representation has a profound impact on classification performance.

#### Bag of Words (BoW)

The simplest representation treats a document as an unordered collection of words. Each document is represented as a vector of word counts:

$$\mathbf{x}_d = [c(w_1, d), c(w_2, d), \ldots, c(w_V, d)]$$

where $c(w_i, d)$ is the count of word $w_i$ in document $d$, and $V$ is the vocabulary size. Despite ignoring word order entirely, BoW remains surprisingly effective for many classification tasks, particularly when combined with strong regularization.

#### TF-IDF

Term Frequency-Inverse Document Frequency refines raw counts by down-weighting words that appear in many documents (and are therefore less informative):

$$\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)$$

where:

$$\text{TF}(w, d) = \frac{c(w, d)}{\sum_{w' \in d} c(w', d)}$$

$$\text{IDF}(w) = \log \frac{N}{1 + |\{d \in D : w \in d\}|}$$

Here $N$ is the total number of documents and $D$ is the corpus. TF-IDF naturally elevates domain-specific terms like "dividend", "impairment", or "delisting" that carry strong classification signal in financial texts, while suppressing generic terms like "the", "and", "is".

#### Word Embeddings

Pre-trained word embeddings (Word2Vec, GloVe, FastText) map words into dense, low-dimensional vectors that capture semantic relationships. A document can be represented by averaging or pooling the embeddings of its constituent words:

$$\mathbf{x}_d = \frac{1}{|d|} \sum_{w \in d} \mathbf{e}(w)$$

where $\mathbf{e}(w) \in \mathbb{R}^k$ is the $k$-dimensional embedding of word $w$. This representation captures semantic similarity — "revenue" and "sales" will have similar embeddings even though they are different tokens.

### Financial Document Categories

Financial documents can be classified along several taxonomies:

- **Document type**: 10-K, 10-Q, 8-K, earnings call transcript, analyst report, press release, news article
- **Sentiment**: positive, negative, neutral
- **Topic**: mergers & acquisitions, earnings, regulatory, legal, management changes, product launches
- **Materiality**: material, non-material (whether the information is likely to affect stock price)
- **Urgency**: time-critical, routine
- **Risk category**: market risk, credit risk, operational risk, regulatory risk

### Hierarchical Classification

Financial documents often belong to a hierarchy of categories. For example, a document might first be classified as "regulatory filing", then as "10-K", then by the specific section (risk factors, MD&A, financial statements). Hierarchical attention networks (Yang et al., 2016) address this by applying attention at both the word and sentence level:

$$\mathbf{h}_i = \overrightarrow{\text{GRU}}(w_i) \| \overleftarrow{\text{GRU}}(w_i)$$

$$\alpha_i = \frac{\exp(\mathbf{u}_w^T \mathbf{h}_i)}{\sum_j \exp(\mathbf{u}_w^T \mathbf{h}_j)}$$

$$\mathbf{s} = \sum_i \alpha_i \mathbf{h}_i$$

The word-level attention $\alpha_i$ highlights which words are most important for classification, while a second level of attention operates over sentence representations to determine which sentences matter most. This produces interpretable classification decisions — a valuable property in regulated financial settings.

## ML Approaches

### Naive Bayes for Fast Baseline

Naive Bayes is the canonical baseline for text classification. Despite its strong independence assumption — that word occurrences are conditionally independent given the class — it performs remarkably well on document classification tasks.

For a document $d$ with words $w_1, w_2, \ldots, w_n$ and candidate class $c$:

$$P(c | d) \propto P(c) \prod_{i=1}^{n} P(w_i | c)$$

The class-conditional word probabilities are estimated from training data with Laplace smoothing:

$$P(w | c) = \frac{c(w, c) + \alpha}{\sum_{w'} c(w', c) + \alpha V}$$

where $\alpha$ is the smoothing parameter (typically 1.0) and $V$ is the vocabulary size. Naive Bayes is extremely fast to train, requires minimal hyperparameter tuning, and serves as a strong baseline against which more complex models should be compared.

### Logistic Regression with TF-IDF

Logistic regression on TF-IDF features is often the strongest "simple" model for document classification. Given a TF-IDF feature vector $\mathbf{x}_d$, the model estimates class probabilities using the softmax function:

$$P(y = c | \mathbf{x}_d) = \frac{\exp(\mathbf{w}_c^T \mathbf{x}_d + b_c)}{\sum_{c'} \exp(\mathbf{w}_{c'}^T \mathbf{x}_d + b_{c'})}$$

The model is trained by minimizing the cross-entropy loss with L2 regularization:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log P(y = c | \mathbf{x}_i) + \lambda \|\mathbf{W}\|_2^2$$

L2 regularization is critical for text classification because the feature space is high-dimensional and sparse, making overfitting a constant risk.

### Support Vector Machines (SVM)

SVMs with linear kernels are particularly effective for text classification because:

1. Text data is typically linearly separable in high-dimensional TF-IDF space
2. SVMs maximize the margin between classes, providing good generalization
3. They handle high-dimensional sparse data efficiently

The SVM objective for binary classification is:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$

For multi-class financial document classification, one-vs-rest or one-vs-one strategies are employed.

### Convolutional Neural Networks (CNN) for Text

CNNs can capture local patterns (n-grams) in text through 1D convolution:

$$h_i = \text{ReLU}(\mathbf{W} \cdot \mathbf{x}_{i:i+k-1} + b)$$

where $\mathbf{x}_{i:i+k-1}$ is the concatenation of $k$ consecutive word embeddings. Multiple filter sizes (e.g., 3, 4, 5) capture different n-gram lengths. Max pooling over each filter's output produces a fixed-size representation regardless of document length:

$$\hat{h} = \max(h_1, h_2, \ldots, h_{n-k+1})$$

This architecture, introduced by Kim (2014), is particularly effective for short financial texts like headlines and social media posts.

## Feature Engineering

### Financial Vocabulary Features

Domain-specific vocabulary provides strong classification signals:

- **Earnings keywords**: "revenue", "EPS", "guidance", "beat", "miss", "outlook"
- **M&A keywords**: "acquisition", "merger", "takeover", "bid", "deal"
- **Regulatory keywords**: "SEC", "compliance", "violation", "fine", "penalty"
- **Risk keywords**: "default", "downgrade", "impairment", "write-off", "restructuring"

Counting the presence and density of these keyword groups creates interpretable features that complement statistical text representations.

### Document Structure Features

Financial documents have characteristic structures that aid classification:

- **Document length**: 10-K filings are typically 50,000+ words; press releases are 500-1,000 words
- **Section headers**: the presence of specific section titles (e.g., "Risk Factors", "Management Discussion and Analysis")
- **Numeric density**: financial statements have high ratios of numbers to words
- **Table density**: some document types (earnings reports) contain many tables
- **Formatting patterns**: legal disclaimers, forward-looking statement warnings

### Temporal Features

For trading applications, the timing of document publication matters:

- **Market hours**: documents released during trading hours vs. after-hours
- **Earnings season**: heightened sensitivity to earnings-related documents
- **Day of week**: Friday evening releases often attempt to bury bad news
- **Relative timing**: how close to the earnings date a document appears

## Applications

### Trading Signal Generation

Document classification generates trading signals in several ways:

1. **Sentiment-driven**: Classifying news as positive/negative and trading in the direction of sentiment before the market fully absorbs the information
2. **Event detection**: Identifying material events (M&A announcements, management changes, regulatory actions) and executing event-driven strategies
3. **Information advantage**: Faster classification of SEC filings as they are published provides a time advantage over manual review

### Risk Management

Document classification supports risk management by:

- **Early warning**: Flagging documents that discuss increased risk exposure, covenant violations, or going-concern doubts
- **Portfolio monitoring**: Continuously scanning news and filings for adverse developments affecting portfolio holdings
- **Counterparty risk**: Monitoring filings and news about counterparties for signs of financial distress

### Regulatory Compliance

Financial institutions use document classification for:

- **Filing categorization**: Automatically routing SEC filings to appropriate review teams
- **Suspicious activity**: Flagging communications that may indicate insider trading or market manipulation
- **KYC/AML**: Classifying customer documents for know-your-customer and anti-money-laundering workflows

## Rust Implementation

Our Rust implementation provides a complete document classification toolkit with the following components:

### TfIdfVectorizer

The `TfIdfVectorizer` struct builds a vocabulary from a training corpus and transforms documents into TF-IDF vectors. It computes term frequencies, document frequencies, and inverse document frequencies, producing sparse feature vectors suitable for classification. The vocabulary is constructed during the `fit` phase and frozen for consistent feature extraction during inference.

### NaiveBayesClassifier

The `NaiveBayesClassifier` implements multinomial Naive Bayes with Laplace smoothing. It learns class-conditional word distributions from labeled training data and predicts class labels by computing log-posterior probabilities. The log-space computation prevents numerical underflow that would occur with direct probability multiplication over large vocabularies.

### DocumentClassifier

The `DocumentClassifier` implements multi-class logistic regression (softmax) with stochastic gradient descent. It accepts TF-IDF feature vectors and produces probability distributions over document categories. Training uses configurable learning rate, number of epochs, and L2 regularization strength.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data and market announcements that serve as real-world documents for classification. The client handles response parsing and error handling.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to obtain market data for generating and evaluating document classification signals:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data for measuring price impact after document classification signals
- **Announcements**: Market announcements and news items that serve as real-world documents to classify by topic and expected impact

The integration demonstrates how document classification can be applied to cryptocurrency market data, classifying market events and measuring their price impact on assets like BTCUSDT.

## References

1. Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical Attention Networks for Document Classification. *Proceedings of NAACL-HLT*, 1480-1489.
2. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *Proceedings of EMNLP*, 1746-1751.
3. Joachims, T. (1998). Text categorization with Support Vector Machines: Learning with many relevant features. *Proceedings of ECML*, 137-142.
4. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.
5. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
