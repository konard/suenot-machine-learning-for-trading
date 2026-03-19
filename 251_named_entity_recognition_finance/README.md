# Chapter 251: Named Entity Recognition for Finance

## Introduction

Named Entity Recognition (NER) is the task of identifying and classifying key entities in unstructured text into predefined categories such as company names, financial instruments, monetary values, dates, and market events. In the financial domain, NER serves as a foundational building block for extracting structured information from earnings reports, SEC filings, news articles, analyst notes, and social media posts.

Financial text is particularly challenging for general-purpose NER models. Company names frequently overlap with common words ("Apple," "Sprint," "Gap"), ticker symbols are short and ambiguous ("A" for Agilent, "IT" for Gartner), and financial instruments have complex naming conventions (e.g., "5-year 3.25% Treasury note due 2028"). Domain-specific NER models trained on financial corpora consistently outperform generic models by 10-20% in F1-score on financial entity extraction tasks.

For algorithmic trading, robust NER enables automated extraction of trading signals from text. Identifying which companies are mentioned in breaking news, linking them to their ticker symbols, and detecting associated monetary values and sentiment provides a structured data feed that can be consumed by quantitative models. This chapter presents a complete NER framework for finance, covering token classification approaches, feature engineering techniques, and a working Rust implementation that processes financial text with Bybit market data integration.

## Key Concepts

### Entity Types in Financial NER

Financial NER systems typically recognize the following entity categories:

- **ORG (Organization)**: Company names, exchanges, regulatory bodies (e.g., "Goldman Sachs", "NYSE", "SEC")
- **TICKER**: Stock ticker symbols and cryptocurrency trading pairs (e.g., "AAPL", "BTCUSDT", "ETH")
- **MONEY**: Monetary values including currencies (e.g., "$1.5 billion", "€500 million", "0.05 BTC")
- **PERCENT**: Percentage values in financial context (e.g., "3.5%", "up 12%")
- **DATE**: Temporal expressions relevant to financial events (e.g., "Q3 2024", "fiscal year 2023", "next Friday")
- **EVENT**: Financial events (e.g., "IPO", "stock split", "earnings call", "margin call")
- **INSTRUMENT**: Financial instruments (e.g., "10-year Treasury", "S&P 500 futures", "put option")

### Sequence Labeling with BIO Tagging

NER is formulated as a sequence labeling problem using the BIO (Beginning, Inside, Outside) tagging scheme. Each token in a sentence receives a label:

- **B-TYPE**: Beginning of an entity of the given type
- **I-TYPE**: Inside (continuation of) an entity of the given type
- **O**: Outside any entity

For example, the sentence "Goldman Sachs reported $2.5B revenue" is labeled:

| Token    | Label   |
|----------|---------|
| Goldman  | B-ORG   |
| Sachs    | I-ORG   |
| reported | O       |
| $2.5B    | B-MONEY |
| revenue  | O       |

This formulation allows multi-token entities (like "Goldman Sachs") to be captured as a single unit, while maintaining token-level predictions that are compatible with standard sequence labeling architectures.

### Conditional Random Fields (CRF)

Conditional Random Fields are a class of discriminative probabilistic models used for structured prediction. Unlike independent token classifiers, CRFs model the conditional probability of the entire label sequence given the input:

$$P(\mathbf{y} | \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \prod_{t=1}^{T} \exp\left(\sum_{k} \lambda_k f_k(y_{t-1}, y_t, \mathbf{x}, t)\right)$$

where $\mathbf{y} = (y_1, \ldots, y_T)$ is the label sequence, $\mathbf{x}$ is the input sequence, $f_k$ are feature functions, $\lambda_k$ are learned weights, and $Z(\mathbf{x})$ is the partition function ensuring the distribution sums to 1.

The key advantage of CRFs for NER is that they enforce label consistency. For example, a CRF learns that B-ORG should be followed by I-ORG or O, but never by I-MONEY. This transition modeling significantly reduces annotation errors compared to independent classifiers.

The Viterbi algorithm finds the optimal label sequence in $O(T \cdot K^2)$ time, where $T$ is the sequence length and $K$ is the number of labels:

$$y^* = \arg\max_{\mathbf{y}} P(\mathbf{y} | \mathbf{x})$$

### Word Embeddings and Contextual Features

Modern NER systems rely on dense vector representations of tokens. Key embedding strategies for financial NER include:

- **Pre-trained word embeddings**: Word2Vec or GloVe vectors trained on financial corpora capture domain-specific semantics. For example, "bullish" and "optimistic" have similar embeddings in a financial word space.
- **Character-level embeddings**: A CNN or LSTM over character sequences captures morphological patterns. This is valuable for financial text where unseen company names and ticker symbols are common.
- **Contextual embeddings**: Transformer-based models like FinBERT produce context-dependent representations where the same word receives different embeddings depending on surrounding text.

### Gazetteers and External Knowledge

A gazetteer is a curated dictionary of known entities. Financial NER benefits greatly from gazetteers because:

1. **Company databases**: Lists of publicly traded companies (from exchanges like NYSE, NASDAQ, Bybit) provide high-precision matching for organization names.
2. **Ticker mappings**: Exhaustive ticker-to-company mappings resolve ambiguity (e.g., "COIN" maps to Coinbase).
3. **Financial lexicons**: Dictionaries of financial terms, instruments, and event types provide strong priors for entity classification.

Gazetteer features are typically encoded as binary indicators appended to the token feature vector: 1 if the token (or n-gram) matches a gazetteer entry, 0 otherwise.

## ML Approaches

### Logistic Regression for Token Classification

The simplest NER approach treats each token independently and applies logistic regression with hand-crafted features:

Given a feature vector $\mathbf{x}_t$ for token $t$, the probability of label $k$ is:

$$P(y_t = k | \mathbf{x}_t) = \frac{\exp(\mathbf{w}_k^T \mathbf{x}_t + b_k)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x}_t + b_j)}$$

Typical features include:
- Current word (lowercased and original)
- Word shape (e.g., "Xx+" for capitalized, "d+" for digits, "$d+.d+" for currency)
- Character prefixes and suffixes (2-4 characters)
- Part-of-speech tag
- Surrounding word context (window of ±2 tokens)
- Gazetteer membership flags

This approach is fast and interpretable but fails to model label dependencies.

### BiLSTM-CRF for Sequence Labeling

The BiLSTM-CRF architecture is a dominant approach for NER. A bidirectional LSTM processes the token sequence in both directions, producing contextual representations that are then fed into a CRF layer:

**Forward LSTM:**
$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$

**Backward LSTM:**
$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$

**Combined representation:**
$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$

The CRF layer on top models label transition probabilities, ensuring globally consistent predictions. The emission scores from the BiLSTM are combined with transition scores in the CRF to find the optimal label sequence.

### Transformer-Based NER (FinBERT)

Transformer models pre-trained on financial text (FinBERT, SEC-BERT) achieve state-of-the-art results on financial NER. The approach fine-tunes a pre-trained transformer by adding a token classification head:

$$\mathbf{h}_t = \text{Transformer}(\mathbf{x})_t$$
$$P(y_t | \mathbf{x}) = \text{softmax}(\mathbf{W} \mathbf{h}_t + \mathbf{b})$$

FinBERT's advantage is that it has already learned financial language patterns during pre-training. Fine-tuning on a relatively small labeled NER dataset (a few thousand sentences) produces models that understand financial jargon, company name conventions, and contextual cues specific to financial documents.

## Feature Engineering

### Token-Level Features

Effective features for financial NER capture multiple aspects of each token:

- **Orthographic features**: Is the token capitalized? All uppercase (common for tickers)? Contains digits? Contains punctuation (e.g., "$", "%")?
- **Word shape**: A condensed representation where uppercase letters become "X", lowercase become "x", digits become "d" (e.g., "$12.5M" → "$dd.dX")
- **Prefix/suffix**: Character n-grams at the beginning and end of the word capture morphological patterns (e.g., suffix "-tion" suggests a noun, prefix "$" suggests money)

### Context Window Features

NER predictions benefit from surrounding context:

- **Neighboring words**: The words immediately before and after the current token provide strong disambiguation cues. "acquired" before a capitalized word strongly suggests B-ORG.
- **Bag-of-words context**: A wider window (±5 tokens) captures topical context. Words like "revenue", "earnings", "shares" indicate a financial context where capitalized words are likely organizations.

### Financial-Specific Features

Domain features significantly improve financial NER:

- **Ticker format detection**: Regular expressions matching common ticker patterns (1-5 uppercase letters, optionally followed by a period and suffix for share classes)
- **Currency symbol adjacency**: Tokens preceded by "$", "€", "£", or "¥" are likely monetary values
- **Percentage patterns**: Tokens matching `\d+\.?\d*%` or followed by "percent" / "basis points"
- **Temporal patterns**: Detection of quarter references (Q1-Q4), fiscal year mentions, date formats
- **Financial keyword proximity**: Distance to anchor words like "acquired", "reported", "traded at", "priced at"

## Applications

### Trading Signal Extraction

NER enables automated extraction of structured trading signals from unstructured news:

1. **Company-event linking**: Identify which company (ORG entity) is associated with which event (EVENT entity) in a news article. "Apple announced a 4-for-1 stock split" → {ORG: Apple, EVENT: stock split}.
2. **Magnitude extraction**: Pair monetary values (MONEY) with their context to quantify the signal. "Revenue beat expectations by $1.2 billion" → magnitude: $1.2B, direction: positive.
3. **Cross-referencing**: Link extracted ORG entities to ticker symbols for automated order routing.

### Risk Monitoring

Continuous NER over news and filing streams enables real-time risk monitoring:

- **Exposure mapping**: Identify all organizations mentioned in negative contexts and check portfolio exposure.
- **Regulatory event detection**: Flag mentions of regulatory actions (SEC investigation, FDA rejection) linked to held positions.
- **Contagion tracking**: When a company entity co-occurs with distress terms, check for counterparty or supply chain exposure.

### Crypto Market Intelligence

For cryptocurrency markets and exchanges like Bybit:

- **Token mention tracking**: Identify cryptocurrency names and tickers in social media and news to gauge attention.
- **Exchange event detection**: Extract mentions of exchange listings, delistings, and trading pair additions.
- **DeFi protocol monitoring**: Recognize DeFi protocol names and associated events (hacks, upgrades, governance votes).

## Rust Implementation

Our Rust implementation provides a complete financial NER toolkit with the following components:

### Token and Entity Types

The `EntityType` enum defines the financial entity categories (ORG, TICKER, MONEY, PERCENT, DATE, EVENT, INSTRUMENT). The `BioTag` enum represents the BIO tagging scheme. The `Token` struct holds the text content along with computed features for each token.

### FinancialNER

The `FinancialNER` struct implements a rule-based and statistical NER system for financial text. It maintains gazetteers for known companies, tickers, and financial terms. The `tokenize` method splits text into tokens with feature extraction. The `predict` method assigns BIO tags to each token using pattern matching and gazetteer lookup, combined with contextual rules.

### NERClassifier

The `NERClassifier` struct implements a multi-class logistic regression model for token classification. It maintains weight matrices for each entity label and trains using stochastic gradient descent. The classifier accepts feature vectors combining token-level, context, and gazetteer features.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline data and ticker information, which is used to build dynamic gazetteers of actively traded symbols and to demonstrate NER on real market data descriptions.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API for two purposes:

- **Dynamic gazetteer construction**: The `/v5/market/tickers` endpoint provides a list of all actively traded symbols. These symbols are added to the ticker gazetteer, improving NER recall on cryptocurrency-related text.
- **Market context generation**: Kline data from `/v5/market/kline` is used to generate realistic financial text (e.g., "BTCUSDT traded at $67,500 with volume of 1,234 BTC") for NER demonstration and testing.

The Bybit API is well-suited for financial NER applications because it provides:
- Comprehensive symbol listings for gazetteer construction
- Real-time price data for generating contextual financial text
- Low-latency responses suitable for real-time text processing pipelines

## References

1. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural architectures for named entity recognition. *Proceedings of NAACL-HLT*, 260-270.
2. Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. *arXiv preprint arXiv:1508.01991*.
3. Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.
4. Salinas Alvarado, J. C., Verspoor, K., & Baldwin, T. (2015). Domain adaption of named entity recognition to support credit risk assessment. *Proceedings of the Australasian Language Technology Association Workshop*, 84-90.
5. Akhtar, M. I., Nesi, P., & Pantaleo, G. (2023). A survey on named entity recognition in the financial domain. *Information Processing & Management*, 60(1), 103133.
