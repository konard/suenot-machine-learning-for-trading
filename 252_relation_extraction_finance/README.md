# Chapter 252: Relation Extraction in Finance

## Introduction

Relation extraction (RE) is a natural language processing task that identifies and classifies semantic relationships between named entities in text. In financial contexts, this means automatically discovering connections such as "Company A acquired Company B," "CEO X leads Organization Y," or "Product Z is manufactured by Company W" from unstructured text sources like news articles, SEC filings, earnings call transcripts, and analyst reports.

Traditional financial analysis relies heavily on structured databases (e.g., Bloomberg, Refinitiv) to track corporate relationships. However, these databases are expensive, often lag behind real events, and miss many implicit or emerging relationships. Relation extraction bridges this gap by mining unstructured text at scale, enabling traders and analysts to build dynamic knowledge graphs that capture the full web of corporate interconnections in near real-time.

This chapter presents a complete framework for financial relation extraction. We cover the key relation types relevant to trading, the NLP models used to extract them, and a working Rust implementation that processes financial text and connects to the Bybit cryptocurrency exchange for practical trading applications.

## Key Concepts

### Financial Relation Types

Financial texts contain several categories of relationships that are valuable for trading and investment:

**Ownership & Investment Relations**:
- `OWNS`: Company A owns shares in Company B
- `INVESTED_IN`: Fund A invested in Company B
- `SUBSIDIARY_OF`: Company A is a subsidiary of Company B
- `STAKE_IN`: Entity holds a percentage stake in a company

**Corporate Action Relations**:
- `ACQUIRED`: Company A acquired Company B
- `MERGED_WITH`: Company A merged with Company B
- `PARTNERED_WITH`: Company A formed a partnership with Company B
- `SPUN_OFF`: Company A spun off division as Company B

**Personnel Relations**:
- `CEO_OF`: Person X is CEO of Company Y
- `FOUNDED_BY`: Company was founded by Person
- `BOARD_MEMBER_OF`: Person serves on the board of Company

**Product & Market Relations**:
- `PRODUCES`: Company produces Product
- `COMPETES_WITH`: Company A competes with Company B
- `SUPPLIES_TO`: Company A supplies to Company B
- `LISTED_ON`: Token/Stock is listed on Exchange

### Distant Supervision

Distant supervision is a technique for generating training data without manual annotation. The core idea is to align a knowledge base (e.g., Wikidata, SEC EDGAR) with a text corpus: if two entities appear in a sentence and a known relation exists between them in the knowledge base, that sentence is labeled as expressing that relation.

Formally, given a knowledge base $\mathcal{K} = \{(e_1, r, e_2)\}$ containing entity-relation triples, and a corpus $\mathcal{C}$, the distant supervision assumption states:

$$\text{If } (e_1, r, e_2) \in \mathcal{K} \text{ and } e_1, e_2 \text{ co-occur in sentence } s, \text{ then } s \text{ expresses } r$$

This assumption is noisy — not every co-occurrence expresses the relation — but with enough data, effective models can be trained. Multi-instance learning techniques help by allowing the model to learn from bags of sentences rather than individual examples, reducing the impact of false positives.

### Dependency Path Features

The shortest dependency path between two entities in a sentence's parse tree is one of the most informative features for relation extraction. The dependency path captures the syntactic relationship between entities while filtering out irrelevant words.

For example, in the sentence "Tesla acquired SolarCity for $2.6 billion," the dependency path between "Tesla" and "SolarCity" would be:

```
Tesla → nsubj → acquired → dobj → SolarCity
```

The path elements — lemmatized words, dependency labels, and POS tags — form a compact representation that generalizes across different surface forms of the same relation. The feature vector for a dependency path typically includes:

- **Edge labels**: The sequence of dependency relations (nsubj, dobj, prep, etc.)
- **Path words**: Lemmatized words along the path
- **Path length**: Number of edges in the path
- **Direction**: Whether each edge is traversed up or down the tree

### Attention-Based Relation Extraction

Modern neural approaches use attention mechanisms to identify which parts of a sentence are most relevant for determining the relation between two entities. Given a sentence with tokens $\{w_1, w_2, \ldots, w_n\}$ and entity positions $(p_1, p_2)$, an attention-based model computes:

$$\alpha_i = \frac{\exp(f(\mathbf{h}_i, \mathbf{h}_{p_1}, \mathbf{h}_{p_2}))}{\sum_{j=1}^{n} \exp(f(\mathbf{h}_j, \mathbf{h}_{p_1}, \mathbf{h}_{p_2}))}$$

$$\mathbf{s} = \sum_{i=1}^{n} \alpha_i \mathbf{h}_i$$

where $\mathbf{h}_i$ is the hidden representation of token $i$, $f$ is a scoring function (typically a bilinear or additive attention), and $\mathbf{s}$ is the attention-weighted sentence representation fed to the classifier.

Entity-aware attention extends this by incorporating entity type embeddings and position embeddings relative to both entities, allowing the model to focus on context words that are specifically relevant to the relation type being predicted.

## ML Approaches

### Pattern-Based Extraction

The simplest approach uses hand-crafted patterns over dependency parses or surface text:

1. **Hearst Patterns**: Templates like "X acquired Y" or "X, a subsidiary of Y" matched against tokenized text
2. **Bootstrapping**: Starting from seed patterns, iteratively discover new patterns from matched instances and new instances from discovered patterns
3. **Regular expressions over parse trees**: Patterns defined over dependency structures rather than surface strings

Pattern-based methods offer high precision but low recall. They are useful as baselines and for bootstrapping training data.

### Convolutional Neural Networks

CNNs for relation extraction process a sentence through the following pipeline:

1. **Input representation**: Each token is represented as the concatenation of its word embedding and position embeddings relative to both entities:
$$\mathbf{x}_i = [\mathbf{e}_{w_i}; \mathbf{e}_{p_i^1}; \mathbf{e}_{p_i^2}]$$

2. **Convolution**: A filter $\mathbf{W} \in \mathbb{R}^{k \times d}$ slides over windows of $k$ tokens:
$$c_i = \text{ReLU}(\mathbf{W} \cdot \mathbf{x}_{i:i+k-1} + b)$$

3. **Max pooling**: Extract the most salient feature from each filter:
$$\hat{c} = \max(c_1, c_2, \ldots, c_{n-k+1})$$

4. **Classification**: The pooled features are fed through a fully connected layer with softmax:
$$P(r | s, e_1, e_2) = \text{softmax}(\mathbf{W}_o \hat{\mathbf{c}} + \mathbf{b}_o)$$

Piecewise CNNs (PCNNs) improve upon standard CNNs by dividing the sentence into three segments around the two entities and applying max pooling separately to each segment, preserving more structural information.

### Transformer-Based Models

Pre-trained language models like BERT and FinBERT have become the state-of-the-art for financial relation extraction. The approach works as follows:

1. **Input formatting**: Insert special markers around entities:
   ```
   [CLS] [E1] Tesla [/E1] announced the acquisition of [E2] SolarCity [/E2] . [SEP]
   ```

2. **Encoding**: Pass through the pre-trained transformer to obtain contextual representations.

3. **Entity representation**: Extract the representations at entity marker positions:
$$\mathbf{h}_{e_1} = \text{Transformer}(\text{[E1]}), \quad \mathbf{h}_{e_2} = \text{Transformer}(\text{[E2]})$$

4. **Classification**: Concatenate entity representations and classify:
$$P(r | s, e_1, e_2) = \text{softmax}(\mathbf{W}_r [\mathbf{h}_{e_1}; \mathbf{h}_{e_2}] + \mathbf{b}_r)$$

Fine-tuning FinBERT on financial relation extraction tasks typically yields F1 scores of 0.75–0.85 on standard benchmarks, significantly outperforming CNN-based approaches.

## Feature Engineering

### Entity Type Features

The types of the two entities strongly constrain which relations are possible:

| Entity 1 Type | Entity 2 Type | Possible Relations |
|---|---|---|
| COMPANY | COMPANY | ACQUIRED, MERGED_WITH, COMPETES_WITH, PARTNERED_WITH |
| PERSON | COMPANY | CEO_OF, FOUNDED_BY, BOARD_MEMBER_OF |
| COMPANY | PRODUCT | PRODUCES, LAUNCHED |
| FUND | COMPANY | INVESTED_IN, STAKE_IN |

Encoding entity types as additional features or using type-constrained decoding can significantly improve precision by eliminating impossible relation predictions.

### Context Window Features

The words immediately surrounding each entity provide strong signals:

- **Left context of entity 1**: Often contains the subject's role or description
- **Between context**: The words between two entities frequently express the relation directly
- **Right context of entity 2**: May contain additional details about the relation (e.g., monetary values, dates)

A typical feature vector includes:
- Bag-of-words for each context window (left, between, right)
- The length of the between context (short distances correlate with explicit relations)
- Presence of specific trigger words (e.g., "acquired," "partnered," "invested")

### Sentence-Level Indicators

Additional sentence-level features improve extraction quality:

- **Tense**: Past tense often indicates completed events; future/conditional tense indicates planned or hypothetical relations
- **Negation**: Detecting negation markers prevents extracting false relations (e.g., "did not acquire")
- **Modality**: Words like "may," "could," "rumored" indicate uncertain relations that should be flagged with lower confidence
- **Source attribution**: Identifying whether the relation is stated as fact or attributed to a source (e.g., "according to sources")

## Applications

### Knowledge Graph Construction

Relation extraction is the backbone of automated financial knowledge graph construction. By processing millions of documents, RE systems can build and maintain a graph where:

- **Nodes** represent entities (companies, people, products, exchanges)
- **Edges** represent relations with timestamps and confidence scores
- **Temporal layers** track how relationships evolve over time

Such knowledge graphs enable:
1. **Supply chain mapping**: Tracing supplier-customer relationships to predict revenue impacts
2. **Contagion analysis**: Identifying how financial stress at one entity propagates through ownership and credit relationships
3. **Competitive intelligence**: Tracking partnerships, acquisitions, and executive movements across an industry

### Event-Driven Trading Signals

Newly extracted relations can generate trading signals:

- **Acquisition announcements**: Detecting "ACQUIRED" relations before they appear in structured databases triggers signals on both acquirer and target stocks
- **Partnership formation**: New "PARTNERED_WITH" relations between companies in complementary sectors can signal revenue growth potential
- **Executive changes**: "CEO_OF" relation changes may predict strategic shifts
- **Crypto listing detection**: Identifying "LISTED_ON" relations for tokens being listed on major exchanges like Bybit generates signals for price appreciation around listing events

The signal strength depends on:
- **Novelty**: Is this a genuinely new relation or already known?
- **Source credibility**: Was this extracted from a reliable source?
- **Confidence score**: How confident is the extraction model?

### Risk Monitoring

Continuous relation extraction supports risk management by:

- **Detecting hidden exposures**: Discovering indirect relationships between portfolio holdings through shared suppliers, customers, or board members
- **Regulatory monitoring**: Tracking ownership relations to ensure compliance with concentration limits
- **Counterparty risk**: Monitoring the health of entities connected to trading counterparties

## Rust Implementation

Our Rust implementation provides a complete relation extraction toolkit for financial text with the following components:

### RelationType

The `RelationType` enum defines the set of financial relations the system can extract, including `Acquired`, `PartneredWith`, `CompetesWith`, `InvestedIn`, `CeoOf`, `ListedOn`, and `SuppliesTo`. Each variant maps to a specific semantic relationship commonly found in financial text.

### Entity

The `Entity` struct represents a named entity with a name, entity type (Company, Person, Product, Exchange), and start/end character offsets in the source text. Entity types constrain which relations are valid for a given entity pair.

### Relation

The `Relation` struct captures an extracted relation consisting of two entities, a relation type, a confidence score, and the source sentence. The confidence score (0.0 to 1.0) reflects the model's certainty in the extraction.

### PatternExtractor

The `PatternExtractor` implements rule-based relation extraction using keyword patterns. It maintains a set of trigger words for each relation type (e.g., "acquired," "bought," "purchased" for the `Acquired` relation) and scans sentences for co-occurring entity pairs and trigger words. This provides a high-precision baseline for relation extraction.

### RelationScorer

The `RelationScorer` implements a simple logistic regression model that scores candidate relations based on features such as entity distance, trigger word presence, entity type compatibility, and context words. It is trained using stochastic gradient descent on labeled examples.

### BybitClient

The `BybitClient` provides async HTTP access to the Bybit V5 API, fetching kline data and order book snapshots. This enables the system to combine relation extraction signals (e.g., new partnership or listing announcements) with real-time market data for trading decisions.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API for market data:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data at configurable intervals. Used to measure price impact after relation-based trading signals.
- **Order book endpoint** (`/v5/market/orderbook`): Provides limit order book snapshots. Used to assess liquidity conditions before executing trades triggered by relation extraction signals.

Trading applications combine relation extraction with Bybit data by:
1. Extracting relations from news and announcements in real-time
2. Mapping extracted entities to Bybit trading pairs (e.g., detecting "Bybit listed TOKEN" maps to TOKENUSDT)
3. Fetching current market data to evaluate entry conditions
4. Generating trade signals with confidence-weighted position sizing

## References

1. Zeng, D., Liu, K., Lai, S., Zhou, G., & Zhao, J. (2014). Relation classification via convolutional deep neural network. *COLING 2014*, 2335-2344.
2. Lin, Y., Shen, S., Liu, Z., Luan, H., & Sun, M. (2016). Neural relation extraction with selective attention over instances. *ACL 2016*, 2124-2133.
3. Wu, S., & He, Y. (2019). Enriching pre-trained language model with entity information for relation classification. *CIKM 2019*, 2361-2364.
4. Alt, C., Hubner, M., & Hennig, L. (2019). Fine-tuning pre-trained transformer language models to distantly supervised relation extraction. *ACL 2019*, 1388-1398.
5. Mintz, M., Bills, S., Snow, R., & Jurafsky, D. (2009). Distant supervision for relation extraction without labeled data. *ACL 2009*, 1003-1011.
6. Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.
