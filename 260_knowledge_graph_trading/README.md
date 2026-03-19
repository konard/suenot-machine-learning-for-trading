# Chapter 260: Knowledge Graph Trading

Knowledge graphs (KGs) encode entities and their relationships as a structured network, enabling machine learning models to reason over interconnected financial information. Unlike tabular features that treat each input independently, KGs capture cross-entity dependencies — such as supply-chain links between companies, sector memberships, executive board overlaps, and macroeconomic exposures — providing a richer context for trading decisions.

## Key Concepts

### Knowledge Graph Fundamentals

A knowledge graph $G = (V, E, R)$ consists of:
- **Nodes** $V$: entities (companies, sectors, commodities, people, events)
- **Edges** $E \subseteq V \times R \times V$: directed relationships between entities
- **Relation types** $R$: semantic labels (e.g., `supplies_to`, `competes_with`, `member_of`)

Each fact is stored as a triple $(h, r, t)$ where $h$ is the head entity, $r$ is the relation, and $t$ is the tail entity. For example: *(AAPL, supplies_to, TSLA)* or *(NVDA, member_of, S&P500)*.

### Knowledge Graph Embeddings

KG embedding models learn continuous vector representations of entities and relations by optimizing a scoring function $f(h, r, t)$ over observed triples.

**TransE** models relations as translations in embedding space:

$$f(h, r, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|$$

The loss encourages $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$ for positive triples and $\mathbf{h} + \mathbf{r} \not\approx \mathbf{t}'$ for corrupted triples.

**DistMult** uses a bilinear scoring function:

$$f(h, r, t) = \mathbf{h}^\top \text{diag}(\mathbf{r}) \, \mathbf{t} = \sum_i h_i \cdot r_i \cdot t_i$$

**ComplEx** extends DistMult to complex-valued embeddings, handling asymmetric relations:

$$f(h, r, t) = \text{Re}\left(\sum_i h_i \cdot r_i \cdot \bar{t}_i\right)$$

### Graph Neural Networks on KGs

Relational Graph Convolutional Networks (R-GCN) aggregate neighbor information through typed edges:

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{r \in R} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} \mathbf{W}_r^{(l)} \mathbf{h}_j^{(l)} + \mathbf{W}_0^{(l)} \mathbf{h}_i^{(l)}\right)$$

where $\mathcal{N}_i^r$ is the set of neighbors of node $i$ under relation $r$, $c_{i,r}$ is a normalization constant, and $\mathbf{W}_r^{(l)}$ are relation-specific weight matrices.

### Temporal Knowledge Graphs

Financial KGs evolve over time. Temporal KG models extend static embeddings with a time component:

$$f(h, r, t, \tau) = \mathbf{h}_\tau^\top \text{diag}(\mathbf{r}_\tau) \, \mathbf{t}_\tau$$

where entity and relation embeddings are time-dependent functions $\mathbf{h}_\tau = g(\mathbf{h}, \tau)$, capturing how corporate relationships, sector allocations, and supply chains change.

## ML Approaches for Trading

### Entity-Based Alpha Signals

KG embeddings provide entity features that complement traditional factor models. Given a stock's embedding $\mathbf{e}_i$, we can train a classifier:

$$P(\text{up}_i \mid \mathbf{e}_i, \mathbf{x}_i) = \sigma(\mathbf{w}^\top [\mathbf{e}_i \| \mathbf{x}_i] + b)$$

where $\mathbf{x}_i$ are conventional features (momentum, value, quality) and $\|$ denotes concatenation.

### Relational Risk Propagation

Supply-chain and credit linkages propagate risk across connected entities. An influence score from entity $j$ to $i$ through path $p$ is:

$$\text{Influence}(j \to i) = \sum_{p \in \text{Paths}(j,i)} \prod_{(u,r,v) \in p} w_{u,r,v}$$

where $w_{u,r,v}$ are edge weights reflecting the strength of each relationship.

### Event Propagation on KGs

Financial events (earnings surprises, regulatory actions, M&A announcements) propagate through the knowledge graph. The propagated impact on entity $i$ from an event at entity $j$ is:

$$\text{Impact}_i = \sum_{j \in \text{EventSources}} \alpha_{ij} \cdot \text{EventScore}_j$$

where $\alpha_{ij}$ is the attention-weighted influence derived from the graph structure.

## Feature Engineering

### Entity Centrality Features

- **Degree centrality**: Number of connections (higher degree = more systemic importance)
- **Betweenness centrality**: Fraction of shortest paths passing through an entity
- **PageRank**: Recursive importance measure based on incoming link quality
- **Eigenvector centrality**: Influence measure based on connection to other influential nodes

### Relationship-Based Features

- **Supply chain depth**: Distance to end consumer or raw material source
- **Sector concentration**: Diversity of cross-sector connections
- **Peer similarity**: Cosine similarity of KG embeddings between companies
- **Contagion score**: Weighted sum of distressed neighbors' risk metrics

### Temporal Graph Features

- **Edge formation rate**: Speed at which new relationships appear
- **Relationship stability**: Duration of existing connections
- **Graph density change**: Temporal evolution of local clustering coefficient

## Applications

### Supply Chain Alpha

Knowledge graphs mapping supplier-customer relationships enable lead-lag trading strategies. When a major supplier reports strong earnings, connected downstream companies often experience delayed positive price reactions, creating exploitable alpha signals.

### Contagion-Aware Risk Management

By propagating credit risk through the KG, portfolio managers can identify hidden exposures. A seemingly diversified portfolio may have concentrated supply-chain risk that only becomes visible through graph analysis.

### Event-Driven Trading

KGs enable systematic propagation of event signals. An FDA approval for a pharmaceutical company propagates to its suppliers, competitors, and partner companies with quantifiable impact weights derived from the graph structure.

## Rust Implementation

The Rust implementation provides five core components:

1. **KnowledgeGraph**: Triple store with entity and relation indexing, neighbor lookup, and path finding
2. **TransEModel**: TransE embedding model with margin-based ranking loss and SGD training
3. **GraphFeatureExtractor**: Computes centrality features (degree, PageRank) and peer similarity from the KG
4. **TradingSignalGenerator**: Combines KG embeddings with market features to generate buy/sell signals
5. **BybitClient**: Async market data client for Bybit V5 API (klines and orderbook)

## Bybit API Integration

The implementation fetches live market data from Bybit:
- **Kline endpoint**: `/v5/market/kline` — OHLCV candlestick data for feature computation
- **Orderbook endpoint**: `/v5/market/orderbook` — real-time bid/ask levels for spread features

Both stock market symbols (via KG entity mapping) and crypto pairs (BTCUSDT, ETHUSDT) are supported.

## References

1. Bordes, A., et al. "Translating Embeddings for Modeling Multi-relational Data." NeurIPS, 2013.
2. Schlichtkrull, M., et al. "Modeling Relational Data with Graph Convolutional Networks." ESWC, 2018.
3. Cheng, D., et al. "Knowledge Graph-Based Event Embedding Framework for Financial Quantitative Investments." AAAI, 2020.
4. Feng, F., et al. "Temporal Relational Ranking for Stock Prediction." ACM TOIS, 2019.
5. Ding, X., et al. "Knowledge-Driven Stock Trend Prediction and Explanation via Temporal Convolutional Network." WWW, 2019.
