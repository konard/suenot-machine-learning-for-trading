# Chapter 258: Temporal Reasoning in Finance

## Introduction

Temporal reasoning in finance is the practice of extracting, interpreting, and leveraging time-related information from financial texts to make informed trading decisions. Financial markets are fundamentally driven by events that unfold over time: earnings announcements have specific dates, economic indicators are released on schedules, contracts expire, and corporate actions follow timelines. The ability to automatically understand and reason about these temporal relationships is a powerful edge for algorithmic trading systems.

Traditional NLP approaches treat financial documents as bags of words or sequences of tokens, ignoring the rich temporal structure embedded in the text. A sentence like "The Federal Reserve will raise rates in March, following the January CPI report that exceeded expectations" contains multiple temporal references, causal relationships, and forward-looking signals that a temporal reasoning system can extract and act upon.

This chapter presents a complete framework for temporal reasoning in financial texts. We cover temporal expression extraction, event ordering, duration estimation, and the application of these techniques to both stock market and cryptocurrency trading using data from the Bybit exchange.

## Key Concepts

### Temporal Expressions in Financial Text

Financial documents are dense with temporal references that fall into several categories:

- **Explicit dates**: "on March 15, 2024", "Q3 2023 earnings"
- **Relative references**: "next quarter", "last fiscal year", "within 30 days"
- **Durations**: "over the past six months", "for the next three years"
- **Recurring events**: "every FOMC meeting", "quarterly earnings cycle"
- **Vague temporal markers**: "soon", "in the near term", "recently"

Each category requires different handling. Explicit dates can be normalized to timestamps directly. Relative references must be resolved against a document creation date or reference time. Durations define intervals rather than points. Vague markers require probabilistic interpretation based on context.

Formally, a temporal expression $\tau$ maps to a time interval $[t_{start}, t_{end}]$ on an absolute timeline. For point expressions, $t_{start} = t_{end}$. The extraction task is to identify spans of text that denote temporal information and normalize them to this canonical form.

### TimeML and Temporal Annotation

TimeML is the standard markup language for temporal information in text. It defines several key annotation types:

- **TIMEX3**: Temporal expressions (dates, times, durations, sets)
- **EVENT**: Occurrences or states described in text
- **TLINK**: Temporal links relating events and times (BEFORE, AFTER, SIMULTANEOUS, INCLUDES, etc.)
- **ALINK**: Aspectual links (INITIATES, TERMINATES, CONTINUES)
- **SLINK**: Subordination links connecting events through modality

For financial applications, the TLINK relations are most valuable. Given events $e_1$ and $e_2$, the temporal relation $R(e_1, e_2) \in \{BEFORE, AFTER, SIMULTANEOUS, INCLUDES, IS\_INCLUDED, BEGINS, ENDS, OVERLAP\}$ defines how they relate in time. Constructing a consistent temporal graph from these pairwise relations enables reasoning about event sequences.

### Allen's Interval Algebra

Allen's interval algebra provides a rigorous mathematical framework for reasoning about temporal intervals. It defines 13 mutually exclusive relations between two time intervals:

$$R \in \{before, after, meets, met\_by, overlaps, overlapped\_by, starts, started\_by, finishes, finished\_by, during, contains, equals\}$$

For two intervals $X = [x_1, x_2]$ and $Y = [y_1, y_2]$:

- $X$ **before** $Y$: $x_2 < y_1$
- $X$ **meets** $Y$: $x_2 = y_1$
- $X$ **overlaps** $Y$: $x_1 < y_1 < x_2 < y_2$
- $X$ **during** $Y$: $y_1 < x_1$ and $x_2 < y_2$
- $X$ **starts** $Y$: $x_1 = y_1$ and $x_2 < y_2$
- $X$ **finishes** $Y$: $x_2 = y_2$ and $y_1 < x_1$
- $X$ **equals** $Y$: $x_1 = y_1$ and $x_2 = y_2$

The remaining six are inverses. Allen's algebra supports constraint propagation through a transitivity table, enabling inference of new relations from known ones. For example, if event $A$ is *before* event $B$, and $B$ *overlaps* event $C$, then $A$ is *before* $C$.

In finance, this framework models scenarios like: "The lockup period *ends before* the secondary offering *begins*" or "The earnings call *overlaps with* the options expiration window."

### Temporal Knowledge Graphs

A temporal knowledge graph (TKG) extends a standard knowledge graph by associating each fact with a time interval during which it is valid:

$$(subject, predicate, object, [t_{start}, t_{end}])$$

For financial applications, a TKG might contain facts like:

- (AAPL, CEO, Tim Cook, [2011-08-24, present])
- (FED, interest_rate, 5.25%, [2023-07-27, 2024-09-18])
- (BTCUSD, trend, bullish, [2024-01-10, 2024-03-14])

Temporal knowledge graphs support queries that are impossible with static graphs:

- "Who was the CEO of Company X when the earnings miss occurred?"
- "What was the interest rate environment during the last crypto bull run?"
- "Which regulatory changes were in effect when the flash crash happened?"

The link prediction task on TKGs estimates the probability of a fact holding at a future time:

$$P((s, p, o, t_{future}) | \mathcal{G}_{\leq t_{now}})$$

This directly maps to financial forecasting: predicting future states of entities based on their temporal history.

## ML Approaches

### BERT-based Temporal Expression Extraction

Fine-tuning BERT for temporal expression extraction treats the problem as a token classification (NER) task. Each token is labeled with BIO tags indicating whether it begins (B-TIMEX), continues (I-TIMEX), or is outside (O) a temporal expression.

Given input tokens $\{w_1, w_2, \ldots, w_n\}$, BERT produces contextual representations $\{\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_n\}$. A classification head maps each representation to tag probabilities:

$$P(y_i | w_1, \ldots, w_n) = \text{softmax}(\mathbf{W} \mathbf{h}_i + \mathbf{b})$$

For financial text, the model must handle domain-specific expressions like "FY2024", "Q3", "T+2 settlement", and "next FOMC". Domain-adaptive pre-training on financial corpora significantly improves performance on these expressions.

### Temporal Relation Classification

Given a pair of events or temporal expressions in context, the temporal relation classifier predicts the Allen relation between them. The input is formed by concatenating the context around both entities with special markers:

$$\text{input} = [\text{CLS}] \; \text{context}_1 \; [E1] \; e_1 \; [/E1] \; \text{context}_2 \; [E2] \; e_2 \; [/E2] \; [\text{SEP}]$$

The [CLS] representation is passed through a multi-class classifier:

$$P(R | e_1, e_2, \text{context}) = \text{softmax}(\mathbf{W}_R \mathbf{h}_{[CLS]} + \mathbf{b}_R)$$

The model is trained on annotated temporal relation datasets (TimeBank, AQUAINT) and fine-tuned on financial text where temporal ordering is critical for understanding causality and event sequences.

### Temporal Graph Neural Networks

For reasoning over temporal knowledge graphs, Temporal Graph Neural Networks (T-GNNs) learn representations that capture both structural and temporal patterns:

$$\mathbf{h}_v^{(l+1)} = \sigma \left( \sum_{(u, r, t) \in \mathcal{N}(v)} \alpha(t) \cdot \mathbf{W}_r^{(l)} \mathbf{h}_u^{(l)} + \mathbf{b}^{(l)} \right)$$

where $\alpha(t)$ is a time-decay attention weight that gives more importance to recent facts:

$$\alpha(t) = \exp\left(-\lambda (t_{now} - t)\right)$$

The temporal attention mechanism allows the network to learn that recent CEO changes matter more than decade-old appointments, while long-term interest rate trends may have persistent effects.

### Sequence-to-Sequence for Timeline Construction

Constructing a complete timeline from a financial document can be framed as a sequence-to-sequence problem. The encoder processes the input text, and the decoder generates a structured timeline of events:

$$\text{Input: } \text{"AAPL reported Q3 earnings on Aug 1. The stock dropped 5\% the next day..."}$$
$$\text{Output: } (e_1, \text{2024-08-01}, \text{earnings\_report}) \rightarrow (e_2, \text{2024-08-02}, \text{price\_drop})$$

The decoder uses temporal-aware positional encodings that capture both sequence position and calendar time, allowing the model to reason about temporal gaps and periodicities.

## Feature Engineering

### Temporal Density Features

The density of temporal references in a financial document correlates with the document's relevance for time-sensitive trading decisions:

$$\text{TemporalDensity}(d) = \frac{|\{\tau \in d : \tau \text{ is a TIMEX3}\}|}{|d|}$$

where $|d|$ is the number of tokens in document $d$. Documents with high temporal density (earnings reports, economic calendars, regulatory filings) tend to contain more actionable trading signals than temporally sparse documents (general commentary, opinion pieces).

### Event Recency Score

The recency of events mentioned in a document affects their market impact. A recency-weighted score for each event is:

$$\text{Recency}(e) = \exp\left(-\lambda \cdot \frac{t_{now} - t_e}{T_{scale}}\right)$$

where $t_e$ is the event time, $t_{now}$ is the current time, $\lambda$ is a decay parameter, and $T_{scale}$ normalizes the time difference (e.g., in trading days). Events with high recency scores are more likely to be driving current price action.

### Forward-Looking Ratio

The ratio of future-pointing temporal expressions to past-pointing ones in a document indicates its forward-looking nature:

$$\text{FLR}(d) = \frac{|\{\tau \in d : t_\tau > t_{now}\}|}{|\{\tau \in d : t_\tau \leq t_{now}\}| + \epsilon}$$

Documents with high FLR contain predictions, guidance, and forward-looking statements that are particularly valuable for trading. Earnings call transcripts typically have high FLR during the guidance section and low FLR during the results review.

### Temporal Coherence Score

The temporal coherence of a document measures how consistently events are ordered in the text relative to their actual chronological order:

$$\text{Coherence}(d) = \frac{\text{number of concordant event pairs}}{\text{total number of event pairs}}$$

A coherence score near 1.0 indicates a chronologically organized narrative, while a low score suggests complex temporal structure (flashbacks, comparisons across time periods) that may require more sophisticated reasoning.

## Applications

### Earnings Event Timeline Trading

Temporal reasoning enables systematic trading around earnings events by constructing complete timelines:

1. **Pre-earnings**: Extract forward-looking statements from analyst reports and company guidance. Identify temporal references to revenue recognition periods, contract renewals, and product launches.
2. **Earnings release**: Parse the earnings report for temporal comparisons (YoY, QoQ growth), duration of trends ("third consecutive quarter of growth"), and future commitments.
3. **Post-earnings**: Monitor analyst revisions and extract temporal expectations for future quarters. Detect shifts in timeline language ("accelerating" vs "stabilizing" vs "decelerating").

Trading signals are generated when temporal analysis reveals mismatches between market expectations (implied by options pricing and analyst consensus) and the actual temporal structure of company fundamentals.

### Regulatory Event Sequencing

Financial regulations follow complex timelines with proposal, comment, and implementation phases. Temporal reasoning extracts these sequences:

- Comment period deadlines
- Phase-in schedules for new requirements
- Compliance milestones
- Grandfather clause expirations

Trading strategies based on regulatory timelines can position ahead of implementation dates when market participants are slow to price in regulatory impacts.

### Cryptocurrency Temporal Patterns

Cryptocurrency markets exhibit temporal patterns distinct from traditional finance:

- **Protocol upgrade timelines**: Hard forks, token migration deadlines, and staking lock-up periods
- **Tokenomics events**: Vesting schedules, token unlock dates, and halving cycles
- **DeFi temporal dynamics**: Liquidity mining epochs, governance voting periods, and protocol fee switches

Temporal reasoning applied to crypto-specific sources (governance forums, protocol documentation, on-chain data) generates trading signals from upcoming tokenomics events.

### Cross-Asset Temporal Arbitrage

When temporal analysis reveals that an event's impact has been priced into one asset but not another, cross-asset arbitrage opportunities arise. For example:

- A rate decision is reflected in bond prices but not yet in currency markets
- A supply chain disruption timeline affects commodity futures before equity markets adjust
- A regulatory timeline impacts one crypto exchange token before another

The temporal reasoning system identifies these leads and lags by comparing event timelines across correlated assets.

## Rust Implementation

Our Rust implementation provides a complete temporal reasoning toolkit for financial text analysis with the following components:

### TemporalExpression

The `TemporalExpression` struct represents an extracted temporal reference with its normalized form. It stores the original text span, the resolved time interval `[start, end]`, the expression type (DATE, TIME, DURATION, SET), and a confidence score. The parser handles financial-specific formats including fiscal quarters (Q1-Q4), fiscal years (FY2024), and settlement conventions (T+1, T+2).

### TemporalRelation

The `TemporalRelation` enum implements Allen's 13 interval relations and supports constraint propagation through a transitivity table. Given known relations between pairs of events, it infers new relations through transitive closure, building a complete temporal graph.

### EventTimeline

The `EventTimeline` struct maintains an ordered sequence of financial events extracted from text. It supports insertion, querying by time range, and gap detection. Events are represented with timestamps, types (EARNINGS, FOMC, HALVING, UNLOCK), and associated entities. The timeline can merge events from multiple sources while maintaining temporal consistency.

### TemporalFeatureExtractor

The `TemporalFeatureExtractor` computes the feature engineering metrics described above: temporal density, event recency scores, forward-looking ratio, and temporal coherence. It accepts raw text and produces a feature vector suitable for downstream ML models.

### TemporalTradingStrategy

The `TemporalTradingStrategy` implements a backtesting framework that generates buy/sell signals based on temporal features. It combines recency-weighted event scores with forward-looking ratios to produce a composite signal. The strategy supports configurable thresholds, position sizing based on signal strength, and risk management through temporal stop-losses (exit when the event's expected impact window expires).

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data from the `/v5/market/kline` endpoint for backtesting and order book snapshots from `/v5/market/orderbook` for live signal generation. The client handles response parsing, error handling, and rate limiting.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to obtain market data for backtesting and live trading:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data at configurable intervals. Used for backtesting temporal trading strategies against historical crypto price data.
- **Order book endpoint** (`/v5/market/orderbook`): Provides real-time order book snapshots for live signal generation and execution.

Cryptocurrency markets are particularly well-suited for temporal reasoning strategies because:
- Markets trade 24/7, making temporal patterns more consistent
- Token unlock schedules and protocol events follow predictable timelines
- On-chain data provides verifiable timestamps for events
- The Bybit API provides fine-grained historical data for comprehensive backtesting

## References

1. Allen, J. F. (1983). Maintaining knowledge about temporal intervals. *Communications of the ACM*, 26(11), 832-843.
2. Pustejovsky, J., Castano, J., Ingria, R., Sauri, R., Gaizauskas, R., Setzer, A., & Katz, G. (2003). TimeML: Robust specification of event and temporal expressions in text. *New Directions in Question Answering*, 3, 28-34.
3. Ning, Q., Wu, H., & Roth, D. (2018). A multi-axis annotation scheme for event temporal relations. *Proceedings of ACL*, 1318-1328.
4. Leeuwenberg, A., & Moens, M. F. (2019). A survey on temporal reasoning for temporal information extraction from text. *Journal of Artificial Intelligence Research*, 66, 341-380.
5. Lacroix, T., Obozinski, G., & Usunier, N. (2020). Tensor decompositions for temporal knowledge base completion. *Proceedings of ICLR*.
