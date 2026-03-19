# Chapter 257: Event Extraction Trading

## Introduction

Event extraction trading is a strategy that uses natural language processing (NLP) to automatically identify and classify market-relevant events from unstructured text sources — news articles, regulatory filings, earnings transcripts, social media posts, and central bank communications. Once extracted, these events serve as inputs to trading models that position ahead of, or react rapidly to, the information embedded in the text.

Traditional sentiment analysis assigns a single polarity score to an entire document. Event extraction goes further: it identifies *what happened*, *to whom*, *when*, and *what type of event* it was. A headline like "Tesla recalls 1.2 million vehicles due to software defect" contains a specific event (product recall), an entity (Tesla), a magnitude (1.2 million vehicles), and a cause (software defect). A sentiment model might simply label this "negative," but an event extraction system provides structured, actionable intelligence that a trading model can reason about.

In modern markets, the speed and accuracy with which a firm processes textual information determines its edge. Discretionary traders cannot read thousands of news articles per day; event extraction systems can. For both equity and cryptocurrency markets, events such as exchange listings, partnership announcements, regulatory actions, and security breaches have well-documented price impact patterns that systematic strategies can exploit.

This chapter presents a complete event extraction trading framework. We cover the NLP techniques for extracting events from text, the machine learning models that classify event types and predict their market impact, and a working Rust implementation that connects to the Bybit cryptocurrency exchange for live event-driven trading.

## Key Concepts

### Event Definition and Taxonomy

A financial event is a discrete occurrence that changes the information set available to market participants. Events can be categorized along several dimensions:

**By source:**
- **Corporate events**: Earnings releases, M&A announcements, management changes, dividend declarations, stock splits, share buybacks
- **Macroeconomic events**: Central bank rate decisions, employment reports, GDP releases, inflation data, trade balance figures
- **Regulatory events**: SEC filings, FDA approvals/rejections, antitrust rulings, sanctions, compliance violations
- **Market structure events**: Exchange listings/delistings, index rebalancing, circuit breakers, trading halts
- **Crypto-specific events**: Token burns, hard forks, exchange hacks, whale movements, protocol upgrades, airdrop announcements

**By predictability:**
- **Scheduled events**: Earnings dates, FOMC meetings, options expiration — the timing is known, but the content is not
- **Unscheduled events**: Product recalls, lawsuits, natural disasters, security breaches — both timing and content are unknown

**By impact horizon:**
- **Immediate impact**: Flash crashes, breaking news, exchange halts — price reacts within seconds
- **Delayed impact**: Regulatory reviews, patent grants, management transitions — full price adjustment takes days or weeks
- **Persistent impact**: Industry disruption, structural regulation, technological breakthroughs — permanent shift in value

### Event Extraction Pipeline

The event extraction pipeline transforms raw text into structured event records through several stages:

**1. Document Acquisition**: Collect text from news feeds (Reuters, Bloomberg), social media (Twitter/X), regulatory databases (SEC EDGAR), and exchange announcements (Bybit, Binance).

**2. Preprocessing**: Tokenize the text, normalize entity names, resolve coreferences (e.g., "the company" → "Tesla"), and segment multi-event documents into individual sentences or passages.

**3. Event Detection**: Determine whether a text span contains an event mention. This is a binary classification task:

$$P(\text{event} | \text{span}) = \sigma(\mathbf{w}^T \phi(\text{span}) + b)$$

where $\phi(\text{span})$ is a feature representation of the text span.

**4. Event Classification**: Assign detected events to predefined categories. Given an event mention $e$, the classifier outputs a distribution over event types:

$$P(\text{type}_k | e) = \frac{\exp(\mathbf{w}_k^T \mathbf{h}_e)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{h}_e)}$$

where $\mathbf{h}_e$ is the contextual representation of the event mention and $K$ is the number of event types.

**5. Argument Extraction**: For each event, extract its arguments — the entities, values, and temporal expressions that parameterize the event. For example, from "Bybit lists SOL/USDT perpetual contract on March 15," we extract:
- Event type: Exchange listing
- Asset: SOL/USDT
- Exchange: Bybit
- Instrument: Perpetual contract
- Date: March 15

**6. Event Linking**: Connect related events across documents and time. An earnings miss followed by an analyst downgrade followed by a management change forms a causal chain that is more informative than any single event.

### Named Entity Recognition (NER) for Finance

Financial NER identifies entities relevant to market analysis:

- **ORG**: Companies, exchanges, central banks, regulatory bodies
- **TICKER**: Stock symbols, cryptocurrency trading pairs
- **MONEY**: Dollar amounts, percentages, price levels
- **DATE**: Temporal expressions for event timing
- **PERSON**: CEOs, fund managers, policymakers
- **EVENT**: Named events (FOMC meeting, halving, earnings call)

Standard NER models trained on general text perform poorly on financial text because of domain-specific entities (ticker symbols, financial instruments) and conventions (abbreviations, jargon). Fine-tuning on financial corpora like SEC filings, earnings transcripts, and crypto news substantially improves accuracy.

### Trigger Word Detection

Event triggers are words or phrases that most clearly express the occurrence of an event. Detecting triggers is the foundation of event extraction:

- "**acquired**" → M&A event
- "**recalled**" → Product recall event
- "**listed**" → Exchange listing event
- "**hacked**" → Security breach event
- "**burned**" → Token burn event
- "**forked**" → Hard fork event

Trigger detection can be formulated as a sequence labeling task using BIO tagging:

$$y_t = \arg\max_{c \in \{B, I, O\}} P(c | \mathbf{x}_t, \mathbf{x}_{t-1}, \ldots, \mathbf{x}_1)$$

where $\mathbf{x}_t$ is the token at position $t$ and $y_t \in \{B\text{-trigger}, I\text{-trigger}, O\}$.

## ML Approaches

### Sequence Labeling with BiLSTM-CRF

The BiLSTM-CRF architecture is a strong baseline for event extraction. A bidirectional LSTM encodes each token in context:

$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$
$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$

A Conditional Random Field (CRF) layer on top enforces valid tag sequences (e.g., I-trigger cannot follow O without a preceding B-trigger):

$$P(\mathbf{y} | \mathbf{x}) = \frac{\exp\left(\sum_{t=1}^{T} \phi(y_{t-1}, y_t, \mathbf{h}_t)\right)}{\sum_{\mathbf{y}'} \exp\left(\sum_{t=1}^{T} \phi(y'_{t-1}, y'_t, \mathbf{h}_t)\right)}$$

### Transformer-Based Event Extraction

Modern event extraction systems leverage pre-trained transformer models (BERT, FinBERT, RoBERTa) for richer contextual representations. The approach typically involves:

1. **Encoding**: Pass the input text through a transformer encoder to obtain contextualized token representations.
2. **Trigger detection**: Apply a token classification head to identify event triggers.
3. **Event typing**: Classify detected triggers into event categories.
4. **Argument extraction**: For each detected event, apply a span extraction head (similar to question answering) to identify event arguments.

The joint loss function combines all subtasks:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{trigger}} + \lambda_2 \mathcal{L}_{\text{type}} + \lambda_3 \mathcal{L}_{\text{argument}}$$

### Event Impact Prediction

Once events are extracted, we need to predict their market impact. This is a regression or classification task that maps event features to expected price changes.

Given an event record $e$ with features $\mathbf{x}_e$ (event type, entity, magnitude, sentiment, historical analogs), the impact prediction model estimates:

$$\hat{r}_e = f(\mathbf{x}_e) = \mathbf{w}^T \mathbf{x}_e + b$$

For classification into impact buckets (strong positive, mild positive, neutral, mild negative, strong negative):

$$P(\text{impact}_k | e) = \text{softmax}(\mathbf{W} \mathbf{x}_e + \mathbf{b})_k$$

Key features for impact prediction include:
- **Event type**: Different event types have different average impacts
- **Entity importance**: Events affecting large-cap stocks move markets more than small-cap events
- **Novelty**: First occurrence of an event type is more impactful than repetitions
- **Timing**: Events during market hours have different dynamics than after-hours events
- **Sentiment intensity**: The strength of language used in the event description
- **Historical analogy**: How similar past events affected the same or similar assets

### Gradient Boosting for Event Scoring

Gradient boosting models (XGBoost, LightGBM) excel at event impact prediction because they handle mixed feature types, capture non-linear interactions, and are robust to noisy labels:

**Feature engineering for event scoring:**
- **One-hot encoded event type**: Binary indicators for each event category
- **Entity embeddings**: Dense vector representations of affected companies or tokens
- **Temporal features**: Day of week, hour, days until next scheduled event
- **Market context**: Current volatility regime, trend direction, volume relative to average
- **Event history**: Number of similar events in the past N days, average impact of recent events

## Feature Engineering

### Text Features

**TF-IDF over event descriptions**: Convert event text into a fixed-dimensional vector using term frequency-inverse document frequency weighting. This captures the relative importance of words within the event description compared to a corpus of historical events.

**Word embeddings**: Average or max-pool pre-trained word vectors (GloVe, FastText, or domain-specific embeddings trained on financial text) to produce a fixed-size representation of the event text.

**Contextual embeddings**: Use the [CLS] token output from a fine-tuned BERT or FinBERT model as a dense representation of the full event description. This captures semantic nuances and context-dependent word meanings.

### Event Sequence Features

Events do not occur in isolation. The sequence of events preceding a trade decision carries information:

- **Event frequency**: Number of events per entity in the last 1, 7, 30 days
- **Event diversity**: Number of distinct event types in the recent window
- **Event momentum**: Ratio of positive to negative events in the recent window
- **Event clustering**: Time gaps between consecutive events (bursts vs. steady flow)
- **Cross-entity events**: Events affecting related entities (supply chain, competitors, sector)

### Market Context Features

Event impact depends on the prevailing market regime:

- **Volatility**: VIX level or realized volatility (events have larger impact in low-volatility regimes)
- **Trend**: Current price relative to moving averages (events aligned with the trend have more persistent impact)
- **Liquidity**: Bid-ask spread and order book depth (events in illiquid markets cause larger dislocations)
- **Positioning**: Open interest, funding rate (crypto), short interest (equities)

## Applications

### News-Driven Alpha Generation

Event extraction enables systematic news trading:

1. **Pre-event positioning**: For scheduled events (earnings, FOMC), use extracted expectations and historical patterns to position before the announcement.
2. **Event reaction**: For unscheduled events, rapidly extract and classify the event, predict its impact, and execute trades within seconds of publication.
3. **Post-event drift**: Many events are not fully priced immediately. Event extraction identifies events with historically persistent price impact and trades the continuation.

### Risk Management

Event extraction supports risk management by:

- **Tail risk detection**: Identifying events that historically precede large drawdowns (regulatory crackdowns, exchange hacks in crypto)
- **Exposure monitoring**: Tracking events affecting portfolio holdings and flagging concentrated risk
- **Hedging triggers**: Automatically suggesting or executing hedges when adverse events are detected

### Crypto-Specific Applications

Cryptocurrency markets are particularly amenable to event-driven trading because:

- **24/7 markets**: Events can be traded at any time without waiting for market open
- **Higher volatility**: Event impact is often larger, providing more trading opportunity
- **Less efficient pricing**: Crypto markets are less efficient than equity markets, so event information is incorporated more slowly
- **Unique event types**: Token burns, protocol upgrades, exchange listings, and whale movements create crypto-specific alpha opportunities

## Rust Implementation

Our Rust implementation provides a complete event extraction trading toolkit with the following components:

### EventExtractor

The `EventExtractor` struct processes raw text and identifies financial events using keyword-based trigger detection. It maintains a configurable dictionary of trigger words mapped to event types and applies pattern matching to classify events from news headlines and article snippets.

### EventClassifier

The `EventClassifier` implements logistic regression for multi-class event type classification. Given a feature vector derived from TF-IDF weights of the event text, it outputs probabilities over predefined event categories (earnings, M&A, regulatory, listing, security breach, token burn, partnership, and others).

### ImpactPredictor

The `ImpactPredictor` estimates the expected price impact of an extracted event. It uses a linear model trained on historical event-impact pairs, incorporating features such as event type, sentiment score, entity importance, and market context. The output is a signed float representing the predicted percentage price change.

### SentimentScorer

The `SentimentScorer` assigns sentiment polarity to event descriptions using a lexicon-based approach. It maintains dictionaries of positive and negative financial terms, computes the net sentiment as the normalized difference between positive and negative word counts, and adjusts for negation and intensifier patterns.

### TradingSignalGenerator

The `TradingSignalGenerator` combines event extraction, classification, impact prediction, and sentiment scoring into actionable trading signals. It processes a stream of text inputs, filters events by confidence and predicted impact thresholds, and outputs buy/sell/hold signals with position sizing based on predicted impact magnitude.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data from the `/v5/market/kline` endpoint and ticker information from the `/v5/market/tickers` endpoint. The client handles response parsing, error handling, and provides market context data needed for event impact assessment.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to obtain market data for event-driven trading:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data at configurable intervals. Used for computing market context features (volatility, trend) and backtesting event-driven strategies.
- **Tickers endpoint** (`/v5/market/tickers`): Provides real-time price, volume, and funding rate data. Used for assessing current market conditions when evaluating event impact.

The Bybit API complements event extraction by providing:
- Real-time market context for event impact assessment
- Historical data for backtesting event-driven strategies
- Cryptocurrency-specific data (funding rates, open interest) relevant to crypto event analysis

## References

1. Chen, Y., Xu, L., Liu, K., Zeng, D., & Zhao, J. (2015). Event extraction via dynamic multi-pooling convolutional neural networks. *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics*, 167-176.
2. Ding, X., Zhang, Y., Liu, T., & Duan, J. (2015). Deep learning for event-driven stock prediction. *Proceedings of the 24th International Joint Conference on Artificial Intelligence*, 2327-2333.
3. Jacobs, G., Lefever, E., & Hoste, V. (2018). Economic event detection in company-specific news text. *Proceedings of the First Workshop on Economics and Natural Language Processing*, 1-10.
4. Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.
5. Yang, S., Feng, D., Qian, T., & Li, Y. (2019). Exploring pre-trained language models for event extraction and generation. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 5284-5294.
6. Xiang, W., & Wang, B. (2019). A survey of event extraction from text. *IEEE Access*, 7, 173111-173137.
