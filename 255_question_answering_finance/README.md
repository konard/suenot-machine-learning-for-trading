# Chapter 255: Question Answering for Finance

## Introduction

Question answering (QA) for finance applies natural language processing techniques to automatically extract answers from financial documents, reports, and data. Unlike general-domain QA, financial question answering requires numerical reasoning, understanding of domain-specific terminology, and the ability to synthesize information from tables, text, and time-series data. This makes it one of the most challenging applications of NLP in the financial domain.

Financial analysts spend a substantial portion of their time reading earnings reports, SEC filings, analyst notes, and market commentary to find specific pieces of information. A QA system automates this process by accepting a natural language question and returning a precise answer extracted from or computed over a corpus of financial documents. For algorithmic trading, QA systems can be integrated into pipelines that convert unstructured text into structured signals, enabling faster reaction to earnings releases, economic reports, and regulatory filings.

This chapter presents the theory behind financial QA, the key architectural patterns used in modern systems, and a working Rust implementation that demonstrates extractive and numerical QA on financial data, with integration to the Bybit cryptocurrency exchange for real-time market context.

## Key Concepts

### Extractive Question Answering

Extractive QA identifies a contiguous span of text in a given context passage that answers the question. Given a question $q$ and a context passage $c = (c_1, c_2, \ldots, c_n)$ of $n$ tokens, the model predicts a start position $s$ and end position $e$ such that the answer is $a = (c_s, c_{s+1}, \ldots, c_e)$.

The probability of the answer span is decomposed as:

$$P(a | q, c) = P(s | q, c) \cdot P(e | q, c, s)$$

In practice, two separate linear heads predict start and end logits over all token positions:

$$\text{start\_logits} = \mathbf{W}_s \mathbf{h} + \mathbf{b}_s$$
$$\text{end\_logits} = \mathbf{W}_e \mathbf{h} + \mathbf{b}_e$$

where $\mathbf{h} \in \mathbb{R}^{n \times d}$ is the hidden representation from the encoder. The span with the highest combined score $\text{start\_logits}_s + \text{end\_logits}_e$ (subject to $s \leq e$) is selected as the answer.

### Numerical Reasoning in Finance

Financial QA frequently requires numerical computation rather than simple span extraction. For example, answering "What was the year-over-year revenue growth?" requires:

1. Extracting the revenue figures for two consecutive years
2. Computing the percentage change: $\text{growth} = \frac{R_{t} - R_{t-1}}{R_{t-1}} \times 100\%$

The FinQA dataset (Chen et al., 2021) formalizes this as a program generation task. The model generates a sequence of operations:

$$\text{Program} = [op_1(arg_1, arg_2), op_2(arg_3, arg_4), \ldots]$$

where each operation $op_i \in \{\text{add}, \text{subtract}, \text{multiply}, \text{divide}, \text{greater}, \text{exp}\}$ and arguments can be constants, table values, or results of previous operations.

### Retrieval-Augmented QA

For large document collections, a two-stage retrieve-then-read pipeline is standard:

1. **Retriever**: Given question $q$, retrieve the top-$k$ most relevant passages from a corpus $\mathcal{D}$ using dense retrieval:

$$\text{score}(q, d) = \mathbf{E}_q(q)^T \mathbf{E}_d(d)$$

where $\mathbf{E}_q$ and $\mathbf{E}_d$ are learned encoders for questions and documents, respectively.

2. **Reader**: For each retrieved passage $d_i$, extract or generate an answer. The final answer is selected based on the combined retriever-reader score:

$$\hat{a} = \arg\max_{a, d_i} \left[ \lambda \cdot \text{score}(q, d_i) + (1 - \lambda) \cdot P(a | q, d_i) \right]$$

This architecture scales to millions of documents while maintaining answer quality.

### Table Question Answering

Financial documents are rich in tables — income statements, balance sheets, cash flow statements. Table QA requires models to:

- Parse the table structure (rows, columns, headers)
- Locate relevant cells based on the question
- Perform aggregation operations (sum, average, comparison) when needed

Given a table $T$ with $m$ rows and $n$ columns, and a question $q$, a table QA model produces an answer by selecting cells and optionally applying an aggregation operator $\text{agg} \in \{\text{NONE}, \text{SUM}, \text{COUNT}, \text{AVERAGE}, \text{MAX}, \text{MIN}\}$:

$$a = \text{agg}\left(\{T_{i,j} : (i,j) \in \text{selected\_cells}\}\right)$$

### Confidence Calibration

In financial applications, knowing when the model does not know is as important as getting the right answer. A well-calibrated QA system should output a confidence score $c$ such that among all predictions where $c \approx p$, approximately $p$ fraction are correct.

Calibration can be measured by Expected Calibration Error (ECE):

$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where $B_m$ are bins of predictions grouped by confidence, $\text{acc}(B_m)$ is the fraction correct in bin $m$, and $\text{conf}(B_m)$ is the average confidence.

## ML Approaches

### Transformer-Based Extractive QA

The dominant approach uses a pre-trained transformer encoder (BERT, RoBERTa, FinBERT) fine-tuned on QA data. The input is formatted as:

$$[\text{CLS}] \; q_1 \; q_2 \; \ldots \; q_m \; [\text{SEP}] \; c_1 \; c_2 \; \ldots \; c_n \; [\text{SEP}]$$

The model produces contextualized representations for each token, and two classification heads predict the start and end positions of the answer span. For financial QA, domain-specific pre-training (e.g., FinBERT trained on financial text) significantly improves performance because the model better understands terms like "EBITDA", "diluted EPS", and "free cash flow".

### Program Generation for Numerical QA

For questions requiring computation, models generate a domain-specific language (DSL) program. The architecture typically uses:

1. An encoder that processes the question and context (table + text)
2. A decoder that generates operations step by step, attending to the encoded representation

At each decoding step $t$, the decoder produces:

$$P(op_t | op_{<t}, q, c) = \text{softmax}(\mathbf{W}_{op} \mathbf{d}_t)$$
$$P(arg_t | op_t, op_{<t}, q, c) = \text{softmax}(\mathbf{W}_{arg} \mathbf{d}_t + \text{copy\_attn}_t)$$

The copy attention mechanism allows the model to select arguments directly from the input table or text, which is crucial for handling the large vocabulary of financial numbers.

### Cross-Encoder Reranking

After initial retrieval, a cross-encoder reranker jointly processes the question and each candidate passage to produce a more accurate relevance score:

$$\text{score}(q, d) = \sigma(\mathbf{w}^T \text{Encoder}([\text{CLS}] \; q \; [\text{SEP}] \; d \; [\text{SEP}]))$$

Cross-encoders are more expensive than bi-encoders (they cannot pre-compute document embeddings) but capture fine-grained interactions between question and passage tokens. In financial QA, this reranking step is especially valuable because subtle differences in context (e.g., which fiscal year a number refers to) dramatically affect answer correctness.

## Feature Engineering

### Document Structure Features

Financial documents have rich structure that aids QA:

- **Section headers**: Identify which part of a filing (Risk Factors, MD&A, Financial Statements) a passage belongs to
- **Table proximity**: Distance from the nearest table, which correlates with numerical answer likelihood
- **Temporal markers**: Fiscal year, quarter, and date references that disambiguate questions about specific time periods
- **Entity mentions**: Company names, ticker symbols, and executive names provide context anchoring

### Question Type Classification

Classifying the question type helps route to the appropriate answering strategy:

- **Factoid**: "What was Apple's revenue in Q3 2024?" → extractive span
- **Numerical**: "What is the gross margin percentage?" → numerical computation
- **Boolean**: "Did revenue increase year-over-year?" → yes/no classification
- **Comparative**: "Which segment had higher growth?" → multi-step reasoning
- **Temporal**: "When did the company first report a profit?" → temporal extraction

### Financial Entity Features

Domain-specific features improve answer extraction:

- **Currency detection**: Identifying monetary amounts and their currencies
- **Percentage recognition**: Distinguishing between absolute numbers and percentages
- **Time period alignment**: Mapping mentions of "last quarter", "FY2023", "YTD" to specific date ranges
- **Unit normalization**: Converting between millions, billions, and raw numbers

## Applications

### Earnings Call Analysis

QA systems can automatically process earnings call transcripts to extract:

1. **Guidance figures**: Revenue and earnings forecasts mentioned by management
2. **Sentiment signals**: How management characterizes business conditions
3. **Risk factors**: New risks or concerns raised during Q&A sessions
4. **Competitive intelligence**: Mentions of competitors and market positioning

A trading system can monitor hundreds of earnings calls simultaneously, extracting key metrics in real-time and generating signals before human analysts can digest the information.

### SEC Filing Analysis

Public companies file extensive reports with the SEC (10-K, 10-Q, 8-K). QA systems enable:

- **Risk monitoring**: Tracking changes in risk factor disclosures across filings
- **Covenant tracking**: Identifying debt covenants and their current status
- **Related party transactions**: Extracting details of insider transactions
- **Revenue recognition changes**: Detecting changes in accounting policies

### Crypto Market Intelligence

For cryptocurrency markets, QA can be applied to:

- **Whitepaper analysis**: Extracting tokenomics, governance mechanisms, and technical specifications
- **Community sentiment**: Answering questions about project developments from Discord/Telegram channels
- **Protocol documentation**: Querying DeFi protocol docs for specific parameters (fees, collateral ratios, liquidation thresholds)
- **Exchange announcements**: Monitoring Bybit and other exchange announcements for listing, delisting, and fee changes

### Portfolio Research Automation

QA systems streamline the research process:

- **Screening**: "Which companies in the S&P 500 increased their dividend this quarter?"
- **Due diligence**: "What are the main risk factors for this company?"
- **Comparative analysis**: "How does Company A's debt-to-equity ratio compare to the industry average?"
- **Event monitoring**: "Has the company announced any share buyback programs?"

## Rust Implementation

Our Rust implementation provides a modular financial QA toolkit with the following components:

### DocumentStore

The `DocumentStore` manages a collection of financial documents, each with metadata (source, date, document type). It supports adding documents, full-text keyword search with TF-IDF scoring, and retrieval by document ID. This forms the retrieval layer of the QA pipeline.

### QuestionClassifier

The `QuestionClassifier` analyzes incoming questions and classifies them by type (factoid, numerical, boolean, comparative, temporal) using keyword patterns and syntactic heuristics. The classification determines which answering strategy the pipeline uses.

### ExtractiveQA

The `ExtractiveQA` module implements span-based answer extraction using keyword matching and proximity scoring. Given a question and a context passage, it identifies the most relevant sentence and extracts a candidate answer span. It returns both the answer text and a confidence score.

### NumericalReasoner

The `NumericalReasoner` handles questions that require computation. It parses financial numbers from text (handling currency symbols, percentages, and magnitude suffixes like "million" and "billion"), identifies the required operation from the question, and executes the computation. Supported operations include growth rate, ratio, difference, sum, and average.

### FinancialQAPipeline

The `FinancialQAPipeline` orchestrates the full QA workflow: classify the question, retrieve relevant documents, apply the appropriate answering strategy, and return a ranked list of answers with confidence scores. It combines the document store, question classifier, extractive QA, and numerical reasoner into a unified interface.

### BybitClient

The `BybitClient` provides async HTTP access to the Bybit V5 API for fetching kline data and order book snapshots. This allows the QA system to incorporate real-time market data when answering questions about current prices, volumes, and market conditions.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to enrich the QA pipeline with live market data:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data. When a question asks about recent price movements ("What was the highest price of BTC today?"), the system fetches klines and extracts the answer directly from market data.
- **Order book endpoint** (`/v5/market/orderbook`): Provides current bid/ask levels. Questions about current spread, liquidity, or order book depth can be answered using live order book snapshots.

This hybrid approach — combining document-based QA with real-time data — is particularly powerful for cryptocurrency markets where information changes rapidly and documents quickly become stale.

## References

1. Chen, Z., Chen, W., Smiley, C., Shah, S., Borber, I., Langlotz, C., & Wang, W. Y. (2021). FinQA: A dataset of numerical reasoning over financial data. *Proceedings of EMNLP 2021*, 7268-7280.
2. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *Proceedings of EMNLP 2016*, 2383-2392.
3. Zhu, F., Lei, W., Huang, Y., Wang, C., Zhang, S., Lv, J., Feng, F., & Chua, T. S. (2021). TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. *Proceedings of ACL 2021*, 3277-3287.
4. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. *Proceedings of EMNLP 2020*, 6769-6781.
5. Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.
6. Herzig, J., Nowak, P. K., Muller, T., Piccinno, F., & Eisenschlos, J. M. (2020). TaPas: Weakly supervised table parsing via pre-training. *Proceedings of ACL 2020*, 4320-4333.
