# Chapter 262: Market Making ML

## Introduction

Market making is the practice of continuously quoting bid and ask prices for a financial instrument, profiting from the spread between them while managing inventory risk. Traditional market making relied on hand-crafted rules and intuition, but modern electronic markets demand faster and more adaptive strategies. Machine learning transforms market making by enabling models to learn optimal quoting policies, predict adverse selection, and dynamically manage inventory from data.

The core challenge of market making is the tension between two competing objectives: capturing spread revenue by quoting aggressively, and avoiding losses from trading against informed counterparties or holding unbalanced inventory during adverse price moves. ML-based market makers learn to navigate this tradeoff by ingesting microstructure signals — order book imbalances, trade flow toxicity, volatility estimates — and adjusting their quoting behavior in real time.

This chapter presents a comprehensive framework for ML-driven market making. We cover the theoretical foundations of optimal quoting, the key features that inform quoting decisions, and the machine learning models used to optimize market making strategies. A complete Rust implementation connects to the Bybit cryptocurrency exchange for live data.

## Key Concepts

### Bid-Ask Spread and Market Maker Profit

A market maker posts a bid price $P^b$ and an ask price $P^a$ simultaneously. The spread $s = P^a - P^b$ represents the market maker's gross profit per round-trip (buying at the bid and selling at the ask). However, the realized profit is reduced by adverse selection — the cost of trading against informed participants who know the asset's true value.

The effective spread earned by a market maker can be decomposed as:

$$\text{Realized Spread} = \frac{s}{2} - \text{Adverse Selection Cost}$$

A profitable market maker must set spreads wide enough to cover adverse selection and inventory costs, but narrow enough to attract order flow and maintain queue priority.

### Avellaneda-Stoikov Framework

The seminal Avellaneda-Stoikov (2008) model provides an analytical foundation for optimal market making. The market maker maximizes expected utility of terminal wealth subject to inventory constraints. The optimal bid and ask quotes are:

$$P^b = S - \delta^b, \quad P^a = S + \delta^a$$

where $S$ is the mid-price and the optimal offsets are:

$$\delta^b = \delta^a = \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right) + \frac{\gamma \sigma^2 (T - t)}{2} q$$

Here $\gamma$ is the risk aversion parameter, $\kappa$ controls the fill rate sensitivity to spread, $\sigma$ is the volatility, $T - t$ is the remaining time horizon, and $q$ is the current inventory. The key insight is that the market maker skews quotes based on inventory: when long, the bid is widened and the ask is tightened to encourage selling.

### Inventory Risk Management

Inventory risk is the exposure a market maker faces from holding a net position. If the market maker accumulates a large long (short) position and the price drops (rises), losses can overwhelm spread revenue. The inventory penalty is modeled as:

$$\text{Inventory Cost} = \gamma \cdot q^2 \cdot \sigma^2 \cdot \Delta t$$

where $q$ is the inventory level, $\sigma$ is volatility, $\Delta t$ is the holding period, and $\gamma$ is the risk aversion coefficient. This quadratic penalty motivates the market maker to keep inventory close to zero.

### Adverse Selection

Adverse selection occurs when a market maker's resting orders are picked off by traders with superior information. The probability of an incoming order being informed is related to the flow toxicity metric VPIN:

$$P(\text{informed}) = \frac{\alpha \mu}{\alpha \mu + 2 \varepsilon}$$

where $\alpha$ is the probability of an information event, $\mu$ is the informed trader arrival rate, and $\varepsilon$ is the uninformed trader arrival rate on each side. When adverse selection is high, the market maker must widen spreads to compensate for the expected losses from trading against informed flow.

### Fill Probability

The probability that a limit order at distance $\delta$ from the mid-price is filled follows an exponential decay:

$$P(\text{fill} | \delta) = A \cdot e^{-\kappa \delta}$$

where $A$ is a normalization constant and $\kappa$ determines how sensitive fill probability is to the order's distance from mid. This creates a fundamental tradeoff: wider spreads yield more profit per fill but fewer fills; tighter spreads attract more flow but earn less per trade.

### Volatility Estimation

Accurate volatility estimation is critical for market making. The realized volatility over a window of $n$ returns $r_i$ is:

$$\hat{\sigma} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (r_i - \bar{r})^2}$$

In high-frequency settings, the Garman-Klass estimator provides a more efficient estimate using OHLC data:

$$\hat{\sigma}_{GK}^2 = \frac{1}{2}(\ln H - \ln L)^2 - (2\ln 2 - 1)(\ln C - \ln O)^2$$

where $H, L, O, C$ are the high, low, open, and close prices respectively. The market maker uses volatility to scale spread width: higher volatility demands wider spreads to compensate for increased inventory risk.

## ML Approaches

### Reinforcement Learning for Optimal Quoting

Reinforcement learning (RL) is the most natural framework for market making because the problem is inherently sequential: the market maker observes the state (prices, inventory, order book), takes an action (quote placement), and receives a reward (PnL). The state space is:

$$\mathbf{s}_t = [q_t, \hat{\sigma}_t, \text{OFI}_t, \text{VPIN}_t, \text{spread}_t, \text{imbalance}_t, T - t]$$

The action space consists of bid and ask offsets $(\delta^b, \delta^a)$ from the mid-price. The reward function balances PnL with inventory risk:

$$r_t = \text{PnL}_t - \gamma \cdot q_t^2$$

A Q-learning agent approximates the optimal action-value function:

$$Q(\mathbf{s}, \mathbf{a}) \leftarrow Q(\mathbf{s}, \mathbf{a}) + \alpha \left[ r + \gamma' \max_{\mathbf{a}'} Q(\mathbf{s}', \mathbf{a}') - Q(\mathbf{s}, \mathbf{a}) \right]$$

where $\alpha$ is the learning rate and $\gamma'$ is the discount factor. After convergence, the market maker selects quotes that maximize expected cumulative reward.

### Logistic Regression for Adverse Selection Detection

A simpler ML component detects when adverse selection is elevated. Given features $\mathbf{x}_t = [\text{VPIN}_t, \text{OFI}_t, \text{spread}_t, \text{volume}_t, \text{imbalance}_t]$, a logistic classifier predicts the probability of an imminent adverse move:

$$P(\text{adverse} | \mathbf{x}_t) = \sigma(\mathbf{w}^T \mathbf{x}_t + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_t + b)}}$$

When the predicted adverse probability exceeds a threshold, the market maker widens spreads or temporarily withdraws from the market.

### Neural Network Spread Predictor

A feed-forward neural network can learn the mapping from market state to optimal spread width. The architecture processes the feature vector through hidden layers:

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\hat{s} = \text{softplus}(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3)$$

The softplus activation ensures a positive spread output. The network is trained on historical data where the target spread is derived from the ex-post optimal spread that would have maximized risk-adjusted PnL.

## Feature Engineering

### Order Book Features

Order book features capture the current state of supply and demand:

- **Bid-ask imbalance**: $\text{Imbalance} = \frac{V^b - V^a}{V^b + V^a}$, ranging from -1 to +1
- **Depth ratio**: Total volume within $k$ levels on each side, normalized
- **Weighted mid-price**: $P_w = \frac{V^a \cdot P^b + V^b \cdot P^a}{V^b + V^a}$, which better reflects the true fair value when imbalance is present
- **Queue position pressure**: Rate of change of volume at the best levels

### Trade Flow Features

Trade flow features summarize recent trading activity:

- **Order flow imbalance (OFI)**: Net change in bid vs ask volume across book updates
- **Cumulative delta**: Running sum of buyer-initiated minus seller-initiated trade volume
- **Trade arrival rate**: Number of trades per unit time, capturing urgency
- **Average trade size**: Mean volume per trade, distinguishing retail from institutional flow

### Volatility and Spread Features

- **Realized volatility**: Rolling standard deviation of mid-price returns
- **Garman-Klass volatility**: More efficient OHLC-based estimator
- **Current spread**: The prevailing bid-ask spread as a fraction of mid-price
- **Spread z-score**: Current spread relative to its rolling distribution

## Applications

### Cryptocurrency Market Making

Cryptocurrency markets are particularly suited for ML-based market making due to their 24/7 operation, high volatility, and fragmented liquidity across exchanges. On platforms like Bybit, a market maker can:

1. Continuously quote BTCUSDT perpetual contracts, earning the bid-ask spread while managing inventory through the funding rate mechanism.
2. Use order book imbalance and VPIN signals to detect informed flow from whales or arbitrageurs and widen spreads accordingly.
3. Adjust position limits based on predicted volatility regimes, reducing exposure before high-impact events.

### Equity Market Making

In equity markets, designated market makers (DMMs) have formal obligations to provide continuous quotes. ML enhances their performance by predicting short-term price movements, optimizing queue position, and managing inventory across correlated securities. Cross-asset signals, such as ETF creation/redemption flow and options market activity, provide additional features.

### Multi-Asset Market Making

Advanced market making strategies operate across multiple correlated instruments simultaneously. A market maker in index futures, for example, must consider the co-movement between the future, its constituent stocks, and related ETFs. ML models can learn these cross-asset relationships and generate hedging signals:

$$q_{\text{hedge}} = -\sum_{i=1}^{N} \beta_i \cdot q_i$$

where $q_i$ is the inventory in asset $i$ and $\beta_i$ is the hedge ratio learned from historical co-movement data.

## Rust Implementation

Our Rust implementation provides a complete ML-driven market making toolkit with the following components:

### AvellanedaStoikov

The `AvellanedaStoikov` struct implements the classical optimal quoting model. It computes bid and ask offsets from the mid-price based on inventory level, volatility, risk aversion, and time horizon. The model's key insight — skewing quotes to manage inventory — forms the baseline against which ML improvements are measured.

### InventoryManager

The `InventoryManager` struct tracks the market maker's current position, computes inventory risk penalties, and enforces position limits. It supports both symmetric and asymmetric inventory penalties, and provides methods for computing the optimal inventory-neutral hedge ratio.

### AdverseSelectionDetector

The `AdverseSelectionDetector` implements logistic regression for real-time adverse selection detection. It ingests features derived from VPIN, OFI, and spread dynamics, and outputs a probability that the current market conditions are toxic. The market maker uses this signal to adjust spread width dynamically.

### MarketMakingRL

The `MarketMakingRL` struct implements a tabular Q-learning agent for optimal quote placement. The state space is discretized into inventory bins, volatility bins, and flow imbalance bins. The agent learns the optimal bid/ask offsets that maximize risk-adjusted PnL over a simulated trading session.

### BybitClient

The `BybitClient` struct provides async HTTP access to the Bybit V5 API. It fetches kline (candlestick) data from the `/v5/market/kline` endpoint and order book snapshots from the `/v5/market/orderbook` endpoint. The client handles response parsing, error handling, and provides the real-time data feed for the market making system.

## Bybit API Integration

The implementation connects to Bybit's V5 REST API to obtain real-time market data:

- **Kline endpoint** (`/v5/market/kline`): Provides OHLCV candlestick data at configurable intervals. Used for volatility estimation and historical backtesting of market making strategies.
- **Order book endpoint** (`/v5/market/orderbook`): Provides a snapshot of the current limit order book with configurable depth. Used for computing bid-ask imbalance, OFI, and calibrating fill probability models.

The Bybit API is well-suited for market making research because it provides:
- Fine-grained intervals (1-minute klines for high-frequency analysis)
- Deep order book snapshots (up to 200 levels)
- Consistent, low-latency responses suitable for real-time quoting systems

## References

1. Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.
2. Gueant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. *Mathematics and Financial Economics*, 7(4), 477-507.
3. Spooner, T., Fearnley, J., Savani, R., & Koukorinis, A. (2018). Market making via reinforcement learning. *Proceedings of AAMAS 2018*.
4. Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *The Review of Financial Studies*, 25(5), 1457-1493.
5. Cartea, A., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
6. Gasperov, B., & Kostanjcar, Z. (2021). Market making with signals through deep reinforcement learning. *IEEE Access*, 9, 61611-61622.
