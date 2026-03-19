# Chapter 211: Neural Architecture Search (NAS) for Trading

## 1. Introduction

Designing neural network architectures for financial trading has traditionally been an art as much as a science. Practitioners spend weeks or months manually experimenting with different layer configurations, activation functions, and connectivity patterns to find models that capture the subtle statistical regularities in market data. Neural Architecture Search (NAS) offers a principled, automated alternative: instead of relying on human intuition and trial-and-error, NAS algorithms systematically explore the space of possible architectures to discover designs that optimize trading-relevant objectives such as prediction accuracy, Sharpe ratio, or risk-adjusted return.

The core idea behind NAS is deceptively simple. We define a **search space** of candidate architectures, a **search strategy** for navigating that space, and a **performance estimation** method for evaluating each candidate. The search algorithm proposes architectures, evaluates them on trading data, and iteratively refines its proposals based on observed performance. Over many iterations, the process converges on architectures that are specifically tailored to the structure of financial time series.

What makes NAS particularly compelling for trading is that financial data has unique characteristics -- non-stationarity, heavy tails, regime changes, low signal-to-noise ratios -- that may favor architectural patterns quite different from those discovered on standard benchmarks like image classification. A NAS system can discover these domain-specific patterns automatically, potentially finding architectures that a human designer would never consider.

In this chapter, we develop a complete NAS framework in Rust, designed specifically for trading applications. We implement evolutionary search over a flexible architecture space that includes dense layers, convolutional layers, recurrent layers, and attention mechanisms. We integrate real market data from the Bybit exchange and demonstrate how to discover architectures optimized for price prediction.

## 2. Mathematical Foundation

### 2.1 Search Space Definition

The search space $\mathcal{A}$ defines the set of all architectures that the NAS algorithm can consider. We represent each architecture as a directed acyclic graph (DAG) where nodes are computational operations and edges represent data flow. Formally, an architecture $a \in \mathcal{A}$ is encoded as a sequence of genes:

$$a = (g_1, g_2, \ldots, g_L)$$

where $L$ is the maximum number of layers and each gene $g_i$ specifies:

- **Layer type** $t_i \in \{$ Dense, Conv1D, LSTM, Attention, Skip, Identity $\}$
- **Hidden dimension** $h_i \in \{16, 32, 64, 128, 256\}$
- **Activation function** $\sigma_i \in \{$ ReLU, Tanh, Sigmoid, GELU, LeakyReLU $\}$
- **Skip connection target** $s_i \in \{0, 1, \ldots, i-1\}$ (connect to a previous layer or none)
- **Dropout rate** $d_i \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$

The total size of the search space is:

$$|\mathcal{A}| = \prod_{i=1}^{L} |T| \cdot |H| \cdot |\Sigma| \cdot i \cdot |D|$$

For $L=6$ layers, this yields over $10^{10}$ possible architectures, making exhaustive search infeasible.

### 2.2 Search Strategies

**Reinforcement Learning (RL)-based Search.** A controller network (typically an RNN) generates architecture descriptions as sequences of tokens. The controller is trained with REINFORCE, using the validation performance of each generated architecture as the reward signal:

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a) \cdot (R(a) - b)]$$

where $\pi_\theta$ is the controller policy, $R(a)$ is the reward (e.g., validation Sharpe ratio), and $b$ is a baseline for variance reduction.

**Evolutionary Search.** A population of architectures is maintained and evolved through mutation and crossover operators. At each generation:

1. **Tournament selection**: select $k$ individuals, keep the best as parent
2. **Mutation**: randomly modify one or more genes of the parent
3. **Crossover**: combine genes from two parents to produce offspring
4. **Evaluation**: train and evaluate the offspring on validation data
5. **Replacement**: insert offspring into the population, removing the weakest

The fitness function $f(a)$ can incorporate multiple objectives:

$$f(a) = \alpha \cdot \text{Accuracy}(a) - \beta \cdot \text{ModelSize}(a) - \gamma \cdot \text{Latency}(a)$$

**Gradient-based Search (DARTS).** The discrete search space is relaxed to a continuous one by placing a softmax over operation choices:

$$\bar{o}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o)}{\sum_{o'} \exp(\alpha_{o'})} \cdot o(x)$$

Architecture parameters $\alpha$ and network weights $w$ are optimized jointly via gradient descent.

### 2.3 Performance Estimation

Fully training each candidate architecture is expensive. Several strategies reduce cost:

- **Weight sharing**: all architectures share a single set of weights in a supernet
- **Early stopping**: terminate training early for unpromising architectures
- **Proxy tasks**: evaluate on a subset of data or a simplified version of the task
- **Performance predictors**: train a surrogate model to predict architecture performance from its encoding

The predicted performance $\hat{R}(a)$ approximates the true performance $R(a)$:

$$\hat{R}(a) = f_\phi(\text{encode}(a))$$

where $f_\phi$ is a learned predictor (e.g., a small neural network or Gaussian process).

## 3. NAS Search Spaces for Trading

### 3.1 Layer Types

Financial time series require architectures that can capture both local patterns and long-range dependencies:

- **Dense (Fully Connected)**: capture arbitrary nonlinear relationships between features at a given time step
- **Conv1D**: detect local temporal patterns such as candlestick formations, momentum signals, and short-term mean reversion
- **LSTM/GRU**: model sequential dependencies and regime persistence across time steps
- **Multi-Head Attention**: learn which historical time steps are most relevant for current predictions, regardless of temporal distance
- **Identity/Skip**: allow information to bypass layers, enabling both shallow and deep effective architectures within the same search space

### 3.2 Skip Connections

Skip connections are especially important for trading models because:

1. Financial signals are often weak and can be destroyed by too many nonlinear transformations
2. Different features may require different amounts of processing
3. Residual connections improve gradient flow during training, which is critical when the target signal has low signal-to-noise ratio

Our search space allows each layer to optionally connect to any earlier layer, enabling the discovery of complex multi-scale architectures.

### 3.3 Activation Functions

Different activation functions serve different roles:

- **ReLU/LeakyReLU**: efficient for general-purpose feature extraction, with LeakyReLU avoiding dead neuron problems
- **Tanh**: bounded output suitable for layers that feed into recurrent units
- **Sigmoid**: useful for gating mechanisms and probability outputs
- **GELU**: smooth approximation to ReLU, often effective in attention-based architectures

The NAS system discovers which activations work best at each layer of the network for trading data.

## 4. Trading Applications

### 4.1 Price Prediction

For price direction or return prediction, the NAS objective is typically:

$$\max_{a \in \mathcal{A}} \text{Accuracy}(a, \mathcal{D}_{\text{val}}) \quad \text{s.t.} \quad \text{Params}(a) \leq B$$

where $\mathcal{D}_{\text{val}}$ is out-of-sample validation data and $B$ is a parameter budget. The search discovers architectures that balance model capacity with the risk of overfitting to noise in financial data.

### 4.2 Volatility Forecasting

Volatility prediction requires architectures sensitive to the clustering and persistence of variance. NAS can discover models that naturally capture GARCH-like dynamics through appropriate combinations of recurrent and attention layers. The fitness function may use mean squared error on realized volatility or a likelihood-based metric.

### 4.3 Portfolio Optimization

For portfolio construction, the NAS objective becomes:

$$\max_{a \in \mathcal{A}} \text{SharpeRatio}\left(\text{Portfolio}(a, \mathcal{D}_{\text{val}})\right)$$

Architectures that output portfolio weights must satisfy constraints (e.g., weights sum to one, no excessive leverage), which can be enforced through appropriate output layers that are fixed across the search space.

### 4.4 Multi-Objective Search

In practice, trading models must balance multiple objectives: prediction accuracy, computational cost (for real-time execution), model size (for deployment), and robustness (for regime changes). We track the **Pareto front** of non-dominated architectures:

An architecture $a$ dominates $a'$ if $a$ is at least as good on all objectives and strictly better on at least one. The Pareto front contains all non-dominated architectures, giving the trader a menu of optimal trade-offs.

## 5. Efficient NAS

### 5.1 Weight Sharing

Weight sharing dramatically reduces the cost of NAS by training a single **supernet** that contains all possible architectures as sub-networks. Each candidate architecture is evaluated by extracting the corresponding subset of weights from the supernet, avoiding the need to train each architecture from scratch.

The supernet weights $W$ are trained by sampling random architectures at each training step:

$$W^* = \arg\min_W \mathbb{E}_{a \sim \mathcal{U}(\mathcal{A})} [\mathcal{L}(a, W, \mathcal{D}_{\text{train}})]$$

### 5.2 One-Shot Methods

One-shot NAS trains the supernet once and then searches for the best sub-network. This reduces search cost from thousands of GPU-hours to a single training run plus a cheap search phase. The key challenge is ensuring that sub-network performance in the supernet correlates with standalone performance.

### 5.3 Proxy Tasks for Trading

To further reduce search cost, we use proxy tasks:

- **Data subsampling**: search on a random 10-20% subset of the full trading history
- **Reduced training epochs**: train each architecture for 5-10 epochs instead of 50-100
- **Simplified features**: use only price and volume instead of the full feature set during search
- **Shorter lookback windows**: use 30-day windows instead of 90-day windows during search

After the search identifies promising architectures, the top candidates are fully trained and evaluated on the complete dataset.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete NAS framework for trading. The key components are:

### 6.1 Architecture Encoding

Each architecture is encoded as a vector of `LayerGene` structs, where each gene specifies the layer type, hidden size, activation function, skip connection target, and dropout rate. This encoding supports efficient mutation and crossover operations.

```rust
pub struct LayerGene {
    pub layer_type: LayerType,
    pub hidden_size: usize,
    pub activation: Activation,
    pub skip_connection: Option<usize>,
    pub dropout: f64,
}
```

### 6.2 Evolutionary Search

The search proceeds through generations. At each generation:

1. Tournament selection picks parents from the current population
2. Mutation randomly modifies one gene in a parent architecture
3. Each offspring is evaluated by simulating training on historical data
4. The population is updated, keeping the best individuals

The evolutionary approach is particularly well-suited for trading NAS because:
- It naturally supports multi-objective optimization via Pareto ranking
- It maintains population diversity, reducing the risk of premature convergence
- It requires no differentiable relaxation of the search space

### 6.3 Bybit Data Integration

We fetch real OHLCV data from the Bybit API for architecture evaluation. The data is preprocessed into features (returns, moving averages, volatility) and binary labels (price direction), providing a realistic evaluation environment for discovered architectures.

### 6.4 Architecture Evaluation

Each candidate architecture is evaluated through a simplified forward pass simulation. In production, this would involve full backpropagation training, but our implementation demonstrates the NAS framework structure using a proxy evaluation that estimates performance based on architecture properties and a lightweight computation on the data.

## 7. Bybit Data Integration

The integration with the Bybit exchange provides real market data for architecture evaluation. We use the public klines endpoint:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

The returned OHLCV candles are processed into features:
- **Log returns**: $r_t = \ln(P_t / P_{t-1})$
- **Moving averages**: SMA and EMA over various windows
- **Volatility**: rolling standard deviation of returns
- **Volume ratios**: current volume relative to moving average

These features form the input to candidate architectures during the NAS evaluation phase. Using real market data ensures that discovered architectures are adapted to the actual statistical properties of cryptocurrency markets, including their characteristic volatility clustering, fat-tailed returns, and microstructure patterns.

## 8. Key Takeaways

1. **NAS automates architecture design**: instead of manually experimenting with network configurations, NAS systematically searches through the space of possible architectures to find designs optimized for trading tasks.

2. **Trading requires specialized search spaces**: financial time series have unique properties (non-stationarity, low SNR, regime changes) that favor different architectural patterns than those found on standard benchmarks. Include Conv1D for local patterns, LSTM for sequential dependencies, and attention for long-range relationships.

3. **Evolutionary NAS is practical for trading**: evolutionary algorithms naturally support multi-objective optimization, maintain diversity, and scale well to complex search spaces without requiring differentiable relaxations.

4. **Efficiency techniques are essential**: weight sharing, proxy tasks, and early stopping reduce the computational cost of NAS from thousands of GPU-hours to practical levels. For trading, data subsampling and reduced training epochs are effective proxies.

5. **Multi-objective optimization matters**: trading models must balance prediction accuracy against model size, inference latency, and robustness. Pareto front tracking provides a principled way to navigate these trade-offs.

6. **Rust provides performance advantages**: the computational intensity of NAS makes Rust's performance characteristics valuable, enabling faster architecture evaluation and shorter search cycles.

7. **Real data evaluation is critical**: architectures should be evaluated on actual market data (e.g., from Bybit) rather than synthetic data, to ensure they capture the true statistical properties of financial markets.

8. **NAS is a complement, not a replacement**: NAS discovers architecture topology, but feature engineering, data preprocessing, risk management, and execution logic still require domain expertise. NAS is most effective when combined with strong financial domain knowledge.
