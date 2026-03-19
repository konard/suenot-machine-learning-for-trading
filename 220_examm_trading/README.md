# Chapter 220: EXAMM Trading — Evolutionary Exploration of Augmenting Memory Models for Financial Markets

## 1. Introduction

The search for optimal neural network architectures has long been a central challenge in machine learning. In the domain of financial time series prediction, this challenge is amplified by the non-stationary, noisy, and adversarial nature of market data. Hand-designed recurrent neural network (RNN) architectures such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) have become standard tools, but they represent only a tiny fraction of the vast space of possible recurrent cell designs. What if we could systematically explore this space and discover architectures specifically suited to financial data?

EXAMM — Evolutionary eXploration of Augmenting Memory Models — is a neuroevolution algorithm that evolves both the topology and the weights of recurrent neural networks. Unlike traditional neural architecture search (NAS) methods that select from a predefined set of building blocks, EXAMM can construct entirely novel recurrent memory cell structures by evolving the internal wiring of recurrent units. This makes it uniquely powerful for trading applications, where the ideal memory dynamics may differ substantially from those captured by standard LSTM or GRU cells.

EXAMM was originally developed by Travis Desell and collaborators at Rochester Institute of Technology for time series prediction tasks in aviation and energy. In this chapter, we adapt and apply the EXAMM framework to financial markets, evolving custom recurrent architectures optimized for price prediction and trading signal generation using Bybit cryptocurrency data.

The core insight behind EXAMM is that the right memory structure for a given prediction task is not something we should assume a priori. Instead, we should let evolution discover it. For financial data — where regime changes, varying volatility, and complex multi-scale temporal dependencies are the norm — this approach offers a principled way to move beyond one-size-fits-all architectures.

## 2. Mathematical Foundation

### 2.1 Neuroevolution of Recurrent Architectures

Neuroevolution refers to the application of evolutionary algorithms to the design and optimization of neural networks. In the context of recurrent networks, this involves evolving both the topology (which neurons connect to which, and through what type of gates or operations) and the parameters (weights and biases) of the network.

A recurrent neural network processes a sequence of inputs $\{x_1, x_2, \ldots, x_T\}$ and maintains a hidden state $h_t$ that evolves over time:

$$h_t = f(h_{t-1}, x_t; \theta)$$

where $f$ is the transition function parameterized by $\theta$. In standard architectures, $f$ is fixed (e.g., the LSTM equations). In EXAMM, $f$ itself is subject to evolution.

### 2.2 Augmenting Topologies

EXAMM builds on the NEAT (NeuroEvolution of Augmenting Topologies) paradigm, extending it to recurrent networks. The key idea is to start with minimal network topologies and gradually increase complexity through mutations that add nodes and connections. This complexification principle ensures that networks are no more complex than necessary for the task at hand.

Each genome in EXAMM encodes a directed graph where:
- **Input nodes** receive external features (e.g., price, volume, technical indicators)
- **Hidden nodes** perform computations using a specific recurrent cell type
- **Output nodes** produce predictions (e.g., next price, trading signal)
- **Edges** carry weighted connections, potentially with recurrent (time-delayed) links

An edge from node $i$ to node $j$ with time delay $d$ means that the output of node $i$ at time $t - d$ feeds into node $j$ at time $t$. When $d = 0$, the connection is feed-forward; when $d \geq 1$, it is recurrent.

### 2.3 Memory Cell Evolution

A distinguishing feature of EXAMM is its ability to evolve the internal structure of memory cells. The algorithm works with several known cell types and can also create novel hybrids:

**Simple RNN:**
$$h_t = \tanh(W_h x_t + U_h h_{t-1} + b_h)$$

**LSTM (Long Short-Term Memory):**
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

**GRU (Gated Recurrent Unit):**
$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$
$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$
$$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**MGU (Minimal Gated Unit):**
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
$$\tilde{h}_t = \tanh(W_h x_t + U_h (f_t \odot h_{t-1}) + b_h)$$
$$h_t = (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t$$

**UGRNN (Update Gate RNN):**
$$\tilde{h}_t = \tanh(W_h x_t + U_h h_{t-1} + b_h)$$
$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Delta-RNN:**
$$d_t = \alpha \odot \tanh(W_d x_t) \odot \tanh(U_d h_{t-1})$$
$$\tilde{h}_t = \tanh(W_h x_t + d_t + b_h)$$
$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

EXAMM can mutate a node from one cell type to another, effectively exploring the space of gating mechanisms. It can also construct hybrid cells by evolving the internal wiring within a node, creating gate structures that do not correspond to any named architecture.

## 3. The EXAMM Algorithm

### 3.1 Island-Based Evolutionary Strategy

EXAMM uses an island model for population management. The population is divided into $K$ islands, each maintaining its own sub-population of genomes. This structure promotes diversity through geographic isolation while allowing periodic migration of successful genomes between islands.

The algorithm proceeds as follows:

1. **Initialization**: Each island is seeded with minimal networks. Different islands may be initialized with different cell types (e.g., Island 1 starts with LSTM cells, Island 2 with GRU cells), encouraging exploration of different regions of architecture space.

2. **Selection**: Within each island, parents are selected using tournament selection. A tournament of size $s$ randomly samples $s$ individuals and selects the one with the best fitness.

3. **Reproduction**: Offspring are created through mutation (more common) or crossover (less common for topological evolution). The offspring replaces the worst individual in the island.

4. **Migration**: Every $M$ generations, the best genome from each island is copied to a random other island, replacing the worst individual there. This spreads successful innovations across the population.

5. **Termination**: Evolution continues for a fixed number of generations or until a fitness plateau is detected.

### 3.2 Speciation

To prevent premature convergence, EXAMM implements speciation within each island. Genomes are grouped into species based on topological similarity. The compatibility distance $\delta$ between two genomes $a$ and $b$ is:

$$\delta(a, b) = c_1 \cdot \frac{E}{N} + c_2 \cdot \frac{D}{N} + c_3 \cdot \bar{W}$$

where $E$ is the number of excess genes, $D$ is the number of disjoint genes, $N$ is the size of the larger genome, $\bar{W}$ is the average weight difference of matching genes, and $c_1, c_2, c_3$ are coefficients controlling the relative importance of each factor.

Fitness sharing within species prevents any single species from dominating the population, maintaining diversity in architectural exploration.

### 3.3 Mutation Operators

EXAMM employs several mutation operators:

- **Add Node**: Splits an existing edge by inserting a new hidden node with a randomly chosen cell type. The original edge is disabled, and two new edges are created connecting through the new node.
- **Add Edge**: Creates a new connection between two previously unconnected nodes. The edge may be feed-forward ($d = 0$) or recurrent ($d \geq 1$).
- **Change Cell Type**: Replaces the cell type of a hidden node (e.g., from GRU to LSTM), re-initializing the cell-specific parameters.
- **Modify Weights**: Applies Gaussian perturbation to the weights of one or more edges: $w' = w + \mathcal{N}(0, \sigma^2)$.
- **Enable/Disable Edge**: Toggles whether an edge is active, allowing the algorithm to explore sub-networks.

### 3.4 Crossover

Crossover between two genomes aligns them by historical innovation markers (similar to NEAT). Matching genes are inherited randomly from either parent; disjoint and excess genes are inherited from the more fit parent. This ensures that offspring inherit compatible structural features.

## 4. Trading Applications

### 4.1 Evolving Custom Recurrent Cells

The primary advantage of EXAMM for trading is the ability to evolve recurrent cell structures that are specifically optimized for financial time series characteristics:

- **Regime-Adaptive Memory**: Financial markets exhibit regime changes (trending vs. mean-reverting, low vs. high volatility). Evolved architectures may develop memory dynamics that naturally adapt to regime shifts, something standard LSTM cells are not explicitly designed for.
- **Multi-Scale Temporal Dependencies**: Price data contains patterns at multiple time scales (tick, minute, hourly, daily). EXAMM can evolve recurrent connections with different time delays, effectively creating multi-scale memory without hand-engineering.
- **Noise Robustness**: Market data is notoriously noisy. Evolution naturally selects architectures that are robust to noise, as fragile architectures will have poor fitness on out-of-sample data.

### 4.2 Discovering Novel Memory Patterns

Through evolution, EXAMM can discover memory cell designs that incorporate unconventional gating patterns. For instance, a cell might evolve a "volatility gate" that modulates information flow based on the variance of recent inputs, or a "momentum gate" that amplifies signals during trending periods.

These novel architectures are not constrained by human intuition about what a recurrent cell "should" look like. The only criterion is prediction performance, which means evolution can exploit subtle statistical regularities in financial data that a human designer might not anticipate.

### 4.3 Feature Processing

For trading, the input features typically include:
- Raw OHLCV (Open, High, Low, Close, Volume) data
- Log returns: $r_t = \log(P_t / P_{t-1})$
- Normalized volume: $v_t = V_t / \text{MA}(V, n)$
- Simple technical indicators serving as auxiliary signals

EXAMM evolves how these features are combined and processed through the recurrent network, discovering which temporal relationships are most predictive.

## 5. EXAMM vs Standard RNN/LSTM

### 5.1 When Evolved Architectures Outperform

Empirical results across multiple domains suggest that EXAMM-evolved architectures outperform standard ones in several scenarios:

1. **Complex Temporal Dependencies**: When the relevant time dependencies are not well-captured by the fixed forget-gate mechanism of LSTM, evolved architectures that develop custom memory dynamics tend to perform better.

2. **Limited Training Data**: Evolved architectures are often more compact than standard LSTM/GRU networks, requiring fewer parameters. This is advantageous when training data is limited, which is common in financial applications with short history or rare events.

3. **Non-Stationary Data**: The diversity of architectures explored by EXAMM increases the chance of finding one that generalizes across distribution shifts, a critical requirement for financial data.

4. **Computational Constraints**: EXAMM often discovers smaller, more efficient architectures that achieve comparable performance to larger standard networks, enabling faster inference for real-time trading.

### 5.2 When Standard Architectures Suffice

Standard LSTM/GRU may be preferred when:
- Abundant training data is available and the task is well-understood
- The computational cost of the evolutionary search is prohibitive
- Interpretability of the architecture is a hard requirement
- The temporal dynamics are simple enough that standard gates suffice

### 5.3 Practical Considerations

The main cost of EXAMM is the evolutionary search itself, which requires evaluating many candidate architectures. However, once the best architecture is found, it can be retrained and deployed like any standard RNN. The search is a one-time cost that pays dividends through improved prediction performance.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete EXAMM framework for trading. The key components are:

### 6.1 Genome Representation

Each genome contains a list of nodes (with cell types) and edges (with weights and time delays). The genome supports mutation and crossover operations that modify the topology while maintaining structural validity.

### 6.2 Cell Type Abstraction

We implement SimpleRNN, LSTM, GRU, and MGU cells, each conforming to a common interface. The forward pass for each cell type takes the current input and previous hidden state, producing the new hidden state. Cells are interchangeable within the genome, enabling the "change cell type" mutation.

### 6.3 Island-Based Evolution

The population is divided into islands, each running semi-independent evolution. Tournament selection picks parents, mutations create offspring, and periodic migration shares the best solutions across islands.

### 6.4 Fitness Evaluation

Fitness is evaluated as the negative mean squared error (MSE) on a validation set of time series predictions. The genome is decoded into a runnable network, the network processes the input sequence, and the predictions are compared against actual values. Lower MSE corresponds to higher fitness.

### 6.5 Bybit Data Integration

We fetch historical BTCUSDT kline data from the Bybit API, extract OHLCV features, compute log returns, and normalize the data for use as training and validation sets.

See the `rust/` directory for the complete implementation with comments explaining each component.

## 7. Bybit Data Integration

The implementation connects to the Bybit V5 public API to fetch historical kline (candlestick) data:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

The response provides OHLCV data which is preprocessed as follows:

1. **Parsing**: Extract timestamp, open, high, low, close, and volume fields from the JSON response.
2. **Log Returns**: Compute $r_t = \log(C_t / C_{t-1})$ where $C_t$ is the close price at time $t$.
3. **Normalization**: Z-score normalize each feature using a rolling window: $\hat{x}_t = (x_t - \mu) / \sigma$.
4. **Windowing**: Create overlapping windows of length $W$ for sequence-to-one prediction.
5. **Train/Validation Split**: Use the first 70% for training and the remaining 30% for validation during fitness evaluation.

This pipeline ensures that the evolutionary search optimizes architectures on realistic, properly preprocessed financial data.

## 8. Key Takeaways

1. **EXAMM evolves both topology and weights** of recurrent neural networks, exploring a much larger architecture space than manual design or simple hyperparameter tuning.

2. **Island-based evolution with speciation** maintains diversity in the population, preventing premature convergence to suboptimal architectures.

3. **Memory cell evolution** is the distinguishing feature: EXAMM can discover novel recurrent cell structures beyond LSTM, GRU, and other standard designs.

4. **For trading applications**, evolved architectures can capture complex, multi-scale temporal dependencies and adapt to regime changes in ways that standard cells may not.

5. **The computational cost** of evolutionary search is a one-time investment. Once the optimal architecture is discovered, it can be deployed efficiently for real-time trading.

6. **Smaller evolved architectures** often outperform larger standard ones, offering both accuracy and speed advantages — critical for low-latency trading systems.

7. **Combining EXAMM with proper data preprocessing** (log returns, normalization, windowing) and Bybit market data creates a complete pipeline for discovering and deploying evolved trading models.

8. **No free lunch**: EXAMM requires careful configuration of population size, mutation rates, number of islands, and migration frequency. These meta-parameters themselves may require tuning for optimal results on financial data.
