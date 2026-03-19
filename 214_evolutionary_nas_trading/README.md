# Chapter 214: Evolutionary NAS Trading

## 1. Introduction

Neural Architecture Search (NAS) has transformed deep learning by automating the design of neural network architectures. Traditionally, practitioners hand-craft network topologies through trial and error, relying on intuition and domain expertise. In quantitative trading, where the search space of potential models is vast and the relationship between architecture choice and profitability is non-obvious, automated architecture discovery becomes especially valuable.

Evolutionary algorithms (EAs) offer a compelling paradigm for NAS. Inspired by biological evolution, these algorithms maintain a population of candidate architectures that undergo selection, crossover, and mutation over successive generations. The fittest individuals — those whose architectures yield the best trading performance — survive and reproduce, while weaker candidates are culled. Over many generations, this process discovers architectures that are well-adapted to the specific characteristics of financial data.

The appeal of evolutionary NAS for trading is multifold. First, financial markets exhibit non-stationary dynamics; an architecture that excels in one regime may falter in another. Evolutionary methods naturally explore diverse regions of the architecture space, producing a population of solutions rather than a single point estimate. Second, trading objectives are inherently multi-dimensional: we care about prediction accuracy, but also about inference latency (for real-time deployment), model robustness (to avoid overfitting), and interpretability. Evolutionary multi-objective optimization handles these competing objectives gracefully through Pareto-based selection.

This chapter develops a complete evolutionary NAS framework for trading in Rust. We encode neural architectures as chromosomes, evolve them using tournament selection and genetic operators, incorporate age-based regularization to maintain population diversity, and apply NSGA-II-inspired Pareto ranking for multi-objective fitness. The system integrates with Bybit exchange data for realistic evaluation.

## 2. Mathematical Foundation

### 2.1 Genetic Algorithms

A genetic algorithm (GA) operates on a population $P = \{x_1, x_2, \ldots, x_N\}$ of $N$ individuals, where each individual $x_i$ encodes a candidate solution. The algorithm iterates through the following cycle:

1. **Evaluation**: Compute the fitness $f(x_i)$ for each individual.
2. **Selection**: Choose parents from the population based on fitness.
3. **Crossover**: Combine pairs of parents to produce offspring.
4. **Mutation**: Apply random perturbations to offspring.
5. **Replacement**: Form the next generation from offspring and (optionally) surviving parents.

The process repeats for $G$ generations or until a convergence criterion is met.

### 2.2 Tournament Selection

Tournament selection is a widely used selection mechanism due to its simplicity and controllable selection pressure. To select one parent, we sample $k$ individuals uniformly at random from the population and choose the one with the highest fitness:

$$\text{selected} = \arg\max_{x \in S_k} f(x), \quad S_k \subset P, \quad |S_k| = k$$

The tournament size $k$ controls selection pressure. With $k = 1$, selection is purely random; as $k$ increases, selection becomes more elitist. A typical choice is $k \in [2, 7]$.

### 2.3 Crossover Operators

Crossover combines genetic material from two parents $x_a$ and $x_b$ to produce offspring. We consider two primary operators:

**Single-point crossover**: Choose a random crossover point $c \in [1, L-1]$ where $L$ is the chromosome length. The offspring inherits genes $1, \ldots, c$ from parent $a$ and genes $c+1, \ldots, L$ from parent $b$:

$$\text{offspring}[i] = \begin{cases} x_a[i] & \text{if } i \leq c \\ x_b[i] & \text{if } i > c \end{cases}$$

**Uniform crossover**: Each gene is independently inherited from either parent with equal probability:

$$\text{offspring}[i] = \begin{cases} x_a[i] & \text{with probability } 0.5 \\ x_b[i] & \text{with probability } 0.5 \end{cases}$$

### 2.4 Mutation

Mutation introduces random changes to maintain genetic diversity. In the context of NAS, mutation operators include:

- **Add layer**: Insert a new layer at a random position with random parameters.
- **Remove layer**: Delete a randomly chosen layer (if the network has more than the minimum).
- **Change layer size**: Modify the number of neurons in a randomly selected layer.
- **Change activation**: Replace the activation function of a random layer.

Each mutation is applied with a probability $p_m$ (typically $0.1$ to $0.3$).

### 2.5 Fitness Evaluation

For trading, fitness $f(x)$ is computed by:

1. Decoding the genome $x$ into a neural network architecture.
2. Training the network on a historical price dataset.
3. Evaluating trading performance on a validation set.

The fitness function can be the Sharpe ratio, total return, or a composite metric. We use a simplified simulation where predicted signals are compared against actual price movements.

### 2.6 Population Diversity

Maintaining diversity is critical to avoid premature convergence. We measure diversity as the average pairwise Hamming distance between genomes:

$$D(P) = \frac{2}{N(N-1)} \sum_{i < j} d_H(x_i, x_j)$$

If diversity falls below a threshold, we increase mutation rates or inject random individuals.

## 3. Neuroevolution

### 3.1 Evolving Topology and Weights

Neuroevolution extends genetic algorithms to evolve both the topology and the weights of neural networks. The NEAT (NeuroEvolution of Augmenting Topologies) framework introduced several key ideas:

- **Complexification**: Start with minimal networks and gradually add complexity. This biases the search toward simpler solutions and avoids the "competing conventions" problem where functionally identical networks have different encodings.
- **Innovation numbers**: Track the history of structural mutations so that crossover can align genes from different parents meaningfully.
- **Speciation**: Group similar individuals into species and protect novel structures from being eliminated before they have a chance to optimize.

In our trading NAS framework, we adopt a simplified form of complexification. Initial genomes encode small networks (1-3 layers), and mutation can add layers over time. This encourages the evolutionary process to first find good shallow architectures and then refine them by adding depth when beneficial.

### 3.2 Architecture Encoding

Each genome is a variable-length list of layer descriptors:

```
Genome = [Layer_1, Layer_2, ..., Layer_n]
Layer_i = (size: u32, activation: Activation)
Activation in {ReLU, Tanh, Sigmoid, LeakyReLU, Swish, Linear}
```

The input dimension is determined by the data, and the output dimension is fixed (e.g., 1 for regression, 3 for classification into buy/hold/sell). The genome encodes only the hidden layers.

## 4. Trading Applications

### 4.1 Evolving Architectures for Alpha Discovery

Alpha discovery — finding predictive signals in market data — is fundamentally a search problem. The space of possible feature transformations and model architectures is enormous. Evolutionary NAS frames this as an optimization problem over architecture space.

Each candidate architecture implicitly defines a feature extraction and prediction pipeline. By evolving architectures on rolling windows of market data, we can discover models that adapt to changing market regimes. The evolutionary process can uncover non-obvious architectural patterns: perhaps a particular combination of layer sizes and activations captures mean-reversion dynamics, while another topology excels at momentum detection.

### 4.2 Multi-Objective Evolution

Trading systems must balance multiple objectives simultaneously:

- **Prediction accuracy**: Measured by directional accuracy or correlation with future returns.
- **Inference latency**: Smaller, simpler models execute faster — critical for high-frequency strategies.
- **Robustness**: The architecture should generalize across different market conditions, not just the training period.

We employ NSGA-II-inspired Pareto ranking. An individual $x_a$ dominates $x_b$ (written $x_a \succ x_b$) if $x_a$ is no worse than $x_b$ on all objectives and strictly better on at least one:

$$x_a \succ x_b \iff \forall i: f_i(x_a) \geq f_i(x_b) \land \exists j: f_j(x_a) > f_j(x_b)$$

The population is sorted into non-dominated fronts $F_1, F_2, \ldots$ where $F_1$ contains individuals not dominated by any other, $F_2$ contains individuals dominated only by those in $F_1$, and so on. Selection prefers individuals from lower-ranked fronts.

Within the same front, we use crowding distance to promote diversity. The crowding distance of an individual measures how isolated it is in objective space — individuals in less crowded regions are preferred to maintain a well-spread Pareto front.

## 5. Aging Evolution

Regularized evolution, introduced by Real et al. (2019), adds an age mechanism to the evolutionary process. Each individual has an age that increments with each generation. Instead of replacing the least fit individual, we remove the oldest individual from the population regardless of its fitness.

The algorithm proceeds as follows:

1. Initialize a population of $N$ random architectures.
2. Evaluate the fitness of all individuals.
3. Repeat for $G$ generations:
   a. Select a parent via tournament selection.
   b. Create an offspring by mutating the parent (no crossover in the simplest variant).
   c. Evaluate the offspring's fitness.
   d. Add the offspring to the population.
   e. Remove the oldest individual from the population.

Age-based removal prevents any single architecture from dominating the population indefinitely, even if it has high fitness. This maintains exploration pressure and is particularly valuable in non-stationary environments like financial markets, where yesterday's optimal architecture may not be tomorrow's.

The simplicity of aging evolution is a practical advantage. It requires fewer hyperparameters than traditional GAs and has been shown to match or exceed the performance of more complex NAS methods including reinforcement learning-based approaches.

## 6. Implementation Walkthrough

Our Rust implementation is organized into several core components:

### 6.1 Genome Encoding

The `Genome` struct contains a vector of `LayerGene` entries, each specifying a layer size and activation function. The genome also tracks its age (generation of creation) and cached fitness scores.

```rust
pub struct LayerGene {
    pub size: usize,
    pub activation: Activation,
}

pub struct Genome {
    pub layers: Vec<LayerGene>,
    pub fitness: Option<f64>,
    pub birth_generation: usize,
    // Multi-objective fitness components
    pub accuracy_score: f64,
    pub latency_score: f64,
    pub robustness_score: f64,
    pub pareto_rank: usize,
    pub crowding_distance: f64,
}
```

### 6.2 Genetic Operators

The `mutate` method applies one of four mutations at random: adding a layer, removing a layer, changing a layer's size, or changing an activation function. The `crossover` function supports both single-point and uniform strategies.

### 6.3 Evolution Engine

The `EvolutionEngine` orchestrates the evolutionary loop. It maintains the population, runs tournament selection, applies genetic operators, evaluates fitness, and tracks statistics per generation. The engine supports both standard generational replacement and aging evolution.

### 6.4 Fitness Evaluation

Fitness is computed by simulating a simple trading strategy: the genome defines a network architecture, we do a forward pass with random weights (as a proxy — in production you would train weights), and measure directional accuracy against actual price movements. Multi-objective fitness combines accuracy, latency (inversely proportional to network size), and robustness (consistency across multiple evaluation windows).

### 6.5 Bybit Integration

The `BybitClient` fetches OHLCV kline data from the Bybit V5 API. This data is preprocessed into feature vectors (returns, volatility, moving average ratios) for architecture evaluation.

```rust
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl BybitClient {
    pub fn fetch_klines(&self, symbol: &str, interval: &str, limit: usize)
        -> Result<Vec<Kline>>;
}
```

## 7. Bybit Data Integration

The implementation connects to Bybit's public API to fetch real market data. The data pipeline works as follows:

1. **Fetch**: Request OHLCV candles for a given symbol (e.g., BTCUSDT) and timeframe.
2. **Preprocess**: Compute features including log returns, rolling volatility, RSI-like momentum indicators, and moving average crossover signals.
3. **Split**: Divide into training, validation, and test sets using temporal ordering (no look-ahead bias).
4. **Evaluate**: Each genome in the population is evaluated on the training/validation split. The test set is reserved for final evaluation of the best architecture.

The Bybit V5 API endpoint used is:
```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

Data is parsed into `Kline` structs and converted to feature matrices for the evolutionary fitness function.

## 8. Key Takeaways

1. **Evolutionary NAS automates architecture design** for trading models, removing the need for manual trial-and-error network design and enabling exploration of architectures that humans might not consider.

2. **Genetic algorithms provide a flexible optimization framework** with well-understood operators (selection, crossover, mutation) that can be tailored to the architecture search space.

3. **Multi-objective evolution via Pareto ranking** naturally handles the competing demands of trading systems: accuracy, speed, and robustness can be optimized simultaneously without reducing them to a single scalar.

4. **Aging evolution offers simplicity and effectiveness**. By removing the oldest individuals rather than the least fit, it maintains exploration pressure and adapts well to non-stationary financial data.

5. **Complexification through neuroevolution** biases the search toward simpler architectures initially, adding complexity only when it improves fitness — an implicit form of regularization that helps prevent overfitting.

6. **Population diversity is essential** for avoiding premature convergence. Tournament selection with moderate pressure, combined with mutation and age-based replacement, maintains a healthy diversity of candidate architectures.

7. **Rust provides performance advantages** for evolutionary NAS, where the inner loop involves evaluating many architectures per generation. Low-level control over memory and computation enables efficient population management and fitness evaluation.

8. **Real market data integration** via Bybit ensures that evolved architectures are tested against actual market dynamics rather than synthetic data, improving the practical relevance of discovered architectures.
