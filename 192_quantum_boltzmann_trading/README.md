# Chapter 192: Quantum Boltzmann Trading

## 1. Introduction

Quantum Boltzmann Machines (QBMs) represent one of the most promising intersections of quantum computing and machine learning for financial modeling. While classical Restricted Boltzmann Machines (RBMs) have been used in finance for decades -- from learning latent structure in asset returns to generative modeling of market scenarios -- they suffer from fundamental sampling limitations. Classical Markov Chain Monte Carlo methods can become trapped in local energy minima, especially when the probability landscape of asset returns is multimodal, as it typically is during regime changes, crises, and tail events.

Quantum Boltzmann Machines overcome these limitations by exploiting quantum mechanical phenomena, particularly quantum tunneling and superposition, to explore the energy landscape more efficiently. Instead of relying solely on thermal fluctuations to escape local minima, a QBM uses transverse-field terms in its Hamiltonian that allow the system to tunnel through energy barriers. This makes QBMs especially well-suited for financial applications where capturing the full joint distribution of asset returns -- including heavy tails, skewness, and nonlinear dependencies -- is critical.

In this chapter, we develop a complete QBM framework for trading. We begin with the mathematical foundations, compare QBMs to classical RBMs, and then build a Rust implementation that fetches real cryptocurrency data from the Bybit exchange, trains a QBM on discretized return distributions, and uses generative sampling to produce synthetic portfolio scenarios. These scenarios can be used for risk assessment, portfolio optimization, and anomaly detection.

## 2. Mathematical Foundation

### 2.1 The Boltzmann Distribution

The Boltzmann distribution is the cornerstone of energy-based models. For a system with state configuration **s**, the probability of observing that state is:

```
P(s) = (1/Z) * exp(-E(s) / T)
```

where `E(s)` is the energy of the configuration, `T` is the temperature, and `Z` is the partition function:

```
Z = sum_s exp(-E(s) / T)
```

The partition function normalizes the distribution and encodes all thermodynamic information about the system. In the context of financial modeling, each state **s** represents a particular pattern of discretized asset returns, and the energy function encodes the learned dependencies between assets.

### 2.2 The Classical RBM Energy Function

A Restricted Boltzmann Machine defines an energy function over visible units **v** (observed data, e.g., discretized returns) and hidden units **h** (latent factors):

```
E(v, h) = -sum_i a_i * v_i - sum_j b_j * h_j - sum_{i,j} v_i * W_{ij} * h_j
```

where `a_i` are visible biases, `b_j` are hidden biases, and `W_{ij}` are the weights connecting visible unit `i` to hidden unit `j`. The restricted connectivity (no intra-layer connections) makes inference tractable.

### 2.3 The Quantum Hamiltonian

A Quantum Boltzmann Machine extends this by introducing quantum mechanical terms. The system is described by a Hamiltonian operator rather than a classical energy function:

```
H = H_classical + H_quantum
```

The classical part encodes the same interactions as the RBM:

```
H_classical = -sum_i a_i * sigma_z^i - sum_j b_j * sigma_z^j - sum_{i,j} W_{ij} * sigma_z^i * sigma_z^j
```

where `sigma_z^i` is the Pauli-Z operator acting on qubit `i`.

### 2.4 The Transverse-Field Ising Model

The quantum part introduces transverse-field terms:

```
H_quantum = -Gamma * sum_i sigma_x^i
```

where `Gamma` is the transverse field strength and `sigma_x^i` is the Pauli-X operator. This term does not commute with `H_classical`, meaning the system exists in superpositions of classical states. The transverse field creates quantum tunneling, allowing the system to pass through energy barriers that would trap a purely classical sampler.

### 2.5 Free Energy and the Partition Function

The quantum partition function is computed via a trace over the density matrix:

```
Z = Tr[exp(-beta * H)]
```

where `beta = 1/T` is the inverse temperature. The free energy is:

```
F = -T * ln(Z)
```

Because the quantum Hamiltonian involves non-commuting operators, computing `Z` exactly requires techniques like the Suzuki-Trotter decomposition, which maps the quantum system to a classical system with an extra "imaginary time" dimension. This is the basis for our simulation approach.

### 2.6 Suzuki-Trotter Decomposition

The Suzuki-Trotter decomposition approximates the quantum partition function by splitting the Hamiltonian into commuting and non-commuting parts:

```
exp(-beta * H) ~ [exp(-beta * H_classical / M) * exp(-beta * H_quantum / M)]^M
```

where `M` is the number of Trotter slices. This maps the d-dimensional quantum system to a (d+1)-dimensional classical system, where the extra dimension represents replicas coupled along the imaginary time axis. The inter-replica coupling strength is:

```
J_perp = -(T/2) * ln(tanh(Gamma / (M * T)))
```

This allows us to simulate the quantum system using classical Monte Carlo methods on the extended system.

## 3. QBM vs Classical RBM

### 3.1 Quantum Tunneling Advantage

The key advantage of QBMs over classical RBMs lies in the exploration of the energy landscape. Consider a double-well potential with a tall, thin barrier. A classical thermal sampler must "climb over" the barrier, requiring energy fluctuations proportional to the barrier height. A quantum sampler can tunnel through the barrier, with tunneling probability depending on the barrier width rather than height.

In financial terms, this means QBMs can more effectively capture:
- **Regime transitions**: Rapid switches between market states (bull/bear) that classical models smooth over
- **Tail dependencies**: Nonlinear correlations that emerge during market stress
- **Multimodal distributions**: Multiple equilibrium states in asset return distributions

### 3.2 Quantum Thermal States

The thermal state of a QBM is a mixed quantum state described by the density matrix:

```
rho = exp(-beta * H) / Z
```

Measurements on this state yield samples from a distribution that naturally incorporates quantum correlations. As the transverse field is annealed to zero, the quantum distribution converges to the classical Boltzmann distribution, but the path through quantum states during training can find better parameters.

### 3.3 Expressiveness

A QBM with `n` qubits can represent distributions over `2^n` states, just like a classical RBM. However, the quantum correlations (entanglement) between units during the sampling process allow the QBM to more efficiently represent certain distributions that would require exponentially many hidden units in a classical RBM. This is particularly relevant for financial distributions that exhibit complex dependency structures.

## 4. Trading Application

### 4.1 Learning Joint Probability Distributions of Asset Returns

The primary application of QBMs in trading is learning the joint probability distribution of asset returns. Given a universe of `N` assets, we discretize each asset's return into `K` bins and represent the joint state as a binary vector. The QBM learns the energy function that assigns low energy (high probability) to return patterns that frequently occur in historical data.

This joint distribution captures:
- Marginal distributions of individual asset returns (including skewness and kurtosis)
- Pairwise correlations between assets
- Higher-order dependencies that linear models miss
- Tail dependencies critical for risk management

### 4.2 Generative Sampling for Portfolio Scenarios

Once trained, the QBM serves as a generative model. We can draw samples from the learned distribution to generate synthetic market scenarios. These scenarios are useful for:

1. **Monte Carlo risk estimation**: Generate thousands of correlated return scenarios for VaR and CVaR computation
2. **Stress testing**: Condition on extreme events in one asset and sample the conditional distribution of others
3. **Portfolio optimization**: Use generated scenarios as inputs to robust optimization frameworks
4. **What-if analysis**: Modify the energy landscape to simulate hypothetical market conditions

### 4.3 Anomaly Detection

The energy function itself serves as an anomaly detector. If a new observation has unusually high energy (low probability under the learned distribution), it signals an anomalous market condition. This can be used for:
- Early warning of regime changes
- Detection of unusual correlation breakdowns
- Identification of potential market manipulation patterns

The quantum advantage is particularly strong here: because QBMs explore the energy landscape more thoroughly during training, they build a more accurate model of "normal" behavior, making anomaly detection more sensitive.

## 5. Training

### 5.1 Quantum Contrastive Divergence

Training a QBM follows the same principle as training a classical RBM: maximize the log-likelihood of the training data. The gradient of the log-likelihood with respect to parameter `theta` is:

```
dL/d_theta = <dE/d_theta>_data - <dE/d_theta>_model
```

The first term (positive phase) is the expected gradient under the data distribution, which is straightforward to compute. The second term (negative phase) requires sampling from the model distribution, which is where the quantum advantage manifests.

In Quantum Contrastive Divergence (QCD), we:
1. Initialize the visible units to a training example
2. Perform `k` steps of quantum-enhanced Gibbs sampling (using the Suzuki-Trotter representation)
3. Use the resulting samples to estimate the model expectation

The quantum tunneling in the Suzuki-Trotter representation allows the sampler to mix faster, meaning fewer steps of CD are needed for accurate gradient estimates.

### 5.2 Variational Parameter Updates

The parameter update rules are:

```
W_{ij} += learning_rate * (<v_i * h_j>_data - <v_i * h_j>_model)
a_i += learning_rate * (<v_i>_data - <v_i>_model)
b_j += learning_rate * (<h_j>_data - <h_j>_model)
Gamma += learning_rate * d_quantum_term / d_Gamma
```

The transverse field strength `Gamma` is typically annealed during training, starting large (strong quantum effects) and decreasing toward zero (approaching classical behavior). This quantum annealing schedule helps the model escape poor local optima during early training.

### 5.3 Practical Training Considerations

- **Learning rate scheduling**: Start with a higher learning rate and decay
- **CD-k steps**: k=1 is often sufficient for QBM due to faster mixing
- **Trotter slices**: M=4 to M=16 provides a good accuracy/speed tradeoff
- **Batch size**: Mini-batch training with batch sizes of 32-64
- **Regularization**: L2 weight decay on `W` prevents overfitting
- **Temperature**: Typically set to T=1 and let the energy scale adapt

## 6. Implementation Walkthrough

Our Rust implementation provides a complete QBM simulator suitable for trading applications. Here is an overview of the key components.

### 6.1 Core Data Structures

The `QuantumBoltzmannMachine` struct holds the model parameters:
- `weights`: 2D array of shape `(n_visible, n_hidden)` for the inter-layer connections
- `visible_bias`: 1D array for visible unit biases
- `hidden_bias`: 1D array for hidden unit biases
- `gamma`: the transverse field strength controlling quantum effects
- `n_trotter`: number of Trotter slices for the Suzuki-Trotter decomposition

### 6.2 Energy Computation

The `compute_energy` method computes the classical energy for a given visible-hidden configuration. This is the core function called millions of times during training and sampling.

### 6.3 Quantum-Enhanced Gibbs Sampling

The `quantum_gibbs_sample` method performs sampling using the Suzuki-Trotter representation. It maintains multiple replicas of the system and alternates between:
1. Within-replica updates (standard Gibbs sampling)
2. Between-replica updates (coupling along the Trotter dimension)

### 6.4 Training Loop

The `train` method implements quantum contrastive divergence:
1. For each mini-batch, compute the positive phase statistics
2. Run quantum Gibbs sampling for the negative phase
3. Update weights, biases, and optionally the transverse field
4. Anneal gamma according to the schedule

### 6.5 Binary Encoding of Returns

Returns are discretized into bins and encoded as binary vectors. For example, with 4 bins per asset and 2 assets, we get an 8-dimensional binary vector. The encoding preserves ordinal information using thermometer encoding: a return in the 3rd bin is encoded as [1, 1, 1, 0].

### 6.6 Bybit Data Integration

The implementation fetches real kline (candlestick) data from the Bybit V5 API. The `BybitClient` struct handles HTTP requests and parses the response into a clean format for the model.

## 7. Bybit Data Integration

The Bybit V5 API provides free access to historical kline data for cryptocurrency pairs. Our integration:

1. **Fetches OHLCV data** for specified trading pairs (e.g., BTCUSDT, ETHUSDT)
2. **Computes log returns** from closing prices: `r_t = ln(P_t / P_{t-1})`
3. **Discretizes returns** into configurable number of bins using quantile-based boundaries
4. **Encodes as binary features** using thermometer encoding for ordinal structure
5. **Handles pagination** for fetching large historical datasets

The API endpoint used is:
```
GET https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={limit}
```

No authentication is required for public market data, making it ideal for research and backtesting.

## 8. Key Takeaways

1. **Quantum Boltzmann Machines** extend classical RBMs by adding transverse-field terms to the Hamiltonian, enabling quantum tunneling through energy barriers during sampling.

2. **The Suzuki-Trotter decomposition** allows us to simulate quantum systems on classical hardware by mapping to a higher-dimensional classical system with inter-replica couplings.

3. **Quantum tunneling provides a sampling advantage** that is particularly valuable for financial distributions with multiple modes, heavy tails, and complex dependency structures.

4. **QBMs learn joint probability distributions** of asset returns that capture marginal properties, correlations, and higher-order dependencies simultaneously.

5. **Generative sampling from trained QBMs** produces synthetic market scenarios useful for risk assessment, stress testing, and portfolio optimization.

6. **Anomaly detection** using the energy function can identify unusual market conditions before they fully materialize.

7. **Quantum contrastive divergence** trains the model using faster-mixing quantum samplers, reducing the number of CD steps needed.

8. **Binary encoding of returns** using thermometer encoding preserves ordinal information and makes the data compatible with the binary units of the Boltzmann machine.

9. **The transverse field strength** acts as a regularization knob: higher values encourage exploration and prevent overfitting to local modes in the data distribution.

10. **Even simulated on classical hardware**, QBMs can outperform classical RBMs on multimodal financial distributions, and the same algorithms will run natively on quantum hardware as it matures.
