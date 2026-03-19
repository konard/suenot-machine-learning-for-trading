# Chapter 188: Quantum GAN for Finance

## 1. Introduction

Generative Adversarial Networks (GANs) have transformed synthetic data generation across domains from computer vision to natural language processing. In financial markets, the ability to generate realistic synthetic data is particularly valuable: historical datasets are limited, extreme market events are rare, and privacy concerns restrict data sharing. **Quantum GANs (QGANs)** extend this paradigm by replacing the classical generator with a parameterized quantum circuit, leveraging the exponentially large Hilbert space to represent complex probability distributions more efficiently than classical counterparts.

A Quantum GAN consists of two components:

1. **Quantum Generator**: A parameterized quantum circuit (PQC) that maps a latent noise input to synthetic financial data. The quantum circuit uses rotation gates and entangling operations to explore a high-dimensional feature space.
2. **Classical Discriminator**: A standard feedforward neural network that distinguishes between real financial data and synthetic samples produced by the quantum generator.

The two components are trained adversarially: the generator learns to produce increasingly realistic data while the discriminator sharpens its ability to detect fakes. This minimax game converges (ideally) to a Nash equilibrium where synthetic data is statistically indistinguishable from real data.

### Why Quantum for Financial Data?

Financial return distributions exhibit characteristics that are notoriously difficult to model classically:

- **Heavy tails** (leptokurtosis): extreme events occur far more often than a Gaussian model predicts.
- **Volatility clustering**: periods of high volatility tend to cluster together.
- **Asymmetric distributions** (negative skew): large drops are more common than large rallies.
- **Non-linear correlations**: dependencies between assets change in different market regimes.

Quantum circuits can represent multi-modal, heavy-tailed distributions naturally through superposition and entanglement. A circuit with *n* qubits can represent distributions over 2^n states, providing an exponential advantage in expressivity per parameter.

## 2. Mathematical Foundation

### 2.1 Quantum Circuit Generator

The quantum generator maps a set of trainable parameters **theta** to a probability distribution over output states. The circuit operates on *n* qubits through a sequence of layers:

**State Preparation:**

Each qubit is initialized in the |0> state. The initial state of the full system is:

|psi_0> = |0>^{otimes n}

**Parameterized Rotation Layer:**

Each qubit *i* in layer *l* undergoes single-qubit rotations:

U_rot(theta_{i,l}) = R_z(theta_{i,l,3}) * R_y(theta_{i,l,2}) * R_x(theta_{i,l,1})

where R_x, R_y, R_z are rotation matrices around the respective axes:

R_x(alpha) = exp(-i * alpha * sigma_x / 2)
R_y(alpha) = exp(-i * alpha * sigma_y / 2)
R_z(alpha) = exp(-i * alpha * sigma_z / 2)

**Entangling Layer:**

After rotations, CNOT gates entangle adjacent qubits:

CNOT_{i, i+1} for i in {0, 1, ..., n-2}

This creates correlations between qubits, enabling the circuit to represent entangled (correlated) distributions.

**Measurement and Output:**

After *L* layers of rotation + entanglement, the final state is:

|psi(theta)> = prod_{l=1}^{L} [U_ent * U_rot(theta_l)] |psi_0>

Measuring each qubit in the computational basis yields bit strings whose frequency distribution approximates the target financial distribution. For continuous outputs, we map measurement probabilities to data values through a post-processing function.

### 2.2 Classical Discriminator

The discriminator is a feedforward neural network D(x; phi) with parameters **phi** that outputs a scalar in [0, 1] representing the probability that input x is real data:

D(x) = sigmoid(W_3 * ReLU(W_2 * ReLU(W_1 * x + b_1) + b_2) + b_3)

### 2.3 Minimax Game

The QGAN training objective follows the standard GAN minimax formulation:

min_theta max_phi V(theta, phi) = E_{x ~ p_data}[log D(x; phi)] + E_{z ~ p_z}[log(1 - D(G(z; theta); phi))]

where:
- p_data is the real financial data distribution
- G(z; theta) is the quantum generator output
- D(x; phi) is the discriminator output

**Generator update:** Adjust theta to minimize log(1 - D(G(z; theta); phi)), or equivalently maximize log D(G(z; theta); phi) (non-saturating loss).

**Discriminator update:** Adjust phi to maximize the full objective V.

### 2.4 Parameter Shift Rule for Quantum Gradients

Since the quantum circuit is not directly differentiable in the classical sense, gradients with respect to circuit parameters are computed using the **parameter shift rule**:

dV/d(theta_i) = [V(theta_i + pi/2) - V(theta_i - pi/2)] / 2

This exact gradient formula exploits the periodicity of rotation gates and enables gradient-based optimization of the quantum circuit.

## 3. Applications in Finance

### 3.1 Data Augmentation for Rare Market Events

Historical financial datasets contain very few examples of extreme events (flash crashes, liquidity crises, Black Swan events). A QGAN trained on real market data can generate synthetic samples that preserve the statistical properties of these rare events, enabling:

- **Stress testing** trading strategies against realistic but never-before-seen scenarios.
- **Risk model calibration** with augmented tail data.
- **Balanced training sets** for ML models that need exposure to extreme market conditions.

### 3.2 Synthetic Orderbook Generation

QGANs can generate synthetic limit order book snapshots that preserve the microstructure properties of real markets:

- Bid-ask spread distributions
- Order size distributions at different price levels
- Correlation between orderbook depth and volatility

This is valuable for backtesting market-making strategies where historical orderbook data is expensive or unavailable.

### 3.3 Privacy-Preserving Data Sharing

Financial institutions can share QGAN-generated synthetic data with researchers and partners without exposing proprietary trading data or client information, while preserving the statistical properties needed for model development.

### 3.4 Monte Carlo Path Generation

Quantum generators can produce synthetic price paths for Monte Carlo simulations that better capture the true distribution of returns compared to parametric models (e.g., geometric Brownian motion), leading to more accurate option pricing and risk metrics.

## 4. Implementation in Rust

Our Rust implementation provides a classical simulation of the quantum generator circuit, paired with a classical discriminator neural network. While a true quantum advantage requires execution on quantum hardware, the classical simulation preserves the algorithmic structure and serves as a development and testing framework.

### Architecture Overview

```
QuantumGenerator
  - n_qubits: number of qubits (determines output dimension)
  - n_layers: depth of the circuit
  - params: Vec<f64> (rotation angles)
  - forward(): simulate circuit, return samples

ClassicalDiscriminator
  - layers: Vec<DenseLayer>
  - forward(): feedforward pass
  - backward(): gradient computation

QGANTrainer
  - generator: QuantumGenerator
  - discriminator: ClassicalDiscriminator
  - train(): alternating updates
  - generate(): produce synthetic samples
```

### Key Implementation Details

1. **Quantum State Simulation**: The quantum state is represented as a complex vector of size 2^n. Gates are applied as matrix multiplications on the state vector.

2. **Parameterized Rotations**: Each rotation gate is a 2x2 unitary matrix. For n qubits, single-qubit gates are extended to the full Hilbert space via tensor products.

3. **CNOT Gates**: Two-qubit entangling gates are applied as 4x4 matrices on the relevant qubit pairs.

4. **Measurement Simulation**: Sampling from the output distribution is performed by computing |amplitude|^2 for each basis state and sampling accordingly.

5. **Gradient Computation**: The parameter shift rule is used for generator gradients; standard backpropagation for discriminator gradients.

## 5. Bybit Crypto Data Integration

The implementation fetches real market data from the Bybit exchange API to:

1. **Obtain training data**: Historical kline (candlestick) data for BTCUSDT is fetched and converted to log returns.
2. **Normalize returns**: Data is scaled to [0, 1] range for GAN training.
3. **Compare statistics**: After training, synthetic data statistics (mean, standard deviation, skewness, kurtosis) are compared against the real data to evaluate generation quality.

### API Endpoint

```
GET https://api.bybit.com/v5/market/kline
  ?category=spot
  &symbol=BTCUSDT
  &interval=60
  &limit=200
```

The response provides OHLCV data which is converted to log returns:

r_t = ln(close_t / close_{t-1})

## 6. Key Takeaways

1. **Quantum GANs combine quantum computing with adversarial training** to generate synthetic financial data that captures complex distributional properties including heavy tails, skewness, and non-linear correlations.

2. **The quantum generator uses parameterized circuits** with rotation gates and entanglement to represent exponentially large probability distributions with a polynomial number of parameters.

3. **The parameter shift rule enables gradient-based optimization** of quantum circuits, making QGANs trainable with standard optimization techniques.

4. **Financial applications include data augmentation** for rare events, synthetic orderbook generation, privacy-preserving data sharing, and improved Monte Carlo simulations.

5. **Classical simulation provides a development framework** while true quantum advantage awaits fault-tolerant quantum hardware. Current noisy intermediate-scale quantum (NISQ) devices can run small-scale QGANs with 5-20 qubits.

6. **Evaluation requires financial-specific metrics** beyond standard GAN metrics: matching of moments (mean, variance, skew, kurtosis), autocorrelation structure, and tail behavior are critical for financial validity.

7. **Hybrid quantum-classical architectures** where the generator is quantum and the discriminator is classical represent the most practical near-term approach, as discriminators benefit less from quantum speedups.

8. **Integration with real exchange data** (e.g., Bybit) enables direct comparison between synthetic and real market distributions, providing a practical validation framework for QGAN-generated financial data.
