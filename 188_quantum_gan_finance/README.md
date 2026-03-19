# Chapter 188: Quantum GAN Finance

## 1. Introduction

Generative Adversarial Networks (GANs) have revolutionized synthetic data generation across many domains, from image synthesis to natural language augmentation. In the financial domain, the ability to produce realistic synthetic market data is enormously valuable: it allows traders and researchers to augment limited training sets, stress-test strategies against rare market events, and explore counterfactual scenarios without risking real capital.

Quantum Generative Adversarial Networks (QGANs) push this paradigm further by replacing the classical generator with a parameterized quantum circuit (PQC). The key insight is that quantum systems naturally produce probability distributions that are exponentially hard to sample from classically. A quantum generator can, in principle, capture complex correlations in financial return distributions -- heavy tails, volatility clustering, and non-Gaussian skewness -- more efficiently than a classical generator of comparable size.

In this chapter we build a hybrid QGAN from first principles: a quantum generator realized as a parameterized circuit, paired with a classical discriminator neural network. We implement the full system in Rust, train it on real Bybit BTCUSDT data, and evaluate the statistical fidelity of the generated synthetic returns.

### Why Quantum for Finance?

Financial return distributions exhibit several properties that make them natural candidates for quantum generation:

- **Heavy tails**: Extreme events occur far more frequently than a Gaussian model predicts. Quantum Born-machine sampling naturally produces distributions with complex tail behavior.
- **Multi-modal structure**: Market regimes (trending, mean-reverting, crisis) create multi-modal return distributions that parameterized quantum circuits can represent compactly.
- **Correlations across time scales**: Entanglement in quantum circuits provides a natural mechanism for encoding cross-temporal dependencies.
- **Exponential state space**: An n-qubit circuit operates in a 2^n dimensional Hilbert space, giving the generator an exponentially large representational capacity relative to its parameter count.

## 2. Mathematical Foundation

### 2.1 The Quantum Generator

The quantum generator is a parameterized quantum circuit (PQC) acting on n qubits. It transforms the initial state |0...0> into a parameterized quantum state:

```
|psi(theta)> = U(theta) |0>^{otimes n}
```

where U(theta) is a unitary operator composed of layers of single-qubit rotations and entangling gates. Each layer consists of:

1. **Rotation gates**: R_y(theta_i) and R_z(theta_i) applied to each qubit, introducing parameterized rotations on the Bloch sphere.
2. **Entangling gates**: CNOT gates connecting adjacent qubits in a ring topology, creating quantum correlations (entanglement) between qubits.

A single rotation gate R_y(theta) acts as:

```
R_y(theta) = | cos(theta/2)  -sin(theta/2) |
             | sin(theta/2)   cos(theta/2) |
```

And the CNOT gate acts on two qubits:

```
CNOT = |1 0 0 0|
       |0 1 0 0|
       |0 0 0 1|
       |0 0 1 0|
```

### 2.2 Born Machine Sampling

After applying the circuit, measurement in the computational basis yields outcome bitstring x with probability given by the Born rule:

```
p(x | theta) = |<x | psi(theta)>|^2
```

This defines a probability distribution over 2^n possible outcomes. By mapping bitstrings to real-valued returns (via binning or linear interpolation), the quantum circuit defines a generative model over financial returns. This is known as a Born machine.

The probability vector is:

```
p_theta = (|alpha_0|^2, |alpha_1|^2, ..., |alpha_{2^n - 1}|^2)
```

where alpha_i are the amplitudes of the quantum state.

### 2.3 The Classical Discriminator

The discriminator is a classical feedforward neural network D(x; phi) that outputs a scalar in [0, 1] representing the probability that input x is drawn from real data rather than the generator. We use a simple two-layer network with sigmoid activation:

```
D(x) = sigmoid(w2 * ReLU(w1 * x + b1) + b2)
```

### 2.4 The Hybrid Training Loop

Training follows the standard GAN minimax objective, adapted for the hybrid setting:

```
min_theta max_phi  E_{x ~ p_real}[log D(x; phi)] + E_{x ~ p_theta}[log(1 - D(x; phi))]
```

The training alternates:

1. **Discriminator step**: Sample a batch from real data and a batch from the quantum generator. Update discriminator parameters phi via gradient ascent on the objective.
2. **Generator step**: Sample from the quantum generator. Compute the loss log(1 - D(G(z))) and update quantum circuit parameters theta via parameter-shift rule gradients.

### 2.5 Parameter-Shift Rule

For quantum circuits with gates of the form exp(-i * theta * P / 2) where P is a Pauli operator, the gradient with respect to parameter theta_k is:

```
d/d(theta_k) f(theta) = [f(theta_k + pi/2) - f(theta_k - pi/2)] / 2
```

This exact gradient formula requires only two circuit evaluations per parameter, making it feasible for hybrid optimization.

## 3. Trading Application

### 3.1 Generating Synthetic Market Data

The primary application is generating synthetic return series that preserve the statistical properties of real market data. A well-trained QGAN produces samples that match:

- Mean return (drift)
- Variance (volatility)
- Skewness (asymmetry of returns)
- Kurtosis (tail heaviness)
- Autocorrelation structure

These synthetic series can be used to:

- **Augment training data** for ML-based trading strategies, especially when historical data is limited.
- **Bootstrap confidence intervals** for strategy performance metrics (Sharpe ratio, maximum drawdown).
- **Generate scenario paths** for risk management and portfolio optimization.

### 3.2 Modeling Rare Market Events

Flash crashes, liquidity crises, and black swan events are by definition rare in historical data. A QGAN trained on market data can learn to generate plausible extreme scenarios by capturing the tail structure of the return distribution. This is particularly valuable because:

- Classical parametric models (e.g., Gaussian, Student-t) impose rigid distributional assumptions.
- Historical simulation is limited by the finite number of observed extreme events.
- The quantum generator's exponential state space allows it to represent complex tail dependencies without explicit parametric assumptions.

### 3.3 Generating Realistic Order Flow

Beyond returns, QGANs can generate synthetic limit order book snapshots, including bid-ask spreads, queue depths, and order arrival rates. This enables:

- Testing execution algorithms against synthetic but realistic market microstructure.
- Simulating market impact for large orders.
- Training reinforcement learning agents in simulated order book environments.

### 3.4 Augmenting Training Sets

When building ML models for alpha generation, overfitting to limited historical data is a constant risk. QGAN-generated synthetic data provides a principled augmentation strategy:

1. Train the QGAN on historical returns.
2. Generate N synthetic return series.
3. Train the alpha model on the combined real + synthetic dataset.
4. Validate on held-out real data.

This approach regularizes the alpha model by exposing it to a broader distribution of plausible market behaviors.

## 4. Architecture

### 4.1 Quantum Generator Circuit Design

Our implementation uses a layered circuit architecture:

```
Layer 1: R_y(theta_1) R_y(theta_2) ... R_y(theta_n)    [rotation]
          CNOT(0,1) CNOT(1,2) ... CNOT(n-1,0)           [entanglement]
Layer 2: R_y(theta_{n+1}) ... R_y(theta_{2n})           [rotation]
          CNOT(0,1) CNOT(1,2) ... CNOT(n-1,0)           [entanglement]
...
Layer L: R_y(theta_{(L-1)n+1}) ... R_y(theta_{Ln})      [rotation]
```

The depth L controls the expressivity of the generator. For financial applications, L = 3-5 layers with n = 4-6 qubits provides a good balance between expressivity and trainability.

### 4.2 Entanglement Layers

The CNOT ring topology creates nearest-neighbor entanglement, which propagates correlations across all qubits after multiple layers. This is crucial for capturing dependencies in the return distribution. The entanglement structure mirrors the way information propagates through financial markets -- local interactions (between related assets or time steps) build up to create global correlations.

### 4.3 Measurement-Based Sampling

After the circuit executes, we measure all qubits in the computational basis. Each measurement outcome is a bitstring of length n, which we map to a real-valued return:

1. Interpret the bitstring as a binary integer k in [0, 2^n - 1].
2. Map to the interval [r_min, r_max] via linear interpolation: r = r_min + k * (r_max - r_min) / (2^n - 1).

The bin width is (r_max - r_min) / (2^n - 1). With n = 6 qubits, we get 64 bins, providing reasonable resolution for return distributions.

## 5. Implementation Walkthrough

The implementation in Rust consists of several core components:

### 5.1 Quantum State Simulator

We simulate the quantum circuit by maintaining the full 2^n complex state vector and applying gate operations as matrix multiplications. This is exact simulation, suitable for small qubit counts (n <= 20).

```rust
pub struct QuantumCircuit {
    num_qubits: usize,
    state: Vec<Complex>,
    params: Vec<f64>,
}
```

The `apply_ry` method applies a rotation gate to a specific qubit by iterating over pairs of amplitudes and applying the 2x2 rotation matrix. The `apply_cnot` method applies the controlled-NOT gate by swapping amplitude pairs conditioned on the control qubit.

### 5.2 Classical Discriminator

The discriminator is a simple two-layer neural network implemented with ndarray for matrix operations:

```rust
pub struct Discriminator {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}
```

It uses ReLU activation in the hidden layer and sigmoid output, trained with binary cross-entropy loss.

### 5.3 Hybrid Training

The training loop alternates between discriminator and generator updates. The generator gradient is computed via the parameter-shift rule: for each parameter, we evaluate the circuit with the parameter shifted by +pi/2 and -pi/2, and take the difference.

### 5.4 Bybit Data Integration

We fetch real BTCUSDT kline data from the Bybit v5 API and compute log returns:

```rust
let returns: Vec<f64> = closes.windows(2)
    .map(|w| (w[1] / w[0]).ln())
    .collect();
```

These returns form the real data distribution that the QGAN learns to replicate.

## 6. Bybit Data Integration

The implementation fetches historical kline (candlestick) data from the Bybit v5 REST API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

The response contains OHLCV data which we parse and convert to log returns. Key considerations for Bybit integration:

- **Rate limiting**: The API allows up to 10 requests per second. Our implementation uses a single batch request.
- **Data quality**: We filter out zero or negative prices, and returns outside [-0.5, 0.5] to remove data anomalies.
- **Time alignment**: Kline timestamps are in milliseconds. We use hourly candles (interval=60) for sufficient data granularity.

The fetched data flows through the following pipeline:

1. Raw OHLCV from Bybit API
2. Extract close prices
3. Compute log returns: r_t = ln(P_t / P_{t-1})
4. Normalize to [r_min, r_max] range
5. Discretize into 2^n bins for Born machine comparison
6. Train QGAN on discretized distribution

## 7. Key Takeaways

1. **Quantum advantage in representation**: A quantum generator with n qubits operates in a 2^n dimensional space, providing exponentially compact representation of complex distributions compared to classical generators of similar parameter count.

2. **Born machines are natural generative models**: The Born rule directly maps quantum states to probability distributions, eliminating the need for explicit density estimation. This is a natural fit for modeling financial return distributions.

3. **Hybrid architecture is practical today**: By combining a quantum generator with a classical discriminator, we can leverage quantum advantages for generation while using well-understood classical methods for discrimination. This hybrid approach works on current simulators and near-term quantum hardware.

4. **Parameter-shift rule enables gradient-based training**: Unlike finite-difference methods, the parameter-shift rule provides exact gradients with only two circuit evaluations per parameter, making training efficient and numerically stable.

5. **Financial data is a good fit**: The statistical properties of financial returns -- heavy tails, multi-modality, complex correlations -- align well with the representational strengths of quantum circuits.

6. **Synthetic data has multiple uses**: QGAN-generated data can augment training sets, model rare events, test execution algorithms, and stress-test trading strategies.

7. **Scalability is the main challenge**: Full state vector simulation scales as O(2^n), limiting current implementations to small qubit counts. As quantum hardware improves, QGANs will be able to handle larger and more complex distributions.

8. **Statistical validation is essential**: Always compare generated data against real data using multiple metrics (mean, variance, skewness, kurtosis, autocorrelation) to ensure the QGAN captures the relevant features of the target distribution.
