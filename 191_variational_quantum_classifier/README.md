# Chapter 191: Variational Quantum Classifier for Trading

## 1. Introduction

Variational Quantum Classifiers (VQCs) represent one of the most promising near-term quantum machine learning algorithms. Unlike Quantum SVMs, which use quantum circuits solely for kernel evaluation, VQCs are end-to-end parameterized quantum models: a quantum circuit with trainable rotation angles is optimized via classical gradient-based methods to perform classification directly. This hybrid quantum-classical approach makes VQCs particularly well-suited for Noisy Intermediate-Scale Quantum (NISQ) devices, where circuit depth must remain shallow.

In financial applications, VQCs can classify market regimes, predict price direction, and detect anomalous patterns. The variational nature of the algorithm allows the model to adapt its quantum feature representation during training, potentially discovering non-linear decision boundaries that are difficult for classical models to learn. The circuit structure — encoding data into quantum states and then applying trainable rotations — creates an expressive hypothesis class that scales exponentially with the number of qubits.

In this chapter, we build a complete Variational Quantum Classifier trading system in Rust. We simulate the quantum circuits classically, but the mathematical framework is fully compatible with execution on real quantum processors. Our system fetches market data from the Bybit exchange, engineers financial features, labels market regimes, trains a VQC, and evaluates its predictive performance.

## 2. Mathematical Foundation

### 2.1 The VQC Algorithm

A Variational Quantum Classifier consists of three components:

1. **Data encoding circuit** U(x): Maps classical input x into a quantum state.
2. **Variational (trainable) circuit** W(theta): A parameterized quantum circuit with trainable angles theta.
3. **Measurement**: Measures one or more qubits to obtain a classification prediction.

The full quantum state before measurement is:

```
|psi(x, theta)> = W(theta) * U(x) |0...0>
```

The prediction is obtained by measuring the expectation value of the Pauli-Z operator on the first qubit:

```
y_pred = <psi(x, theta)| Z_0 |psi(x, theta)>
```

This expectation value lies in [-1, +1], providing a natural binary classification output. For multi-class problems, we measure multiple qubits or use one-vs-rest encoding.

### 2.2 Data Encoding Circuit U(x)

We use angle encoding, where each feature x_k is encoded as a rotation angle on qubit k:

```
U(x) = tensor_product( Ry(x_k) ) for k = 1, ..., n
```

where Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]] is the Y-rotation gate. Features are normalized to [0, pi] before encoding.

For richer data encoding, we can apply the encoding circuit multiple times (data re-uploading), interleaving it with variational layers:

```
|psi> = W_L(theta_L) * U(x) * ... * W_1(theta_1) * U(x) |0...0>
```

### 2.3 Variational Circuit W(theta)

The variational circuit consists of L layers, each containing:

**Single-qubit rotations:**
```
R(theta) = Rz(theta_3) * Ry(theta_2) * Rz(theta_1)
```
applied to each qubit, providing full single-qubit rotational freedom (3 parameters per qubit per layer).

**Entangling gates:**
CNOT gates between adjacent qubits in a linear or circular topology:
```
CNOT(q_k, q_{k+1})  for k = 0, ..., n-2
```

The total number of trainable parameters is 3 * n * L (for L layers and n qubits).

### 2.4 Cost Function and Optimization

The cost function is the binary cross-entropy (or mean squared error) between predictions and labels:

```
C(theta) = (1/N) * sum_i [ (1 - y_i * f(x_i, theta))^2 ]
```

where f(x_i, theta) = <Z_0> is the expectation value for input x_i.

We optimize theta using the parameter-shift rule, a quantum-native gradient estimation technique:

```
dC/d(theta_k) = [C(theta_k + pi/2) - C(theta_k - pi/2)] / 2
```

This requires 2 circuit evaluations per parameter per gradient step. In our classical simulation, we can also compute gradients via finite differences or automatic differentiation.

### 2.5 Expressibility and Entangling Capability

The power of a VQC depends on two properties:

1. **Expressibility**: How uniformly the circuit can explore the space of quantum states. More layers and entangling gates increase expressibility.
2. **Entangling capability**: The degree to which the circuit can create entanglement between qubits. This is essential for capturing correlations between features.

A balance must be struck: too few layers lead to underfitting, while too many layers cause barren plateaus (vanishing gradients in the cost landscape).

## 3. VQC vs Classical and Other Quantum Methods

### 3.1 Comparison Table

| Aspect | Classical NN | QSVM | VQC |
|--------|-------------|------|-----|
| Model type | Parameterized layers | Kernel-based | Parameterized quantum circuit |
| Training | Backpropagation | Convex optimization | Parameter-shift rule |
| Feature space | Fixed architecture | Quantum kernel | Learned quantum representation |
| Parameters | O(d * h * L) | O(N^2) kernel matrix | O(n * L) rotations |
| Scalability | Good | Limited by kernel matrix | Good (shallow circuits) |
| NISQ compatibility | N/A | Requires deep circuits | Designed for NISQ |

### 3.2 Advantages of VQC for Trading

1. **Adaptive feature learning**: Unlike QSVM where the feature map is fixed, VQC learns the optimal quantum feature representation during training.
2. **Shallow circuits**: VQC can achieve good performance with shallow circuits (few layers), making it suitable for noisy quantum hardware.
3. **Scalable training**: Training cost scales with the number of parameters, not the size of the training set (unlike kernel methods).
4. **Data re-uploading**: By re-encoding data between variational layers, VQC can approximate arbitrary functions (universal approximation).

## 4. Trading Application

### 4.1 Market Regime Classification

We classify market states into three regimes based on forward returns:

- **Bull (+1)**: Forward return exceeds a positive threshold (e.g., +0.5%)
- **Bear (-1)**: Forward return falls below a negative threshold (e.g., -0.5%)
- **Sideways (0)**: Forward return within the threshold band

For binary classification with VQC, we merge sideways and bear into a single class (-1).

### 4.2 Feature Engineering

Our feature set for VQC includes:

1. **Log returns**: r_t = ln(P_t / P_{t-1}) capturing price momentum
2. **Realized volatility**: Rolling standard deviation of returns
3. **RSI-like momentum**: Ratio of average gains to average losses
4. **Volume ratio**: Current volume relative to moving average volume
5. **Price position**: Normalized position within recent high/low range

Features are normalized to [0, pi] for angle encoding into the quantum circuit.

### 4.3 Trading Strategy

The VQC output (expectation value of Z on first qubit) directly provides a trading signal:

- **Long signal**: f(x) > threshold (e.g., 0.0)
- **Short/flat signal**: f(x) <= threshold

The continuous output can also be used for position sizing: stronger signals lead to larger positions.

## 5. Implementation Walkthrough

### 5.1 Project Structure

```
191_variational_quantum_classifier/
  rust/
    Cargo.toml
    src/
      lib.rs              # Core VQC implementation
    examples/
      trading_example.rs  # Full trading pipeline
```

### 5.2 Quantum State Simulation

The quantum state is represented as a complex vector of dimension 2^n. Gates are applied by computing their matrix representation and multiplying with the state vector:

```rust
// Apply a single-qubit gate to qubit `target` in an n-qubit system
fn apply_single_gate(state: &mut Vec<(f64, f64)>, gate: [[f64; 4]], target: usize, n_qubits: usize) {
    // For each pair of amplitudes affected by the gate
    // Compute new amplitudes using the 2x2 gate matrix
}
```

### 5.3 Variational Circuit

Each layer of the variational circuit applies:
1. Ry and Rz rotations with trainable parameters to each qubit
2. CNOT gates between adjacent qubits for entanglement

```rust
fn apply_variational_layer(state: &mut Vec<(f64, f64)>, params: &[f64], n_qubits: usize) {
    // Single-qubit rotations: Rz * Ry * Rz per qubit (3 params each)
    for q in 0..n_qubits {
        apply_rz(state, params[3*q], q, n_qubits);
        apply_ry(state, params[3*q + 1], q, n_qubits);
        apply_rz(state, params[3*q + 2], q, n_qubits);
    }
    // Entangling CNOT ladder
    for q in 0..n_qubits-1 {
        apply_cnot(state, q, q+1, n_qubits);
    }
}
```

### 5.4 Training with Parameter-Shift Rule

```rust
fn parameter_shift_gradient(params: &[f64], data: &[Vec<f64>], labels: &[f64]) -> Vec<f64> {
    let mut grad = vec![0.0; params.len()];
    for k in 0..params.len() {
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();
        params_plus[k] += PI / 2.0;
        params_minus[k] -= PI / 2.0;
        grad[k] = (cost(&params_plus, data, labels) - cost(&params_minus, data, labels)) / 2.0;
    }
    grad
}
```

### 5.5 Bybit Integration

Data is fetched from the Bybit public API:

```rust
let url = format!(
    "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
    symbol, interval, limit
);
```

The response contains OHLCV candles parsed into structured data for feature engineering.

## 6. Bybit Data Integration

The Bybit API provides historical kline (candlestick) data for our VQC. We fetch BTCUSDT perpetual futures data at various timeframes.

Key considerations:

- **Rate limiting**: Bybit public endpoints have generous rate limits; we implement reasonable delays.
- **Data quality**: We handle missing candles and verify timestamp continuity.
- **Normalization**: Raw prices are converted to returns and indicators, then normalized to [0, pi] for quantum encoding.
- **Temporal splitting**: Earlier data for training, later data for testing, to avoid look-ahead bias.

The pipeline:
1. Fetch N candles of BTCUSDT from Bybit
2. Compute features (returns, volatility, RSI, volume ratio, price position)
3. Label each candle with a market regime
4. Normalize features to [0, pi]
5. Split into train/test sets
6. Initialize VQC parameters randomly
7. Train VQC using parameter-shift rule
8. Predict on test set and evaluate accuracy

## 7. Key Takeaways

1. **VQCs are hybrid quantum-classical models** that combine parameterized quantum circuits with classical optimization. They are the quantum analog of neural networks, with trainable rotation angles playing the role of weights.

2. **Data encoding is crucial**. Angle encoding maps features to qubit rotations, and data re-uploading between variational layers increases the model's expressiveness, enabling universal function approximation.

3. **The parameter-shift rule** provides exact quantum gradients using only two circuit evaluations per parameter. This makes VQC training compatible with both quantum hardware and classical simulation.

4. **Shallow circuits suffice** for many classification tasks. This is important for NISQ devices where circuit depth is limited by decoherence and gate errors.

5. **Market regime classification** benefits from VQC's ability to learn non-linear decision boundaries in quantum feature space. The adaptive nature of the variational circuit means the model can discover useful feature interactions during training.

6. **Barren plateaus** are a key challenge: for randomly initialized deep circuits, gradients can vanish exponentially with the number of qubits. Strategies to mitigate this include layer-wise training, identity initialization, and limiting circuit depth.

7. **Classical simulation is tractable** for small qubit counts (2-4 qubits). Our Rust implementation simulates quantum circuits exactly, allowing full pipeline development and testing without quantum hardware.

8. **Rust provides performance and safety** for the computationally intensive tasks of state vector simulation and gradient computation. The strong type system catches errors at compile time, and zero-cost abstractions keep performance high.
