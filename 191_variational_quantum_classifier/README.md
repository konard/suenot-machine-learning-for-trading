# Chapter 191: Variational Quantum Classifier for Trading

## 1. Introduction

The Variational Quantum Classifier (VQC) represents one of the most promising near-term applications of quantum computing to machine learning. Unlike fault-tolerant quantum algorithms that require millions of error-corrected qubits, VQCs operate within the constraints of today's Noisy Intermediate-Scale Quantum (NISQ) devices by leveraging hybrid quantum-classical optimization. In the context of financial markets, VQCs offer a compelling approach to classification tasks such as predicting whether an asset's price will move up or down, detecting market regimes, and identifying trading signals.

The core idea behind a VQC is deceptively simple: encode classical data into a quantum state, apply a parameterized quantum circuit (the "variational ansatz"), measure the output, and interpret the measurement probabilities as class predictions. The circuit parameters are then optimized using a classical optimizer to minimize a loss function, much like training a neural network. What makes VQCs interesting for trading is their ability to explore exponentially large Hilbert spaces with relatively few parameters, potentially capturing complex nonlinear relationships in market data that classical models might miss.

This chapter provides a comprehensive treatment of VQCs applied to financial prediction. We begin with the mathematical foundations, move through architecture design, discuss training strategies, and conclude with a complete Rust implementation that fetches real market data from the Bybit exchange and trains a VQC to predict price direction for BTCUSDT.

## 2. Mathematical Foundation

### 2.1 Parameterized Quantum Circuits as Classifiers

A parameterized quantum circuit (PQC) is a unitary transformation $U(\theta)$ that depends on a set of tunable parameters $\theta = (\theta_1, \theta_2, \ldots, \theta_p)$. Given an $n$-qubit system, the circuit acts on the initial state $|0\rangle^{\otimes n}$ to produce:

$$|\psi(\theta)\rangle = U(\theta)|0\rangle^{\otimes n}$$

For classification, we define a mapping from the measurement outcome probabilities to class labels. The probability of measuring a specific bitstring $z$ is:

$$P(z|\theta) = |\langle z | \psi(\theta) \rangle|^2$$

In the simplest binary classification setup, we measure the first qubit and use $P(|0\rangle)$ as the probability of class 0 and $P(|1\rangle)$ as the probability of class 1.

### 2.2 Data Encoding

Before a quantum circuit can process classical data, we must encode it into a quantum state. Two primary encoding strategies are used:

**Angle Encoding:** Each feature $x_i$ is encoded as a rotation angle on a dedicated qubit:

$$|x\rangle = \bigotimes_{i=1}^{n} R_Y(x_i)|0\rangle$$

where $R_Y(\theta) = \exp(-i\theta Y/2)$ is a rotation about the Y-axis. This approach is straightforward and requires one qubit per feature. Angle encoding is computationally efficient, requiring only $O(n)$ gates for $n$ features.

**Amplitude Encoding:** The feature vector $\mathbf{x} = (x_1, \ldots, x_{2^n})$ is encoded into the amplitudes of an $n$-qubit state:

$$|x\rangle = \sum_{i=0}^{2^n - 1} x_i |i\rangle$$

where $\mathbf{x}$ is normalized such that $\|\mathbf{x}\|^2 = 1$. Amplitude encoding is exponentially compact (encoding $2^n$ features in $n$ qubits) but requires more complex state preparation circuits.

For trading applications, angle encoding is preferred due to its simplicity and robustness to noise. We typically work with a small number of engineered features (returns, volatility, momentum), making angle encoding a natural fit.

### 2.3 Variational Ansatz

The variational ansatz defines the structure of the parameterized circuit. A common design alternates between layers of single-qubit rotations and entangling gates:

**Single-qubit layer:** Apply $R_Y(\theta_{l,i})$ and $R_Z(\theta_{l,i}')$ rotations to each qubit $i$ in layer $l$:

$$U_{\text{rot}}^{(l)} = \bigotimes_{i=1}^{n} R_Z(\theta_{l,i}') R_Y(\theta_{l,i})$$

**Entangling layer:** Apply CNOT gates in a linear or circular pattern to create correlations between qubits:

$$U_{\text{ent}} = \prod_{i=1}^{n-1} \text{CNOT}(i, i+1)$$

A full ansatz with $L$ layers takes the form:

$$U(\theta) = \prod_{l=1}^{L} U_{\text{ent}} \cdot U_{\text{rot}}^{(l)}$$

The expressibility of the ansatz (its ability to represent arbitrary unitaries) increases with the number of layers, but so does the risk of barren plateaus in the optimization landscape.

### 2.4 Measurement-Based Classification

After applying the full circuit (encoding + ansatz), we perform a computational basis measurement on the first qubit. The probability of obtaining outcome $|0\rangle$ on the first qubit is:

$$p_0 = \sum_{z: z_1 = 0} |\langle z | \psi \rangle|^2$$

For binary classification (e.g., price up vs. price down), we interpret $p_0$ as the probability of class 0 (price down) and $p_1 = 1 - p_0$ as the probability of class 1 (price up).

### 2.5 Cross-Entropy Loss

The model is trained to minimize the binary cross-entropy loss:

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{j=1}^{N} \left[ y_j \log(p_1^{(j)}) + (1 - y_j) \log(1 - p_1^{(j)}) \right]$$

where $y_j \in \{0, 1\}$ is the true label and $p_1^{(j)}$ is the predicted probability of class 1 for sample $j$. This loss function is differentiable with respect to the circuit parameters, enabling gradient-based optimization.

## 3. VQC Architecture

The complete VQC architecture consists of three sequential stages:

### 3.1 Feature Map Circuit

The feature map circuit encodes classical input data into a quantum state. For trading with $n$ features:

```
|0> -- RY(x_1) --
|0> -- RY(x_2) --
|0> -- RY(x_3) --
...
|0> -- RY(x_n) --
```

To enhance the feature map's expressiveness, we can apply data re-uploading, where the encoding is repeated between variational layers. This has been shown to improve the model's capacity to learn complex decision boundaries.

### 3.2 Variational Circuit

The variational circuit applies trainable rotations and entanglement:

```
-- RY(t1) -- RZ(t2) -- o --- ... ---
                        |
-- RY(t3) -- RZ(t4) -- X -- o -- ...
                             |
-- RY(t5) -- RZ(t6) --------X -- ...
```

Each layer adds $2n$ parameters ($n$ for RY and $n$ for RZ rotations). With $L$ layers, the total parameter count is $2nL$. For a 4-qubit, 3-layer VQC, this gives 24 trainable parameters, which is compact enough for efficient classical optimization yet expressive enough for useful classification.

### 3.3 Measurement

The final step measures the first qubit in the computational basis. The squared amplitudes of the statevector components where the first qubit is $|0\rangle$ or $|1\rangle$ give us the class probabilities directly.

## 4. Trading Application

### 4.1 Binary Classification: Up/Down Prediction

The most natural trading application of a VQC is predicting the direction of the next price movement. Given a set of technical features computed from historical OHLCV data, the VQC outputs the probability that the next candle will close higher than it opened.

**Feature Engineering Pipeline:**
1. **Returns:** Log returns over different lookback periods (1, 5, 10 candles)
2. **Volatility:** Rolling standard deviation of returns
3. **Momentum:** Rate of change (ROC) and relative strength indicators
4. **Volume:** Normalized volume changes

Features are scaled to the range $[-\pi, \pi]$ for angle encoding, ensuring they map naturally to qubit rotation angles.

**Label Construction:**
- Label = 1 if the next candle's close > current close (price goes up)
- Label = 0 otherwise (price goes down or stays flat)

### 4.2 Multi-Class Regime Detection

For more sophisticated applications, the VQC can be extended to multi-class classification by measuring multiple qubits. For example, a 2-qubit measurement yields four possible outcomes corresponding to four market regimes:

| Outcome | Regime |
|---------|--------|
| $\|00\rangle$ | Low volatility, bearish |
| $\|01\rangle$ | Low volatility, bullish |
| $\|10\rangle$ | High volatility, bearish |
| $\|11\rangle$ | High volatility, bullish |

This enables regime-aware trading strategies that adapt position sizing and risk management to current market conditions.

## 5. Training

### 5.1 Hybrid Quantum-Classical Optimization

VQC training follows a hybrid loop:

1. **Forward pass:** Run the quantum circuit with current parameters $\theta$ to obtain predictions
2. **Loss computation:** Calculate the cross-entropy loss on the classical computer
3. **Gradient estimation:** Compute gradients of the loss with respect to $\theta$
4. **Parameter update:** Use a classical optimizer (Adam, SGD) to update $\theta$
5. **Repeat** until convergence

This hybrid approach is essential because quantum circuits cannot perform backpropagation natively. Instead, gradients are estimated using the parameter shift rule.

### 5.2 Parameter Shift Rule

The parameter shift rule provides exact analytical gradients for quantum circuits containing Pauli rotation gates. For a parameter $\theta_k$ appearing in a rotation gate, the gradient of an expectation value $f(\theta)$ is:

$$\frac{\partial f}{\partial \theta_k} = \frac{f(\theta_k + \pi/2) - f(\theta_k - \pi/2)}{2}$$

This requires two circuit evaluations per parameter. For $p$ parameters, the full gradient vector requires $2p$ circuit evaluations. While this is more expensive than classical backpropagation, it provides exact gradients (not estimates) and works natively with quantum hardware.

In practice, for VQC training on financial data:
- Each gradient step requires $2p$ forward passes through the quantum circuit
- For a 4-qubit, 3-layer VQC with 24 parameters, this means 48 circuit evaluations per training step
- On a simulator, this is fast; on real quantum hardware, batching and circuit optimization become important

### 5.3 Avoiding Barren Plateaus

Barren plateaus are a well-known challenge in VQC training: as the number of qubits grows, the gradient landscape can become exponentially flat, making optimization difficult. Strategies to mitigate this include:

- **Shallow circuits:** Using fewer layers reduces the risk of barren plateaus
- **Local cost functions:** Measuring only a subset of qubits rather than global observables
- **Parameter initialization:** Starting parameters near zero or using structured initialization
- **Layer-wise training:** Training one layer at a time, freezing previously trained layers

For trading applications with small feature sets (4-8 features), barren plateaus are generally not a concern.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete VQC framework with the following components:

### 6.1 Statevector Simulator

We implement a full statevector simulator that tracks the quantum state as a vector of $2^n$ complex amplitudes. Single-qubit gates are applied by computing the tensor product structure, and two-qubit gates (CNOT) are applied through controlled operations on the statevector.

```rust
// Initialize |00...0> state
let mut statevector = vec![Complex::zero(); 1 << num_qubits];
statevector[0] = Complex::one();

// Apply RY gate to qubit i
apply_ry(&mut statevector, qubit, angle);

// Apply CNOT gate
apply_cnot(&mut statevector, control, target);
```

### 6.2 VQC Structure

The VQC is structured as:
1. **Angle encoding:** Apply RY rotations with feature values
2. **Variational layers:** Alternate between RY/RZ rotations and CNOT entanglement
3. **Measurement:** Extract probability of first qubit being |1>

```rust
let vqc = VQC::new(num_qubits, num_layers);
let prob_class1 = vqc.forward(&features, &params);
```

### 6.3 Training Loop

The training loop implements gradient descent with the parameter shift rule:

```rust
for epoch in 0..num_epochs {
    let gradients = vqc.compute_gradients(&features, &labels, &params);
    for i in 0..params.len() {
        params[i] -= learning_rate * gradients[i];
    }
}
```

### 6.4 Bybit Data Integration

We fetch real market data from the Bybit API:

```rust
let candles = fetch_bybit_klines("BTCUSDT", "15", 500).await?;
let (features, labels) = engineer_features(&candles);
```

The feature engineering pipeline computes returns, volatility, and momentum indicators, scales them appropriately, and constructs binary labels for price direction.

## 7. Bybit Data Integration

The implementation connects to the Bybit v5 public API to fetch historical kline (candlestick) data. The endpoint `https://api.bybit.com/v5/market/kline` provides OHLCV data for any trading pair and timeframe.

**Data Flow:**
1. Fetch raw OHLCV data from Bybit API
2. Parse JSON response into structured candle data
3. Compute technical features from the candle series
4. Scale features to $[-\pi, \pi]$ for angle encoding
5. Generate binary labels based on next-candle direction
6. Split into training and test sets

**Feature Engineering Details:**

| Feature | Computation | Rationale |
|---------|-------------|-----------|
| Return (1-bar) | $\ln(C_t / C_{t-1})$ | Short-term momentum |
| Return (5-bar) | $\ln(C_t / C_{t-5})$ | Medium-term trend |
| Volatility | $\text{std}(\text{ret}_{t-9:t})$ | Risk regime indicator |
| Volume change | $(V_t - V_{t-1}) / V_{t-1}$ | Participation signal |

Features are normalized using min-max scaling to $[-\pi, \pi]$, which maps naturally to rotation angles in the quantum circuit. This normalization is critical because qubit rotations are periodic with period $2\pi$, and we want to use the full range of the rotation to maximize the circuit's discriminative power.

## 8. Key Takeaways

1. **VQCs are hybrid models** that combine quantum circuit execution with classical optimization. They are well-suited for NISQ-era devices and can be simulated classically for small problem sizes.

2. **Angle encoding is practical** for trading features. With 4-8 engineered features, we need only 4-8 qubits, which is well within the capabilities of current quantum hardware and trivial to simulate.

3. **The parameter shift rule** enables exact gradient computation for quantum circuits, making gradient-based optimization feasible without requiring quantum backpropagation.

4. **For small feature sets**, VQCs have a comparable number of parameters to simple neural networks, but they explore the exponentially large Hilbert space, potentially learning different decision boundaries.

5. **Barren plateaus** are the primary training challenge for VQCs, but for the small circuits used in trading applications (4-8 qubits, 2-4 layers), they are rarely problematic.

6. **Real-world performance** of VQCs on financial data is still an active research area. Current evidence suggests they perform comparably to classical models of similar complexity, with potential advantages in specific regimes.

7. **The Rust implementation** provides a complete, self-contained VQC framework including statevector simulation, training, and Bybit data integration, suitable for experimentation and further development.

8. **Practical considerations:** VQCs should be viewed as part of a broader toolkit. For production trading, ensemble methods combining quantum and classical classifiers may offer the best risk-adjusted performance.
