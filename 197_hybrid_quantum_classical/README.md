# Chapter 197: Hybrid Quantum-Classical Computing for Trading

## 1. Introduction

The promise of quantum computing for financial applications has generated enormous interest, yet the current era of Noisy Intermediate-Scale Quantum (NISQ) devices presents a fundamental challenge: standalone quantum computers are too error-prone and too limited in qubit count to solve real-world trading problems on their own. The hybrid quantum-classical paradigm addresses this limitation head-on by combining the strengths of both computational models into a single, practical pipeline.

In a hybrid quantum-classical system, a classical computer handles the tasks it excels at — data ingestion, preprocessing, optimization loop control, and final decision-making — while a quantum processor tackles specific subroutines where quantum mechanics offers a potential advantage, such as exploring exponentially large feature spaces, sampling from complex distributions, or performing certain linear algebra operations more efficiently. The classical computer orchestrates the workflow, delegates targeted computations to the quantum device, collects the results, and integrates them into a broader machine learning or optimization framework.

For trading specifically, this paradigm is compelling because financial markets generate massive volumes of classical data that must be cleaned, normalized, and feature-engineered before any model — quantum or classical — can process it. At the same time, the combinatorial complexity of portfolio optimization, the high-dimensional correlations between assets, and the non-linear dynamics of price movements are precisely the kinds of problems where quantum subroutines may provide an edge. By treating the quantum processor as a specialized accelerator within a classical pipeline, hybrid approaches can deliver practical value today rather than waiting for fault-tolerant quantum computers that may be years away.

This chapter explores the architecture, mathematics, and implementation of hybrid quantum-classical systems for trading. We build a complete pipeline in Rust that fetches live market data from Bybit, performs classical feature engineering, passes the features through a parameterized quantum circuit for transformation, and uses a classical postprocessing layer for trading signal generation. The quantum circuit parameters are optimized using the Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm, a gradient-free optimizer well-suited to the noisy landscapes characteristic of quantum computations.

## 2. Architecture

The hybrid quantum-classical architecture follows a three-stage pipeline pattern that cleanly separates responsibilities between the classical and quantum components.

### 2.1 Classical Preprocessing

The first stage operates entirely on classical hardware. Raw market data — OHLCV candles, order book snapshots, trade streams — is ingested and transformed into a compact feature vector suitable for quantum processing. This stage includes:

- **Data normalization**: Scaling features to the range [0, pi] or [-pi, pi] for angle encoding into quantum gates
- **Dimensionality reduction**: Compressing high-dimensional feature spaces to match the available qubit count. PCA, autoencoders, or simple feature selection can be used
- **Feature engineering**: Computing technical indicators (RSI, MACD, Bollinger Bands), returns, volatility measures, and cross-asset correlations
- **Windowing**: Organizing time-series data into fixed-length windows for sequential processing

The output of this stage is a vector of real numbers with dimensionality matching the quantum circuit's input capacity (number of qubits times encoding depth).

### 2.2 Quantum Circuit Layer

The second stage runs on a quantum processor (or a classical simulator thereof). A parameterized quantum circuit — also called a variational ansatz — transforms the classical feature vector into a quantum state, manipulates that state through trainable gates, and produces measurement outcomes. The circuit has three sub-components:

- **Data encoding**: Classical features are encoded into the quantum state using rotation gates (RX, RY, RZ). Each feature value becomes the angle of a rotation gate applied to one qubit. This is sometimes called angle encoding or amplitude encoding depending on the strategy
- **Variational layers**: Parameterized gates (rotations and entangling operations like CNOT) create entanglement between qubits and introduce trainable parameters. Multiple layers of alternating rotation and entanglement gates increase the circuit's expressibility
- **Measurement**: All qubits are measured in the computational basis. The measurement outcomes are collected over multiple shots to estimate expectation values

The quantum circuit acts as a non-linear feature transformer. By entangling qubits that encode different market features, the circuit can implicitly compute correlations and higher-order interactions that would require exponentially many terms in a classical polynomial expansion.

### 2.3 Classical Postprocessing

The third stage takes the quantum measurement results — typically expectation values of Pauli-Z operators on each qubit — and feeds them into a classical model for final prediction. This can be as simple as a linear classifier or as complex as a neural network. The classical postprocessor:

- Interprets measurement statistics as transformed features
- Applies a trainable linear or non-linear layer to produce trading signals
- Computes the loss function (e.g., cross-entropy for direction prediction)
- Feeds gradients or loss values back to the optimizer

### 2.4 Optimization Loop

The optimizer sits outside the three-stage pipeline and iteratively adjusts the trainable parameters in both the quantum circuit and the classical postprocessing layer. For quantum parameters, gradient-free optimizers like COBYLA or SPSA are preferred because quantum gradients are expensive to compute (requiring the parameter-shift rule) and inherently noisy. The optimization loop repeats until convergence or a computational budget is exhausted.

```
                    +------------------+
                    |   Classical      |
  Raw Data ------->|   Preprocessing  |------+
                    +------------------+      |
                                              v
                    +------------------+      |
                    |   Quantum        |<-----+
                    |   Circuit        |
                    +------------------+
                           |
                           v
                    +------------------+
                    |   Classical      |
                    |   Postprocessing |----> Prediction
                    +------------------+
                           |
                           v
                    +------------------+
                    |   Optimizer      |----> Update Parameters
                    |   (SPSA/COBYLA)  |
                    +------------------+
```

## 3. Mathematical Foundation

### 3.1 Variational Quantum Circuits

A variational quantum circuit (VQC) is parameterized by a vector theta in R^p, where p is the number of trainable rotation angles. The circuit implements a unitary operation U(theta) on an n-qubit system. Given input features x in R^n encoded via a feature map S(x), the quantum state is:

```
|psi(x, theta)> = U(theta) S(x) |0>^n
```

The feature map S(x) typically uses product-of-rotations encoding:

```
S(x) = tensor_product_{i=1}^{n} RY(x_i)
```

where RY(alpha) = [[cos(alpha/2), -sin(alpha/2)], [sin(alpha/2), cos(alpha/2)]].

The variational ansatz U(theta) consists of L layers, each containing single-qubit rotations and two-qubit entangling gates:

```
U(theta) = product_{l=1}^{L} W_l(theta_l) V_l
```

where W_l contains parameterized RY and RZ rotations and V_l contains CNOT gates in a ladder or all-to-all pattern.

### 3.2 Cost Function and Expectation Values

The output of the circuit is obtained by measuring expectation values of observables. For a Pauli-Z measurement on qubit i:

```
f_i(x, theta) = <psi(x, theta)| Z_i |psi(x, theta)>
```

These expectation values, bounded in [-1, 1], serve as transformed features. The final prediction is:

```
y_hat = sigma(w^T f(x, theta) + b)
```

where sigma is the sigmoid function, w and b are classical parameters, and f is the vector of expectation values.

The cost function for binary classification (e.g., price direction) is the binary cross-entropy:

```
L(theta, w, b) = -1/N sum_{i=1}^{N} [y_i log(y_hat_i) + (1 - y_i) log(1 - y_hat_i)]
```

### 3.3 SPSA Optimizer

The Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm estimates the gradient using only two function evaluations per iteration, regardless of the number of parameters. This makes it ideal for quantum circuits where each function evaluation requires running the circuit many times.

At iteration k, SPSA:

1. Generates a random perturbation vector Delta_k where each component is +1 or -1 with equal probability
2. Evaluates the cost at theta_k + c_k * Delta_k and theta_k - c_k * Delta_k
3. Estimates the gradient as:

```
g_k = (L(theta_k + c_k * Delta_k) - L(theta_k - c_k * Delta_k)) / (2 * c_k) * Delta_k^{-1}
```

4. Updates parameters: theta_{k+1} = theta_k - a_k * g_k

The sequences a_k = a / (k + A)^alpha and c_k = c / k^gamma are chosen to satisfy convergence conditions. Typical values are alpha = 0.602, gamma = 0.101.

### 3.4 Barren Plateaus

A critical challenge in variational quantum circuits is the barren plateau phenomenon: as the number of qubits grows, the gradient of the cost function vanishes exponentially. For a random parameterized circuit on n qubits:

```
Var[partial L / partial theta_i] in O(1 / 2^n)
```

This means that for large circuits, the optimizer sees an essentially flat landscape and cannot find a descent direction. Mitigation strategies include:

- **Shallow circuits**: Limiting the depth to O(log n) layers
- **Local cost functions**: Using cost functions that depend on few-qubit observables rather than global properties
- **Structured ansatze**: Using problem-inspired circuit architectures rather than random ones
- **Parameter initialization**: Starting from parameters that produce states close to the solution
- **Layer-wise training**: Training one layer at a time before training the full circuit

For trading applications with 4-8 qubits, barren plateaus are not yet a severe problem, but awareness of this issue is important for scaling.

## 4. Trading Applications

### 4.1 Quantum-Enhanced Feature Extraction

The most immediately practical application is using a quantum circuit as a non-linear feature transformer. Classical features (returns, volatility, RSI) are encoded into the quantum state, processed through entangling gates, and measured. The measurement results capture correlations and non-linearities that a classical linear model would miss.

This approach works because an n-qubit circuit with L layers of entangling gates can implicitly represent functions in a feature space of dimension up to 2^n, far exceeding what classical polynomial feature expansion can achieve with the same number of parameters.

In trading, this is valuable for:
- Detecting non-obvious correlations between technical indicators
- Capturing regime changes through quantum interference patterns
- Creating compact, expressive feature representations from high-dimensional market data

### 4.2 Quantum Layers in Neural Networks

A quantum circuit can replace one or more layers in a classical neural network. The resulting "dressed quantum circuit" benefits from the expressive power of both paradigms:

- Classical layers handle initial feature extraction and final classification
- The quantum layer performs a transformation in exponentially large Hilbert space
- Back-propagation through the classical layers is standard; quantum parameters use SPSA or the parameter-shift rule

This pattern is particularly suited to scenarios where the dataset is small (as quantum circuits can generalize well with few parameters) or where specific non-linear relationships are suspected.

### 4.3 Hybrid Portfolio Optimization

Portfolio optimization — finding asset weights that maximize return for a given risk level — can be formulated as a quadratic unconstrained binary optimization (QUBO) problem. The hybrid approach:

1. Classical preprocessing: Estimate covariance matrices, expected returns, and constraints from historical data
2. Quantum subroutine: Use the Variational Quantum Eigensolver (VQE) or QAOA to find approximate solutions to the QUBO
3. Classical postprocessing: Round and refine the quantum solution, apply real-world constraints (transaction costs, position limits), and execute trades

For small portfolios (up to ~20 assets on current hardware), this can explore the solution space more efficiently than classical heuristics.

## 5. Design Patterns

### 5.1 When to Use Quantum vs. Classical

Use quantum components when:
- The problem has combinatorial structure (portfolio selection, order routing)
- Non-linear feature interactions are important but the number of features matches qubit count
- The dataset is small enough that quantum generalization advantage matters
- You need to sample from complex probability distributions

Keep it classical when:
- Data volume is large and needs efficient batch processing
- The feature space is low-dimensional and well-understood
- Latency requirements are below quantum circuit execution time
- The problem is well-solved by existing classical methods

### 5.2 Data Encoding Strategies

**Angle encoding** maps each feature to one rotation gate angle. One qubit per feature. Simple and effective for NISQ devices.

**Amplitude encoding** maps 2^n features into the amplitudes of n qubits. More compact but requires complex state preparation circuits.

**Basis encoding** maps binary features to computational basis states. Efficient for discrete variables.

For trading, angle encoding is recommended as the starting point because:
- Features are naturally real-valued (prices, returns, indicators)
- The number of relevant features (4-8) matches available qubit counts
- State preparation circuits are shallow, reducing noise

### 5.3 Measurement Interpretation

Quantum measurements produce probabilistic outcomes. To get reliable expectation values, the circuit must be run many times (shots). The number of shots determines the precision:

- 100 shots: rough estimates, fast iteration during development
- 1000 shots: reasonable precision for most trading applications
- 10000 shots: high precision, but slow on real quantum hardware

In simulation, exact expectation values can be computed without sampling, which is the approach we use in our Rust implementation.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete hybrid quantum-classical pipeline for trading. The key components are:

### 6.1 Quantum Circuit Simulation

We simulate a parameterized quantum circuit using statevector representation. Each qubit is represented by a 2-element complex vector, and multi-qubit states are formed through tensor products. The circuit applies:

1. **RY encoding gates**: Each input feature is encoded as a Y-rotation angle on one qubit
2. **Variational layers**: Parameterized RY rotations followed by CNOT entangling gates
3. **Measurement**: Expectation values of Pauli-Z on each qubit are computed analytically from the statevector

The `QuantumLayer` struct manages the circuit parameters and provides a `forward` method that takes classical features and returns expectation values.

### 6.2 Classical Layers

The `ClassicalPreprocessor` normalizes input features to the range [0, pi] using min-max scaling, suitable for angle encoding. The `ClassicalPostprocessor` applies a linear transformation followed by sigmoid activation to produce a probability output.

### 6.3 SPSA Optimizer

The `SPSAOptimizer` implements the SPSA algorithm with configurable hyperparameters. It maintains the learning rate and perturbation schedules and provides a `step` method that performs one optimization iteration.

### 6.4 End-to-End Training

The `HybridModel` struct combines all components and provides `train` and `predict` methods. Training iterates over the dataset, computing forward passes through the full pipeline and updating parameters via SPSA.

```rust
// Simplified usage pattern:
let model = HybridModel::new(num_features, num_qubits, num_layers);
model.train(&features, &labels, num_epochs);
let predictions = model.predict(&test_features);
```

### 6.5 Key Implementation Details

**Complex number arithmetic**: Quantum states require complex number operations. We implement a minimal `Complex` struct with multiplication, addition, and magnitude operations.

**Tensor products**: Multi-qubit states are computed via Kronecker products of individual qubit states. For n qubits, the statevector has 2^n complex amplitudes.

**CNOT gates**: The controlled-NOT gate operates on the full 2^n dimensional statevector, flipping the target qubit conditioned on the control qubit being |1>.

**Pauli-Z expectation**: For qubit i, we sum |amplitude|^2 for basis states where qubit i is |0> (contributing +1) and subtract the sum for states where qubit i is |1>.

## 7. Bybit Data Integration

The implementation fetches real market data from the Bybit public API. The `BybitClient` struct provides methods to:

- Fetch OHLCV (kline) data for any trading pair and timeframe
- Parse the JSON response into structured candle data
- Compute technical features: returns, volatility (rolling standard deviation), RSI, and a simple moving average ratio

The API endpoint used is:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

This returns hourly candles without authentication requirements, making it suitable for research and backtesting.

Features computed from raw candle data:
1. **Log returns**: ln(close_t / close_{t-1})
2. **Volatility**: Rolling 10-period standard deviation of returns
3. **RSI**: 14-period Relative Strength Index, normalized to [0, 1]
4. **SMA ratio**: Close price divided by 20-period simple moving average, capturing trend

These four features map naturally to a 4-qubit quantum circuit, providing a clean demonstration of the hybrid pipeline.

## 8. Key Takeaways

1. **Hybrid is practical today**: While standalone quantum computing for trading remains aspirational, hybrid quantum-classical pipelines can be built and tested now using quantum simulators, and deployed on real quantum hardware as it matures.

2. **Architecture matters**: The classical-quantum-classical sandwich pattern cleanly separates concerns. Classical components handle data engineering and decision-making; quantum components provide non-linear feature transformation.

3. **SPSA is essential**: Gradient-free optimizers like SPSA are critical for quantum parameter optimization because quantum gradients are expensive and noisy. SPSA's two-evaluation-per-iteration cost makes it uniquely suited to this setting.

4. **Start small**: With 4-8 qubits and 2-3 variational layers, hybrid models can already demonstrate meaningful feature extraction. Scaling beyond this requires careful attention to barren plateaus and circuit depth.

5. **Data encoding is a design choice**: Angle encoding provides the simplest and most robust mapping from financial features to quantum states. Amplitude encoding is more compact but harder to prepare.

6. **Quantum advantage is problem-dependent**: Not every trading problem benefits from quantum subroutines. Combinatorial optimization (portfolio selection) and non-linear feature extraction are the most promising near-term applications.

7. **Simulation enables development**: Classical simulation of quantum circuits allows full development, testing, and validation of hybrid trading systems without quantum hardware access. The Rust implementation in this chapter runs entirely on classical hardware.

8. **Integration with existing systems**: The hybrid approach is designed to plug into existing trading infrastructure. The quantum circuit is just another feature transformer or optimizer in the pipeline — it does not require replacing the entire system.

The field of quantum machine learning for trading is evolving rapidly. By understanding the hybrid paradigm and building practical implementations today, traders and quant developers position themselves to leverage quantum advantage as hardware improves. The key is to identify specific subroutines where quantum processing adds value and to engineer clean interfaces between classical and quantum components.
