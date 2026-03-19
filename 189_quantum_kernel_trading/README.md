# Chapter 189: Quantum Kernel Trading

## 1. Introduction

Quantum computing promises to revolutionize many fields, and financial markets represent one of the most compelling application domains. At the heart of this revolution lies a concept called **quantum kernels** -- mathematical functions that leverage quantum mechanical principles to detect patterns in data that classical methods struggle to find.

In classical machine learning, kernel methods such as Support Vector Machines (SVMs) have long been used for classification and regression tasks. The kernel trick allows algorithms to operate in high-dimensional feature spaces without explicitly computing coordinates in those spaces. Quantum kernels extend this idea by using quantum circuits as feature maps, projecting data into exponentially large Hilbert spaces where previously invisible patterns become separable.

For financial markets, this capability is particularly relevant. Price movements are driven by complex, non-linear interactions among thousands of variables -- order flow dynamics, macroeconomic shifts, sentiment cascades, and regime transitions. Classical feature engineering can capture some of these relationships, but quantum kernels offer a fundamentally different approach: encoding financial data into quantum states and measuring similarity in a space whose dimensionality grows exponentially with the number of qubits.

This chapter explores the theory behind quantum kernels, their mathematical foundations, and how to apply them to real trading problems. We build a complete implementation in Rust that simulates quantum kernel computations and applies them to market regime detection using live data from the Bybit exchange.

## 2. Mathematical Foundation

### 2.1 Quantum Feature Maps

A quantum feature map is a parameterized quantum circuit $U(\mathbf{x})$ that maps a classical data point $\mathbf{x} \in \mathbb{R}^d$ to a quantum state in a $2^n$-dimensional Hilbert space:

$$\mathbf{x} \mapsto |\phi(\mathbf{x})\rangle = U(\mathbf{x})|0\rangle^{\otimes n}$$

where $n$ is the number of qubits and $|0\rangle^{\otimes n}$ is the initial state of all qubits set to zero. The unitary operator $U(\mathbf{x})$ encodes the classical features into the quantum state through rotations and entanglement operations.

The key insight is that while the input data lives in $\mathbb{R}^d$, the quantum state lives in $\mathbb{C}^{2^n}$ -- a space whose dimension grows exponentially with the number of qubits. This exponential blowup is precisely what gives quantum kernels their potential advantage over classical methods.

### 2.2 The Kernel Trick in Hilbert Space

In classical kernel methods, the kernel function $K(\mathbf{x}, \mathbf{y})$ computes the inner product of data points mapped to a feature space:

$$K(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle$$

For quantum kernels, we define the **fidelity kernel** as the squared overlap between two quantum states:

$$K(\mathbf{x}, \mathbf{y}) = |\langle \phi(\mathbf{x}) | \phi(\mathbf{y}) \rangle|^2$$

This quantity has a direct physical interpretation: it is the probability that a quantum state prepared as $|\phi(\mathbf{x})\rangle$ would be measured as $|\phi(\mathbf{y})\rangle$. When $\mathbf{x} = \mathbf{y}$, the kernel equals 1. When the data points are maximally different in quantum state space, the kernel approaches 0.

The fidelity kernel satisfies the requirements of a valid kernel function (Mercer's condition): it is symmetric and positive semi-definite. This means it can be used as a drop-in replacement for any classical kernel in algorithms like SVMs.

### 2.3 Computing the Inner Product

The inner product $\langle \phi(\mathbf{x}) | \phi(\mathbf{y}) \rangle$ is computed as:

$$\langle \phi(\mathbf{x}) | \phi(\mathbf{y}) \rangle = \langle 0|^{\otimes n} U^\dagger(\mathbf{x}) U(\mathbf{y}) |0\rangle^{\otimes n}$$

In simulation, this means we construct the state vectors for both inputs, then compute their complex inner product. On real quantum hardware, this would be estimated through repeated measurements using the **swap test** or **compute-uncompute** approach.

### 2.4 Quantum Kernel Alignment

Not all quantum kernels are equally useful for a given task. **Quantum kernel alignment** measures how well a kernel matrix matches the ideal kernel defined by the labels:

$$\text{KA}(K, K^*) = \frac{\langle K, K^* \rangle_F}{\|K\|_F \|K^*\|_F}$$

where $K^*_{ij} = y_i y_j$ is the ideal kernel matrix, $\langle \cdot, \cdot \rangle_F$ is the Frobenius inner product, and $\|\cdot\|_F$ is the Frobenius norm. A kernel alignment close to 1 indicates the quantum kernel is well-suited for the classification task.

In practice, one can optimize the parameters of the quantum feature map circuit to maximize kernel alignment with the training data, effectively performing a form of quantum feature engineering.

## 3. Trading Application

### 3.1 Market Regime Detection

Financial markets alternate between distinct regimes: low-volatility trending periods, high-volatility mean-reverting periods, and crisis periods with extreme tail events. Detecting these regimes is crucial for strategy selection -- a momentum strategy thrives in trending markets but suffers during mean reversion, and vice versa.

Quantum kernels offer a natural framework for regime detection because:

1. **Non-linear separability**: Market regimes are defined by complex interactions among features (volatility clustering, correlation breakdown, volume spikes). Quantum feature maps project these into spaces where regimes become linearly separable.
2. **Expressiveness**: With $n$ qubits, the feature space has dimension $2^n$, allowing the kernel to capture interactions among all subsets of features simultaneously.
3. **Noise resilience**: The squared fidelity kernel $|\langle \phi(\mathbf{x})|\phi(\mathbf{y})\rangle|^2$ provides a smooth similarity measure that is naturally robust to small perturbations in the input data.

### 3.2 Non-Linear Pattern Recognition in Price Data

Traditional technical analysis relies on hand-crafted indicators (RSI, MACD, Bollinger Bands) that capture specific patterns. Quantum kernels can discover patterns that span multiple features simultaneously. For example, a quantum kernel might identify that a particular combination of low volume, narrowing Bollinger Bands, and declining RSI reliably precedes a volatility expansion -- a pattern too complex for any single classical indicator.

The workflow for quantum kernel-based pattern recognition is:

1. **Feature engineering**: Compute financial features from raw OHLCV data (returns, volatility, volume ratios, technical indicators).
2. **Quantum encoding**: Map features to quantum states using a parameterized circuit.
3. **Kernel computation**: Build the kernel matrix $K_{ij} = |\langle \phi(\mathbf{x}_i)|\phi(\mathbf{x}_j)\rangle|^2$ for all pairs of data points.
4. **Classification**: Use the kernel matrix with an SVM or kernel perceptron to classify market regimes.
5. **Prediction**: For new data, compute kernel values against the training set and classify.

### 3.3 Support Vector Classification with Quantum Kernels

An SVM with a quantum kernel solves the same optimization problem as a classical SVM:

$$\min_{\alpha} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) - \sum_i \alpha_i$$

subject to $0 \leq \alpha_i \leq C$ and $\sum_i \alpha_i y_i = 0$. The only difference is that $K$ is now the quantum fidelity kernel. The decision function for a new point $\mathbf{x}$ is:

$$f(\mathbf{x}) = \text{sign}\left(\sum_i \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

In our implementation, we use a simplified kernel perceptron approach that iteratively updates weights based on misclassified samples, avoiding the complexity of full SMO while still leveraging the quantum kernel's discriminative power.

## 4. Quantum Feature Map Design

### 4.1 Angle Encoding

Angle encoding maps each classical feature to a rotation angle on a separate qubit:

$$U(\mathbf{x}) = \bigotimes_{i=1}^{n} R_Y(x_i) = \bigotimes_{i=1}^{n} \begin{pmatrix} \cos(x_i/2) & -\sin(x_i/2) \\ \sin(x_i/2) & \cos(x_i/2) \end{pmatrix}$$

This produces the state:

$$|\phi(\mathbf{x})\rangle = \bigotimes_{i=1}^{n} \left[\cos(x_i/2)|0\rangle + \sin(x_i/2)|1\rangle\right]$$

Angle encoding requires $n$ qubits for $n$ features and produces states in a $2^n$-dimensional space. It is the simplest encoding and serves as our primary approach.

The fidelity kernel under angle encoding has a closed-form expression:

$$K(\mathbf{x}, \mathbf{y}) = \prod_{i=1}^{n} \cos^2\left(\frac{x_i - y_i}{2}\right)$$

This product structure means the kernel is sensitive to differences in every feature dimension, and a large difference in any single feature drives the kernel toward zero.

### 4.2 Amplitude Encoding

Amplitude encoding represents the entire feature vector as the amplitudes of a quantum state:

$$|\phi(\mathbf{x})\rangle = \frac{1}{\|\mathbf{x}\|} \sum_{i=0}^{2^n - 1} x_i |i\rangle$$

This encoding is exponentially more compact -- only $\lceil \log_2 d \rceil$ qubits are needed for $d$ features -- but requires more complex circuits to prepare. In simulation, it is straightforward to implement directly.

### 4.3 Encoding Financial Features

Financial features require preprocessing before quantum encoding:

1. **Normalization**: Features must be scaled to $[0, \pi]$ for angle encoding or normalized to unit length for amplitude encoding.
2. **Feature selection**: Since each qubit adds exponential cost to simulation, we select the most informative features: log returns, rolling volatility, relative volume, and momentum.
3. **Temporal encoding**: We can encode sequences of features across time steps, creating a richer representation that captures temporal dynamics.

## 5. Implementation Walkthrough

Our Rust implementation consists of several key components. Let us walk through the architecture and core algorithms.

### 5.1 Quantum State Simulation

We represent quantum states as vectors of complex amplitudes with $2^n$ entries. The `QuantumState` struct manages this representation:

```rust
pub struct QuantumState {
    pub amplitudes: Vec<Complex>,
    pub num_qubits: usize,
}
```

Each amplitude is a complex number with real and imaginary parts. The state must satisfy the normalization condition: the sum of squared magnitudes equals 1.

### 5.2 Quantum Feature Map

The `QuantumFeatureMap` applies angle encoding to transform classical feature vectors into quantum states. For each feature, it applies a Y-rotation to the corresponding qubit:

```rust
pub fn encode(&self, features: &[f64]) -> QuantumState {
    // Start from |0...0> state
    // Apply R_Y(x_i) to qubit i for each feature x_i
    // Result: tensor product of single-qubit states
}
```

The encoding scales each feature to the range $[0, \pi]$ before applying rotations, ensuring the full Bloch sphere is utilized.

### 5.3 Fidelity Kernel

The core kernel computation calculates $K(\mathbf{x}, \mathbf{y}) = |\langle \phi(\mathbf{x})|\phi(\mathbf{y})\rangle|^2$:

```rust
pub fn fidelity_kernel(state1: &QuantumState, state2: &QuantumState) -> f64 {
    let inner = inner_product(&state1.amplitudes, &state2.amplitudes);
    inner.norm_squared()  // |<phi(x)|phi(y)>|^2
}
```

For a dataset of $N$ points, we construct the full $N \times N$ kernel matrix, which is then passed to the classifier.

### 5.4 Kernel Perceptron Classifier

We implement a kernel perceptron that operates directly on the precomputed kernel matrix:

```rust
pub fn train(&mut self, kernel_matrix: &Array2<f64>, labels: &[f64], epochs: usize) {
    for _ in 0..epochs {
        for i in 0..n_samples {
            let prediction = self.predict_with_kernel(kernel_matrix, labels, i);
            if prediction != labels[i] {
                self.alphas[i] += 1.0;
            }
        }
    }
}
```

This algorithm updates a weight for each training point whenever it is misclassified, building a decision boundary in the quantum feature space.

### 5.5 Market Regime Labeling

We define market regimes based on rolling volatility:

- **High volatility regime (+1)**: Rolling standard deviation of returns exceeds the median.
- **Low volatility regime (-1)**: Rolling standard deviation is below the median.

This simple labeling scheme captures the most fundamental market regime distinction and provides clear training targets for the quantum kernel classifier.

## 6. Bybit Data Integration

Our implementation fetches real market data from the Bybit exchange using their public REST API. The data pipeline works as follows:

### 6.1 API Endpoint

We use the Bybit V5 API endpoint for kline (candlestick) data:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

This returns OHLCV data for BTCUSDT perpetual futures at 1-hour intervals.

### 6.2 Data Processing Pipeline

1. **Fetch**: HTTP GET request to Bybit API, parse JSON response.
2. **Parse**: Extract timestamp, open, high, low, close, and volume fields.
3. **Feature engineering**:
   - Log returns: $r_t = \ln(C_t / C_{t-1})$
   - Rolling volatility: $\sigma_t = \text{std}(r_{t-w+1}, \ldots, r_t)$ with window $w = 20$
   - Relative volume: $v_t = V_t / \text{mean}(V_{t-w+1}, \ldots, V_t)$
   - Momentum: $m_t = C_t / C_{t-w} - 1$
4. **Normalize**: Scale all features to $[0, \pi]$ for quantum angle encoding.
5. **Label**: Assign regime labels based on volatility threshold.

### 6.3 Error Handling

The implementation uses Rust's `anyhow` crate for error handling, providing clean error propagation from network failures, JSON parsing errors, and numerical issues.

## 7. Key Takeaways

1. **Quantum kernels extend classical kernel methods** by using quantum circuits as feature maps, projecting data into exponentially large Hilbert spaces where complex patterns become linearly separable.

2. **The fidelity kernel** $K(\mathbf{x}, \mathbf{y}) = |\langle \phi(\mathbf{x})|\phi(\mathbf{y})\rangle|^2$ is a valid Mercer kernel that measures quantum state overlap, providing a natural similarity metric for classification.

3. **Angle encoding** offers a simple yet effective way to map financial features to quantum states, with each feature controlling a qubit rotation. The resulting kernel captures non-linear interactions across all feature combinations.

4. **Market regime detection** is a natural application for quantum kernels because regimes are defined by complex multi-feature interactions that benefit from high-dimensional feature spaces.

5. **Simulation is practical today**: While real quantum hardware is still noisy and limited, quantum kernel methods can be simulated classically for moderate numbers of qubits (up to ~20-25 qubits), allowing researchers to develop and validate quantum trading strategies now.

6. **Quantum kernel alignment** provides a principled way to evaluate and optimize quantum feature maps for specific trading tasks, bridging the gap between circuit design and financial objective.

7. **Integration with real data** through exchanges like Bybit demonstrates that quantum kernel methods can be applied to live market conditions, not just academic benchmarks.

8. **The quantum advantage question remains open**: For the system sizes simulable today, quantum kernels may not outperform well-tuned classical methods. However, they provide a framework for leveraging future quantum hardware and can already offer unique perspectives on market structure through their exponentially rich feature spaces.
