# Chapter 221: Adversarial Training for Robust Trading Models

## 1. Introduction

Machine learning models deployed in financial markets face a uniquely hostile environment. Unlike image classification or natural language processing, where adversarial inputs are largely theoretical concerns, trading models operate in an arena where adversarial actors — market manipulators, high-frequency front-runners, and sophisticated institutional players — actively seek to exploit predictable behavior. Adversarial training offers a principled framework for building models that remain robust under these conditions.

Adversarial training was originally developed in the deep learning community as a defense against adversarial examples — carefully crafted perturbations to inputs that cause models to make incorrect predictions. The seminal work by Goodfellow et al. (2014) on the Fast Gradient Sign Method (FGSM) demonstrated that neural networks are surprisingly brittle: imperceptibly small perturbations to input data can cause dramatic changes in model outputs. Madry et al. (2018) later formalized adversarial training as a robust optimization problem, establishing Projected Gradient Descent (PGD) as a strong attack and defense method.

In the context of algorithmic trading, adversarial training serves multiple purposes. First, it hardens models against deliberate market manipulation — spoofing, layering, and wash trading that inject misleading signals into order book data. Second, it improves robustness to the natural noise and non-stationarity of financial data, where distribution shifts occur regularly due to regime changes, policy announcements, and liquidity events. Third, it provides a systematic framework for stress-testing trading strategies against worst-case scenarios.

This chapter develops the mathematical foundations of adversarial training, explains why trading models are particularly vulnerable to adversarial perturbations, surveys the major adversarial training methods, and provides a complete Rust implementation integrated with live Bybit market data.

## 2. Mathematical Foundation

### Adversarial Examples

Given a model $f_\theta$ parameterized by weights $\theta$, an input $x$, and a true label $y$, an adversarial example is a perturbed input $x' = x + \delta$ such that:

$$f_\theta(x + \delta) \neq y \quad \text{while} \quad \|\delta\|_p \leq \epsilon$$

The perturbation $\delta$ is bounded by an $\ell_p$-norm ball of radius $\epsilon$, ensuring the adversarial example remains "close" to the original input. In trading, $\epsilon$ represents the magnitude of price manipulation or data noise that a model should tolerate.

### Min-Max Formulation

Adversarial training is formulated as a min-max optimization problem:

$$\min_\theta \max_{\|\delta\|_p \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y)$$

The inner maximization finds the worst-case perturbation $\delta^*$ that maximizes the loss, while the outer minimization updates the model parameters $\theta$ to minimize the loss on these worst-case inputs. This saddle-point formulation ensures the trained model performs well even under adversarial conditions.

### Fast Gradient Sign Method (FGSM)

FGSM is a single-step attack that generates adversarial examples by taking one step in the direction of the gradient of the loss with respect to the input:

$$\delta = \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f_\theta(x), y))$$

FGSM is computationally cheap (requires only one forward and one backward pass) but produces relatively weak adversarial examples. Its simplicity makes it useful for fast adversarial training and as a baseline attack.

### Projected Gradient Descent (PGD)

PGD is an iterative attack that applies multiple small FGSM steps, projecting back onto the $\epsilon$-ball after each step:

$$x^{(t+1)} = \Pi_{B_\epsilon(x)} \left( x^{(t)} + \alpha \cdot \text{sign}(\nabla_{x^{(t)}} \mathcal{L}(f_\theta(x^{(t)}), y)) \right)$$

where $\Pi_{B_\epsilon(x)}$ denotes projection onto the $\ell_\infty$-ball of radius $\epsilon$ centered at $x$, and $\alpha$ is the step size. PGD with random restarts is considered the strongest first-order attack and forms the basis of PGD Adversarial Training (PGD-AT).

### TRADES Objective

The TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization) method decomposes the adversarial training objective into two terms:

$$\min_\theta \mathcal{L}(f_\theta(x), y) + \beta \cdot \max_{\|\delta\| \leq \epsilon} D_{KL}(f_\theta(x) \| f_\theta(x + \delta))$$

The first term ensures accuracy on clean data, while the second term (weighted by $\beta$) penalizes the KL divergence between predictions on clean and adversarial inputs. This formulation explicitly trades off clean accuracy for adversarial robustness, with $\beta$ controlling the balance.

## 3. Why Trading Models Need Robustness

### Market Manipulation

Financial markets are rife with manipulation tactics that inject adversarial signals into data:

- **Spoofing**: Placing large orders with no intention of execution to create false impressions of supply/demand. These fake orders perturb order book features that models use for prediction.
- **Layering**: Placing multiple orders at different price levels to create artificial depth, then canceling them once the market moves.
- **Wash trading**: Simultaneously buying and selling to inflate volume statistics, misleading volume-based indicators.
- **Momentum ignition**: Aggressive trading to trigger momentum-following algorithms, then trading against the induced move.

A model trained only on "clean" historical data will learn to trust these manipulated signals, leading to predictable and exploitable behavior.

### Noisy Data

Financial data is inherently noisy. Tick data contains measurement artifacts, exchange clock synchronization errors, and stale quotes. Even without deliberate manipulation, the signal-to-noise ratio in financial time series is extremely low. Models must be robust to this baseline noise level to avoid overfitting to spurious patterns.

### Distribution Shift

Financial markets undergo continuous regime changes. Volatility regimes shift, correlations break down during crises, and market microstructure evolves as new participants and technologies enter. A model that performs well under one distribution may fail catastrophically when conditions change. Adversarial training provides a form of distributional robustness by training the model to perform well under worst-case perturbations within a neighborhood of the training distribution.

### Adversarial Market Participants

In competitive markets, other participants actively seek to exploit predictable algorithmic behavior. If a model's trading signals can be inferred from its market footprint, adversaries can front-run or trade against it. Adversarial training helps build models whose predictions are stable under the types of perturbations that adversarial participants might introduce.

## 4. Adversarial Training Methods

### PGD Adversarial Training (PGD-AT)

PGD-AT (Madry et al., 2018) is the standard adversarial training method. For each training batch:

1. Generate adversarial examples using PGD with $K$ steps
2. Compute the loss on the adversarial examples
3. Update model parameters via gradient descent on the adversarial loss

PGD-AT produces highly robust models but is computationally expensive — each training step requires $K$ forward-backward passes for the PGD attack plus one pass for the parameter update.

### TRADES

TRADES separates the goals of accuracy and robustness. The clean loss ensures the model learns correct predictions on unperturbed data, while the KL divergence term ensures predictions remain stable under perturbation. The hyperparameter $\beta$ controls the accuracy-robustness tradeoff: larger $\beta$ increases robustness at the cost of clean accuracy.

### Free Adversarial Training

Free adversarial training (Shafahi et al., 2019) reduces the computational overhead by simultaneously updating both the adversarial perturbation and model parameters in a single backward pass. Instead of running $K$ PGD steps per training step, it replays each minibatch $m$ times, accumulating perturbation updates across replays. This achieves comparable robustness to PGD-AT at the cost of standard training.

### You Only Propagate Once (YOPO)

YOPO (Zhang et al., 2019) exploits the observation that adversarial perturbations primarily affect the first layer of the network. It decouples the adversarial update from the full network backward pass, performing multiple perturbation updates using only the first-layer gradient while updating the full network less frequently. This further reduces computational cost.

## 5. Trading Applications

### Robust Price Prediction

Price prediction models must contend with noisy inputs — bid-ask spreads, volume spikes, and microstructure effects. An adversarially trained price predictor learns features that are stable under small perturbations to input features:

- If OHLCV data is perturbed by $\pm \epsilon$ (simulating noise or manipulation), the model's prediction should remain consistent
- Adversarial training acts as a strong regularizer, preventing the model from relying on fragile, noise-sensitive features

### Manipulation-Resistant Signals

Trading signals derived from order book data are particularly vulnerable to spoofing. An adversarially trained signal generator has seen worst-case perturbations to order book features during training and learns to discount potentially manipulated signals:

- Order book imbalance features become more robust when the model has been trained against adversarial perturbations that simulate spoofing
- Volume-weighted features become resistant to wash trading artifacts

### Stress-Testing Trading Strategies

Even without using adversarial training for model fitting, the PGD attack framework provides a systematic method for stress-testing strategies:

1. Define the input space (market conditions, features)
2. Define the perturbation budget (how much conditions can deviate)
3. Use PGD to find the worst-case scenario within the budget
4. Evaluate strategy performance under these worst-case conditions

This is more principled than traditional Monte Carlo stress testing because it actively searches for failure modes rather than sampling randomly.

## 6. Implementation Walkthrough (Rust)

Our Rust implementation provides a complete adversarial training framework for trading models. The core components are:

### Neural Network with Gradient Computation

We implement a simple feedforward neural network with explicit gradient computation. The network uses ReLU activations and supports forward and backward passes needed for both standard training and adversarial attacks.

```rust
pub struct NeuralNetwork {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
}
```

The backward pass computes gradients with respect to both parameters (for training) and inputs (for adversarial attacks). This dual gradient computation is the key enabler for adversarial training.

### FGSM Attack

The FGSM implementation computes input gradients in a single backward pass and applies the sign perturbation:

```rust
pub fn fgsm_attack(
    model: &NeuralNetwork,
    x: &Array2<f64>,
    y: &Array1<f64>,
    epsilon: f64,
) -> Array2<f64>
```

### PGD Attack

The PGD attack wraps FGSM in an iterative loop with projection:

```rust
pub fn pgd_attack(
    model: &NeuralNetwork,
    x: &Array2<f64>,
    y: &Array1<f64>,
    epsilon: f64,
    alpha: f64,
    num_steps: usize,
) -> Array2<f64>
```

### Adversarial Training Loop

The training loop alternates between generating adversarial examples and updating model parameters:

```rust
pub fn adversarial_train(
    model: &mut NeuralNetwork,
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    epochs: usize,
    epsilon: f64,
    learning_rate: f64,
)
```

### TRADES-Style Regularization

Our TRADES implementation adds a KL divergence penalty between clean and adversarial predictions:

```rust
pub fn trades_train(
    model: &mut NeuralNetwork,
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    epochs: usize,
    epsilon: f64,
    beta: f64,
    learning_rate: f64,
)
```

## 7. Bybit Data Integration

The implementation includes a Bybit API client that fetches real-time OHLCV data for any trading pair. The data pipeline:

1. **Fetch**: Retrieves kline (candlestick) data from Bybit's public API v5
2. **Parse**: Converts JSON responses into structured `Candle` objects
3. **Feature Engineering**: Computes normalized returns, volatility ratios, volume changes, and price range features
4. **Label Generation**: Creates binary labels based on future price direction

```rust
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>>
```

The feature engineering pipeline normalizes all inputs to similar scales, which is important for adversarial training since the perturbation budget $\epsilon$ is applied uniformly across features. Without normalization, features with larger magnitudes would be relatively less perturbed, creating an uneven robustness profile.

## 8. Key Takeaways

1. **Adversarial training is a min-max optimization**: The model is trained to minimize loss under worst-case perturbations, providing provable robustness guarantees within the perturbation budget.

2. **Trading models are naturally adversarial environments**: Market manipulation, noisy data, and competitive participants create adversarial conditions that standard training does not prepare models for.

3. **FGSM vs PGD tradeoff**: FGSM is fast but weak; PGD is strong but expensive. For production trading systems, free adversarial training or YOPO offer good compromises.

4. **TRADES balances accuracy and robustness**: The $\beta$ parameter in TRADES explicitly controls the tradeoff between clean accuracy and adversarial robustness — essential for trading where both matter.

5. **Adversarial training as regularization**: Even without explicit adversarial threats, adversarial training acts as a powerful regularizer that reduces overfitting and improves generalization to out-of-distribution market conditions.

6. **Stress testing via PGD**: The PGD framework provides a principled method for finding worst-case scenarios for any trading strategy, superior to random Monte Carlo simulations.

7. **Epsilon selection matters**: The perturbation budget $\epsilon$ should reflect realistic noise/manipulation levels in the target market. Too small and the model is not robust enough; too large and the model sacrifices too much clean accuracy.

8. **Robustness evaluation is essential**: Always evaluate models under attack at multiple $\epsilon$ values. A model that appears accurate on clean data may be completely fragile under even small perturbations.

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples. arXiv:1412.6572.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR.
- Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). Theoretically Principled Trade-off between Robustness and Accuracy. ICML.
- Shafahi, A., Najibi, M., Ghiasi, A., Xu, Z., Dickerson, J., Studer, C., Davis, L.S., Taylor, G., & Goldstein, T. (2019). Adversarial Training for Free! NeurIPS.
- Zhang, D., Zhang, T., Lu, Y., Zhu, Z., & Dong, B. (2019). You Only Propagate Once: Painless Adversarial Training Using Maximal Principle. NeurIPS.
