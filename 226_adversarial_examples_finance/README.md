# Chapter 226: Adversarial Examples in Finance

## 1. Introduction

Adversarial examples represent one of the most fascinating and dangerous vulnerabilities in modern machine learning systems. In the context of financial markets, an adversarial example is a carefully crafted perturbation of input data — so small that it appears indistinguishable from legitimate market data — that causes a trading model to produce dramatically incorrect predictions or decisions.

The concept originated in computer vision, where researchers demonstrated that adding imperceptible noise to an image could cause a state-of-the-art classifier to misidentify objects with high confidence. In finance, the stakes are considerably higher. A trading model that misclassifies a market regime, misprices an asset, or generates incorrect buy/sell signals due to adversarial manipulation can result in catastrophic financial losses.

Financial markets are inherently adversarial environments. Unlike static image classification tasks, trading involves active participants who have direct financial incentives to exploit weaknesses in other participants' models. Market manipulation — spoofing, wash trading, quote stuffing — can be understood as a form of adversarial attack on other traders' algorithms. This chapter formalizes this connection and explores both the attack surface and the defense mechanisms available to quantitative traders.

The key insight is that machine learning models used in trading — whether they predict price direction, estimate volatility, or classify market regimes — are vulnerable to the same classes of adversarial attacks that plague computer vision and natural language processing models. Understanding these vulnerabilities is essential for building robust trading systems.

We will explore the mathematical foundations of adversarial examples, catalog the specific threat vectors relevant to financial data, implement attack algorithms in Rust for performance-critical applications, and connect this chapter to the defense techniques covered in Chapters 221-225.

## 2. Mathematical Foundation

### Threat Models and Perturbation Budgets

An adversarial example is formally defined as follows. Given a model `f`, an input `x`, and its true label `y`, an adversarial example `x' = x + delta` satisfies two conditions:

1. `f(x') != y` (the model is fooled)
2. `||delta||_p <= epsilon` (the perturbation is bounded)

The choice of L_p norm defines the threat model:

**L_inf (Maximum Norm):** The perturbation is bounded by `max(|delta_i|) <= epsilon`. Each feature can be changed by at most epsilon. In financial terms, this means no single feature (price, volume, indicator) is altered by more than a specified amount. This is the most commonly used threat model because it allows small changes to many features simultaneously, which mirrors realistic market noise.

**L_2 (Euclidean Norm):** The total perturbation energy is bounded: `sqrt(sum(delta_i^2)) <= epsilon`. This allows larger changes to a few features as long as the overall perturbation magnitude remains small. In financial data, this corresponds to scenarios where a few data points experience noticeable deviations while most remain unchanged.

**L_1 (Manhattan Norm):** The sum of absolute perturbations is bounded: `sum(|delta_i|) <= epsilon`. This produces sparse perturbations — most features remain unchanged while a few are significantly altered. This models targeted manipulation of specific data points, such as spoofing a single price level.

### Perturbation Budgets in Finance

Setting appropriate epsilon values for financial data requires domain expertise. Unlike image pixels that range from 0 to 255, financial features span diverse scales:

- **Price data (OHLCV):** Perturbation budgets should reflect realistic tick sizes and market microstructure noise. For BTC/USDT, a perturbation of 0.1% of the current price (roughly $50 at $50,000) represents a realistic noise level.
- **Volume data:** Volume is inherently noisy, with natural variations of 10-50% between similar periods. Perturbation budgets can be proportionally larger.
- **Technical indicators:** Since indicators are derived from price and volume, their perturbation budgets should be consistent with underlying data noise.
- **Order book data:** Bid-ask spreads, depth imbalances, and queue positions all have natural variability that defines reasonable perturbation bounds.

### White-Box vs Black-Box Attacks

**White-box attacks** assume the adversary has full access to the model architecture, weights, and gradients. This is the strongest threat model and produces the most effective attacks. In trading, this corresponds to a competitor who has reverse-engineered your model.

**Black-box attacks** assume the adversary can only query the model and observe its outputs. This is more realistic in practice — a market participant can observe the behavior of other algorithms through their market impact but cannot directly inspect their internals. Black-box attacks typically use transfer attacks (adversarial examples generated for a surrogate model often fool the target) or query-based optimization.

## 3. Financial Adversarial Examples

### Manipulated OHLCV Data

OHLCV (Open, High, Low, Close, Volume) data forms the backbone of most technical trading systems. Adversarial perturbations to this data can take several forms:

- **Price manipulation:** Tiny adjustments to closing prices that shift technical indicator signals. For example, perturbing a close price by 0.05% could flip a moving average crossover signal.
- **Volume spoofing:** Artificial inflation or deflation of reported volume to trigger volume-based signals. A model that relies on volume breakouts is particularly vulnerable.
- **High/Low manipulation:** Adjusting the high or low of a candle changes volatility estimates, Bollinger Band widths, and range-based indicators.

### Spoofed Order Books

Order book data provides a real-time view of market supply and demand. Adversarial attacks on order book features include:

- **Layering:** Placing large orders at multiple price levels to create the illusion of strong support or resistance, then canceling before execution.
- **Depth imbalance manipulation:** Strategically placing and canceling orders to shift the bid-ask imbalance ratio that many ML models use as a predictive feature.
- **Queue position manipulation:** Flooding the order queue to affect execution priority signals.

### Fake News Injection

NLP-based trading models that process news feeds and social media are vulnerable to:

- **Synthetic news articles:** AI-generated fake news designed to trigger sentiment-based trading signals.
- **Social media manipulation:** Coordinated posting campaigns to shift social sentiment indicators.
- **Selective information injection:** Genuine but misleading information timed to maximize model confusion.

## 4. Attack Methods

### FGSM (Fast Gradient Sign Method)

The simplest and fastest gradient-based attack. Given input `x`, true label `y`, and loss function `L`:

```
x' = x + epsilon * sign(grad_x L(f(x), y))
```

FGSM computes the gradient of the loss with respect to the input and takes a single step in the direction that maximizes the loss. It is computationally cheap — requiring only one forward and one backward pass — making it ideal for real-time adversarial robustness evaluation.

In financial applications, FGSM reveals the most sensitive features: those whose gradients have the largest magnitude are the features where small perturbations have the greatest impact on model predictions.

### PGD (Projected Gradient Descent)

PGD is an iterative version of FGSM that produces stronger adversarial examples:

```
x_{t+1} = Proj_{B(x, epsilon)} (x_t + alpha * sign(grad_x L(f(x_t), y)))
```

At each iteration, PGD takes a small FGSM step with step size alpha, then projects the result back onto the epsilon-ball around the original input. After `T` iterations, this produces adversarial examples that are closer to the worst-case perturbation within the threat model.

PGD is considered the standard "first-order adversary" — if a model is robust to PGD attacks, it is generally robust to other first-order attacks.

### C&W (Carlini & Wagner) Attack

The C&W attack formulates adversarial example generation as an optimization problem:

```
minimize ||delta||_p + c * max(max_{i != y}(Z(x')_i) - Z(x')_y, -kappa)
```

where `Z(x')` are the logits (pre-softmax outputs) and `kappa` is a confidence parameter. This attack is slower but produces smaller perturbations than FGSM or PGD, making it harder to detect.

For financial models, C&W is particularly relevant because it finds the minimum perturbation needed to flip a trading decision, which quantifies how far from a decision boundary the current market state lies.

### DeepFool

DeepFool iteratively finds the closest decision boundary and pushes the input across it:

1. At each iteration, compute the minimal perturbation to cross the nearest class boundary.
2. Apply this perturbation and repeat until the prediction changes.

DeepFool produces minimal L_2 perturbations, which is useful for understanding model sensitivity — features with small DeepFool perturbations are the most vulnerable.

### AutoAttack

AutoAttack is an ensemble of attacks (APGD-CE, APGD-DLR, FAB, Square) that provides a reliable estimate of model robustness. It combines white-box and black-box methods and is parameter-free, making it the standard benchmark for adversarial robustness evaluation.

## 5. Real-World Implications

### Market Manipulation as Adversarial Attack

The connection between adversarial machine learning and market manipulation is profound. Traditional market manipulation techniques can be reinterpreted through the lens of adversarial examples:

**Spoofing** — placing orders with the intent to cancel before execution — is an adversarial perturbation of the order book feature space. The spoofer adds fake liquidity (perturbation) that changes the order book imbalance (input features) and causes other algorithms to make incorrect predictions about future price direction.

**Wash trading** — simultaneously buying and selling the same asset — is an adversarial perturbation of volume features. The inflated volume triggers volume-based signals in other traders' models without representing genuine market interest.

**Quote stuffing** — rapidly submitting and canceling orders to create congestion — is a denial-of-service attack that degrades the quality of input features available to competing algorithms.

**Pump and dump schemes** — coordinated buying to artificially inflate prices followed by selling — represent a sustained adversarial attack on price-based features that exploits momentum-following models.

### Regulatory Considerations

Market manipulation is illegal under regulations including the Dodd-Frank Act (US), Market Abuse Regulation (EU), and similar laws globally. Understanding adversarial attacks on trading models serves two purposes:

1. **Defense:** Building models that are robust to manipulation attempts.
2. **Detection:** Identifying when incoming data may have been adversarially manipulated.

### Financial Impact

The financial impact of adversarial vulnerability can be significant. A trading model that processes $10M in daily volume and is fooled by an adversarial perturbation even 1% of the time faces expected losses proportional to its position size and the magnitude of the incorrect signal. For high-frequency strategies, even brief periods of adversarial vulnerability can result in substantial losses.

## 6. Defense Overview

The defense techniques covered in Chapters 221-225 directly address adversarial vulnerability:

- **Chapter 221: Adversarial Training** — Training on adversarial examples to build robust models. This is the most direct defense: by including adversarial perturbations in the training data, the model learns to ignore them.
- **Chapter 222: Certified Defenses** — Providing provable guarantees that no perturbation within a given budget can change the model's prediction. This offers the strongest theoretical protection but may sacrifice some accuracy.
- **Chapter 223: Input Preprocessing** — Detecting and removing adversarial perturbations before they reach the model. Techniques include feature squeezing, input smoothing, and statistical anomaly detection.
- **Chapter 224: Ensemble Methods** — Using multiple diverse models to reduce adversarial transferability. An adversarial example crafted for one model is less likely to fool an ensemble of architecturally diverse models.
- **Chapter 225: Detection Methods** — Identifying adversarial inputs at inference time. This includes statistical tests, auxiliary detection networks, and consistency checks across multiple feature representations.

The combination of these techniques creates a layered defense that is significantly more robust than any single approach.

## 7. Implementation Walkthrough

Our Rust implementation provides a complete framework for generating and analyzing adversarial examples on financial data. The core components are:

### Neural Network with Gradient Computation

We implement a simple feedforward neural network with explicit gradient computation for the input layer. This enables all gradient-based attack methods. The network uses ReLU activations and supports configurable architecture.

```rust
// Core prediction with gradient computation
let output = network.forward(&input);
let gradient = network.input_gradient(&input, target);
```

### FGSM Attack

The FGSM implementation computes the sign of the input gradient and scales by epsilon:

```rust
let gradient = network.input_gradient(&input, target);
let perturbation = gradient.mapv(|g| g.signum() * epsilon);
let adversarial = &input + &perturbation;
```

### PGD Attack

PGD iterates FGSM steps and projects back onto the epsilon-ball:

```rust
for _ in 0..num_steps {
    let grad = network.input_gradient(&current, target);
    current = &current + &grad.mapv(|g| g.signum() * alpha);
    // Project back onto L_inf ball
    current = project_linf(&current, &original, epsilon);
}
```

### DeepFool-like Minimal Perturbation

Our DeepFool implementation iteratively finds the minimal perturbation that crosses the decision boundary by taking small steps along the gradient direction until the prediction changes.

### Perturbation Analysis

The framework analyzes which features are most vulnerable by computing feature-wise gradient magnitudes and perturbation sensitivities. This information is critical for prioritizing defenses.

## 8. Bybit Data Integration

The implementation fetches real market data from the Bybit API to test adversarial attacks on realistic inputs. We use the public klines endpoint:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

This returns OHLCV candles that serve as input features for the trading model. The data is normalized and fed through the neural network, then adversarial examples are generated and analyzed.

Key aspects of the integration:
- **Feature engineering:** Raw OHLCV data is transformed into returns, volatility estimates, and momentum features.
- **Normalization:** Features are z-score normalized so that perturbation budgets are comparable across features.
- **Realistic perturbation bounds:** Epsilon values are calibrated to match realistic market noise levels for each feature type.

## 9. Key Takeaways

1. **Financial markets are adversarial by nature.** Unlike benign deployment environments, trading models face active adversaries with financial incentives to exploit model weaknesses.

2. **Adversarial examples are real threats.** Small perturbations to market data — within the range of natural market noise — can cause trading models to make dramatically incorrect predictions.

3. **FGSM and PGD are practical evaluation tools.** Fast gradient-based attacks allow real-time assessment of model robustness and identification of vulnerable features.

4. **Market manipulation is adversarial attack.** Spoofing, wash trading, and other manipulation techniques can be formally understood as adversarial perturbations of input features.

5. **Perturbation budgets must be domain-specific.** Unlike image classification where epsilon = 8/255 is standard, financial perturbation budgets must reflect realistic market microstructure noise.

6. **Feature sensitivity analysis is essential.** Understanding which input features are most vulnerable to adversarial perturbation guides both model design and defense priorities.

7. **Layered defenses are necessary.** No single defense (adversarial training, certified robustness, input preprocessing, ensembles, detection) is sufficient alone. Combining techniques from Chapters 221-225 provides comprehensive protection.

8. **Rust enables performance-critical robustness testing.** Adversarial robustness evaluation requires many forward and backward passes; Rust's performance makes this feasible even for large-scale production systems.

9. **Continuous monitoring is required.** The adversarial threat landscape evolves as market participants adapt. Regular re-evaluation of model robustness is essential.

10. **Understanding attacks improves defense.** Studying attack methods does not promote market manipulation — it is essential for building trading systems that are resilient to manipulation attempts.
