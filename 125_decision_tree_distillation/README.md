# Decision Tree Distillation for Trading: Extracting Interpretable Rules from Complex Models

Decision Tree Distillation is a powerful model interpretability technique that extracts simple, human-readable decision rules from complex "black-box" machine learning models. In algorithmic trading, this approach bridges the gap between high-performing ensemble models or neural networks and the need for transparent, explainable trading decisions.

The core idea is elegantly simple: train a complex model that achieves high predictive accuracy, then train a simpler decision tree to mimic the complex model's predictions. The resulting decision tree "distills" the knowledge learned by the complex model into an interpretable format, revealing the underlying decision logic in terms of if-then rules.

In trading applications, Decision Tree Distillation addresses critical challenges:
- **Regulatory Compliance**: Financial regulations increasingly require explainability in algorithmic trading systems
- **Risk Management**: Understanding why a model recommends a trade helps assess and manage risk
- **Model Validation**: Distilled rules reveal whether a model has learned sensible patterns or spurious correlations
- **Operational Trust**: Traders and portfolio managers can review and approve trading logic expressed as clear rules

## Content

1. [Understanding Decision Tree Distillation](#understanding-decision-tree-distillation)
    * [The Knowledge Distillation Framework](#the-knowledge-distillation-framework)
    * [Why Distill to Decision Trees](#why-distill-to-decision-trees)
    * [Soft Labels vs Hard Labels](#soft-labels-vs-hard-labels)
2. [Distillation Algorithm](#distillation-algorithm)
    * [Mathematical Foundation](#mathematical-foundation)
    * [Teacher-Student Framework](#teacher-student-framework)
    * [Fidelity vs Interpretability Trade-off](#fidelity-vs-interpretability-trade-off)
3. [Decision Tree Distillation for Trading](#decision-tree-distillation-for-trading)
    * [Distilling Trading Signal Models](#distilling-trading-signal-models)
    * [Rule Extraction for Risk Management](#rule-extraction-for-risk-management)
    * [Regime-Specific Distillation](#regime-specific-distillation)
4. [Code Examples](#code-examples)
    * [Python Implementation](#python-implementation)
    * [Rust Implementation](#rust-implementation)
5. [Practical Applications](#practical-applications)
    * [Extracting Trading Rules from Neural Networks](#extracting-trading-rules-from-neural-networks)
    * [Ensemble Model Interpretation](#ensemble-model-interpretation)
    * [Real-Time Rule-Based Trading](#real-time-rule-based-trading)
6. [Backtesting Distilled Models](#backtesting-distilled-models)
7. [References](#references)

## Understanding Decision Tree Distillation

### The Knowledge Distillation Framework

Knowledge distillation, introduced by Hinton et al. (2015), is a technique for transferring knowledge from a large, complex model (the "teacher") to a smaller, simpler model (the "student"). The key insight is that the teacher model's predictions contain more information than just the final class labels—they encode the relative similarities and relationships between classes.

In the context of trading model interpretation, we adapt this framework:

```
Teacher Model (Complex):
  - Random Forest with 500 trees
  - Gradient Boosting Machine
  - Deep Neural Network
  - Ensemble of multiple models

Student Model (Interpretable):
  - Decision Tree (depth 3-10)
  - Rule List
  - Linear Model
```

The student model learns to mimic the teacher's behavior, not the original training labels. This is crucial because:
1. The teacher has already learned to ignore noise in the training data
2. The teacher's soft predictions provide richer information than hard labels
3. The student can achieve higher accuracy by learning from the teacher than from raw data

### Why Distill to Decision Trees

Decision trees are ideal student models for trading applications because they produce:

**Explicit Decision Rules**:
```
IF RSI_14 < 30 AND MACD_histogram > 0 AND Volume_ratio > 1.2
THEN Buy (confidence: 0.78)
```

**Hierarchical Feature Importance**:
- Root node split = most important feature
- Deeper splits = refinement conditions
- Path length = decision complexity

**Natural Thresholds**:
- Split points reveal critical values (e.g., RSI < 30)
- These thresholds can be validated against trading intuition
- Anomalous thresholds may indicate data issues

**Audit Trail**:
- Every prediction can be traced through the tree
- Compliance officers can review and approve specific paths
- Changes in model behavior are immediately visible in the rules

### Soft Labels vs Hard Labels

Traditional model training uses hard labels (0 or 1 for classification). Distillation typically uses soft labels—the teacher model's probability outputs:

**Hard Labels (Original Data)**:
```
Sample 1: Price went UP   → Label: 1
Sample 2: Price went DOWN → Label: 0
```

**Soft Labels (Teacher Predictions)**:
```
Sample 1: Teacher predicts UP with P(UP)=0.73   → Label: 0.73
Sample 2: Teacher predicts DOWN with P(DOWN)=0.85 → Label: 0.15
```

Soft labels preserve uncertainty information:
- A prediction of 0.51 vs 0.99 both map to "UP" in hard labels
- Soft labels distinguish between confident and uncertain predictions
- The student learns which cases are easy vs difficult for the teacher

In trading, this is particularly valuable because market predictions are inherently uncertain, and understanding when a model is confident vs uncertain is crucial for position sizing and risk management.

## Distillation Algorithm

### Mathematical Foundation

Given a teacher model `T` and training data `X`, the distillation process seeks a student decision tree `S` that minimizes:

```
L(S) = Σᵢ L_distill(S(xᵢ), T(xᵢ)) + λ · Complexity(S)
```

Where:
- `L_distill` is the distillation loss (e.g., cross-entropy, MSE)
- `T(xᵢ)` is the teacher's prediction (soft label) for sample `xᵢ`
- `Complexity(S)` is a regularization term (tree depth, number of leaves)
- `λ` controls the trade-off between fidelity and simplicity

For classification with soft labels, the distillation loss is often:

```
L_distill = -Σᵢ Σⱼ T(xᵢ)ⱼ · log(S(xᵢ)ⱼ)
```

Where `j` indexes over classes.

### Teacher-Student Framework

The distillation process follows these steps:

```
1. TRAIN teacher model on original data
   T = TrainComplex(X_train, y_train)

2. GENERATE soft labels using teacher
   y_soft = T.predict_proba(X_train)

3. TRAIN student decision tree on soft labels
   S = DecisionTree(max_depth=d)
   S.fit(X_train, y_soft)

4. EVALUATE fidelity and accuracy
   fidelity = agreement(S.predict(X_test), T.predict(X_test))
   accuracy = agreement(S.predict(X_test), y_test)
```

The fidelity metric measures how well the student mimics the teacher, while accuracy measures performance on the original task. High fidelity with high accuracy indicates successful distillation.

### Fidelity vs Interpretability Trade-off

There is a fundamental trade-off between how faithfully the student mimics the teacher and how interpretable the student remains:

```
Tree Depth    Fidelity    Interpretability
─────────────────────────────────────────
    2           60%           Excellent
    3           72%           Very Good
    5           85%           Good
    7           92%           Moderate
   10           96%           Poor
   15           99%           Very Poor
```

For trading applications, we typically target:
- **High-stakes decisions**: Depth 3-5 (must be fully reviewable)
- **Research/development**: Depth 7-10 (balance of insight and accuracy)
- **Automated systems**: Depth 5-7 (reviewable but detailed)

The optimal depth depends on:
1. Complexity of the underlying trading strategy
2. Regulatory requirements for explainability
3. Number of features in the model
4. Required fidelity threshold

## Decision Tree Distillation for Trading

### Distilling Trading Signal Models

Consider a complex ensemble model that predicts buy/sell signals based on technical indicators:

```python
# Teacher: Complex ensemble
teacher_features = ['RSI_14', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
                   'Volume_SMA_20', 'ATR_14', 'ADX_14', 'CCI_20', 'ROC_10']

# Teacher predicts: BUY (0.73), HOLD (0.15), SELL (0.12)

# Distilled Decision Tree reveals:
IF RSI_14 < 35:
    IF MACD > MACD_signal:
        IF Volume_SMA_20 > 1.5:
            → BUY (probability: 0.78)
        ELSE:
            → HOLD (probability: 0.52)
    ELSE:
        → HOLD (probability: 0.61)
ELSE IF RSI_14 > 70:
    IF ADX_14 > 25:
        → SELL (probability: 0.71)
    ELSE:
        → HOLD (probability: 0.58)
ELSE:
    → HOLD (probability: 0.64)
```

The distilled tree reveals that the complex ensemble primarily relies on:
1. RSI for identifying oversold/overbought conditions
2. MACD crossover for momentum confirmation
3. Volume for signal strength validation
4. ADX for trend strength in sell decisions

### Rule Extraction for Risk Management

Distilled decision trees provide explicit rules that can be monitored for risk management:

**Position Sizing Rules**:
```
Tree Path Confidence → Position Size
─────────────────────────────────────
     > 0.80          →  Full position
   0.65 - 0.80       →  75% position
   0.50 - 0.65       →  50% position
     < 0.50          →  No position
```

**Risk Alerts**:
```
IF distilled_rule uses feature NOT in approved_list:
    ALERT: Model may be using unexpected data

IF tree_depth_for_decision > max_approved_depth:
    ALERT: Decision path too complex for auto-execution

IF leaf_node_samples < minimum_samples:
    ALERT: Insufficient historical support for this rule
```

### Regime-Specific Distillation

Market regimes (trending, ranging, volatile) may require different decision logic. Regime-specific distillation trains separate trees for each regime:

```
Regime Detection:
  IF Volatility_20d > 0.3 AND ADX < 20:
      regime = "High Volatility, No Trend"
  ELIF ADX > 25:
      regime = "Trending"
  ELSE:
      regime = "Ranging"

Distilled Trees by Regime:

TRENDING REGIME:
  IF MACD > MACD_signal AND ADX > 30:
      → BUY
  IF MACD < MACD_signal AND ADX > 30:
      → SELL

RANGING REGIME:
  IF RSI < 30 AND Price < BB_lower:
      → BUY
  IF RSI > 70 AND Price > BB_upper:
      → SELL

VOLATILE REGIME:
  IF ATR_14 > 2 * ATR_14_avg:
      → REDUCE_POSITION
  ELSE:
      → Use normal rules with 50% size
```

## Code Examples

### Python Implementation

The Python implementation in `python/` provides:

- `python/model.py`: Core distillation implementation
  - `DistillationModel` class for training and extracting rules
  - Support for various teacher models (sklearn, XGBoost, LightGBM)
  - Rule extraction and visualization utilities

- `python/backtest.py`: Backtesting framework
  - Compare teacher vs distilled model performance
  - Track rule usage and confidence levels
  - Performance attribution by decision path

- `python/data_loader.py`: Data fetching utilities
  - Yahoo Finance integration for stocks
  - Bybit API for cryptocurrency data
  - Technical indicator computation

Example usage:

```python
from model import DistillationModel
from data_loader import load_stock_data, load_crypto_data

# Load data
stock_data = load_stock_data('AAPL', period='2y')
crypto_data = load_crypto_data('BTCUSDT', interval='1h', limit=5000)

# Train teacher model (complex ensemble)
from sklearn.ensemble import GradientBoostingClassifier
teacher = GradientBoostingClassifier(n_estimators=200, max_depth=10)
teacher.fit(X_train, y_train)

# Distill to interpretable decision tree
distiller = DistillationModel(max_depth=5, min_samples_leaf=50)
distiller.fit(X_train, teacher)

# Extract and display rules
rules = distiller.extract_rules()
for rule in rules:
    print(f"IF {rule.conditions} THEN {rule.prediction} (conf: {rule.confidence:.2f})")

# Evaluate fidelity
fidelity = distiller.fidelity_score(X_test, teacher)
print(f"Fidelity to teacher: {fidelity:.2%}")
```

### Rust Implementation

The Rust implementation in `rust/` provides high-performance distillation suitable for production:

- `rust/src/lib.rs`: Core library with distillation algorithms
- `rust/src/model/`: Decision tree and distillation implementations
- `rust/src/data/`: Data loading from Bybit API
- `rust/src/backtest/`: Backtesting framework
- `rust/examples/`: Runnable examples

Run the basic example:
```bash
cd rust
cargo run --example basic_distillation
```

Run the trading strategy example:
```bash
cargo run --example trading_strategy
```

## Practical Applications

### Extracting Trading Rules from Neural Networks

Neural networks often achieve the best predictive performance but are completely opaque. Distillation reveals what the network has learned:

```python
# Neural network teacher
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

teacher_nn = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # BUY, HOLD, SELL
])
teacher_nn.fit(X_train, y_train_onehot, epochs=100)

# Distill to decision tree
soft_labels = teacher_nn.predict(X_train)
distiller = DistillationModel(max_depth=6)
distiller.fit(X_train, soft_labels)

# The distilled tree reveals what patterns the neural network learned
print(distiller.tree_to_text())
```

### Ensemble Model Interpretation

Ensemble models (Random Forest, Gradient Boosting) combine hundreds of trees, making interpretation difficult. A single distilled tree summarizes the ensemble's collective wisdom:

```
Original Ensemble:
  - 500 decision trees
  - Each tree: depth 10-15
  - Total: ~5000+ decision nodes

Distilled Tree:
  - 1 decision tree
  - Depth: 5
  - Total: 31 decision nodes
  - Fidelity: 89%
```

The distilled tree captures 89% of the ensemble's behavior with 0.6% of the complexity.

### Real-Time Rule-Based Trading

Distilled trees enable efficient real-time trading systems:

```
Complex Model Pipeline:
  Input → Feature Engineering → Ensemble (500 trees) → Prediction
  Latency: 50-200ms

Distilled Model Pipeline:
  Input → Feature Engineering → Decision Tree (5 levels) → Prediction
  Latency: 1-5ms
```

The distilled model provides:
- 10-40x faster inference
- Deterministic execution path
- Easy hardware implementation (FPGA-compatible)
- Reduced memory footprint

## Backtesting Distilled Models

Backtesting should compare both accuracy and trading performance:

```python
from backtest import DistillationBacktester

backtester = DistillationBacktester(
    teacher_model=teacher,
    student_model=distiller,
    data=test_data,
    initial_capital=100000
)

results = backtester.run()

print("Performance Comparison:")
print(f"Teacher Sharpe Ratio: {results['teacher_sharpe']:.2f}")
print(f"Student Sharpe Ratio: {results['student_sharpe']:.2f}")
print(f"Agreement Rate: {results['agreement_rate']:.2%}")
print(f"Disagreement Impact: ${results['disagreement_pnl']:,.2f}")
```

Key metrics to track:
- **Fidelity**: How often teacher and student agree
- **Accuracy Retention**: Student accuracy vs teacher accuracy
- **Sharpe Ratio Retention**: Risk-adjusted return comparison
- **Disagreement Analysis**: When and why models disagree
- **Rule Stability**: Do extracted rules change over time?

## References

1. **Distilling a Neural Network Into a Soft Decision Tree**
   - Authors: Nicholas Frosst, Geoffrey Hinton
   - URL: https://arxiv.org/abs/1711.09784
   - Year: 2017
   - Introduces soft decision trees trained via distillation

2. **Distilling Knowledge from Deep Networks with Applications to Healthcare Domain**
   - Authors: Zhengping Che, Sanjay Purushotham, Robber Khemani, Yan Liu
   - URL: https://arxiv.org/abs/1512.03542
   - Year: 2015
   - Knowledge distillation for interpretable models

3. **Born Again Neural Networks**
   - Authors: Tommaso Furlanello et al.
   - URL: https://arxiv.org/abs/1805.04770
   - Year: 2018
   - Self-distillation and knowledge transfer

4. **Interpretable Machine Learning**
   - Author: Christoph Molnar
   - URL: https://christophm.github.io/interpretable-ml-book/
   - Comprehensive guide to interpretable ML techniques

5. **Model Compression**
   - Authors: Cristian Bucila, Rich Caruana, Alexandru Niculescu-Mizil
   - Year: 2006
   - Early work on model compression via distillation

## Data Sources

- **Yahoo Finance / yfinance**: Historical stock prices and fundamental data
- **Bybit API**: Cryptocurrency market data (OHLCV, order book)
- **Alpha Vantage**: Alternative stock data source
- **Kaggle**: Various financial datasets for experimentation

## Libraries and Tools

### Python
- `scikit-learn`: Decision tree implementation, ensemble models
- `xgboost`, `lightgbm`: Gradient boosting implementations
- `tensorflow`, `pytorch`: Neural network teachers
- `pandas`, `numpy`: Data manipulation
- `yfinance`: Yahoo Finance data API
- `matplotlib`, `graphviz`: Tree visualization

### Rust
- `ndarray`: N-dimensional arrays
- `linfa`: Machine learning toolkit (decision trees)
- `polars`: Fast DataFrames
- `reqwest`: HTTP client for API requests
- `serde`: Serialization/deserialization
- `plotters`: Visualization
