# Chapter 119: Rule Extraction Trading — Extracting Interpretable Rules from Black-Box Models

In this chapter, we explore rule extraction techniques that transform opaque black-box machine learning models into interpretable decision rules for trading applications. While neural networks and ensemble methods achieve excellent predictive performance, their lack of transparency creates challenges for risk management, regulatory compliance, and strategy validation. Rule extraction bridges this gap by distilling complex models into human-readable if-then rules.

We will learn how to extract decision rules from trained neural networks and gradient boosting models, evaluate rule fidelity and coverage, and deploy these rules for transparent algorithmic trading strategies. The chapter covers pedagogical methods (model-agnostic approaches), decompositional methods (architecture-specific techniques), and hybrid approaches that combine both paradigms.

## Content

1. [Rule Extraction: From Black-Box to Transparency](#rule-extraction-from-black-box-to-transparency)
2. [Pedagogical Rule Extraction Methods](#pedagogical-rule-extraction-methods)
   * [TREPAN Algorithm](#trepan-algorithm)
   * [Rule Extraction via Sequential Covering](#rule-extraction-via-sequential-covering)
3. [Decompositional Rule Extraction](#decompositional-rule-extraction)
   * [Decision Diagram Extraction from Neural Networks](#decision-diagram-extraction-from-neural-networks)
   * [Rule Extraction from Decision Trees and Ensembles](#rule-extraction-from-decision-trees-and-ensembles)
4. [Code Example: Building a Rule Extraction Pipeline](#code-example-building-a-rule-extraction-pipeline)
   * [Data Preparation: Stock and Crypto Data](#data-preparation-stock-and-crypto-data)
   * [Training Black-Box Models](#training-black-box-models)
   * [Extracting Rules from Neural Networks](#extracting-rules-from-neural-networks)
   * [Extracting Rules from Gradient Boosting](#extracting-rules-from-gradient-boosting)
5. [Rule Evaluation Metrics](#rule-evaluation-metrics)
6. [Code Example: Trading Strategy with Extracted Rules](#code-example-trading-strategy-with-extracted-rules)
   * [Rule-Based Signal Generation](#rule-based-signal-generation)
   * [Backtesting the Rule Strategy](#backtesting-the-rule-strategy)
7. [Rust Implementation for Production](#rust-implementation-for-production)

## Rule Extraction: From Black-Box to Transparency

Rule extraction is the process of deriving symbolic, human-interpretable knowledge from trained machine learning models. In trading applications, this serves several critical purposes:

- **Regulatory Compliance**: Financial regulators increasingly require explainability for algorithmic trading decisions
- **Risk Management**: Understanding why a model makes predictions helps identify potential failure modes
- **Strategy Validation**: Domain experts can verify that extracted rules align with market intuition
- **Debugging**: Rules reveal what patterns the model has learned, including spurious correlations

### Types of Rule Extraction

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Pedagogical** | Treats model as black-box, learns rules from input-output pairs | Model-agnostic, simple | May miss internal structure |
| **Decompositional** | Analyzes model architecture directly | Captures exact behavior | Architecture-specific |
| **Eclectic** | Combines pedagogical and decompositional | Best of both worlds | More complex |

## Pedagogical Rule Extraction Methods

Pedagogical methods treat the trained model as an oracle and extract rules by observing its input-output behavior.

### TREPAN Algorithm

TREPAN (Trees Paraphrasing Networks) builds a decision tree that mimics the neural network's behavior:

```python
def trepan_extract(model, X_train, max_depth=10):
    """
    Extract decision tree rules from any black-box model.

    Args:
        model: Trained black-box model with predict method
        X_train: Training features
        max_depth: Maximum tree depth

    Returns:
        Fitted decision tree that approximates the model
    """
    from sklearn.tree import DecisionTreeClassifier

    # Get model predictions as pseudo-labels
    y_pseudo = model.predict(X_train)

    # Train interpretable tree on pseudo-labels
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train, y_pseudo)

    return tree
```

### Rule Extraction via Sequential Covering

Sequential covering iteratively extracts rules that cover subsets of the data:

```python
def sequential_covering(model, X, feature_names, min_coverage=0.05):
    """
    Extract rules using sequential covering algorithm.
    """
    rules = []
    uncovered = np.ones(len(X), dtype=bool)

    while uncovered.sum() / len(X) > min_coverage:
        rule = find_best_rule(model, X[uncovered], feature_names)
        if rule is None:
            break
        rules.append(rule)
        uncovered = uncovered & ~rule.covers(X)

    return rules
```

## Decompositional Rule Extraction

Decompositional methods analyze the internal structure of the model to extract rules.

### Decision Diagram Extraction from Neural Networks

Based on the research paper "Extracting Rules from Neural Networks as Decision Diagrams" (arXiv:2104.06411), we can convert neural network computations into Binary Decision Diagrams (BDDs):

```python
class NeuralNetworkToRules:
    """
    Extract rules from neural networks using decision diagram conversion.
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def extract_layer_rules(self, layer_idx, threshold=0.5):
        """
        Extract rules from a specific layer by analyzing activation patterns.
        """
        weights = self.model.layers[layer_idx].get_weights()[0]
        biases = self.model.layers[layer_idx].get_weights()[1]

        rules = []
        for neuron_idx in range(weights.shape[1]):
            w = weights[:, neuron_idx]
            b = biases[neuron_idx]

            # Create rule: if sum(w_i * x_i) + b > threshold
            conditions = []
            for feat_idx, weight in enumerate(w):
                if abs(weight) > 0.1:  # Significant weight
                    conditions.append({
                        'feature': self.feature_names[feat_idx],
                        'weight': weight
                    })

            rules.append({
                'conditions': conditions,
                'bias': b,
                'threshold': threshold
            })

        return rules
```

### Rule Extraction from Decision Trees and Ensembles

For tree-based models, rules can be extracted directly from the tree structure:

```python
def extract_tree_rules(tree, feature_names):
    """
    Extract if-then rules from a decision tree.
    """
    tree_ = tree.tree_
    rules = []

    def recurse(node, path):
        if tree_.feature[node] != -2:  # Not a leaf
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            # Left branch: feature <= threshold
            left_path = path + [(feature, '<=', threshold)]
            recurse(tree_.children_left[node], left_path)

            # Right branch: feature > threshold
            right_path = path + [(feature, '>', threshold)]
            recurse(tree_.children_right[node], right_path)
        else:
            # Leaf node - create rule
            prediction = tree_.value[node].argmax()
            rules.append({'conditions': path, 'prediction': prediction})

    recurse(0, [])
    return rules
```

## Code Example: Building a Rule Extraction Pipeline

### Data Preparation: Stock and Crypto Data

We use dual data sources for comprehensive testing:

```python
import yfinance as yf
import pandas as pd
import numpy as np

def prepare_stock_data(ticker='SPY', period='2y'):
    """
    Download and prepare stock market data with technical indicators.
    """
    df = yf.download(ticker, period=period)

    # Technical indicators
    df['returns'] = df['Close'].pct_change()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['rsi'] = compute_rsi(df['Close'], 14)
    df['volatility'] = df['returns'].rolling(20).std()

    # Target: next day direction
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)

    return df.dropna()

def prepare_crypto_data(symbol='BTCUSDT', interval='1h', limit=1000):
    """
    Fetch cryptocurrency data from Bybit API.
    """
    import requests

    url = "https://api.bybit.com/v5/market/kline"
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add features
    df['returns'] = df['close'].pct_change()
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)

    return df.dropna()
```

### Training Black-Box Models

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

def train_models(X, y):
    """
    Train neural network and gradient boosting models.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Neural Network
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        max_iter=500,
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    return nn_model, gb_model, scaler, X_test, y_test
```

### Extracting Rules from Neural Networks

```python
def extract_nn_rules(nn_model, X_train, feature_names, max_rules=20):
    """
    Extract interpretable rules from neural network using TREPAN-like approach.
    """
    from sklearn.tree import DecisionTreeClassifier

    # Generate pseudo-labels from NN
    y_pseudo = nn_model.predict(X_train)

    # Fit interpretable tree
    tree = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        max_leaf_nodes=max_rules
    )
    tree.fit(X_train, y_pseudo)

    # Extract rules
    rules = extract_tree_rules(tree, feature_names)

    # Calculate fidelity
    tree_preds = tree.predict(X_train)
    fidelity = (tree_preds == y_pseudo).mean()

    return rules, fidelity, tree
```

### Extracting Rules from Gradient Boosting

```python
def extract_gb_rules(gb_model, feature_names, importance_threshold=0.05):
    """
    Extract rules from gradient boosting ensemble.
    """
    all_rules = []

    for tree_idx, tree in enumerate(gb_model.estimators_.ravel()):
        rules = extract_tree_rules(tree, feature_names)

        # Weight rules by tree importance
        for rule in rules:
            rule['tree_idx'] = tree_idx
            rule['weight'] = 1.0 / len(gb_model.estimators_)

        all_rules.extend(rules)

    # Filter by feature importance
    importances = gb_model.feature_importances_
    important_features = set(
        feature_names[i] for i, imp in enumerate(importances)
        if imp >= importance_threshold
    )

    filtered_rules = [
        rule for rule in all_rules
        if all(cond[0] in important_features for cond in rule['conditions'])
    ]

    return consolidate_rules(filtered_rules)
```

## Rule Evaluation Metrics

```python
def evaluate_rules(rules, model, X, y, feature_names):
    """
    Evaluate extracted rules against original model and ground truth.
    """
    metrics = {}

    # Fidelity: how well rules match model predictions
    model_preds = model.predict(X)
    rule_preds = apply_rules(rules, X, feature_names)
    metrics['fidelity'] = (model_preds == rule_preds).mean()

    # Accuracy: how well rules predict ground truth
    metrics['accuracy'] = (y == rule_preds).mean()

    # Coverage: fraction of samples covered by at least one rule
    coverage_mask = np.zeros(len(X), dtype=bool)
    for rule in rules:
        coverage_mask |= rule_covers(rule, X, feature_names)
    metrics['coverage'] = coverage_mask.mean()

    # Complexity: average number of conditions per rule
    metrics['avg_conditions'] = np.mean([
        len(rule['conditions']) for rule in rules
    ])

    # Rule count
    metrics['n_rules'] = len(rules)

    return metrics
```

## Code Example: Trading Strategy with Extracted Rules

### Rule-Based Signal Generation

```python
class RuleBasedStrategy:
    """
    Trading strategy using extracted rules.
    """

    def __init__(self, rules, feature_names):
        self.rules = rules
        self.feature_names = feature_names

    def generate_signal(self, features):
        """
        Generate trading signal based on extracted rules.

        Returns:
            1 for buy, -1 for sell, 0 for hold
        """
        buy_score = 0
        sell_score = 0

        for rule in self.rules:
            if self._rule_matches(rule, features):
                if rule['prediction'] == 1:  # Bullish
                    buy_score += rule.get('weight', 1.0)
                else:  # Bearish
                    sell_score += rule.get('weight', 1.0)

        # Generate signal based on score difference
        score_diff = buy_score - sell_score

        if score_diff > 0.5:
            return 1
        elif score_diff < -0.5:
            return -1
        else:
            return 0

    def _rule_matches(self, rule, features):
        """Check if all conditions in a rule are satisfied."""
        for feature, operator, threshold in rule['conditions']:
            feat_idx = self.feature_names.index(feature)
            value = features[feat_idx]

            if operator == '<=' and value > threshold:
                return False
            if operator == '>' and value <= threshold:
                return False

        return True

    def explain_signal(self, features):
        """
        Provide human-readable explanation for the signal.
        """
        explanations = []

        for rule in self.rules:
            if self._rule_matches(rule, features):
                conditions_str = ' AND '.join([
                    f"{feat} {op} {thresh:.4f}"
                    for feat, op, thresh in rule['conditions']
                ])
                direction = "BUY" if rule['prediction'] == 1 else "SELL"
                explanations.append(f"Rule matched: IF {conditions_str} THEN {direction}")

        return explanations
```

### Backtesting the Rule Strategy

```python
def backtest_rule_strategy(rules, X, y, prices, feature_names,
                            initial_capital=100000):
    """
    Backtest the rule-based trading strategy.
    """
    strategy = RuleBasedStrategy(rules, feature_names)

    capital = initial_capital
    position = 0
    returns = []
    trades = []

    for i in range(len(X) - 1):
        signal = strategy.generate_signal(X[i])
        price = prices[i]
        next_price = prices[i + 1]

        # Execute trades
        if signal == 1 and position <= 0:  # Buy signal
            position = capital / price
            trades.append({
                'idx': i,
                'type': 'BUY',
                'price': price,
                'explanation': strategy.explain_signal(X[i])
            })
        elif signal == -1 and position >= 0:  # Sell signal
            if position > 0:
                capital = position * price
            position = -capital / price
            trades.append({
                'idx': i,
                'type': 'SELL',
                'price': price,
                'explanation': strategy.explain_signal(X[i])
            })

        # Calculate returns
        if position > 0:
            ret = (next_price - price) / price
        elif position < 0:
            ret = (price - next_price) / price
        else:
            ret = 0

        returns.append(ret)

    # Calculate metrics
    returns = np.array(returns)
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    max_dd = compute_max_drawdown(returns)
    total_return = (1 + returns).prod() - 1

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'n_trades': len(trades),
        'trades': trades
    }
```

## Rust Implementation for Production

The `rust_examples/` directory contains a high-performance Rust implementation for production rule extraction and execution:

```rust
// Example: Extracting rules from a decision tree in Rust
use ndarray::Array2;

pub struct Rule {
    pub conditions: Vec<Condition>,
    pub prediction: i32,
    pub confidence: f64,
}

pub struct Condition {
    pub feature_idx: usize,
    pub operator: Operator,
    pub threshold: f64,
}

pub enum Operator {
    LessOrEqual,
    GreaterThan,
}

impl Rule {
    pub fn matches(&self, features: &[f64]) -> bool {
        self.conditions.iter().all(|cond| {
            let value = features[cond.feature_idx];
            match cond.operator {
                Operator::LessOrEqual => value <= cond.threshold,
                Operator::GreaterThan => value > cond.threshold,
            }
        })
    }

    pub fn to_string(&self, feature_names: &[String]) -> String {
        let conditions: Vec<String> = self.conditions.iter().map(|c| {
            let op = match c.operator {
                Operator::LessOrEqual => "<=",
                Operator::GreaterThan => ">",
            };
            format!("{} {} {:.4}", feature_names[c.feature_idx], op, c.threshold)
        }).collect();

        let prediction = if self.prediction == 1 { "BUY" } else { "SELL" };
        format!("IF {} THEN {}", conditions.join(" AND "), prediction)
    }
}
```

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Fidelity** | Agreement between rules and original model | > 90% |
| **Accuracy** | Rule prediction accuracy on test data | > 55% |
| **Coverage** | Fraction of samples covered by rules | > 95% |
| **Complexity** | Average conditions per rule | < 5 |
| **Sharpe Ratio** | Risk-adjusted returns of rule strategy | > 1.0 |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% |

## Dependencies

### Python
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=2.0.0
yfinance>=0.2.0
requests>=2.28.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

### Rust
See `rust_examples/Cargo.toml` for full dependencies.

## Expected Outcomes

After completing this chapter, you will be able to:

1. **Extract interpretable rules** from neural networks and ensemble models
2. **Evaluate rule quality** using fidelity, coverage, and complexity metrics
3. **Build transparent trading strategies** based on extracted rules
4. **Explain trading decisions** in human-readable terms
5. **Deploy rule-based systems** in production with Rust

## References

1. **Extracting Rules from Neural Networks as Decision Diagrams**
   - URL: https://arxiv.org/abs/2104.06411
   - Year: 2021
   - Key insight: Neural networks can be converted to Binary Decision Diagrams

2. **TREPAN: Extracting Tree-Structured Representations of Trained Networks**
   - Authors: Craven & Shavlik
   - Key insight: Decision trees can approximate neural network behavior

3. **Interpretable Machine Learning: A Guide for Making Black Box Models Explainable**
   - Author: Christoph Molnar
   - URL: https://christophm.github.io/interpretable-ml-book/

4. **Born Again Trees: From Deep Forests to Interpretable Trees**
   - URL: https://arxiv.org/abs/2003.11132
   - Key insight: Ensemble knowledge can be distilled into single trees

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Understanding of decision trees and neural networks
- Familiarity with Python machine learning libraries
- Basic knowledge of trading strategies
- Optional: Rust programming for production implementation

## Disclaimers

- **Not Financial Advice**: This material is for educational purposes only. Past performance does not guarantee future results.
- **Model Limitations**: Extracted rules are approximations; they may not capture all behaviors of the original model.
- **Market Risk**: All trading strategies involve risk of financial loss. Always use proper risk management.
- **Data Quality**: Strategy performance depends on data quality and market conditions.
