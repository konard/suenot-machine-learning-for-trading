"""
Rule Extraction Module

This module provides algorithms for extracting interpretable if-then rules
from black-box machine learning models such as neural networks and gradient boosting.

Key Classes:
- Rule: Represents a single trading rule
- RuleExtractor: Base class for rule extraction algorithms
- TrepanExtractor: TREPAN-like pedagogical rule extraction
- TreeRuleExtractor: Direct extraction from decision trees
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class Condition:
    """A single condition in a rule."""
    feature_idx: int
    operator: str  # '<=' or '>'
    threshold: float
    feature_name: Optional[str] = None

    def matches(self, features: np.ndarray) -> bool:
        """Check if this condition is satisfied."""
        value = features[self.feature_idx]
        if self.operator == '<=':
            return value <= self.threshold
        else:  # '>'
            return value > self.threshold

    def to_string(self, feature_names: Optional[List[str]] = None) -> str:
        """Convert condition to readable string."""
        name = self.feature_name
        if name is None and feature_names is not None:
            name = feature_names[self.feature_idx]
        if name is None:
            name = f"feature_{self.feature_idx}"
        return f"{name} {self.operator} {self.threshold:.4f}"


@dataclass
class Rule:
    """A trading rule consisting of conditions and a prediction."""
    conditions: List[Condition]
    prediction: int  # 1 for BUY, -1 for SELL, 0 for HOLD
    confidence: float = 0.5
    weight: float = 1.0
    source_tree: Optional[int] = None

    def matches(self, features: np.ndarray) -> bool:
        """Check if all conditions are satisfied."""
        return all(cond.matches(features) for cond in self.conditions)

    def to_string(self, feature_names: Optional[List[str]] = None) -> str:
        """Convert rule to readable string."""
        if not self.conditions:
            return f"DEFAULT: {self._pred_str()} (confidence: {self.confidence:.2f})"

        conditions_str = " AND ".join(
            cond.to_string(feature_names) for cond in self.conditions
        )
        return f"IF {conditions_str} THEN {self._pred_str()} (confidence: {self.confidence:.2f})"

    def _pred_str(self) -> str:
        if self.prediction == 1:
            return "BUY"
        elif self.prediction == -1:
            return "SELL"
        return "HOLD"

    @property
    def complexity(self) -> int:
        """Number of conditions in this rule."""
        return len(self.conditions)


class RuleSet:
    """A collection of rules with evaluation methods."""

    def __init__(self, rules: List[Rule], feature_names: Optional[List[str]] = None):
        self.rules = rules
        self.feature_names = feature_names or []

    def predict(self, features: np.ndarray) -> int:
        """Make prediction using weighted voting."""
        buy_score = 0.0
        sell_score = 0.0

        for rule in self.rules:
            if rule.matches(features):
                vote = rule.weight * rule.confidence
                if rule.prediction == 1:
                    buy_score += vote
                elif rule.prediction == -1:
                    sell_score += vote

        if buy_score > sell_score and buy_score > 0.5:
            return 1
        elif sell_score > buy_score and sell_score > 0.5:
            return -1
        return 0

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Make predictions for multiple samples."""
        return np.array([self.predict(f) for f in features])

    def explain(self, features: np.ndarray) -> List[str]:
        """Get explanations for matching rules."""
        return [
            rule.to_string(self.feature_names)
            for rule in self.rules
            if rule.matches(features)
        ]

    def calculate_fidelity(self, features: np.ndarray, model_predictions: np.ndarray) -> float:
        """Calculate how well rules match model predictions."""
        if len(features) == 0:
            return 0.0

        rule_preds = self.predict_batch(features)
        return np.mean(rule_preds == model_predictions)

    def calculate_coverage(self, features: np.ndarray) -> float:
        """Calculate fraction of samples covered by at least one rule."""
        if len(features) == 0:
            return 0.0

        covered = sum(
            1 for f in features
            if any(rule.matches(f) for rule in self.rules)
        )
        return covered / len(features)

    def calculate_accuracy(self, features: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate accuracy against true labels."""
        if len(features) == 0:
            return 0.0

        rule_preds = self.predict_batch(features)
        return np.mean(rule_preds == true_labels)

    @property
    def average_complexity(self) -> float:
        """Average number of conditions per rule."""
        if not self.rules:
            return 0.0
        return np.mean([rule.complexity for rule in self.rules])

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)


class TrepanExtractor:
    """
    TREPAN-like rule extraction from black-box models.

    This extracts rules by training a decision tree to mimic
    the black-box model's predictions.
    """

    def __init__(
        self,
        feature_names: List[str],
        max_depth: int = 6,
        min_samples_leaf: int = 50,
        max_leaf_nodes: Optional[int] = None
    ):
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes

    def extract(
        self,
        model: Any,
        X: np.ndarray,
        use_proba: bool = False
    ) -> RuleSet:
        """
        Extract rules from a black-box model.

        Args:
            model: A model with predict() method
            X: Feature matrix
            use_proba: Whether to use predict_proba if available

        Returns:
            RuleSet containing extracted rules
        """
        # Get model predictions as pseudo-labels
        if use_proba and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            y_pseudo = np.argmax(proba, axis=1)
        else:
            y_pseudo = model.predict(X)

        # Train surrogate decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42
        )
        tree.fit(X, y_pseudo)

        # Extract rules from tree
        rules = self._extract_tree_rules(tree)

        # Calculate fidelity
        tree_preds = tree.predict(X)
        fidelity = np.mean(tree_preds == y_pseudo)
        print(f"Rule fidelity to model: {fidelity:.2%}")

        return RuleSet(rules, self.feature_names)

    def extract_from_predictions(
        self,
        X: np.ndarray,
        predictions: np.ndarray
    ) -> RuleSet:
        """
        Extract rules from pre-computed predictions.

        Args:
            X: Feature matrix
            predictions: Model predictions

        Returns:
            RuleSet containing extracted rules
        """
        # Train surrogate decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42
        )
        tree.fit(X, predictions)

        # Extract rules
        rules = self._extract_tree_rules(tree)

        return RuleSet(rules, self.feature_names)

    def _extract_tree_rules(self, tree: DecisionTreeClassifier) -> List[Rule]:
        """Extract rules from a fitted decision tree."""
        tree_ = tree.tree_
        rules = []

        def recurse(node: int, path: List[Condition]):
            # Check if leaf node
            if tree_.feature[node] == -2:
                # Get prediction and confidence
                value = tree_.value[node].ravel()
                prediction = int(np.argmax(value))
                total = np.sum(value)
                confidence = value[prediction] / total if total > 0 else 0.5

                # Map class to trading signal
                # Assuming 0 -> SELL (-1), 1 -> BUY (1)
                signal = 1 if prediction == 1 else -1

                if len(path) > 0 or True:  # Include default rules
                    rules.append(Rule(
                        conditions=list(path),
                        prediction=signal,
                        confidence=confidence
                    ))
                return

            # Get split information
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else None

            # Left branch: feature <= threshold
            left_cond = Condition(
                feature_idx=feature_idx,
                operator='<=',
                threshold=threshold,
                feature_name=feature_name
            )
            recurse(tree_.children_left[node], path + [left_cond])

            # Right branch: feature > threshold
            right_cond = Condition(
                feature_idx=feature_idx,
                operator='>',
                threshold=threshold,
                feature_name=feature_name
            )
            recurse(tree_.children_right[node], path + [right_cond])

        recurse(0, [])
        return rules


class GradientBoostingExtractor:
    """Extract rules from gradient boosting ensemble."""

    def __init__(
        self,
        feature_names: List[str],
        importance_threshold: float = 0.05,
        max_rules: int = 50
    ):
        self.feature_names = feature_names
        self.importance_threshold = importance_threshold
        self.max_rules = max_rules

    def extract(self, model: GradientBoostingClassifier) -> RuleSet:
        """
        Extract rules from a gradient boosting model.

        Args:
            model: Fitted GradientBoostingClassifier

        Returns:
            RuleSet containing extracted rules
        """
        # Get feature importances
        importances = model.feature_importances_
        important_features = set(
            i for i, imp in enumerate(importances)
            if imp >= self.importance_threshold
        )

        # Extract rules from each tree
        all_rules = []
        n_trees = len(model.estimators_.ravel())

        for tree_idx, tree in enumerate(model.estimators_.ravel()):
            tree_rules = self._extract_single_tree(tree, tree_idx)

            # Filter rules to only use important features
            for rule in tree_rules:
                if all(cond.feature_idx in important_features for cond in rule.conditions):
                    rule.weight = 1.0 / n_trees
                    all_rules.append(rule)

        # Consolidate similar rules
        consolidated = self._consolidate_rules(all_rules)

        # Keep top rules
        consolidated.sort(key=lambda r: r.confidence * r.weight, reverse=True)
        consolidated = consolidated[:self.max_rules]

        return RuleSet(consolidated, self.feature_names)

    def _extract_single_tree(self, tree, tree_idx: int) -> List[Rule]:
        """Extract rules from a single tree in the ensemble."""
        tree_ = tree.tree_
        rules = []

        def recurse(node: int, path: List[Condition]):
            if tree_.feature[node] == -2:
                value = tree_.value[node].ravel()
                prediction = 1 if value[0] > 0 else -1
                confidence = min(abs(value[0]) / 10.0, 1.0)  # Normalize

                rules.append(Rule(
                    conditions=list(path),
                    prediction=prediction,
                    confidence=confidence,
                    source_tree=tree_idx
                ))
                return

            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else None

            left_cond = Condition(feature_idx, '<=', threshold, feature_name)
            recurse(tree_.children_left[node], path + [left_cond])

            right_cond = Condition(feature_idx, '>', threshold, feature_name)
            recurse(tree_.children_right[node], path + [right_cond])

        recurse(0, [])
        return rules

    def _consolidate_rules(self, rules: List[Rule]) -> List[Rule]:
        """Merge rules with identical conditions."""
        rule_dict: Dict[str, Rule] = {}

        for rule in rules:
            key = self._rule_key(rule)

            if key in rule_dict:
                existing = rule_dict[key]
                existing.weight += rule.weight
                existing.confidence = (existing.confidence + rule.confidence) / 2
            else:
                rule_dict[key] = Rule(
                    conditions=rule.conditions.copy(),
                    prediction=rule.prediction,
                    confidence=rule.confidence,
                    weight=rule.weight
                )

        return list(rule_dict.values())

    def _rule_key(self, rule: Rule) -> str:
        """Generate a unique key for a rule based on its conditions."""
        parts = sorted([
            f"{c.feature_idx}:{c.operator}:{int(c.threshold * 1000)}"
            for c in rule.conditions
        ])
        parts.append(f"pred:{rule.prediction}")
        return "|".join(parts)


def extract_rules_from_nn(
    model,
    X_train: np.ndarray,
    feature_names: List[str],
    max_depth: int = 6,
    min_samples_leaf: int = 50
) -> Tuple[RuleSet, float]:
    """
    Convenience function to extract rules from a neural network.

    Args:
        model: A trained neural network with predict method
        X_train: Training features
        feature_names: Names of features
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf

    Returns:
        Tuple of (RuleSet, fidelity score)
    """
    extractor = TrepanExtractor(
        feature_names=feature_names,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )

    ruleset = extractor.extract(model, X_train)

    # Calculate fidelity
    model_preds = model.predict(X_train)
    fidelity = ruleset.calculate_fidelity(X_train, model_preds)

    return ruleset, fidelity


def evaluate_rules(
    rules: RuleSet,
    X: np.ndarray,
    y_true: np.ndarray,
    model_predictions: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate a rule set on test data.

    Args:
        rules: RuleSet to evaluate
        X: Feature matrix
        y_true: True labels
        model_predictions: Optional model predictions for fidelity

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': rules.calculate_accuracy(X, y_true),
        'coverage': rules.calculate_coverage(X),
        'n_rules': len(rules),
        'avg_complexity': rules.average_complexity
    }

    if model_predictions is not None:
        metrics['fidelity'] = rules.calculate_fidelity(X, model_predictions)

    return metrics


if __name__ == "__main__":
    # Example usage
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_classification

    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    y = 2 * y - 1  # Convert to -1, 1

    feature_names = ['RSI', 'MACD', 'SMA_Ratio', 'Volatility', 'Volume_Change']

    # Train a neural network
    nn = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    nn.fit(X, y)

    # Extract rules
    extractor = TrepanExtractor(feature_names, max_depth=5, min_samples_leaf=30)
    ruleset = extractor.extract(nn, X)

    print("\nExtracted Rules:")
    print("-" * 60)
    for i, rule in enumerate(ruleset.rules[:10]):
        print(f"Rule {i+1}: {rule.to_string(feature_names)}")

    # Evaluate
    metrics = evaluate_rules(ruleset, X, y, nn.predict(X))
    print("\nMetrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.3f}")
        else:
            print(f"  {name}: {value}")
