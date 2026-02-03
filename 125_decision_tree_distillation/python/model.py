"""
Decision Tree Distillation Model

This module implements decision tree distillation for extracting interpretable
rules from complex machine learning models used in trading.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any, Dict
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_text, _tree
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error


@dataclass
class DistilledRule:
    """Represents a single rule extracted from a distilled decision tree."""
    conditions: List[str]
    prediction: str
    confidence: float
    samples: int
    path_indices: List[int]

    def __str__(self) -> str:
        conditions_str = " AND ".join(self.conditions) if self.conditions else "TRUE"
        return f"IF {conditions_str} THEN {self.prediction} (conf: {self.confidence:.2f}, n={self.samples})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary representation."""
        return {
            'conditions': self.conditions,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'samples': self.samples,
            'path_indices': self.path_indices
        }


@dataclass
class DistillationResult:
    """Results from distillation process."""
    fidelity: float
    accuracy: float
    rules: List[DistilledRule]
    feature_importance: Dict[str, float]
    tree_depth: int
    n_leaves: int

    def summary(self) -> str:
        """Return a summary of distillation results."""
        return f"""
Distillation Results:
  Fidelity to Teacher: {self.fidelity:.2%}
  Accuracy on Labels:  {self.accuracy:.2%}
  Tree Depth:          {self.tree_depth}
  Number of Leaves:    {self.n_leaves}
  Number of Rules:     {len(self.rules)}

Top Features:
{self._format_features()}
"""

    def _format_features(self) -> str:
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        return "\n".join(
            f"  {name}: {importance:.3f}"
            for name, importance in sorted_features
        )


class DistillationModel:
    """
    Decision Tree Distillation for Trading Models.

    This class implements knowledge distillation from complex models
    (teacher) to simple decision trees (student) for interpretability.

    Attributes:
        max_depth: Maximum depth of the distilled tree
        min_samples_leaf: Minimum samples required in leaf nodes
        min_samples_split: Minimum samples required to split
        task: 'classification' or 'regression'
        feature_names: Names of input features
        class_names: Names of output classes (for classification)

    Example:
        >>> from model import DistillationModel
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>>
        >>> # Train complex teacher model
        >>> teacher = GradientBoostingClassifier(n_estimators=200)
        >>> teacher.fit(X_train, y_train)
        >>>
        >>> # Distill to interpretable tree
        >>> distiller = DistillationModel(max_depth=5)
        >>> distiller.fit(X_train, teacher)
        >>>
        >>> # Extract and display rules
        >>> rules = distiller.extract_rules()
        >>> for rule in rules:
        ...     print(rule)
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 50,
        min_samples_split: int = 100,
        task: str = 'classification',
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        use_soft_labels: bool = True,
        temperature: float = 1.0,
    ):
        """
        Initialize the distillation model.

        Args:
            max_depth: Maximum depth of distilled tree (default: 5)
            min_samples_leaf: Minimum samples in leaf nodes (default: 50)
            min_samples_split: Minimum samples to split (default: 100)
            task: 'classification' or 'regression' (default: 'classification')
            feature_names: List of feature names for rule extraction
            class_names: List of class names (e.g., ['SELL', 'HOLD', 'BUY'])
            use_soft_labels: Use probability outputs from teacher (default: True)
            temperature: Temperature for softening probabilities (default: 1.0)
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.task = task
        self.feature_names = feature_names
        self.class_names = class_names or ['SELL', 'HOLD', 'BUY']
        self.use_soft_labels = use_soft_labels
        self.temperature = temperature

        self.student_: Optional[BaseEstimator] = None
        self.teacher_: Optional[BaseEstimator] = None
        self._fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        teacher: BaseEstimator,
        y: Optional[np.ndarray] = None,
    ) -> 'DistillationModel':
        """
        Fit the distilled decision tree to mimic the teacher model.

        Args:
            X: Training features
            teacher: Trained complex model (teacher)
            y: Optional original labels for accuracy evaluation

        Returns:
            self: Fitted DistillationModel
        """
        self.teacher_ = teacher

        # Set feature names if not provided
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Convert to numpy if DataFrame
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        # Generate soft labels from teacher
        if self.task == 'classification':
            if self.use_soft_labels and hasattr(teacher, 'predict_proba'):
                soft_labels = teacher.predict_proba(X_arr)
                # Apply temperature scaling
                if self.temperature != 1.0:
                    soft_labels = self._apply_temperature(soft_labels)
                # Train on probability of positive class for binary
                if soft_labels.shape[1] == 2:
                    y_distill = soft_labels[:, 1]
                    self.student_ = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        min_samples_split=self.min_samples_split,
                    )
                else:
                    # Multi-class: train on argmax but use soft labels info
                    y_distill = np.argmax(soft_labels, axis=1)
                    self.student_ = DecisionTreeClassifier(
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        min_samples_split=self.min_samples_split,
                    )
            else:
                y_distill = teacher.predict(X_arr)
                self.student_ = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                )
        else:
            y_distill = teacher.predict(X_arr)
            self.student_ = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
            )

        # Fit student on teacher's predictions
        self.student_.fit(X_arr, y_distill)
        self._fitted = True

        return self

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to soften/sharpen probabilities."""
        log_probs = np.log(probs + 1e-10)
        scaled = log_probs / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the distilled tree.

        Args:
            X: Input features

        Returns:
            predictions: Model predictions
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        predictions = self.student_.predict(X_arr)

        # Convert regression output to classes for classification
        if self.task == 'classification' and isinstance(self.student_, DecisionTreeRegressor):
            predictions = (predictions > 0.5).astype(int)

        return predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            X: Input features

        Returns:
            probabilities: Class probabilities
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        if isinstance(self.student_, DecisionTreeRegressor):
            # Convert regression output to pseudo-probabilities
            raw_pred = self.student_.predict(X_arr)
            proba = np.column_stack([1 - raw_pred, raw_pred])
            return np.clip(proba, 0, 1)
        else:
            return self.student_.predict_proba(X_arr)

    def fidelity_score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        teacher: Optional[BaseEstimator] = None,
    ) -> float:
        """
        Calculate fidelity: how well student mimics teacher.

        Args:
            X: Test features
            teacher: Teacher model (uses stored teacher if None)

        Returns:
            fidelity: Agreement rate between teacher and student
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        teacher = teacher or self.teacher_
        if teacher is None:
            raise ValueError("No teacher model available")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        teacher_pred = teacher.predict(X_arr)
        student_pred = self.predict(X_arr)

        if self.task == 'classification':
            return accuracy_score(teacher_pred, student_pred)
        else:
            # For regression, use correlation
            return np.corrcoef(teacher_pred, student_pred)[0, 1]

    def extract_rules(self) -> List[DistilledRule]:
        """
        Extract human-readable rules from the distilled tree.

        Returns:
            rules: List of DistilledRule objects
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        tree = self.student_.tree_
        rules = []

        def recurse(node: int, conditions: List[str], path: List[int]):
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                # Leaf node
                if isinstance(self.student_, DecisionTreeClassifier):
                    class_idx = np.argmax(tree.value[node])
                    prediction = self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx)
                    confidence = tree.value[node].flatten()[class_idx] / tree.value[node].sum()
                else:
                    raw_value = tree.value[node][0, 0]
                    if self.task == 'classification':
                        class_idx = 1 if raw_value > 0.5 else 0
                        prediction = self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx)
                        confidence = raw_value if class_idx == 1 else 1 - raw_value
                    else:
                        prediction = f"{raw_value:.4f}"
                        confidence = 1.0

                rules.append(DistilledRule(
                    conditions=list(conditions),
                    prediction=prediction,
                    confidence=float(confidence),
                    samples=int(tree.n_node_samples[node]),
                    path_indices=list(path)
                ))
            else:
                # Internal node
                feature_name = self.feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                # Left child (<=)
                left_cond = f"{feature_name} <= {threshold:.4f}"
                recurse(
                    tree.children_left[node],
                    conditions + [left_cond],
                    path + [tree.children_left[node]]
                )

                # Right child (>)
                right_cond = f"{feature_name} > {threshold:.4f}"
                recurse(
                    tree.children_right[node],
                    conditions + [right_cond],
                    path + [tree.children_right[node]]
                )

        recurse(0, [], [0])
        return rules

    def tree_to_text(self) -> str:
        """
        Get text representation of the decision tree.

        Returns:
            text: Tree structure as text
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return export_text(self.student_, feature_names=self.feature_names)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the distilled tree.

        Returns:
            importance: Dictionary mapping feature names to importance scores
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = self.student_.feature_importances_
        return dict(zip(self.feature_names, importance))

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        teacher: Optional[BaseEstimator] = None,
    ) -> DistillationResult:
        """
        Comprehensive evaluation of the distilled model.

        Args:
            X: Test features
            y: True labels
            teacher: Teacher model for fidelity calculation

        Returns:
            result: DistillationResult with all metrics
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        teacher = teacher or self.teacher_
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        # Calculate fidelity
        fidelity = self.fidelity_score(X_arr, teacher) if teacher else 0.0

        # Calculate accuracy
        student_pred = self.predict(X_arr)
        if self.task == 'classification':
            accuracy = accuracy_score(y, student_pred)
        else:
            accuracy = 1 - mean_squared_error(y, student_pred)

        # Extract rules and features
        rules = self.extract_rules()
        feature_importance = self.get_feature_importance()

        return DistillationResult(
            fidelity=fidelity,
            accuracy=accuracy,
            rules=rules,
            feature_importance=feature_importance,
            tree_depth=self.student_.get_depth(),
            n_leaves=self.student_.get_n_leaves(),
        )


def distill_trading_model(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    teacher_model: Optional[BaseEstimator] = None,
    max_depth: int = 5,
    feature_names: Optional[List[str]] = None,
) -> Tuple[DistillationModel, DistillationResult]:
    """
    Convenience function to distill a trading model end-to-end.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        teacher_model: Pre-trained teacher (creates GBM if None)
        max_depth: Maximum depth for distilled tree
        feature_names: Names of features

    Returns:
        distiller: Fitted DistillationModel
        result: DistillationResult with evaluation metrics

    Example:
        >>> from model import distill_trading_model
        >>> from data_loader import load_stock_data, compute_features
        >>>
        >>> # Load and prepare data
        >>> data = load_stock_data('AAPL', period='2y')
        >>> X, y = compute_features(data)
        >>>
        >>> # Split data
        >>> split = int(len(X) * 0.8)
        >>> X_train, X_test = X[:split], X[split:]
        >>> y_train, y_test = y[:split], y[split:]
        >>>
        >>> # Distill
        >>> distiller, result = distill_trading_model(
        ...     X_train, y_train, X_test, y_test,
        ...     max_depth=5
        ... )
        >>> print(result.summary())
    """
    from sklearn.ensemble import GradientBoostingClassifier

    # Convert to numpy
    X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    # Set feature names
    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train_arr.shape[1])]

    # Train teacher if not provided
    if teacher_model is None:
        teacher_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
        )
        teacher_model.fit(X_train_arr, y_train)

    # Create and fit distiller
    distiller = DistillationModel(
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=['DOWN', 'UP'],
    )
    distiller.fit(X_train, teacher_model, y_train)

    # Evaluate
    result = distiller.evaluate(X_test, y_test, teacher_model)

    return distiller, result


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Decision Tree Distillation Example")
    print("=" * 50)

    # Generate synthetic trading data
    np.random.seed(42)
    n_samples = 2000

    # Create features similar to trading indicators
    rsi = np.random.uniform(0, 100, n_samples)
    macd = np.random.normal(0, 1, n_samples)
    volume_ratio = np.random.uniform(0.5, 2.0, n_samples)
    volatility = np.random.uniform(0.01, 0.05, n_samples)

    X = np.column_stack([rsi, macd, volume_ratio, volatility])
    feature_names = ['RSI_14', 'MACD', 'Volume_Ratio', 'Volatility']

    # Create labels based on a complex rule (simulating a complex model)
    y = (
        (rsi < 30) & (macd > 0) |
        (rsi < 40) & (macd > 0.5) & (volume_ratio > 1.5)
    ).astype(int)

    # Split data
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train teacher model
    from sklearn.ensemble import GradientBoostingClassifier
    teacher = GradientBoostingClassifier(n_estimators=100, max_depth=8)
    teacher.fit(X_train, y_train)
    print(f"Teacher accuracy: {teacher.score(X_test, y_test):.2%}")

    # Distill to interpretable tree
    distiller = DistillationModel(
        max_depth=4,
        feature_names=feature_names,
        class_names=['DOWN', 'UP'],
    )
    distiller.fit(X_train, teacher)

    # Evaluate
    result = distiller.evaluate(X_test, y_test, teacher)
    print(result.summary())

    # Print extracted rules
    print("\nExtracted Trading Rules:")
    print("-" * 50)
    for rule in result.rules:
        print(rule)

    # Print tree structure
    print("\nTree Structure:")
    print("-" * 50)
    print(distiller.tree_to_text())
