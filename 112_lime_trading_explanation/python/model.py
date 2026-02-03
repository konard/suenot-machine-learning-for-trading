"""
Trading Models for LIME Explanation

Provides wrapper classes for common machine learning models used in trading,
with a consistent interface for LIME explanations.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod


class TradingModel(ABC):
    """
    Abstract base class for trading models.

    Provides a consistent interface for model training, prediction,
    and LIME-compatible prediction functions.
    """

    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize the trading model.

        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'logistic')
            **kwargs: Additional arguments passed to the underlying model
        """
        self.model_type = model_type
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TradingModel':
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        pass


class RandomForestTrader(TradingModel):
    """
    Random Forest classifier for trading signals.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Random Forest trading model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            random_state: Random seed for reproducibility
        """
        super().__init__(model_type='random_forest')

        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestTrader':
        """
        Train the Random Forest model.

        Args:
            X: Feature DataFrame
            y: Target Series (0/1 for down/up)

        Returns:
            self
        """
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (0 or 1)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities [P(down), P(up)]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from the model.

        Returns:
            Series with feature importance values
        """
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)


class XGBoostTrader(TradingModel):
    """
    XGBoost classifier for trading signals.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize XGBoost trading model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            random_state: Random seed
        """
        super().__init__(model_type='xgboost')

        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Please install xgboost: pip install xgboost")

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss',
            **kwargs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostTrader':
        """Train the XGBoost model."""
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from the model."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)


class LogisticTrader(TradingModel):
    """
    Logistic Regression classifier for trading signals.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Logistic Regression trading model.

        Args:
            C: Inverse regularization strength
            penalty: Regularization type ('l1', 'l2', 'elasticnet')
            random_state: Random seed
        """
        super().__init__(model_type='logistic')

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("Please install scikit-learn: pip install scikit-learn")

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            random_state=random_state,
            max_iter=1000,
            **kwargs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LogisticTrader':
        """Train the Logistic Regression model."""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)
        self.model.fit(X_scaled, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from coefficients."""
        importance = np.abs(self.model.coef_[0])
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)


def create_model(model_type: str = 'random_forest', **kwargs) -> TradingModel:
    """
    Factory function to create trading models.

    Args:
        model_type: Type of model ('random_forest', 'xgboost', 'logistic')
        **kwargs: Arguments passed to the model constructor

    Returns:
        TradingModel instance
    """
    models = {
        'random_forest': RandomForestTrader,
        'xgboost': XGBoostTrader,
        'logistic': LogisticTrader
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](**kwargs)


class ModelEvaluator:
    """
    Utilities for evaluating trading model performance.
    """

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score
            )
        except ImportError:
            raise ImportError("Please install scikit-learn")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

        return metrics

    @staticmethod
    def classification_metrics_proba(
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics using probabilities.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        try:
            from sklearn.metrics import roc_auc_score, log_loss
        except ImportError:
            raise ImportError("Please install scikit-learn")

        metrics = {
            'roc_auc': roc_auc_score(y_true, y_proba[:, 1]),
            'log_loss': log_loss(y_true, y_proba)
        }

        return metrics
