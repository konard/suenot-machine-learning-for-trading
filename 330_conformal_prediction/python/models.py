"""
Base Prediction Models for Conformal Prediction

Provides various regression models to be used as base predictors.
"""

import numpy as np
from typing import Optional, List, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


class QuantileGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting Regressor for quantile prediction.

    Wrapper around sklearn's GradientBoostingRegressor with quantile loss.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 10,
        random_state: Optional[int] = None
    ):
        """
        Initialize quantile gradient boosting regressor.

        Args:
            quantile: Quantile to predict (0.0 to 1.0)
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
        """
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model = GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileGradientBoostingRegressor':
        """Fit the model."""
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict quantiles."""
        return self.model.predict(X)


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble of multiple regression models.

    Combines predictions from multiple models for improved robustness.
    """

    def __init__(
        self,
        n_estimators_gb: int = 100,
        n_estimators_rf: int = 100,
        max_depth: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize ensemble regressor.

        Args:
            n_estimators_gb: Number of gradient boosting estimators
            n_estimators_rf: Number of random forest estimators
            max_depth: Maximum depth for tree models
            random_state: Random seed
        """
        self.n_estimators_gb = n_estimators_gb
        self.n_estimators_rf = n_estimators_rf
        self.max_depth = max_depth
        self.random_state = random_state

        self.models = [
            GradientBoostingRegressor(
                n_estimators=n_estimators_gb,
                max_depth=max_depth,
                random_state=random_state
            ),
            RandomForestRegressor(
                n_estimators=n_estimators_rf,
                max_depth=max_depth,
                random_state=random_state
            ),
            Ridge(alpha=1.0),
        ]

        self.weights = np.array([0.5, 0.3, 0.2])  # Default weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleRegressor':
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted average of models."""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all models."""
        return np.array([model.predict(X) for model in self.models])


class AdaptiveRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge regression with adaptive regularization.

    Automatically selects regularization strength via cross-validation.
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        cv: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize adaptive ridge regressor.

        Args:
            alphas: List of alpha values to try
            cv: Number of cross-validation folds
            random_state: Random seed
        """
        self.alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0]
        self.cv = cv
        self.random_state = random_state

        from sklearn.linear_model import RidgeCV
        self.model = RidgeCV(alphas=self.alphas, cv=cv)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveRidgeRegressor':
        """Fit with automatic alpha selection."""
        self.model.fit(X, y)
        self.best_alpha_ = self.model.alpha_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict values."""
        return self.model.predict(X)


def create_quantile_pair(
    alpha: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 3,
    random_state: Optional[int] = None
) -> Tuple[QuantileGradientBoostingRegressor, QuantileGradientBoostingRegressor]:
    """
    Create a pair of quantile regressors for CQR.

    Args:
        alpha: Miscoverage rate
        n_estimators: Number of estimators
        max_depth: Max tree depth
        random_state: Random seed

    Returns:
        Tuple of (lower_quantile_model, upper_quantile_model)
    """
    q_low = QuantileGradientBoostingRegressor(
        quantile=alpha / 2,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    q_high = QuantileGradientBoostingRegressor(
        quantile=1 - alpha / 2,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    return q_low, q_high


def get_default_model(
    model_type: str = 'gradient_boosting',
    **kwargs
) -> BaseEstimator:
    """
    Get a default model by type.

    Args:
        model_type: Type of model ('gradient_boosting', 'random_forest',
                   'ridge', 'ensemble')
        **kwargs: Additional model parameters

    Returns:
        Configured model instance
    """
    if model_type == 'gradient_boosting':
        return GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', None)
        )

    elif model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=kwargs.get('random_state', None)
        )

    elif model_type == 'ridge':
        return Ridge(
            alpha=kwargs.get('alpha', 1.0),
            random_state=kwargs.get('random_state', None)
        )

    elif model_type == 'ensemble':
        return EnsembleRegressor(
            n_estimators_gb=kwargs.get('n_estimators_gb', 100),
            n_estimators_rf=kwargs.get('n_estimators_rf', 100),
            max_depth=kwargs.get('max_depth', 5),
            random_state=kwargs.get('random_state', None)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    np.random.seed(42)

    # Generate test data
    n = 500
    X = np.random.randn(n, 10)
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + np.random.randn(n) * 0.3

    X_train, y_train = X[:400], y[:400]
    X_test, y_test = X[400:], y[400:]

    # Test Gradient Boosting
    print("Testing Gradient Boosting...")
    gb = get_default_model('gradient_boosting')
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)
    rmse_gb = np.sqrt(np.mean((y_test - pred_gb) ** 2))
    print(f"  RMSE: {rmse_gb:.4f}")

    # Test Quantile Models
    print("\nTesting Quantile Gradient Boosting...")
    q_low, q_high = create_quantile_pair(alpha=0.1)
    q_low.fit(X_train, y_train)
    q_high.fit(X_train, y_train)

    pred_low = q_low.predict(X_test)
    pred_high = q_high.predict(X_test)

    coverage = np.mean((y_test >= pred_low) & (y_test <= pred_high))
    print(f"  Coverage: {coverage:.3f}")

    # Test Ensemble
    print("\nTesting Ensemble...")
    ensemble = get_default_model('ensemble')
    ensemble.fit(X_train, y_train)
    pred_ens = ensemble.predict(X_test)
    rmse_ens = np.sqrt(np.mean((y_test - pred_ens) ** 2))
    print(f"  RMSE: {rmse_ens:.4f}")
