"""
Synthetic Data Generation for Mamba4Cast Training

This module provides synthetic time series generation for training
Mamba4Cast models. The model learns to generalize from these synthetic
patterns to real-world time series.

The diversity of synthetic data is key to zero-shot generalization.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    seq_length: int = 200
    horizon: int = 24
    n_features: int = 1
    seed: Optional[int] = None


class SyntheticDataGenerator:
    """
    Generate diverse synthetic time series for Mamba4Cast training.

    The key to zero-shot generalization is training on a diverse
    distribution of synthetic time series that captures:
    - Trends (linear, exponential, polynomial)
    - Seasonality (various frequencies)
    - Mean reversion
    - Volatility clustering
    - Regime changes
    - Random walks
    - Combined patterns

    Example:
        generator = SyntheticDataGenerator(seq_length=200, horizon=24)
        X, y = generator.generate_batch(batch_size=32)
    """

    def __init__(
        self,
        seq_length: int = 200,
        horizon: int = 24,
        n_features: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Initialize the generator.

        Args:
            seq_length: Length of context sequence
            horizon: Forecast horizon
            n_features: Number of features
            seed: Random seed for reproducibility
        """
        self.seq_length = seq_length
        self.horizon = horizon
        self.n_features = n_features

        if seed is not None:
            np.random.seed(seed)

        # Process weights for sampling
        self._process_weights = {
            'ar': 0.15,
            'ma': 0.10,
            'arma': 0.15,
            'random_walk': 0.10,
            'trend': 0.10,
            'seasonal': 0.10,
            'mean_reverting': 0.10,
            'garch': 0.10,
            'regime_switching': 0.05,
            'combined': 0.05,
        }

    def generate_ar(
        self,
        length: int,
        order: int = 3,
    ) -> np.ndarray:
        """
        Generate AR(p) process.

        Args:
            length: Sequence length
            order: AR order

        Returns:
            AR time series
        """
        # Random AR coefficients (ensure stationarity)
        coeffs = np.random.uniform(-0.5, 0.5, order)
        coeffs = coeffs / (np.sum(np.abs(coeffs)) + 0.5)

        # Generate series
        noise_scale = np.random.uniform(0.1, 1.0)
        noise = np.random.randn(length) * noise_scale

        series = np.zeros(length)
        series[:order] = noise[:order]

        for t in range(order, length):
            series[t] = np.dot(coeffs, series[t-order:t][::-1]) + noise[t]

        return series

    def generate_ma(
        self,
        length: int,
        order: int = 3,
    ) -> np.ndarray:
        """
        Generate MA(q) process.

        Args:
            length: Sequence length
            order: MA order

        Returns:
            MA time series
        """
        # Random MA coefficients
        coeffs = np.random.uniform(-0.8, 0.8, order + 1)

        # Generate noise
        noise_scale = np.random.uniform(0.1, 1.0)
        noise = np.random.randn(length + order) * noise_scale

        # Generate series
        series = np.convolve(noise, coeffs, mode='valid')[:length]

        return series

    def generate_arma(
        self,
        length: int,
        p: int = 2,
        q: int = 2,
    ) -> np.ndarray:
        """
        Generate ARMA(p,q) process.

        Args:
            length: Sequence length
            p: AR order
            q: MA order

        Returns:
            ARMA time series
        """
        ar_coeffs = np.random.uniform(-0.4, 0.4, p)
        ar_coeffs = ar_coeffs / (np.sum(np.abs(ar_coeffs)) + 0.6)

        ma_coeffs = np.random.uniform(-0.6, 0.6, q + 1)

        noise_scale = np.random.uniform(0.1, 1.0)
        noise = np.random.randn(length + q) * noise_scale

        ma_part = np.convolve(noise, ma_coeffs, mode='valid')[:length]

        series = np.zeros(length)
        series[:p] = ma_part[:p]

        for t in range(p, length):
            series[t] = np.dot(ar_coeffs, series[t-p:t][::-1]) + ma_part[t]

        return series

    def generate_random_walk(
        self,
        length: int,
        drift: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate random walk with optional drift.

        Args:
            length: Sequence length
            drift: Drift term (default: random)

        Returns:
            Random walk time series
        """
        if drift is None:
            drift = np.random.uniform(-0.01, 0.01)

        noise_scale = np.random.uniform(0.1, 1.0)
        increments = drift + np.random.randn(length) * noise_scale

        return np.cumsum(increments)

    def generate_trend(
        self,
        length: int,
        trend_type: str = 'random',
    ) -> np.ndarray:
        """
        Generate trend series.

        Args:
            length: Sequence length
            trend_type: 'linear', 'exponential', 'polynomial', or 'random'

        Returns:
            Trend time series
        """
        t = np.arange(length)

        if trend_type == 'random':
            trend_type = np.random.choice(['linear', 'exponential', 'polynomial'])

        if trend_type == 'linear':
            slope = np.random.uniform(-0.1, 0.1)
            intercept = np.random.uniform(-10, 10)
            trend = slope * t + intercept

        elif trend_type == 'exponential':
            rate = np.random.uniform(-0.01, 0.01)
            scale = np.random.uniform(1, 10)
            trend = scale * np.exp(rate * t)

        else:  # polynomial
            degree = np.random.randint(2, 4)
            coeffs = np.random.uniform(-0.001, 0.001, degree + 1)
            trend = np.polyval(coeffs, t)

        # Add noise
        noise_scale = np.abs(trend).mean() * 0.1 + 0.1
        trend += np.random.randn(length) * noise_scale

        return trend

    def generate_seasonal(
        self,
        length: int,
        n_harmonics: int = 3,
    ) -> np.ndarray:
        """
        Generate seasonal series with multiple harmonics.

        Args:
            length: Sequence length
            n_harmonics: Number of seasonal components

        Returns:
            Seasonal time series
        """
        t = np.arange(length)
        series = np.zeros(length)

        for _ in range(n_harmonics):
            period = np.random.uniform(10, 50)
            amplitude = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)

            series += amplitude * np.sin(2 * np.pi * t / period + phase)

        # Add trend and noise
        trend = np.random.uniform(-0.01, 0.01) * t
        noise = np.random.randn(length) * np.random.uniform(0.1, 0.5)

        return series + trend + noise

    def generate_mean_reverting(
        self,
        length: int,
        mean: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate Ornstein-Uhlenbeck (mean-reverting) process.

        Args:
            length: Sequence length
            mean: Long-term mean
            speed: Mean reversion speed

        Returns:
            Mean-reverting time series
        """
        if mean is None:
            mean = np.random.uniform(-5, 5)
        if speed is None:
            speed = np.random.uniform(0.01, 0.1)

        sigma = np.random.uniform(0.1, 1.0)

        series = np.zeros(length)
        series[0] = mean + np.random.randn() * sigma

        for t in range(1, length):
            series[t] = (
                series[t-1] +
                speed * (mean - series[t-1]) +
                sigma * np.random.randn()
            )

        return series

    def generate_garch(
        self,
        length: int,
    ) -> np.ndarray:
        """
        Generate GARCH(1,1) process with volatility clustering.

        Args:
            length: Sequence length

        Returns:
            GARCH time series
        """
        omega = np.random.uniform(0.001, 0.01)
        alpha = np.random.uniform(0.05, 0.15)
        beta = np.random.uniform(0.7, 0.9)

        # Ensure alpha + beta < 1 for stationarity
        if alpha + beta >= 1:
            scale = 0.95 / (alpha + beta)
            alpha *= scale
            beta *= scale

        returns = np.zeros(length)
        variance = np.zeros(length)
        variance[0] = omega / (1 - alpha - beta)

        for t in range(1, length):
            variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
            returns[t] = np.sqrt(variance[t]) * np.random.randn()

        # Convert returns to price-like series
        return np.cumsum(returns) + np.random.uniform(10, 100)

    def generate_regime_switching(
        self,
        length: int,
        n_regimes: int = 2,
    ) -> np.ndarray:
        """
        Generate regime-switching time series.

        Args:
            length: Sequence length
            n_regimes: Number of regimes

        Returns:
            Regime-switching time series
        """
        # Regime parameters
        means = np.random.uniform(-0.5, 0.5, n_regimes)
        stds = np.random.uniform(0.1, 1.0, n_regimes)

        # Transition probabilities
        transition_prob = np.random.uniform(0.01, 0.05)

        # Generate regimes
        regime = np.zeros(length, dtype=int)
        regime[0] = np.random.randint(n_regimes)

        for t in range(1, length):
            if np.random.random() < transition_prob:
                regime[t] = np.random.randint(n_regimes)
            else:
                regime[t] = regime[t-1]

        # Generate series
        increments = np.array([
            means[r] + stds[r] * np.random.randn()
            for r in regime
        ])

        return np.cumsum(increments) + np.random.uniform(10, 100)

    def generate_combined(
        self,
        length: int,
    ) -> np.ndarray:
        """
        Generate combined patterns (trend + seasonal + noise).

        Args:
            length: Sequence length

        Returns:
            Combined time series
        """
        t = np.arange(length)

        # Trend component
        slope = np.random.uniform(-0.05, 0.05)
        trend = slope * t

        # Seasonal component
        period = np.random.uniform(15, 40)
        amplitude = np.random.uniform(1, 5)
        seasonal = amplitude * np.sin(2 * np.pi * t / period)

        # Noise component
        noise_scale = np.random.uniform(0.2, 1.0)
        noise = np.random.randn(length) * noise_scale

        # Base level
        base = np.random.uniform(10, 100)

        return base + trend + seasonal + noise

    def generate_single(
        self,
        process_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single synthetic time series with context and target.

        Args:
            process_type: Type of process (default: random)

        Returns:
            Tuple of (context, target) arrays
        """
        total_length = self.seq_length + self.horizon

        if process_type is None:
            # Sample process type according to weights
            types = list(self._process_weights.keys())
            weights = list(self._process_weights.values())
            process_type = np.random.choice(types, p=weights)

        # Generate based on type
        generators = {
            'ar': lambda: self.generate_ar(total_length),
            'ma': lambda: self.generate_ma(total_length),
            'arma': lambda: self.generate_arma(total_length),
            'random_walk': lambda: self.generate_random_walk(total_length),
            'trend': lambda: self.generate_trend(total_length),
            'seasonal': lambda: self.generate_seasonal(total_length),
            'mean_reverting': lambda: self.generate_mean_reverting(total_length),
            'garch': lambda: self.generate_garch(total_length),
            'regime_switching': lambda: self.generate_regime_switching(total_length),
            'combined': lambda: self.generate_combined(total_length),
        }

        series = generators.get(process_type, generators['arma'])()

        # Split into context and target
        context = series[:self.seq_length].reshape(-1, 1)
        target = series[self.seq_length:].reshape(-1, 1)

        return context.astype(np.float32), target.astype(np.float32)

    def generate_batch(
        self,
        batch_size: int,
        process_types: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of synthetic time series.

        Args:
            batch_size: Number of series to generate
            process_types: List of process types to use (default: random)

        Returns:
            Tuple of (X, y) arrays with shapes:
            - X: (batch_size, seq_length, n_features)
            - y: (batch_size, horizon, n_features)
        """
        X_list = []
        y_list = []

        for i in range(batch_size):
            if process_types is not None:
                ptype = process_types[i % len(process_types)]
            else:
                ptype = None

            x, y = self.generate_single(ptype)
            X_list.append(x)
            y_list.append(y)

        X = np.stack(X_list)
        y = np.stack(y_list)

        return X, y

    def create_dataset(
        self,
        n_samples: int,
        include_metadata: bool = False,
    ):
        """
        Create a complete dataset for training.

        Args:
            n_samples: Number of samples
            include_metadata: Whether to include process type info

        Returns:
            Dictionary with 'X', 'y', and optionally 'process_types'
        """
        X_list = []
        y_list = []
        types_list = []

        process_types = list(self._process_weights.keys())

        for i in range(n_samples):
            ptype = np.random.choice(
                process_types,
                p=list(self._process_weights.values())
            )

            x, y = self.generate_single(ptype)
            X_list.append(x)
            y_list.append(y)
            types_list.append(ptype)

        result = {
            'X': np.stack(X_list),
            'y': np.stack(y_list),
        }

        if include_metadata:
            result['process_types'] = types_list

        return result
