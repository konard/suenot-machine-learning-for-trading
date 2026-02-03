"""
Zero-Shot Forecasting Utilities for Mamba4Cast

This module provides high-level utilities for zero-shot forecasting
with Mamba4Cast models.
"""

import numpy as np
from typing import Optional, Dict, List, Union
import torch

from .mamba4cast_model import Mamba4CastForecaster


class ZeroShotForecaster:
    """
    High-level interface for zero-shot time series forecasting.

    This class wraps Mamba4CastForecaster with convenience methods
    for common forecasting tasks.

    Example:
        forecaster = ZeroShotForecaster.load_pretrained()
        result = forecaster.forecast(prices, horizon=24)
        signals = forecaster.trading_signals(prices)
    """

    def __init__(
        self,
        model: Mamba4CastForecaster,
        device: str = "cpu",
    ):
        """
        Initialize the forecaster.

        Args:
            model: Mamba4CastForecaster model
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @classmethod
    def create(
        cls,
        preset: str = "default",
        device: str = "cpu",
        **model_kwargs,
    ) -> "ZeroShotForecaster":
        """
        Create a new forecaster with specified configuration.

        Args:
            preset: Model preset ('small', 'default', 'large')
            device: Device for inference
            **model_kwargs: Additional model arguments

        Returns:
            Configured ZeroShotForecaster
        """
        from .mamba4cast_model import create_mamba4cast_model

        model = create_mamba4cast_model(preset=preset, **model_kwargs)
        return cls(model, device)

    def _prepare_input(
        self,
        data: Union[np.ndarray, List, torch.Tensor],
        context_length: int,
    ) -> torch.Tensor:
        """Prepare input data for model."""
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        # Ensure correct shape: (batch, seq, features)
        if data.dim() == 1:
            data = data.unsqueeze(-1)
        if data.dim() == 2:
            data = data.unsqueeze(0)

        # Take last context_length points
        data = data[:, -context_length:, :]

        return data.to(self.device)

    @torch.no_grad()
    def forecast(
        self,
        data: Union[np.ndarray, List, torch.Tensor],
        horizon: int = 24,
        context_length: int = 100,
        return_confidence: bool = False,
    ) -> Union[np.ndarray, Dict]:
        """
        Generate zero-shot forecast.

        Args:
            data: Input time series
            horizon: Number of steps to forecast
            context_length: Historical context to use
            return_confidence: Whether to return confidence estimates

        Returns:
            Forecast array or dict with forecast and confidence
        """
        self.model.eval()

        x = self._prepare_input(data, context_length)
        forecast = self.model(x, horizon=horizon)

        result = forecast.squeeze(0).cpu().numpy()

        if return_confidence:
            # Simple confidence based on prediction stability
            confidence = self._estimate_confidence(x, horizon)
            return {
                'forecast': result,
                'confidence': confidence,
            }

        return result

    def _estimate_confidence(
        self,
        x: torch.Tensor,
        horizon: int,
        n_samples: int = 10,
    ) -> np.ndarray:
        """Estimate confidence via dropout-based uncertainty."""
        self.model.train()  # Enable dropout

        forecasts = []
        for _ in range(n_samples):
            with torch.no_grad():
                f = self.model(x, horizon=horizon)
                forecasts.append(f.cpu().numpy())

        self.model.eval()

        forecasts = np.stack(forecasts)
        std = forecasts.std(axis=0)

        # Convert std to confidence (inverse relationship)
        max_std = std.max() + 1e-6
        confidence = 1 - (std / max_std)

        return confidence.squeeze()

    @torch.no_grad()
    def multi_horizon_forecast(
        self,
        data: Union[np.ndarray, List, torch.Tensor],
        horizons: List[int],
        context_length: int = 100,
    ) -> Dict[int, np.ndarray]:
        """
        Generate forecasts for multiple horizons.

        Args:
            data: Input time series
            horizons: List of forecast horizons
            context_length: Historical context to use

        Returns:
            Dictionary mapping horizon to forecast
        """
        max_horizon = max(horizons)
        x = self._prepare_input(data, context_length)

        full_forecast = self.model(x, horizon=max_horizon)
        full_forecast = full_forecast.squeeze(0).cpu().numpy()

        results = {}
        for h in horizons:
            results[h] = full_forecast[:h, :]

        return results

    @torch.no_grad()
    def trading_signals(
        self,
        data: Union[np.ndarray, List, torch.Tensor],
        horizon: int = 24,
        context_length: int = 100,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ) -> Dict:
        """
        Generate trading signals from forecasts.

        Args:
            data: Input time series
            horizon: Forecast horizon
            context_length: Historical context
            buy_threshold: Return threshold for buy signal
            sell_threshold: Return threshold for sell signal

        Returns:
            Dictionary with signals and analysis
        """
        x = self._prepare_input(data, context_length)

        # Get current price
        current_price = float(x[0, -1, 0].cpu())

        # Generate forecast
        forecast = self.model(x, horizon=horizon)
        forecast = forecast.squeeze(0).cpu().numpy()

        # Analyze signals for each step
        signals = []
        for h in range(horizon):
            pred_price = float(forecast[h, 0])
            expected_return = (pred_price - current_price) / current_price

            if expected_return > buy_threshold:
                signal = 'BUY'
            elif expected_return < sell_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            signals.append({
                'horizon': h + 1,
                'signal': signal,
                'predicted_price': pred_price,
                'expected_return': expected_return,
                'confidence': 1 - abs(expected_return) / 0.1,  # Simple heuristic
            })

        # Summary signal (based on end of horizon)
        final_return = (forecast[-1, 0] - current_price) / current_price
        if final_return > buy_threshold:
            summary_signal = 'BUY'
        elif final_return < sell_threshold:
            summary_signal = 'SELL'
        else:
            summary_signal = 'HOLD'

        return {
            'current_price': current_price,
            'summary_signal': summary_signal,
            'expected_return': float(final_return),
            'signals': signals,
            'forecast': forecast,
        }

    @torch.no_grad()
    def cross_asset_forecast(
        self,
        assets: Dict[str, np.ndarray],
        horizon: int = 24,
        context_length: int = 100,
    ) -> Dict[str, Dict]:
        """
        Generate forecasts for multiple assets.

        Args:
            assets: Dictionary mapping asset names to price data
            horizon: Forecast horizon
            context_length: Historical context

        Returns:
            Dictionary with forecasts for each asset
        """
        results = {}

        for name, data in assets.items():
            try:
                forecast = self.forecast(
                    data, horizon=horizon, context_length=context_length
                )

                current_price = data[-1] if len(data.shape) == 1 else data[-1, 0]
                final_price = forecast[-1, 0]
                expected_return = (final_price - current_price) / current_price

                results[name] = {
                    'current_price': float(current_price),
                    'forecast': forecast,
                    'expected_return': float(expected_return),
                    'predicted_final_price': float(final_price),
                }
            except Exception as e:
                results[name] = {
                    'error': str(e),
                }

        return results

    @torch.no_grad()
    def scenario_analysis(
        self,
        data: Union[np.ndarray, List, torch.Tensor],
        horizon: int = 24,
        context_length: int = 100,
        perturbation_pcts: List[float] = [-0.05, -0.02, 0, 0.02, 0.05],
    ) -> Dict:
        """
        Perform scenario analysis with perturbed inputs.

        Args:
            data: Input time series
            horizon: Forecast horizon
            context_length: Historical context
            perturbation_pcts: Percentage changes to apply

        Returns:
            Dictionary with scenario forecasts
        """
        x = self._prepare_input(data, context_length)
        base_price = float(x[0, -1, 0].cpu())

        scenarios = {}

        for pct in perturbation_pcts:
            # Perturb the last value
            x_perturbed = x.clone()
            x_perturbed[0, -1, 0] *= (1 + pct)

            forecast = self.model(x_perturbed, horizon=horizon)
            forecast = forecast.squeeze(0).cpu().numpy()

            perturbed_price = base_price * (1 + pct)
            final_price = forecast[-1, 0]
            expected_return = (final_price - perturbed_price) / perturbed_price

            scenarios[f'{pct:+.1%}'] = {
                'starting_price': perturbed_price,
                'final_price': float(final_price),
                'expected_return': float(expected_return),
                'forecast': forecast,
            }

        return {
            'base_price': base_price,
            'scenarios': scenarios,
        }


def quick_forecast(
    data: Union[np.ndarray, List],
    horizon: int = 24,
    context_length: int = 100,
    preset: str = "small",
) -> np.ndarray:
    """
    Convenience function for quick zero-shot forecasting.

    Args:
        data: Input time series
        horizon: Steps to forecast
        context_length: Historical context
        preset: Model preset

    Returns:
        Forecast array
    """
    forecaster = ZeroShotForecaster.create(preset=preset)
    return forecaster.forecast(data, horizon=horizon, context_length=context_length)
