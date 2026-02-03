//! High-level forecaster implementation.
//!
//! This module provides the main interface for zero-shot forecasting.

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

use super::Mamba4CastBlock;

/// Trading signal generated from forecast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Signal type: BUY, SELL, or HOLD
    pub signal: String,
    /// Expected return
    pub expected_return: f32,
    /// Predicted price
    pub predicted_price: f32,
    /// Confidence level (0-1)
    pub confidence: f32,
}

/// Result of a forecast operation.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Forecasted values
    pub forecast: Vec<f32>,
    /// Current price (last context value)
    pub current_price: f32,
    /// Expected return at end of horizon
    pub expected_return: f32,
}

/// Mamba4Cast forecaster for zero-shot time series prediction.
///
/// This struct provides high-level methods for generating forecasts
/// and trading signals from time series data.
pub struct Mamba4CastForecaster {
    /// Model dimension
    d_model: usize,
    /// Number of layers
    n_layers: usize,
    /// SSM state dimension
    d_state: usize,
    /// Maximum forecast horizon
    max_horizon: usize,
    /// Model layers
    layers: Vec<Mamba4CastBlock>,
    /// Input projection weights
    input_proj: Array2<f32>,
    /// Forecast head weights
    forecast_head: Array2<f32>,
}

impl Mamba4CastForecaster {
    /// Create a new forecaster.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension
    /// * `n_layers` - Number of Mamba layers
    /// * `d_state` - SSM state dimension
    /// * `max_horizon` - Maximum forecast horizon
    pub fn new(
        d_model: usize,
        n_layers: usize,
        d_state: usize,
        max_horizon: usize,
    ) -> Self {
        // Create layers
        let layers: Vec<Mamba4CastBlock> = (0..n_layers)
            .map(|_| Mamba4CastBlock::new(d_model, d_state, 4, 2, max_horizon))
            .collect();

        // Initialize projection weights
        let input_proj = Array2::from_shape_fn(
            (1, d_model),
            |_| rand::random::<f32>() * 0.02 - 0.01
        );

        let forecast_head = Array2::from_shape_fn(
            (d_model, max_horizon),
            |_| rand::random::<f32>() * 0.02 - 0.01
        );

        Self {
            d_model,
            n_layers,
            d_state,
            max_horizon,
            layers,
            input_proj,
            forecast_head,
        }
    }

    /// Generate zero-shot forecast for a time series.
    ///
    /// # Arguments
    ///
    /// * `time_series` - Input time series values
    /// * `context_length` - Number of historical points to use
    /// * `horizon` - Number of steps to forecast
    ///
    /// # Returns
    ///
    /// Vector of forecasted values
    pub fn zero_shot_forecast(
        &self,
        time_series: &[f32],
        context_length: usize,
        horizon: usize,
    ) -> Vec<f32> {
        // Get context
        let start = time_series.len().saturating_sub(context_length);
        let context: Vec<f32> = time_series[start..].to_vec();

        // Normalize
        let (normalized, mean, std) = self.normalize(&context);

        // Forward pass
        let forecast_normalized = self.forward(&normalized, horizon.min(self.max_horizon));

        // Denormalize
        forecast_normalized
            .iter()
            .map(|x| x * std + mean)
            .collect()
    }

    /// Generate forecast result with metadata.
    pub fn forecast(&self, time_series: &[f32], context_length: usize, horizon: usize) -> ForecastResult {
        let forecast = self.zero_shot_forecast(time_series, context_length, horizon);
        let current_price = *time_series.last().unwrap_or(&0.0);
        let final_price = *forecast.last().unwrap_or(&current_price);
        let expected_return = if current_price != 0.0 {
            (final_price - current_price) / current_price
        } else {
            0.0
        };

        ForecastResult {
            forecast,
            current_price,
            expected_return,
        }
    }

    /// Generate trading signals from forecast.
    ///
    /// # Arguments
    ///
    /// * `time_series` - Input time series
    /// * `context_length` - Historical context length
    /// * `horizon` - Forecast horizon
    /// * `threshold` - Signal threshold (default 0.01 = 1%)
    pub fn trading_signals(
        &self,
        time_series: &[f32],
        context_length: usize,
        horizon: usize,
        threshold: f32,
    ) -> TradingSignal {
        let result = self.forecast(time_series, context_length, horizon);

        let signal = if result.expected_return > threshold {
            "BUY".to_string()
        } else if result.expected_return < -threshold {
            "SELL".to_string()
        } else {
            "HOLD".to_string()
        };

        let confidence = 1.0 - (result.expected_return.abs() / 0.1).min(1.0);

        TradingSignal {
            signal,
            expected_return: result.expected_return,
            predicted_price: *result.forecast.last().unwrap_or(&0.0),
            confidence,
        }
    }

    /// Normalize time series data.
    fn normalize(&self, data: &[f32]) -> (Vec<f32>, f32, f32) {
        let n = data.len() as f32;
        let mean: f32 = data.iter().sum::<f32>() / n;
        let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt().max(1e-6);

        let normalized: Vec<f32> = data.iter().map(|x| (x - mean) / std).collect();

        (normalized, mean, std)
    }

    /// Forward pass through the model.
    fn forward(&self, context: &[f32], horizon: usize) -> Vec<f32> {
        let seq_len = context.len();

        // Create input tensor (batch=1, seq_len, features=1)
        let mut x = Array3::<f32>::zeros((1, seq_len, self.d_model));

        // Project input
        for t in 0..seq_len {
            for d in 0..self.d_model {
                x[[0, t, d]] = context[t] * self.input_proj[[0, d]];
            }
        }

        // Process through layers with residual connections
        for layer in &self.layers {
            let y = layer.forward(&x);
            // Add residual
            for b in 0..1 {
                for t in 0..seq_len {
                    for d in 0..self.d_model {
                        x[[b, t, d]] += y[[b, t, d]];
                    }
                }
            }
        }

        // Use last hidden state for forecasting
        let last_hidden: Vec<f32> = (0..self.d_model)
            .map(|d| x[[0, seq_len - 1, d]])
            .collect();

        // Project to forecast
        let mut forecast = vec![0.0f32; horizon];
        for h in 0..horizon {
            for d in 0..self.d_model {
                forecast[h] += last_hidden[d] * self.forecast_head[[d, h]];
            }
        }

        forecast
    }

    /// Get model configuration.
    pub fn config(&self) -> ModelConfig {
        ModelConfig {
            d_model: self.d_model,
            n_layers: self.n_layers,
            d_state: self.d_state,
            max_horizon: self.max_horizon,
        }
    }
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub max_horizon: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecaster_creation() {
        let forecaster = Mamba4CastForecaster::new(64, 4, 16, 96);
        let config = forecaster.config();

        assert_eq!(config.d_model, 64);
        assert_eq!(config.n_layers, 4);
        assert_eq!(config.d_state, 16);
        assert_eq!(config.max_horizon, 96);
    }

    #[test]
    fn test_zero_shot_forecast() {
        let forecaster = Mamba4CastForecaster::new(32, 2, 8, 48);

        let time_series: Vec<f32> = (0..150)
            .map(|i| 100.0 + (i as f32) * 0.1 + rand::random::<f32>() * 2.0)
            .collect();

        let forecast = forecaster.zero_shot_forecast(&time_series, 100, 24);

        assert_eq!(forecast.len(), 24);
    }

    #[test]
    fn test_trading_signals() {
        let forecaster = Mamba4CastForecaster::new(32, 2, 8, 48);

        let time_series: Vec<f32> = (0..150)
            .map(|i| 100.0 + (i as f32) * 0.1)
            .collect();

        let signal = forecaster.trading_signals(&time_series, 100, 24, 0.01);

        assert!(["BUY", "SELL", "HOLD"].contains(&signal.signal.as_str()));
    }

    #[test]
    fn test_normalize() {
        let forecaster = Mamba4CastForecaster::new(32, 2, 8, 48);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (normalized, mean, std) = forecaster.normalize(&data);

        assert_eq!(normalized.len(), 5);
        assert!((mean - 3.0).abs() < 0.001);

        // Check normalized mean is ~0
        let norm_mean: f32 = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!(norm_mean.abs() < 0.001);
    }
}
