//! # Trading Signals
//!
//! Trading signal generation using RealNVP density estimation.
//!
//! Signals are generated based on:
//! - Log probability of current market state
//! - Direction in latent space
//! - Confidence based on density

use ndarray::{Array1, Array2};
use crate::flow::RealNVP;
use crate::utils::{Candle, MarketState, compute_market_state};
use crate::config;

/// Type of trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalType {
    /// Long (buy) signal
    Long,
    /// Short (sell) signal
    Short,
    /// No signal (hold/neutral)
    Neutral,
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal type
    pub signal_type: SignalType,
    /// Signal strength (-1.0 to 1.0)
    pub strength: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Log probability of current state
    pub log_prob: f64,
    /// Whether state is in-distribution
    pub in_distribution: bool,
    /// Latent representation
    pub latent: Array1<f64>,
    /// Mean of generated scenarios
    pub scenario_mean: f64,
    /// Std of generated scenarios
    pub scenario_std: f64,
}

impl TradingSignal {
    /// Create a neutral (no trade) signal
    pub fn neutral(log_prob: f64, latent: Array1<f64>) -> Self {
        Self {
            signal_type: SignalType::Neutral,
            strength: 0.0,
            confidence: 0.0,
            log_prob,
            in_distribution: false,
            latent,
            scenario_mean: 0.0,
            scenario_std: 0.0,
        }
    }

    /// Get position size recommendation (-1.0 to 1.0)
    pub fn position_size(&self) -> f64 {
        match self.signal_type {
            SignalType::Long => self.strength * self.confidence,
            SignalType::Short => -self.strength * self.confidence,
            SignalType::Neutral => 0.0,
        }
    }

    /// Check if should trade
    pub fn should_trade(&self) -> bool {
        self.in_distribution && self.confidence > 0.1 && self.strength.abs() > 0.1
    }
}

/// RealNVP-based trader
pub struct RealNVPTrader {
    /// RealNVP flow model
    model: RealNVP,
    /// Feature dimension
    feature_dim: usize,
    /// Log probability threshold for in-distribution
    log_prob_threshold: f64,
    /// Position scaling factor
    position_scale: f64,
    /// Latent index for return signal
    return_latent_idx: usize,
    /// Feature mean for normalization
    feature_mean: Option<Array1<f64>>,
    /// Feature std for normalization
    feature_std: Option<Array1<f64>>,
    /// Lookback period for features
    lookback: usize,
}

impl RealNVPTrader {
    /// Create a new RealNVP trader
    pub fn new(model: RealNVP) -> Self {
        let feature_dim = model.dim();
        Self {
            model,
            feature_dim,
            log_prob_threshold: config::DEFAULT_LOG_PROB_THRESHOLD,
            position_scale: 1.0,
            return_latent_idx: 0,
            feature_mean: None,
            feature_std: None,
            lookback: config::DEFAULT_LOOKBACK,
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        model: RealNVP,
        log_prob_threshold: f64,
        position_scale: f64,
        lookback: usize,
    ) -> Self {
        let feature_dim = model.dim();
        Self {
            model,
            feature_dim,
            log_prob_threshold,
            position_scale,
            return_latent_idx: 0,
            feature_mean: None,
            feature_std: None,
            lookback,
        }
    }

    /// Fit the normalizer on training data
    pub fn fit_normalizer(&mut self, features: &Array2<f64>) {
        let mean = features.mean_axis(ndarray::Axis(0)).unwrap();
        let std = features.std_axis(ndarray::Axis(0), 0.0);

        // Add small epsilon to avoid division by zero
        let std = std.mapv(|s| if s < 1e-8 { 1.0 } else { s });

        self.feature_mean = Some(mean);
        self.feature_std = Some(std);
    }

    /// Normalize features
    pub fn normalize(&self, features: &Array1<f64>) -> Array1<f64> {
        match (&self.feature_mean, &self.feature_std) {
            (Some(mean), Some(std)) => (features - mean) / std,
            _ => features.clone(),
        }
    }

    /// Generate signal from market state array
    pub fn generate_signal_from_state(&self, state: &Array1<f64>) -> TradingSignal {
        let x = self.normalize(state);

        // Get log probability and latent representation
        let (z, log_det) = self.model.forward(&x);
        let log_prob = self.model.log_prob(&x);

        // Check if in distribution
        let in_distribution = log_prob > self.log_prob_threshold;

        if !in_distribution {
            return TradingSignal::neutral(log_prob, z);
        }

        // Generate scenarios for risk analysis
        let scenarios = self.model.sample(100);
        let scenario_returns: Vec<f64> = scenarios.column(0).iter().cloned().collect();
        let scenario_mean = scenario_returns.iter().sum::<f64>() / scenario_returns.len() as f64;
        let scenario_std = {
            let variance: f64 = scenario_returns.iter()
                .map(|x| (x - scenario_mean).powi(2))
                .sum::<f64>() / scenario_returns.len() as f64;
            variance.sqrt()
        };

        // Determine signal from latent space direction
        let latent_value = z[self.return_latent_idx];
        let signal_direction = if latent_value.abs() > 0.5 {
            latent_value.signum()
        } else {
            0.0
        };

        // Calculate confidence based on density
        let confidence = ((log_prob - self.log_prob_threshold) / 10.0).clamp(0.0, 1.0);

        let signal_type = if signal_direction > 0.0 {
            SignalType::Long
        } else if signal_direction < 0.0 {
            SignalType::Short
        } else {
            SignalType::Neutral
        };

        let strength = signal_direction.abs() * self.position_scale;

        TradingSignal {
            signal_type,
            strength,
            confidence,
            log_prob,
            in_distribution,
            latent: z,
            scenario_mean,
            scenario_std,
        }
    }

    /// Generate signal from candles
    pub fn generate_signal(&self, candles: &[Candle]) -> TradingSignal {
        if candles.len() < self.lookback {
            return TradingSignal::neutral(
                f64::NEG_INFINITY,
                Array1::zeros(self.feature_dim),
            );
        }

        // Compute market state from recent candles
        let recent = &candles[candles.len() - self.lookback..];
        let state = compute_market_state(recent);

        self.generate_signal_from_state(&state)
    }

    /// Generate scenarios for risk analysis
    pub fn generate_scenarios(&self, n_scenarios: usize) -> Array2<f64> {
        let scenarios = self.model.sample(n_scenarios);

        // Denormalize if we have statistics
        match (&self.feature_mean, &self.feature_std) {
            (Some(mean), Some(std)) => {
                let mut denormalized = scenarios.clone();
                for mut row in denormalized.rows_mut() {
                    for (i, val) in row.iter_mut().enumerate() {
                        *val = *val * std[i] + mean[i];
                    }
                }
                denormalized
            }
            _ => scenarios,
        }
    }

    /// Compute Value at Risk using generated scenarios
    pub fn compute_var(&self, alpha: f64, n_scenarios: usize) -> f64 {
        let scenarios = self.generate_scenarios(n_scenarios);

        // Assume first feature is return
        let mut returns: Vec<f64> = scenarios.column(0).iter().cloned().collect();
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = (alpha * returns.len() as f64) as usize;
        let var = returns[idx.min(returns.len() - 1)];

        if var < 0.0 { -var } else { 0.0 }
    }

    /// Get model reference
    pub fn model(&self) -> &RealNVP {
        &self.model
    }

    /// Get mutable model reference
    pub fn model_mut(&mut self) -> &mut RealNVP {
        &mut self.model
    }

    /// Set log probability threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.log_prob_threshold = threshold;
    }

    /// Set position scale
    pub fn set_position_scale(&mut self, scale: f64) {
        self.position_scale = scale;
    }
}

/// Risk metrics from RealNVP analysis
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    /// Value at Risk (5%)
    pub var_5: f64,
    /// Value at Risk (1%)
    pub var_1: f64,
    /// Expected Shortfall / CVaR (5%)
    pub cvar_5: f64,
    /// Maximum expected loss from scenarios
    pub max_loss: f64,
    /// Probability of loss
    pub prob_loss: f64,
}

impl RealNVPTrader {
    /// Compute comprehensive risk metrics
    pub fn compute_risk_metrics(&self, n_scenarios: usize) -> RiskMetrics {
        let scenarios = self.generate_scenarios(n_scenarios);
        let mut returns: Vec<f64> = scenarios.column(0).iter().cloned().collect();
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = returns.len() as f64;

        // VaR at 5% and 1%
        let idx_5 = (0.05 * n) as usize;
        let idx_1 = (0.01 * n) as usize;

        let var_5 = -returns[idx_5.min(returns.len() - 1)].min(0.0);
        let var_1 = -returns[idx_1.min(returns.len() - 1)].min(0.0);

        // CVaR (Expected Shortfall)
        let cvar_5 = if idx_5 > 0 {
            -returns[..idx_5].iter().sum::<f64>() / idx_5 as f64
        } else {
            var_5
        };

        // Max loss
        let max_loss = -returns[0].min(0.0);

        // Probability of loss
        let prob_loss = returns.iter().filter(|&&r| r < 0.0).count() as f64 / n;

        RiskMetrics {
            var_5,
            var_1,
            cvar_5,
            max_loss,
            prob_loss,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_signal_position_size() {
        let latent = Array1::zeros(4);

        let long_signal = TradingSignal {
            signal_type: SignalType::Long,
            strength: 0.8,
            confidence: 0.7,
            log_prob: -5.0,
            in_distribution: true,
            latent: latent.clone(),
            scenario_mean: 0.01,
            scenario_std: 0.02,
        };

        assert!((long_signal.position_size() - 0.56).abs() < 1e-10);

        let short_signal = TradingSignal {
            signal_type: SignalType::Short,
            strength: 0.5,
            confidence: 0.6,
            log_prob: -5.0,
            in_distribution: true,
            latent: latent.clone(),
            scenario_mean: -0.01,
            scenario_std: 0.02,
        };

        assert!((short_signal.position_size() - (-0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_realnvp_trader_creation() {
        let flow = RealNVP::new(10, 64, 4);
        let trader = RealNVPTrader::new(flow);

        assert_eq!(trader.feature_dim, 10);
    }

    #[test]
    fn test_signal_generation() {
        let flow = RealNVP::new(4, 32, 4);
        let trader = RealNVPTrader::new(flow);

        let state = Array1::from_vec(vec![0.01, 0.02, 0.5, 1.0]);
        let signal = trader.generate_signal_from_state(&state);

        assert!(signal.log_prob.is_finite());
    }
}
