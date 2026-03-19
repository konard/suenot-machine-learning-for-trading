//! HiPPO-based trading strategy implementation.

use crate::model::hippo::HiPPOLegS;
use crate::trading::signals::{generate_signals, TradingSignal};

/// Configuration for the HiPPO trading strategy.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Number of HiPPO polynomial coefficients.
    pub n_hippo: usize,
    /// Discretization time step.
    pub dt: f64,
    /// Minimum trend magnitude for signal generation.
    pub trend_threshold: f64,
    /// Whether to apply volatility-based position scaling.
    pub vol_filter: bool,
    /// Maximum position size multiplier.
    pub max_position: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            n_hippo: 64,
            dt: 1.0,
            trend_threshold: 0.0,
            vol_filter: true,
            max_position: 1.0,
        }
    }
}

/// HiPPO-based trading strategy.
///
/// Uses HiPPO-LegS polynomial coefficients to generate trading signals:
/// - c1 (linear coefficient) determines trend direction
/// - c2 (quadratic coefficient) provides acceleration information
/// - Higher-order coefficient variance acts as volatility filter
pub struct TradingStrategy {
    config: StrategyConfig,
    hippo: HiPPOLegS,
}

impl TradingStrategy {
    /// Create a new trading strategy.
    pub fn new(config: StrategyConfig) -> Self {
        let hippo = HiPPOLegS::new(config.n_hippo);
        Self { config, hippo }
    }

    /// Create a strategy with default configuration.
    pub fn default_strategy() -> Self {
        Self::new(StrategyConfig::default())
    }

    /// Generate trading signals from a price series.
    ///
    /// # Arguments
    /// * `prices` - Slice of price values.
    ///
    /// # Returns
    /// Vector of trading signals, one per price point.
    pub fn generate_signals(&self, prices: &[f64]) -> Vec<TradingSignal> {
        // Normalize prices
        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let std: f64 = {
            let var: f64 = prices.iter().map(|&p| (p - mean).powi(2)).sum::<f64>()
                / prices.len() as f64;
            var.sqrt().max(1e-10)
        };
        let normalized: Vec<f64> = prices.iter().map(|&p| (p - mean) / std).collect();

        // Run HiPPO (cap dt for numerical stability with large N)
        let effective_dt = self.config.dt.min(1.0 / (self.config.n_hippo as f64 + 1.0));
        let history = self.hippo.process_sequence(&normalized, effective_dt);

        // Generate signals from coefficients
        let mut signals = generate_signals(&history, self.config.trend_threshold);

        // Apply volatility filter
        if self.config.vol_filter && !history.is_empty() {
            let n = self.config.n_hippo;
            if n > 4 {
                let half = n / 2;
                let vol_proxy: Vec<f64> = history
                    .iter()
                    .map(|coeffs| {
                        let high = &coeffs[half..];
                        let m: f64 = high.iter().sum::<f64>() / high.len() as f64;
                        high.iter().map(|&c| (c - m).powi(2)).sum::<f64>() / high.len() as f64
                    })
                    .collect();

                let median_vol = {
                    let mut sorted = vol_proxy.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    sorted[sorted.len() / 2]
                };
                let vol_threshold = median_vol * 2.0;

                for (i, signal) in signals.iter_mut().enumerate() {
                    if vol_proxy[i] > vol_threshold {
                        signal.position *= 0.5;
                    }
                }
            }
        }

        // Clamp positions
        for signal in &mut signals {
            signal.position = signal.position.clamp(
                -self.config.max_position,
                self.config.max_position,
            );
        }

        signals
    }

    /// Get positions as a simple f64 vector (for backtesting).
    pub fn get_positions(&self, prices: &[f64]) -> Vec<f64> {
        self.generate_signals(prices)
            .iter()
            .map(|s| s.position)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_data;

    #[test]
    fn test_strategy_creation() {
        let strategy = TradingStrategy::default_strategy();
        assert_eq!(strategy.config.n_hippo, 64);
    }

    #[test]
    fn test_strategy_signals() {
        let config = StrategyConfig {
            n_hippo: 16,
            dt: 0.1,
            trend_threshold: 0.0,
            vol_filter: false,
            max_position: 1.0,
        };
        let strategy = TradingStrategy::new(config);

        let data = generate_synthetic_data("BTCUSDT", 200, 50000.0, 0.02, 42);
        let prices = data.close_prices();
        let signals = strategy.generate_signals(&prices);

        assert_eq!(signals.len(), prices.len());
        // Should have some non-zero signals
        let non_flat: usize = signals.iter().filter(|s| !s.is_flat()).count();
        assert!(non_flat > 0, "Strategy should generate some signals");
    }
}
