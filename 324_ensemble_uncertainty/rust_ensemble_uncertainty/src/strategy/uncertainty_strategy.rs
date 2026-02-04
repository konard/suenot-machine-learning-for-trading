//! Uncertainty-aware trading strategy.

use super::signal::{Signal, SignalGenerator, SignalType};
use std::collections::HashMap;

/// Uncertainty-aware trading strategy
pub struct UncertaintyStrategy {
    generator: SignalGenerator,
    use_kelly: bool,
    kelly_fraction: f64,
}

impl UncertaintyStrategy {
    /// Create new uncertainty strategy
    pub fn new(
        min_prediction_threshold: f64,
        max_uncertainty_threshold: f64,
        min_confidence_threshold: f64,
    ) -> Self {
        Self {
            generator: SignalGenerator::new(
                min_prediction_threshold,
                max_uncertainty_threshold,
                min_confidence_threshold,
            ),
            use_kelly: true,
            kelly_fraction: 0.5,
        }
    }

    /// Enable or disable Kelly criterion
    pub fn with_kelly(mut self, use_kelly: bool, fraction: f64) -> Self {
        self.use_kelly = use_kelly;
        self.kelly_fraction = fraction;
        self
    }

    /// Generate signal with optional Kelly adjustment
    pub fn generate_signal(
        &self,
        symbol: &str,
        prediction: f64,
        uncertainty: f64,
        current_price: Option<f64>,
        historical_win_rate: Option<f64>,
    ) -> Signal {
        let mut signal = self.generator.generate_signal(symbol, prediction, uncertainty, current_price);

        // Apply Kelly criterion adjustment if enabled
        if self.use_kelly && signal.signal_type != SignalType::Hold {
            if let Some(win_rate) = historical_win_rate {
                let kelly_size = self.kelly_criterion(
                    win_rate,
                    prediction.abs(),
                    uncertainty,
                );
                signal.position_size = signal.position_size.min(kelly_size);
            }
        }

        signal
    }

    /// Calculate Kelly criterion position size
    fn kelly_criterion(&self, win_prob: f64, win_size: f64, loss_size: f64) -> f64 {
        if loss_size <= 0.0 {
            return 0.0;
        }

        // Kelly formula: f* = (p*b - q) / b
        let b = win_size / (loss_size + 1e-8);
        let q = 1.0 - win_prob;
        let kelly = (win_prob * b - q) / b;

        // Apply fractional Kelly
        (kelly * self.kelly_fraction).max(0.0)
    }

    /// Generate signals for portfolio
    pub fn generate_portfolio_signals(
        &self,
        symbols: &[&str],
        predictions: &[f64],
        uncertainties: &[f64],
        current_prices: &[f64],
    ) -> Vec<Signal> {
        let signals = self.generator.generate_signals(
            symbols,
            predictions,
            uncertainties,
            Some(current_prices),
        );

        self.generator.filter_signals(signals, Some(10))
    }

    /// Allocate portfolio based on signals
    pub fn allocate_portfolio(
        &self,
        signals: &[Signal],
        max_position_per_asset: f64,
        max_total_exposure: f64,
    ) -> HashMap<String, f64> {
        let mut allocations = HashMap::new();
        let mut total_exposure = 0.0;

        // Sort by confidence
        let mut sorted_signals = signals.to_vec();
        sorted_signals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        for signal in sorted_signals {
            if signal.signal_type == SignalType::Hold {
                continue;
            }

            let proposed_size = signal.position_size.min(max_position_per_asset);
            let remaining_capacity = max_total_exposure - total_exposure;

            if remaining_capacity <= 0.0 {
                break;
            }

            let final_size = proposed_size.min(remaining_capacity);
            if final_size > 0.01 {
                allocations.insert(signal.symbol.clone(), final_size);
                total_exposure += final_size;
            }
        }

        allocations
    }
}

impl Default for UncertaintyStrategy {
    fn default() -> Self {
        Self::new(0.001, 0.03, 0.6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy() {
        let strategy = UncertaintyStrategy::new(0.001, 0.05, 0.6);

        let signal = strategy.generate_signal(
            "BTCUSDT",
            0.02,
            0.01,
            Some(50000.0),
            Some(0.55),
        );

        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_portfolio_allocation() {
        let strategy = UncertaintyStrategy::new(0.001, 0.05, 0.6);

        let signals = vec![
            Signal::new("BTCUSDT".to_string(), SignalType::Long, 0.02, 0.01, 0.8, 0.5),
            Signal::new("ETHUSDT".to_string(), SignalType::Long, 0.015, 0.015, 0.7, 0.4),
            Signal::new("SOLUSDT".to_string(), SignalType::Hold, 0.005, 0.04, 0.3, 0.0),
        ];

        let allocations = strategy.allocate_portfolio(&signals, 0.25, 0.8);

        assert!(allocations.contains_key("BTCUSDT"));
        assert!(allocations.contains_key("ETHUSDT"));
        assert!(!allocations.contains_key("SOLUSDT")); // Hold signal
    }
}
