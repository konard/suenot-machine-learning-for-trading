//! Trading strategy implementation.

use super::signals::TradingSignal;
use crate::deeplift::{Attribution, DeepLIFT};
use crate::model::TradingNetwork;

/// Trading strategy with DeepLIFT explanations.
#[derive(Debug)]
pub struct TradingStrategy {
    /// Neural network model
    pub model: TradingNetwork,
    /// DeepLIFT explainer
    pub explainer: DeepLIFT,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Signal threshold
    pub threshold: f64,
}

impl TradingStrategy {
    /// Create a new trading strategy.
    pub fn new(
        model: TradingNetwork,
        reference: Vec<f64>,
        feature_names: Vec<String>,
        threshold: f64,
    ) -> Self {
        let mut explainer =
            DeepLIFT::new(reference, crate::deeplift::AttributionRule::Rescale);
        explainer.set_network(model.weights.clone(), model.biases.clone());

        Self {
            model,
            explainer,
            feature_names,
            threshold,
        }
    }

    /// Generate trading signal for given features.
    pub fn generate_signal(&self, features: &[f64]) -> TradingSignal {
        let prediction = self.model.forward(features);
        TradingSignal::from_prediction(prediction, self.threshold)
    }

    /// Generate signal with explanation.
    pub fn generate_signal_with_explanation(
        &self,
        features: &[f64],
    ) -> (TradingSignal, Attribution) {
        let prediction = self.model.forward(features);
        let signal = TradingSignal::from_prediction(prediction, self.threshold);
        let attribution = self
            .explainer
            .attribute(features, self.feature_names.clone());

        (signal, attribution)
    }

    /// Get top contributing features for a signal.
    pub fn explain_signal(&self, features: &[f64], n: usize) -> Vec<(String, f64)> {
        let attribution = self
            .explainer
            .attribute(features, self.feature_names.clone());
        attribution.top_features(n)
    }
}

/// Strategy performance metrics.
#[derive(Debug, Clone, Default)]
pub struct StrategyMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: usize,
}

impl std::fmt::Display for StrategyMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Strategy Metrics:")?;
        writeln!(f, "  Total Return: {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "  Sharpe Ratio: {:.3}", self.sharpe_ratio)?;
        writeln!(f, "  Max Drawdown: {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Win Rate: {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  Num Trades: {}", self.num_trades)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::features::feature_names;

    #[test]
    fn test_strategy_creation() {
        let model = TradingNetwork::new(vec![11, 32, 1]);
        let reference = vec![0.0; 11];
        let names = feature_names();

        let strategy = TradingStrategy::new(model, reference, names, 0.001);
        assert_eq!(strategy.threshold, 0.001);
    }

    #[test]
    fn test_signal_generation() {
        let model = TradingNetwork::new(vec![11, 32, 1]);
        let reference = vec![0.0; 11];
        let names = feature_names();

        let strategy = TradingStrategy::new(model, reference, names, 0.001);
        let features = vec![0.01, 0.02, 0.03, 0.0, 0.0, 0.01, 0.02, 0.5, 0.001, 0.0, 0.1];

        let signal = strategy.generate_signal(&features);
        // Just verify it doesn't panic and returns a valid signal
        let _ = signal.position();
    }
}
