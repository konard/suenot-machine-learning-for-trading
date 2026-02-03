//! SHAP-enhanced trading strategy.

use crate::model::shap::{LinearModel, ShapExplainer};
use crate::trading::signals::{SignalGenerator, TradingSignal};

/// SHAP-enhanced trading strategy.
pub struct TradingStrategy {
    /// Signal generator for feature computation.
    signal_generator: SignalGenerator,
    /// Trading model.
    model: LinearModel,
    /// SHAP explainer.
    explainer: Option<ShapExplainer>,
    /// Feature names.
    feature_names: Vec<String>,
    /// Minimum confidence for trading.
    min_confidence: f64,
    /// Minimum SHAP stability for trading.
    min_stability: f64,
}

impl TradingStrategy {
    /// Create a new trading strategy.
    pub fn new(
        model: LinearModel,
        feature_names: Vec<String>,
        min_confidence: f64,
        min_stability: f64,
    ) -> Self {
        Self {
            signal_generator: SignalGenerator::default(),
            model,
            explainer: None,
            feature_names,
            min_confidence,
            min_stability,
        }
    }

    /// Initialize SHAP explainer with background data.
    pub fn init_explainer(&mut self, background_data: Vec<Vec<f64>>, base_value: f64) {
        self.explainer = Some(ShapExplainer::new(
            self.feature_names.clone(),
            base_value,
            background_data,
            100, // n_samples for KernelSHAP
        ));
    }

    /// Generate trading signal with SHAP explanation.
    pub fn generate_signal(&self, prices: &[f64], volumes: &[f64]) -> Option<TradingSignal> {
        let features = self.signal_generator.compute_features(prices, volumes);
        if features.is_empty() {
            return None;
        }

        // Get model prediction
        let probability = self.model.predict(&features);

        // Get SHAP explanation if explainer is available
        let shap_values = self.explainer.as_ref().map(|exp| {
            let explanation = exp.explain(|x| self.model.predict(x), &features);
            explanation.shap_values
        });

        Some(TradingSignal::new(probability, shap_values))
    }

    /// Check if signal should be executed.
    pub fn should_execute(&self, signal: &TradingSignal) -> bool {
        signal.is_actionable(self.min_confidence, self.min_stability)
    }

    /// Get position recommendation.
    pub fn recommended_position(&self, signal: &TradingSignal, max_position: f64) -> f64 {
        if !self.should_execute(signal) {
            return 0.0;
        }

        signal.direction as f64 * signal.position_size(max_position)
    }
}

/// Builder for creating trading strategies.
pub struct StrategyBuilder {
    weights: Option<Vec<f64>>,
    bias: f64,
    feature_names: Vec<String>,
    min_confidence: f64,
    min_stability: f64,
}

impl StrategyBuilder {
    /// Create a new strategy builder.
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: 0.0,
            feature_names: vec![
                "returns_1".to_string(),
                "returns_5".to_string(),
                "returns_10".to_string(),
                "volatility".to_string(),
                "rsi".to_string(),
                "volume_ratio".to_string(),
                "price_ma_ratio".to_string(),
            ],
            min_confidence: 0.1,
            min_stability: 0.3,
        }
    }

    /// Set model weights.
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set model bias.
    pub fn bias(mut self, bias: f64) -> Self {
        self.bias = bias;
        self
    }

    /// Set feature names.
    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Set minimum confidence threshold.
    pub fn min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Set minimum SHAP stability threshold.
    pub fn min_stability(mut self, stability: f64) -> Self {
        self.min_stability = stability;
        self
    }

    /// Build the trading strategy.
    pub fn build(self) -> TradingStrategy {
        let weights = self.weights.unwrap_or_else(|| {
            vec![0.5, 0.3, 0.2, -0.4, 0.1, 0.15, 0.25] // Default weights
        });

        let model = LinearModel::new(weights, self.bias);

        TradingStrategy::new(
            model,
            self.feature_names,
            self.min_confidence,
            self.min_stability,
        )
    }
}

impl Default for StrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}
