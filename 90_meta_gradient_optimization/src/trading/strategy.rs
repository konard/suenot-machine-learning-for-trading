//! Trading strategy implementation using meta-gradient optimization.
//!
//! Provides an online trading strategy that adapts its learning dynamics
//! based on meta-gradient feedback.

use crate::meta_gradient::optimizer::{MetaGradientOptimizer, TaskData};
use crate::trading::signals::{ConfidenceSignal, TradingSignal};

/// Trading strategy with meta-gradient optimization
pub struct TradingStrategy {
    optimizer: MetaGradientOptimizer,
    prediction_threshold: f64,
    feature_buffer: Vec<Vec<f64>>,
    label_buffer: Vec<f64>,
    adaptation_window: usize,
    validation_window: usize,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(
        optimizer: MetaGradientOptimizer,
        prediction_threshold: f64,
        adaptation_window: usize,
        validation_window: usize,
    ) -> Self {
        Self {
            optimizer,
            prediction_threshold,
            feature_buffer: Vec::new(),
            label_buffer: Vec::new(),
            adaptation_window,
            validation_window,
        }
    }

    /// Add new observation to buffer
    pub fn observe(&mut self, features: Vec<f64>, label: f64) {
        self.feature_buffer.push(features);
        self.label_buffer.push(label);

        let max_buffer = self.adaptation_window + self.validation_window + 10;
        if self.feature_buffer.len() > max_buffer {
            let start = self.feature_buffer.len() - max_buffer;
            self.feature_buffer = self.feature_buffer[start..].to_vec();
            self.label_buffer = self.label_buffer[start..].to_vec();
        }
    }

    /// Generate trading signal for current features
    pub fn generate_signal(&mut self, current_features: &[f64]) -> Option<ConfidenceSignal> {
        let total_needed = self.adaptation_window + self.validation_window;
        if self.feature_buffer.len() < total_needed {
            return None;
        }

        let buf_len = self.feature_buffer.len();
        let train_start = buf_len - total_needed;
        let val_start = buf_len - self.validation_window;

        // Create task for meta-gradient update
        let task = TaskData::new(
            self.feature_buffer[train_start..val_start].to_vec(),
            self.label_buffer[train_start..val_start].to_vec(),
            self.feature_buffer[val_start..].to_vec(),
            self.label_buffer[val_start..].to_vec(),
        );

        // Meta-gradient update
        self.optimizer.meta_train_step(&[task.clone()]);

        // Adapt model and predict
        let adapted = self.optimizer.adapt(
            &task.train_features,
            &task.train_labels,
            None,
        );

        let prediction = adapted.predict(current_features);

        Some(ConfidenceSignal::from_prediction(
            prediction,
            self.prediction_threshold,
        ))
    }

    /// Get current learning rates
    pub fn learning_rates(&self) -> Vec<f64> {
        self.optimizer.get_learning_rates()
    }

    /// Get reference to optimizer
    pub fn optimizer(&self) -> &MetaGradientOptimizer {
        &self.optimizer
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.feature_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::TradingModel;

    #[test]
    fn test_strategy_creation() {
        let model = TradingModel::new(4, 16, 1);
        let optimizer = MetaGradientOptimizer::new(model, 0.01, 0.001, 3, false, false);
        let strategy = TradingStrategy::new(optimizer, 0.001, 20, 5);

        assert_eq!(strategy.buffer_size(), 0);
    }

    #[test]
    fn test_observe_and_buffer() {
        let model = TradingModel::new(4, 16, 1);
        let optimizer = MetaGradientOptimizer::new(model, 0.01, 0.001, 3, false, false);
        let mut strategy = TradingStrategy::new(optimizer, 0.001, 20, 5);

        for i in 0..30 {
            strategy.observe(vec![0.1 * i as f64; 4], 0.01);
        }

        assert!(strategy.buffer_size() <= 35); // max_buffer = 20 + 5 + 10
    }

    #[test]
    fn test_generate_signal_insufficient_data() {
        let model = TradingModel::new(4, 16, 1);
        let optimizer = MetaGradientOptimizer::new(model, 0.01, 0.001, 3, false, false);
        let mut strategy = TradingStrategy::new(optimizer, 0.001, 20, 5);

        // Not enough data
        for _ in 0..10 {
            strategy.observe(vec![0.1; 4], 0.01);
        }

        let signal = strategy.generate_signal(&[0.1, 0.2, 0.3, 0.4]);
        assert!(signal.is_none());
    }

    #[test]
    fn test_generate_signal_with_data() {
        let model = TradingModel::new(4, 8, 1);
        let optimizer = MetaGradientOptimizer::new(model, 0.01, 0.001, 2, false, false);
        let mut strategy = TradingStrategy::new(optimizer, 0.001, 20, 5);

        for i in 0..30 {
            strategy.observe(vec![0.01 * i as f64; 4], 0.005);
        }

        let signal = strategy.generate_signal(&[0.1, 0.2, 0.3, 0.4]);
        assert!(signal.is_some());
    }
}
