//! Trading signal generation for FNet.

/// Trading signal type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold/no action
    Hold,
}

/// Trading signal with metadata.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal type
    pub signal: Signal,
    /// Prediction value
    pub prediction: f64,
    /// Confidence score [0, 1]
    pub confidence: f64,
    /// Timestamp
    pub timestamp: i64,
    /// Current price
    pub price: f64,
}

impl TradingSignal {
    /// Create a new trading signal.
    pub fn new(
        signal: Signal,
        prediction: f64,
        confidence: f64,
        timestamp: i64,
        price: f64,
    ) -> Self {
        Self {
            signal,
            prediction,
            confidence,
            timestamp,
            price,
        }
    }

    /// Check if this is an actionable signal (not Hold).
    pub fn is_actionable(&self) -> bool {
        self.signal != Signal::Hold
    }
}

/// Signal generator configuration.
#[derive(Debug, Clone)]
pub struct SignalGeneratorConfig {
    /// Threshold for prediction to trigger signal
    pub threshold: f64,
    /// Minimum confidence to act on signal
    pub confidence_threshold: f64,
    /// Position sizing (fraction of capital)
    pub position_size: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Maximum holding period (in timesteps)
    pub max_holding_period: usize,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.001,
            confidence_threshold: 0.4,
            position_size: 1.0,
            stop_loss: 0.02,
            take_profit: 0.04,
            max_holding_period: 24,
        }
    }
}

/// Signal generator using FNet predictions.
pub struct SignalGenerator {
    config: SignalGeneratorConfig,
    /// Historical predictions for confidence estimation
    prediction_history: Vec<f64>,
    /// Historical actual returns for confidence calibration
    return_history: Vec<f64>,
}

impl SignalGenerator {
    /// Create a new signal generator.
    pub fn new(config: SignalGeneratorConfig) -> Self {
        Self {
            config,
            prediction_history: Vec::new(),
            return_history: Vec::new(),
        }
    }

    /// Generate a trading signal from prediction.
    pub fn generate_signal(
        &mut self,
        prediction: f64,
        timestamp: i64,
        price: f64,
    ) -> TradingSignal {
        // Calculate confidence based on prediction magnitude
        let confidence = self.calculate_confidence(prediction);

        // Determine signal
        let signal = if confidence < self.config.confidence_threshold {
            Signal::Hold
        } else if prediction > self.config.threshold {
            Signal::Buy
        } else if prediction < -self.config.threshold {
            Signal::Sell
        } else {
            Signal::Hold
        };

        // Update history
        self.prediction_history.push(prediction);

        TradingSignal::new(signal, prediction, confidence, timestamp, price)
    }

    /// Update with actual return for calibration.
    pub fn update_return(&mut self, actual_return: f64) {
        self.return_history.push(actual_return);

        // Keep limited history
        if self.return_history.len() > 1000 {
            self.return_history.remove(0);
        }
        if self.prediction_history.len() > 1000 {
            self.prediction_history.remove(0);
        }
    }

    /// Calculate confidence score.
    fn calculate_confidence(&self, prediction: f64) -> f64 {
        // Simple confidence based on prediction magnitude
        let base_confidence = (prediction.abs() / self.config.threshold).min(1.0);

        // Adjust based on historical accuracy if available
        if self.prediction_history.len() < 10 || self.return_history.len() < 10 {
            return base_confidence;
        }

        // Calculate recent accuracy
        let recent_preds = &self.prediction_history[self.prediction_history.len() - 10..];
        let recent_returns = &self.return_history[self.return_history.len() - 10..];

        let correct_direction: usize = recent_preds
            .iter()
            .zip(recent_returns.iter())
            .filter(|(&p, &r)| (p > 0.0 && r > 0.0) || (p < 0.0 && r < 0.0))
            .count();

        let accuracy = correct_direction as f64 / 10.0;

        // Blend base confidence with accuracy
        base_confidence * 0.5 + accuracy * 0.5
    }

    /// Get signal generator configuration.
    pub fn config(&self) -> &SignalGeneratorConfig {
        &self.config
    }

    /// Get prediction accuracy if enough history.
    pub fn get_accuracy(&self) -> Option<f64> {
        if self.prediction_history.len() < 20 || self.return_history.len() < 20 {
            return None;
        }

        let min_len = self.prediction_history.len().min(self.return_history.len());
        // Only count as correct when both prediction and return have same non-zero sign
        // Zero predictions should not be counted as correct by default
        let correct: usize = self.prediction_history[..min_len]
            .iter()
            .zip(self.return_history[..min_len].iter())
            .filter(|(&p, &r)| (p > 0.0 && r > 0.0) || (p < 0.0 && r < 0.0))
            .count();

        Some(correct as f64 / min_len as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let config = SignalGeneratorConfig::default();
        let mut generator = SignalGenerator::new(config);

        // Strong positive prediction should give Buy
        let signal = generator.generate_signal(0.02, 1700000000, 50000.0);
        assert_eq!(signal.signal, Signal::Buy);
        assert!(signal.confidence > 0.0);

        // Strong negative prediction should give Sell
        let signal = generator.generate_signal(-0.02, 1700003600, 49500.0);
        assert_eq!(signal.signal, Signal::Sell);

        // Weak prediction should give Hold
        let signal = generator.generate_signal(0.0001, 1700007200, 49800.0);
        assert_eq!(signal.signal, Signal::Hold);
    }

    #[test]
    fn test_confidence_calibration() {
        let config = SignalGeneratorConfig::default();
        let mut generator = SignalGenerator::new(config);

        // Add some history with correct predictions
        for i in 0..20 {
            generator.prediction_history.push(0.01);
            generator.return_history.push(0.005); // Same direction
        }

        let accuracy = generator.get_accuracy();
        assert!(accuracy.is_some());
        assert_eq!(accuracy.unwrap(), 1.0); // All correct direction
    }

    #[test]
    fn test_trading_signal_actionable() {
        let buy = TradingSignal::new(Signal::Buy, 0.01, 0.8, 1700000000, 50000.0);
        assert!(buy.is_actionable());

        let hold = TradingSignal::new(Signal::Hold, 0.0001, 0.2, 1700000000, 50000.0);
        assert!(!hold.is_actionable());
    }
}
