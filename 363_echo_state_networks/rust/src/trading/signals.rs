//! Signal generation for trading

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TradingSignal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold/neutral
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl TradingSignal {
    /// Convert to position direction (-1, 0, 1)
    pub fn to_direction(&self) -> f64 {
        match self {
            TradingSignal::StrongBuy => 1.0,
            TradingSignal::Buy => 1.0,
            TradingSignal::Hold => 0.0,
            TradingSignal::Sell => -1.0,
            TradingSignal::StrongSell => -1.0,
        }
    }

    /// Convert to position size multiplier
    pub fn to_size_multiplier(&self) -> f64 {
        match self {
            TradingSignal::StrongBuy => 1.0,
            TradingSignal::Buy => 0.5,
            TradingSignal::Hold => 0.0,
            TradingSignal::Sell => 0.5,
            TradingSignal::StrongSell => 1.0,
        }
    }

    /// Create from continuous prediction value
    pub fn from_prediction(value: f64, thresholds: &SignalThresholds) -> Self {
        if value >= thresholds.strong_buy {
            TradingSignal::StrongBuy
        } else if value >= thresholds.buy {
            TradingSignal::Buy
        } else if value <= thresholds.strong_sell {
            TradingSignal::StrongSell
        } else if value <= thresholds.sell {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }
}

/// Thresholds for converting predictions to signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalThresholds {
    pub strong_buy: f64,
    pub buy: f64,
    pub sell: f64,
    pub strong_sell: f64,
}

impl Default for SignalThresholds {
    fn default() -> Self {
        Self {
            strong_buy: 0.7,
            buy: 0.55,
            sell: -0.55,
            strong_sell: -0.7,
        }
    }
}

/// Signal generator from ESN predictions
pub struct SignalGenerator {
    /// Signal thresholds
    thresholds: SignalThresholds,
    /// Smoothing window for predictions
    smoothing_window: usize,
    /// Recent predictions buffer
    prediction_buffer: Vec<f64>,
    /// Confidence threshold
    confidence_threshold: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalGenerator {
    /// Create new signal generator
    pub fn new() -> Self {
        Self {
            thresholds: SignalThresholds::default(),
            smoothing_window: 3,
            prediction_buffer: Vec::new(),
            confidence_threshold: 0.0,
        }
    }

    /// Set thresholds
    pub fn with_thresholds(mut self, thresholds: SignalThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Set smoothing window
    pub fn with_smoothing(mut self, window: usize) -> Self {
        self.smoothing_window = window;
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Generate signal from prediction
    pub fn generate(&mut self, prediction: &Array1<f64>) -> (TradingSignal, f64) {
        let value = prediction[0];

        // Add to buffer
        self.prediction_buffer.push(value);
        if self.prediction_buffer.len() > self.smoothing_window {
            self.prediction_buffer.remove(0);
        }

        // Smooth prediction
        let smoothed = if self.prediction_buffer.len() >= self.smoothing_window {
            self.prediction_buffer.iter().sum::<f64>() / self.prediction_buffer.len() as f64
        } else {
            value
        };

        // Calculate confidence (based on prediction magnitude)
        let confidence = smoothed.abs();

        // Generate signal
        let signal = if confidence < self.confidence_threshold {
            TradingSignal::Hold
        } else {
            TradingSignal::from_prediction(smoothed, &self.thresholds)
        };

        (signal, confidence)
    }

    /// Generate signal with ensemble uncertainty
    pub fn generate_with_uncertainty(
        &mut self,
        mean: &Array1<f64>,
        variance: &Array1<f64>,
    ) -> (TradingSignal, f64) {
        let prediction = mean[0];
        let uncertainty = variance[0].sqrt();

        // Adjust confidence based on uncertainty
        let base_confidence = prediction.abs();
        let adjusted_confidence = base_confidence / (1.0 + uncertainty);

        // More conservative signals when uncertain
        let adjusted_thresholds = if uncertainty > 0.3 {
            SignalThresholds {
                strong_buy: self.thresholds.strong_buy * 1.2,
                buy: self.thresholds.buy * 1.2,
                sell: self.thresholds.sell * 1.2,
                strong_sell: self.thresholds.strong_sell * 1.2,
            }
        } else {
            self.thresholds.clone()
        };

        let signal = TradingSignal::from_prediction(prediction, &adjusted_thresholds);
        (signal, adjusted_confidence)
    }

    /// Reset buffer
    pub fn reset(&mut self) {
        self.prediction_buffer.clear();
    }
}

/// Signal filter for reducing noise
pub struct SignalFilter {
    /// Minimum consecutive signals required
    min_consecutive: usize,
    /// Current signal
    current_signal: TradingSignal,
    /// Consecutive count
    consecutive_count: usize,
    /// Last confirmed signal
    confirmed_signal: TradingSignal,
}

impl SignalFilter {
    /// Create new signal filter
    pub fn new(min_consecutive: usize) -> Self {
        Self {
            min_consecutive,
            current_signal: TradingSignal::Hold,
            consecutive_count: 0,
            confirmed_signal: TradingSignal::Hold,
        }
    }

    /// Filter signal
    pub fn filter(&mut self, signal: TradingSignal) -> TradingSignal {
        if signal == self.current_signal {
            self.consecutive_count += 1;
        } else {
            self.current_signal = signal;
            self.consecutive_count = 1;
        }

        if self.consecutive_count >= self.min_consecutive {
            self.confirmed_signal = self.current_signal;
        }

        self.confirmed_signal
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.current_signal = TradingSignal::Hold;
        self.consecutive_count = 0;
        self.confirmed_signal = TradingSignal::Hold;
    }
}

/// Combine multiple signals
pub struct SignalCombiner {
    /// Weights for each signal source
    weights: Vec<f64>,
    /// Voting threshold
    voting_threshold: f64,
}

impl SignalCombiner {
    /// Create new combiner with equal weights
    pub fn new(n_sources: usize) -> Self {
        let weight = 1.0 / n_sources as f64;
        Self {
            weights: vec![weight; n_sources],
            voting_threshold: 0.5,
        }
    }

    /// Create with custom weights
    pub fn with_weights(weights: Vec<f64>) -> Self {
        let sum: f64 = weights.iter().sum();
        let normalized: Vec<f64> = weights.iter().map(|w| w / sum).collect();
        Self {
            weights: normalized,
            voting_threshold: 0.5,
        }
    }

    /// Combine signals using weighted voting
    pub fn combine(&self, signals: &[TradingSignal]) -> TradingSignal {
        if signals.len() != self.weights.len() {
            return TradingSignal::Hold;
        }

        let mut buy_weight = 0.0;
        let mut sell_weight = 0.0;

        for (signal, weight) in signals.iter().zip(&self.weights) {
            match signal {
                TradingSignal::StrongBuy | TradingSignal::Buy => buy_weight += weight,
                TradingSignal::StrongSell | TradingSignal::Sell => sell_weight += weight,
                TradingSignal::Hold => {}
            }
        }

        if buy_weight >= self.voting_threshold {
            TradingSignal::Buy
        } else if sell_weight >= self.voting_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_prediction() {
        let thresholds = SignalThresholds::default();

        assert_eq!(TradingSignal::from_prediction(0.8, &thresholds), TradingSignal::StrongBuy);
        assert_eq!(TradingSignal::from_prediction(0.6, &thresholds), TradingSignal::Buy);
        assert_eq!(TradingSignal::from_prediction(0.0, &thresholds), TradingSignal::Hold);
        assert_eq!(TradingSignal::from_prediction(-0.6, &thresholds), TradingSignal::Sell);
        assert_eq!(TradingSignal::from_prediction(-0.8, &thresholds), TradingSignal::StrongSell);
    }

    #[test]
    fn test_signal_generator() {
        let mut generator = SignalGenerator::new();

        let pred = Array1::from_vec(vec![0.8]);
        let (signal, _) = generator.generate(&pred);

        assert_eq!(signal, TradingSignal::StrongBuy);
    }

    #[test]
    fn test_signal_filter() {
        let mut filter = SignalFilter::new(3);

        // First two signals don't change confirmed
        assert_eq!(filter.filter(TradingSignal::Buy), TradingSignal::Hold);
        assert_eq!(filter.filter(TradingSignal::Buy), TradingSignal::Hold);

        // Third consecutive confirms
        assert_eq!(filter.filter(TradingSignal::Buy), TradingSignal::Buy);
    }

    #[test]
    fn test_signal_combiner() {
        let combiner = SignalCombiner::new(3);

        let signals = vec![TradingSignal::Buy, TradingSignal::Buy, TradingSignal::Sell];
        assert_eq!(combiner.combine(&signals), TradingSignal::Buy);

        let signals2 = vec![TradingSignal::Hold, TradingSignal::Hold, TradingSignal::Buy];
        assert_eq!(combiner.combine(&signals2), TradingSignal::Hold);
    }
}
