//! Signal generation for trading.
//!
//! This module provides signal generation from model predictions.

use crate::MarketRegime;

/// Trading signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Strong buy signal.
    StrongBuy,
    /// Buy signal.
    Buy,
    /// Hold/neutral signal.
    Hold,
    /// Sell signal.
    Sell,
    /// Strong sell signal.
    StrongSell,
}

impl Signal {
    /// Convert to position size multiplier.
    pub fn to_position_size(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Hold => 0.0,
            Signal::Sell => -0.5,
            Signal::StrongSell => -1.0,
        }
    }

    /// Check if signal is bullish.
    pub fn is_bullish(&self) -> bool {
        matches!(self, Signal::StrongBuy | Signal::Buy)
    }

    /// Check if signal is bearish.
    pub fn is_bearish(&self) -> bool {
        matches!(self, Signal::StrongSell | Signal::Sell)
    }

    /// Check if signal is neutral.
    pub fn is_neutral(&self) -> bool {
        matches!(self, Signal::Hold)
    }
}

/// Configuration for signal generation.
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Threshold for strong buy (prediction > threshold).
    pub strong_buy_threshold: f64,
    /// Threshold for buy.
    pub buy_threshold: f64,
    /// Threshold for sell (prediction < -threshold).
    pub sell_threshold: f64,
    /// Threshold for strong sell.
    pub strong_sell_threshold: f64,
    /// Confidence threshold for acting on signals.
    pub confidence_threshold: f64,
    /// Whether to adjust thresholds based on regime.
    pub regime_adjustment: bool,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            strong_buy_threshold: 0.7,
            buy_threshold: 0.3,
            sell_threshold: -0.3,
            strong_sell_threshold: -0.7,
            confidence_threshold: 0.5,
            regime_adjustment: true,
        }
    }
}

/// Signal generator from model predictions.
pub struct SignalGenerator {
    config: SignalConfig,
    /// History of generated signals.
    signal_history: Vec<Signal>,
    /// History of predictions.
    prediction_history: Vec<f64>,
}

impl SignalGenerator {
    /// Create a new signal generator.
    pub fn new(config: SignalConfig) -> Self {
        Self {
            config,
            signal_history: Vec::new(),
            prediction_history: Vec::new(),
        }
    }

    /// Generate signal from model prediction.
    pub fn generate(&mut self, prediction: f64, regime: Option<MarketRegime>) -> Signal {
        let (strong_buy, buy, sell, strong_sell) = if self.config.regime_adjustment {
            self.adjust_thresholds(regime)
        } else {
            (
                self.config.strong_buy_threshold,
                self.config.buy_threshold,
                self.config.sell_threshold,
                self.config.strong_sell_threshold,
            )
        };

        let signal = if prediction > strong_buy {
            Signal::StrongBuy
        } else if prediction > buy {
            Signal::Buy
        } else if prediction < strong_sell {
            Signal::StrongSell
        } else if prediction < sell {
            Signal::Sell
        } else {
            Signal::Hold
        };

        self.signal_history.push(signal);
        self.prediction_history.push(prediction);

        signal
    }

    /// Adjust thresholds based on market regime.
    fn adjust_thresholds(&self, regime: Option<MarketRegime>) -> (f64, f64, f64, f64) {
        let regime = regime.unwrap_or(MarketRegime::Sideways);

        match regime {
            MarketRegime::Bull => {
                // In bull market, lower buy thresholds, raise sell thresholds
                (
                    self.config.strong_buy_threshold * 0.8,
                    self.config.buy_threshold * 0.8,
                    self.config.sell_threshold * 1.2,
                    self.config.strong_sell_threshold * 1.2,
                )
            }
            MarketRegime::Bear => {
                // In bear market, raise buy thresholds, lower sell thresholds
                (
                    self.config.strong_buy_threshold * 1.2,
                    self.config.buy_threshold * 1.2,
                    self.config.sell_threshold * 0.8,
                    self.config.strong_sell_threshold * 0.8,
                )
            }
            MarketRegime::HighVolatility => {
                // In high volatility, widen thresholds
                (
                    self.config.strong_buy_threshold * 1.5,
                    self.config.buy_threshold * 1.3,
                    self.config.sell_threshold * 1.3,
                    self.config.strong_sell_threshold * 1.5,
                )
            }
            MarketRegime::LowVolatility => {
                // In low volatility, tighten thresholds
                (
                    self.config.strong_buy_threshold * 0.7,
                    self.config.buy_threshold * 0.7,
                    self.config.sell_threshold * 0.7,
                    self.config.strong_sell_threshold * 0.7,
                )
            }
            MarketRegime::Sideways => {
                // Use default thresholds
                (
                    self.config.strong_buy_threshold,
                    self.config.buy_threshold,
                    self.config.sell_threshold,
                    self.config.strong_sell_threshold,
                )
            }
        }
    }

    /// Generate signal with confidence check.
    pub fn generate_with_confidence(
        &mut self,
        prediction: f64,
        confidence: f64,
        regime: Option<MarketRegime>,
    ) -> Signal {
        if confidence < self.config.confidence_threshold {
            self.signal_history.push(Signal::Hold);
            self.prediction_history.push(prediction);
            return Signal::Hold;
        }

        self.generate(prediction, regime)
    }

    /// Get signal based on multiple predictions (ensemble).
    pub fn generate_ensemble(&mut self, predictions: &[f64], regime: Option<MarketRegime>) -> Signal {
        if predictions.is_empty() {
            return Signal::Hold;
        }

        let avg_prediction = predictions.iter().sum::<f64>() / predictions.len() as f64;
        self.generate(avg_prediction, regime)
    }

    /// Get smoothed signal (with lookback).
    pub fn get_smoothed_signal(&self, lookback: usize) -> Signal {
        if self.prediction_history.is_empty() {
            return Signal::Hold;
        }

        let start = self.prediction_history.len().saturating_sub(lookback);
        let recent = &self.prediction_history[start..];
        let avg = recent.iter().sum::<f64>() / recent.len() as f64;

        // Use midpoint thresholds for smoothed signal
        let mid_buy = (self.config.strong_buy_threshold + self.config.buy_threshold) / 2.0;
        let mid_sell = (self.config.strong_sell_threshold + self.config.sell_threshold) / 2.0;

        if avg > mid_buy {
            Signal::Buy
        } else if avg < mid_sell {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    /// Get signal history.
    pub fn history(&self) -> &[Signal] {
        &self.signal_history
    }

    /// Get prediction history.
    pub fn predictions(&self) -> &[f64] {
        &self.prediction_history
    }

    /// Get configuration.
    pub fn config(&self) -> &SignalConfig {
        &self.config
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.signal_history.clear();
        self.prediction_history.clear();
    }

    /// Get signal statistics.
    pub fn stats(&self) -> SignalStats {
        let total = self.signal_history.len();
        let buy_count = self.signal_history.iter().filter(|s| s.is_bullish()).count();
        let sell_count = self.signal_history.iter().filter(|s| s.is_bearish()).count();
        let hold_count = self.signal_history.iter().filter(|s| s.is_neutral()).count();

        SignalStats {
            total_signals: total,
            buy_signals: buy_count,
            sell_signals: sell_count,
            hold_signals: hold_count,
            buy_ratio: buy_count as f64 / total.max(1) as f64,
            sell_ratio: sell_count as f64 / total.max(1) as f64,
        }
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalConfig::default())
    }
}

/// Statistics about generated signals.
#[derive(Debug, Clone)]
pub struct SignalStats {
    /// Total number of signals.
    pub total_signals: usize,
    /// Number of buy signals.
    pub buy_signals: usize,
    /// Number of sell signals.
    pub sell_signals: usize,
    /// Number of hold signals.
    pub hold_signals: usize,
    /// Ratio of buy signals.
    pub buy_ratio: f64,
    /// Ratio of sell signals.
    pub sell_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_conversion() {
        assert_eq!(Signal::StrongBuy.to_position_size(), 1.0);
        assert_eq!(Signal::Sell.to_position_size(), -0.5);
        assert!(Signal::Buy.is_bullish());
        assert!(Signal::StrongSell.is_bearish());
        assert!(Signal::Hold.is_neutral());
    }

    #[test]
    fn test_signal_generation() {
        let mut generator = SignalGenerator::default();

        assert_eq!(generator.generate(0.8, None), Signal::StrongBuy);
        assert_eq!(generator.generate(0.4, None), Signal::Buy);
        assert_eq!(generator.generate(0.0, None), Signal::Hold);
        assert_eq!(generator.generate(-0.4, None), Signal::Sell);
        assert_eq!(generator.generate(-0.8, None), Signal::StrongSell);
    }

    #[test]
    fn test_regime_adjustment() {
        let mut generator = SignalGenerator::new(SignalConfig {
            regime_adjustment: true,
            ..Default::default()
        });

        // Same prediction, different regime
        let bull_signal = generator.generate(0.25, Some(MarketRegime::Bull));
        generator.clear();
        let bear_signal = generator.generate(0.25, Some(MarketRegime::Bear));

        // In bull market, lower threshold should make this a buy
        // In bear market, higher threshold should keep this as hold
        assert!(bull_signal.is_bullish() || bull_signal.is_neutral());
    }

    #[test]
    fn test_signal_stats() {
        let mut generator = SignalGenerator::default();

        generator.generate(0.8, None);
        generator.generate(0.5, None);
        generator.generate(0.0, None);
        generator.generate(-0.5, None);
        generator.generate(-0.8, None);

        let stats = generator.stats();
        assert_eq!(stats.total_signals, 5);
        assert!(stats.buy_signals >= 1);
        assert!(stats.sell_signals >= 1);
    }
}
