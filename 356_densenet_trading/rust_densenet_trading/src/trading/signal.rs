//! Trading Signals
//!
//! Trading signal generation and management.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Trading signal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalDirection {
    Long,
    Short,
    Neutral,
}

impl SignalDirection {
    /// Convert to position multiplier
    pub fn to_multiplier(&self) -> f64 {
        match self {
            SignalDirection::Long => 1.0,
            SignalDirection::Short => -1.0,
            SignalDirection::Neutral => 0.0,
        }
    }

    /// Create from prediction class
    pub fn from_class(class: usize) -> Self {
        match class {
            0 => SignalDirection::Short,
            2 => SignalDirection::Long,
            _ => SignalDirection::Neutral,
        }
    }
}

impl std::fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalDirection::Long => write!(f, "LONG"),
            SignalDirection::Short => write!(f, "SHORT"),
            SignalDirection::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// A trading signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Signal direction
    pub direction: SignalDirection,

    /// Confidence level [0, 1]
    pub confidence: f64,

    /// Timestamp when signal was generated
    pub timestamp: u64,

    /// Current price when signal was generated
    pub price: f64,

    /// Model probabilities [short, hold, long]
    pub probabilities: Vec<f64>,

    /// Suggested position size [0, 1]
    pub position_size: f64,

    /// Expected return (if available)
    pub expected_return: Option<f64>,

    /// Stop loss price (if available)
    pub stop_loss: Option<f64>,

    /// Take profit price (if available)
    pub take_profit: Option<f64>,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(
        direction: SignalDirection,
        confidence: f64,
        timestamp: u64,
        price: f64,
        probabilities: Vec<f64>,
    ) -> Self {
        Self {
            direction,
            confidence,
            timestamp,
            price,
            probabilities,
            position_size: confidence, // Default to confidence
            expected_return: None,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Create from model prediction
    pub fn from_prediction(
        probabilities: &[f64],
        confidence: f64,
        timestamp: u64,
        price: f64,
        threshold: f64,
    ) -> Self {
        let direction = if confidence < threshold {
            SignalDirection::Neutral
        } else if probabilities.len() >= 3 {
            if probabilities[2] > probabilities[0] {
                SignalDirection::Long
            } else {
                SignalDirection::Short
            }
        } else {
            SignalDirection::Neutral
        };

        Self::new(
            direction,
            confidence,
            timestamp,
            price,
            probabilities.to_vec(),
        )
    }

    /// Set stop loss
    pub fn with_stop_loss(mut self, stop_loss: f64) -> Self {
        self.stop_loss = Some(stop_loss);
        self
    }

    /// Set take profit
    pub fn with_take_profit(mut self, take_profit: f64) -> Self {
        self.take_profit = Some(take_profit);
        self
    }

    /// Set position size
    pub fn with_position_size(mut self, size: f64) -> Self {
        self.position_size = size.max(0.0).min(1.0);
        self
    }

    /// Set expected return
    pub fn with_expected_return(mut self, expected: f64) -> Self {
        self.expected_return = Some(expected);
        self
    }

    /// Check if signal is actionable (not neutral with sufficient confidence)
    pub fn is_actionable(&self, min_confidence: f64) -> bool {
        self.direction != SignalDirection::Neutral && self.confidence >= min_confidence
    }

    /// Calculate stop loss based on ATR
    pub fn calculate_stop_loss(&self, atr: f64, multiplier: f64) -> f64 {
        match self.direction {
            SignalDirection::Long => self.price - atr * multiplier,
            SignalDirection::Short => self.price + atr * multiplier,
            SignalDirection::Neutral => self.price,
        }
    }

    /// Calculate take profit based on risk-reward ratio
    pub fn calculate_take_profit(&self, stop_loss: f64, risk_reward: f64) -> f64 {
        let risk = (self.price - stop_loss).abs();
        let reward = risk * risk_reward;

        match self.direction {
            SignalDirection::Long => self.price + reward,
            SignalDirection::Short => self.price - reward,
            SignalDirection::Neutral => self.price,
        }
    }

    /// Get risk in percentage
    pub fn risk_percent(&self) -> Option<f64> {
        self.stop_loss.map(|sl| ((self.price - sl) / self.price).abs() * 100.0)
    }

    /// Get potential reward in percentage
    pub fn reward_percent(&self) -> Option<f64> {
        self.take_profit.map(|tp| ((tp - self.price) / self.price).abs() * 100.0)
    }

    /// Get risk-reward ratio
    pub fn risk_reward_ratio(&self) -> Option<f64> {
        match (self.risk_percent(), self.reward_percent()) {
            (Some(risk), Some(reward)) if risk > 0.0 => Some(reward / risk),
            _ => None,
        }
    }
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Signal({}, conf={:.2}%, price={:.2}, size={:.2}%)",
            self.direction,
            self.confidence * 100.0,
            self.price,
            self.position_size * 100.0
        )
    }
}

/// Signal history for tracking
#[derive(Debug, Clone, Default)]
pub struct SignalHistory {
    pub signals: Vec<TradingSignal>,
}

impl SignalHistory {
    pub fn new() -> Self {
        Self { signals: Vec::new() }
    }

    pub fn push(&mut self, signal: TradingSignal) {
        self.signals.push(signal);
    }

    pub fn len(&self) -> usize {
        self.signals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    pub fn last(&self) -> Option<&TradingSignal> {
        self.signals.last()
    }

    /// Get signals in a time range
    pub fn in_range(&self, start: u64, end: u64) -> Vec<&TradingSignal> {
        self.signals
            .iter()
            .filter(|s| s.timestamp >= start && s.timestamp <= end)
            .collect()
    }

    /// Count signals by direction
    pub fn count_by_direction(&self) -> (usize, usize, usize) {
        let mut long = 0;
        let mut short = 0;
        let mut neutral = 0;

        for signal in &self.signals {
            match signal.direction {
                SignalDirection::Long => long += 1,
                SignalDirection::Short => short += 1,
                SignalDirection::Neutral => neutral += 1,
            }
        }

        (long, short, neutral)
    }

    /// Average confidence
    pub fn avg_confidence(&self) -> f64 {
        if self.signals.is_empty() {
            return 0.0;
        }
        self.signals.iter().map(|s| s.confidence).sum::<f64>() / self.signals.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_direction() {
        assert_eq!(SignalDirection::Long.to_multiplier(), 1.0);
        assert_eq!(SignalDirection::Short.to_multiplier(), -1.0);
        assert_eq!(SignalDirection::Neutral.to_multiplier(), 0.0);
    }

    #[test]
    fn test_signal_from_prediction() {
        let probs = vec![0.1, 0.2, 0.7]; // High long probability
        let signal = TradingSignal::from_prediction(&probs, 0.8, 1000, 42000.0, 0.3);

        assert_eq!(signal.direction, SignalDirection::Long);
        assert!((signal.confidence - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_signal_neutral_low_confidence() {
        let probs = vec![0.4, 0.2, 0.4];
        let signal = TradingSignal::from_prediction(&probs, 0.2, 1000, 42000.0, 0.5);

        assert_eq!(signal.direction, SignalDirection::Neutral);
    }

    #[test]
    fn test_stop_loss_calculation() {
        let signal = TradingSignal::new(
            SignalDirection::Long,
            0.8,
            1000,
            100.0,
            vec![0.1, 0.1, 0.8],
        );

        let sl = signal.calculate_stop_loss(2.0, 1.5);
        assert!((sl - 97.0).abs() < 0.001); // 100 - 2*1.5 = 97
    }

    #[test]
    fn test_take_profit_calculation() {
        let signal = TradingSignal::new(
            SignalDirection::Long,
            0.8,
            1000,
            100.0,
            vec![0.1, 0.1, 0.8],
        );

        let tp = signal.calculate_take_profit(97.0, 2.0); // 2:1 reward
        assert!((tp - 106.0).abs() < 0.001); // risk=3, reward=6
    }

    #[test]
    fn test_signal_history() {
        let mut history = SignalHistory::new();

        history.push(TradingSignal::new(
            SignalDirection::Long,
            0.8,
            1000,
            100.0,
            vec![],
        ));
        history.push(TradingSignal::new(
            SignalDirection::Short,
            0.7,
            2000,
            101.0,
            vec![],
        ));

        let (long, short, neutral) = history.count_by_direction();
        assert_eq!(long, 1);
        assert_eq!(short, 1);
        assert_eq!(neutral, 0);

        assert!((history.avg_confidence() - 0.75).abs() < 0.001);
    }
}
