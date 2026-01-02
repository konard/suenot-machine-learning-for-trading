//! Signal generation from GNN predictions

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold / No action
    Hold,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::Hold => write!(f, "HOLD"),
        }
    }
}

/// A trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Symbol
    pub symbol: String,
    /// Signal type
    pub signal_type: SignalType,
    /// Signal strength (0 to 1)
    pub strength: f64,
    /// Confidence level (0 to 1)
    pub confidence: f64,
    /// Current price
    pub price: f64,
    /// Target price (optional)
    pub target_price: Option<f64>,
    /// Stop loss price (optional)
    pub stop_loss: Option<f64>,
    /// Expected return
    pub expected_return: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Reason/explanation
    pub reason: String,
}

impl Signal {
    /// Create a new signal
    pub fn new(
        symbol: impl Into<String>,
        signal_type: SignalType,
        strength: f64,
        confidence: f64,
        price: f64,
        timestamp: u64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            signal_type,
            strength: strength.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            price,
            target_price: None,
            stop_loss: None,
            expected_return: 0.0,
            timestamp,
            reason: String::new(),
        }
    }

    /// Set target price
    pub fn with_target(mut self, target: f64) -> Self {
        self.target_price = Some(target);
        self.expected_return = (target - self.price) / self.price;
        self
    }

    /// Set stop loss
    pub fn with_stop_loss(mut self, stop: f64) -> Self {
        self.stop_loss = Some(stop);
        self
    }

    /// Set reason
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Check if signal is actionable (not hold)
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Hold && self.strength > 0.0
    }

    /// Get risk-reward ratio
    pub fn risk_reward(&self) -> Option<f64> {
        match (self.target_price, self.stop_loss) {
            (Some(target), Some(stop)) => {
                let reward = (target - self.price).abs();
                let risk = (self.price - stop).abs();
                if risk > 0.0 {
                    Some(reward / risk)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Signal generator from GNN outputs
#[derive(Debug)]
pub struct SignalGenerator {
    /// Signal history
    history: VecDeque<Signal>,
    /// Maximum history size
    max_history: usize,
    /// Thresholds for signal generation
    pub buy_threshold: f64,
    pub sell_threshold: f64,
    /// Minimum difference from neutral
    pub min_edge: f64,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            max_history: 1000,
            buy_threshold: 0.55,
            sell_threshold: 0.55,
            min_edge: 0.1,
        }
    }

    /// Configure thresholds
    pub fn with_thresholds(mut self, buy: f64, sell: f64, edge: f64) -> Self {
        self.buy_threshold = buy;
        self.sell_threshold = sell;
        self.min_edge = edge;
        self
    }

    /// Generate signal from direction probabilities
    pub fn generate(
        &mut self,
        symbol: &str,
        price: f64,
        direction_probs: (f64, f64, f64), // (down, neutral, up)
        confidence: f64,
        timestamp: u64,
    ) -> Signal {
        let (p_down, p_neutral, p_up) = direction_probs;

        // Determine signal type and strength
        let (signal_type, strength) = if p_up > self.buy_threshold && p_up - p_down > self.min_edge
        {
            (SignalType::Buy, p_up)
        } else if p_down > self.sell_threshold && p_down - p_up > self.min_edge {
            (SignalType::Sell, p_down)
        } else {
            (SignalType::Hold, p_neutral)
        };

        // Create signal
        let mut signal = Signal::new(symbol, signal_type, strength, confidence, price, timestamp);

        // Set targets based on probabilities
        match signal_type {
            SignalType::Buy => {
                let expected_move = (p_up * 0.03 - p_down * 0.02) * confidence;
                signal = signal
                    .with_target(price * (1.0 + expected_move.max(0.01)))
                    .with_stop_loss(price * (1.0 - 0.02))
                    .with_reason(format!(
                        "GNN prediction: {:.1}% up vs {:.1}% down (conf: {:.1}%)",
                        p_up * 100.0,
                        p_down * 100.0,
                        confidence * 100.0
                    ));
            }
            SignalType::Sell => {
                let expected_move = (p_down * 0.03 - p_up * 0.02) * confidence;
                signal = signal
                    .with_target(price * (1.0 - expected_move.max(0.01)))
                    .with_stop_loss(price * (1.0 + 0.02))
                    .with_reason(format!(
                        "GNN prediction: {:.1}% down vs {:.1}% up (conf: {:.1}%)",
                        p_down * 100.0,
                        p_up * 100.0,
                        confidence * 100.0
                    ));
            }
            SignalType::Hold => {
                signal = signal.with_reason(format!(
                    "No clear edge: up={:.1}%, down={:.1}%, neutral={:.1}%",
                    p_up * 100.0,
                    p_down * 100.0,
                    p_neutral * 100.0
                ));
            }
        }

        // Store in history
        self.history.push_back(signal.clone());
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        signal
    }

    /// Generate signal from raw prediction scores
    pub fn generate_from_scores(
        &mut self,
        symbol: &str,
        price: f64,
        score: f64, // Raw score, positive = bullish
        confidence: f64,
        timestamp: u64,
    ) -> Signal {
        // Convert score to probabilities using softmax-like transformation
        let scaled = score.tanh();
        let p_up = (scaled + 1.0) / 2.0;
        let p_down = 1.0 - p_up;
        let p_neutral = 1.0 - (p_up - 0.5).abs() * 2.0;

        self.generate(symbol, price, (p_down, p_neutral, p_up), confidence, timestamp)
    }

    /// Get signal history for a symbol
    pub fn symbol_history(&self, symbol: &str) -> Vec<&Signal> {
        self.history.iter().filter(|s| s.symbol == symbol).collect()
    }

    /// Get recent signals
    pub fn recent_signals(&self, count: usize) -> Vec<&Signal> {
        self.history.iter().rev().take(count).collect()
    }

    /// Get signal statistics
    pub fn stats(&self) -> SignalStats {
        let total = self.history.len();
        let buys = self.history.iter().filter(|s| s.signal_type == SignalType::Buy).count();
        let sells = self
            .history
            .iter()
            .filter(|s| s.signal_type == SignalType::Sell)
            .count();
        let holds = total - buys - sells;

        let avg_confidence = if total > 0 {
            self.history.iter().map(|s| s.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };

        let avg_strength = if total > 0 {
            self.history.iter().map(|s| s.strength).sum::<f64>() / total as f64
        } else {
            0.0
        };

        SignalStats {
            total_signals: total,
            buy_signals: buys,
            sell_signals: sells,
            hold_signals: holds,
            avg_confidence,
            avg_strength,
        }
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Signal statistics
#[derive(Debug, Clone)]
pub struct SignalStats {
    pub total_signals: usize,
    pub buy_signals: usize,
    pub sell_signals: usize,
    pub hold_signals: usize,
    pub avg_confidence: f64,
    pub avg_strength: f64,
}

/// Combine multiple signals
pub fn combine_signals(signals: &[Signal]) -> Option<Signal> {
    if signals.is_empty() {
        return None;
    }

    let total_weight: f64 = signals.iter().map(|s| s.confidence).sum();
    if total_weight == 0.0 {
        return None;
    }

    // Weighted vote
    let mut buy_weight = 0.0;
    let mut sell_weight = 0.0;
    let mut hold_weight = 0.0;

    for signal in signals {
        let w = signal.confidence * signal.strength;
        match signal.signal_type {
            SignalType::Buy => buy_weight += w,
            SignalType::Sell => sell_weight += w,
            SignalType::Hold => hold_weight += w,
        }
    }

    let total = buy_weight + sell_weight + hold_weight;
    let (signal_type, strength) = if buy_weight >= sell_weight && buy_weight >= hold_weight {
        (SignalType::Buy, buy_weight / total)
    } else if sell_weight >= buy_weight && sell_weight >= hold_weight {
        (SignalType::Sell, sell_weight / total)
    } else {
        (SignalType::Hold, hold_weight / total)
    };

    let avg_price = signals.iter().map(|s| s.price).sum::<f64>() / signals.len() as f64;
    let avg_confidence = total_weight / signals.len() as f64;
    let timestamp = signals.iter().map(|s| s.timestamp).max().unwrap_or(0);

    Some(Signal {
        symbol: signals[0].symbol.clone(),
        signal_type,
        strength,
        confidence: avg_confidence,
        price: avg_price,
        target_price: None,
        stop_loss: None,
        expected_return: 0.0,
        timestamp,
        reason: "Combined signal from multiple sources".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new("BTCUSDT", SignalType::Buy, 0.8, 0.9, 50000.0, 1000);
        assert_eq!(signal.signal_type, SignalType::Buy);
        assert!(signal.is_actionable());
    }

    #[test]
    fn test_signal_generator() {
        let mut gen = SignalGenerator::new();

        // Strong bullish signal
        let signal = gen.generate("BTCUSDT", 50000.0, (0.1, 0.2, 0.7), 0.8, 1000);
        assert_eq!(signal.signal_type, SignalType::Buy);

        // Strong bearish signal
        let signal = gen.generate("BTCUSDT", 50000.0, (0.7, 0.2, 0.1), 0.8, 2000);
        assert_eq!(signal.signal_type, SignalType::Sell);

        // Neutral signal
        let signal = gen.generate("BTCUSDT", 50000.0, (0.3, 0.4, 0.3), 0.8, 3000);
        assert_eq!(signal.signal_type, SignalType::Hold);
    }

    #[test]
    fn test_signal_stats() {
        let mut gen = SignalGenerator::new();

        gen.generate("TEST", 100.0, (0.1, 0.2, 0.7), 0.8, 1000);
        gen.generate("TEST", 100.0, (0.7, 0.2, 0.1), 0.8, 2000);
        gen.generate("TEST", 100.0, (0.3, 0.4, 0.3), 0.8, 3000);

        let stats = gen.stats();
        assert_eq!(stats.total_signals, 3);
        assert_eq!(stats.buy_signals, 1);
        assert_eq!(stats.sell_signals, 1);
    }

    #[test]
    fn test_combine_signals() {
        let signals = vec![
            Signal::new("TEST", SignalType::Buy, 0.8, 0.9, 100.0, 1000),
            Signal::new("TEST", SignalType::Buy, 0.7, 0.8, 101.0, 1001),
            Signal::new("TEST", SignalType::Hold, 0.5, 0.6, 100.5, 1002),
        ];

        let combined = combine_signals(&signals).unwrap();
        assert_eq!(combined.signal_type, SignalType::Buy);
    }

    #[test]
    fn test_risk_reward() {
        let signal = Signal::new("TEST", SignalType::Buy, 0.8, 0.9, 100.0, 1000)
            .with_target(106.0) // 6% profit
            .with_stop_loss(98.0); // 2% loss

        let rr = signal.risk_reward().unwrap();
        assert!((rr - 3.0).abs() < 0.1); // 3:1 risk-reward
    }
}
