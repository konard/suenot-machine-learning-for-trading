//! Trading signal generation from pattern matches.

use crate::graph::FinancialGraph;
use crate::mining::{Pattern, PatternMatch, PatternMatcher, PatternType, PatternChanges};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Strong buy signal
    StrongBuy,
    /// Regular buy signal
    Buy,
    /// Hold / neutral
    Hold,
    /// Regular sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl SignalType {
    /// Convert to numeric value (-2 to +2)
    pub fn to_value(&self) -> i32 {
        match self {
            SignalType::StrongBuy => 2,
            SignalType::Buy => 1,
            SignalType::Hold => 0,
            SignalType::Sell => -1,
            SignalType::StrongSell => -2,
        }
    }

    /// Check if signal is bullish
    pub fn is_bullish(&self) -> bool {
        matches!(self, SignalType::StrongBuy | SignalType::Buy)
    }

    /// Check if signal is bearish
    pub fn is_bearish(&self) -> bool {
        matches!(self, SignalType::StrongSell | SignalType::Sell)
    }
}

/// A trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Signal type
    pub signal_type: SignalType,
    /// Symbol(s) the signal applies to
    pub symbols: Vec<String>,
    /// Signal strength (0-1)
    pub strength: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Reason for signal
    pub reason: String,
    /// Pattern that generated this signal
    pub pattern_type: Option<PatternType>,
    /// Timestamp
    pub timestamp: i64,
}

impl Signal {
    /// Create a new signal
    pub fn new(signal_type: SignalType, symbols: Vec<String>, reason: impl Into<String>) -> Self {
        Self {
            signal_type,
            symbols,
            strength: 0.5,
            confidence: 0.5,
            reason: reason.into(),
            pattern_type: None,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Set signal strength
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set pattern type
    pub fn with_pattern(mut self, pattern_type: PatternType) -> Self {
        self.pattern_type = Some(pattern_type);
        self
    }

    /// Get combined score (strength * confidence)
    pub fn score(&self) -> f64 {
        self.strength * self.confidence
    }
}

/// Signal generator from pattern matching
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Pattern matcher
    matcher: PatternMatcher,
    /// Previous graph for change detection
    previous_graph: Option<FinancialGraph>,
    /// Signal thresholds
    strong_threshold: f64,
    weak_threshold: f64,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(patterns: Vec<Pattern>) -> Self {
        Self {
            matcher: PatternMatcher::new(patterns),
            previous_graph: None,
            strong_threshold: 0.85,
            weak_threshold: 0.7,
        }
    }

    /// Set thresholds
    pub fn with_thresholds(mut self, strong: f64, weak: f64) -> Self {
        self.strong_threshold = strong;
        self.weak_threshold = weak;
        self
    }

    /// Generate signals from current graph
    pub fn generate(&mut self, graph: &FinancialGraph) -> Vec<Signal> {
        let mut signals = Vec::new();

        // Get current pattern matches
        let current_matches = self.matcher.find_matches(graph);

        // Generate signals from current patterns
        for pm in &current_matches {
            if let Some(signal) = self.pattern_to_signal(pm) {
                signals.push(signal);
            }
        }

        // Compare with previous graph if available
        if let Some(ref prev) = self.previous_graph {
            let changes = self.matcher.compare_graphs(prev, graph);
            signals.extend(self.changes_to_signals(&changes));
        }

        // Update previous graph
        self.previous_graph = Some(graph.clone());

        // Aggregate signals by symbol
        signals = self.aggregate_signals(signals);

        signals
    }

    /// Convert a pattern match to a signal
    fn pattern_to_signal(&self, pm: &PatternMatch) -> Option<Signal> {
        let avg_weight = pm.avg_weight();

        let signal_type = match pm.pattern.pattern_type {
            PatternType::Triangle => {
                if avg_weight >= self.strong_threshold {
                    SignalType::StrongBuy
                } else if avg_weight >= self.weak_threshold {
                    SignalType::Buy
                } else {
                    SignalType::Hold
                }
            }
            PatternType::Clique => {
                if avg_weight >= self.strong_threshold {
                    SignalType::StrongBuy
                } else {
                    SignalType::Buy
                }
            }
            PatternType::Star => SignalType::Hold, // Need more context
            PatternType::Chain => SignalType::Hold,
            _ => SignalType::Hold,
        };

        if signal_type == SignalType::Hold {
            return None;
        }

        Some(
            Signal::new(
                signal_type,
                pm.symbols.clone(),
                format!("{} pattern detected", pm.pattern.pattern_type.name()),
            )
            .with_strength(avg_weight)
            .with_confidence(0.5 + avg_weight * 0.5)
            .with_pattern(pm.pattern.pattern_type),
        )
    }

    /// Convert pattern changes to signals
    fn changes_to_signals(&self, changes: &PatternChanges) -> Vec<Signal> {
        let mut signals = Vec::new();

        // New patterns appearing = bullish
        for pm in &changes.appeared {
            let signal_type = match pm.pattern.pattern_type {
                PatternType::Triangle | PatternType::Clique => SignalType::Buy,
                _ => SignalType::Hold,
            };

            if signal_type != SignalType::Hold {
                signals.push(
                    Signal::new(
                        signal_type,
                        pm.symbols.clone(),
                        format!("New {} pattern formed", pm.pattern.pattern_type.name()),
                    )
                    .with_strength(pm.avg_weight())
                    .with_pattern(pm.pattern.pattern_type),
                );
            }
        }

        // Patterns disappearing = bearish
        for pm in &changes.disappeared {
            let signal_type = match pm.pattern.pattern_type {
                PatternType::Triangle | PatternType::Clique => SignalType::Sell,
                _ => SignalType::Hold,
            };

            if signal_type != SignalType::Hold {
                signals.push(
                    Signal::new(
                        signal_type,
                        pm.symbols.clone(),
                        format!("{} pattern broken", pm.pattern.pattern_type.name()),
                    )
                    .with_strength(0.6)
                    .with_pattern(pm.pattern.pattern_type),
                );
            }
        }

        signals
    }

    /// Aggregate multiple signals for the same symbols
    fn aggregate_signals(&self, signals: Vec<Signal>) -> Vec<Signal> {
        let mut by_symbol: HashMap<String, Vec<&Signal>> = HashMap::new();

        for signal in &signals {
            for symbol in &signal.symbols {
                by_symbol
                    .entry(symbol.clone())
                    .or_insert_with(Vec::new)
                    .push(signal);
            }
        }

        let mut aggregated = Vec::new();

        for (symbol, symbol_signals) in by_symbol {
            if symbol_signals.is_empty() {
                continue;
            }

            // Compute weighted average signal
            let total_score: f64 = symbol_signals.iter().map(|s| s.score()).sum();
            let weighted_value: f64 = symbol_signals
                .iter()
                .map(|s| s.signal_type.to_value() as f64 * s.score())
                .sum();

            let avg_value = if total_score > 0.0 {
                weighted_value / total_score
            } else {
                0.0
            };

            let signal_type = if avg_value >= 1.5 {
                SignalType::StrongBuy
            } else if avg_value >= 0.5 {
                SignalType::Buy
            } else if avg_value <= -1.5 {
                SignalType::StrongSell
            } else if avg_value <= -0.5 {
                SignalType::Sell
            } else {
                SignalType::Hold
            };

            if signal_type != SignalType::Hold {
                let avg_strength: f64 = symbol_signals.iter().map(|s| s.strength).sum::<f64>()
                    / symbol_signals.len() as f64;

                aggregated.push(
                    Signal::new(
                        signal_type,
                        vec![symbol],
                        format!("Aggregated from {} patterns", symbol_signals.len()),
                    )
                    .with_strength(avg_strength)
                    .with_confidence(total_score / symbol_signals.len() as f64),
                );
            }
        }

        aggregated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new(SignalType::Buy, vec!["BTCUSDT".to_string()], "Test signal")
            .with_strength(0.8)
            .with_confidence(0.9);

        assert_eq!(signal.signal_type, SignalType::Buy);
        assert_eq!(signal.strength, 0.8);
        assert_eq!(signal.confidence, 0.9);
        assert!(signal.signal_type.is_bullish());
    }

    #[test]
    fn test_signal_score() {
        let signal = Signal::new(SignalType::Buy, vec!["BTCUSDT".to_string()], "Test")
            .with_strength(0.8)
            .with_confidence(0.5);

        assert!((signal.score() - 0.4).abs() < 0.001);
    }
}
