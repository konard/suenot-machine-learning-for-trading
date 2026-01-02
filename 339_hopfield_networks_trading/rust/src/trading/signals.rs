//! Trading signal generation based on Hopfield Network pattern matching
//!
//! This module provides utilities for converting pattern matching results
//! into actionable trading signals.

use crate::hopfield::{HopfieldNetwork, RetrievalResult};
use crate::trading::patterns::{LabeledPattern, PatternLabel};
use crate::trading::MarketRegime;

/// Trading signal
#[derive(Debug, Clone)]
pub struct Signal {
    /// Signal direction and strength (-1.0 to 1.0)
    pub strength: f64,
    /// Signal type
    pub signal_type: SignalType,
    /// Confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Recommended position size (0.0 to 1.0)
    pub position_size: f64,
    /// Stop loss percentage
    pub stop_loss: Option<f64>,
    /// Take profit percentage
    pub take_profit: Option<f64>,
    /// Signal metadata
    pub metadata: SignalMetadata,
}

/// Signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Enter long position
    Long,
    /// Enter short position
    Short,
    /// Exit current position
    Exit,
    /// Hold current position
    Hold,
    /// No clear signal
    None,
}

impl SignalType {
    /// Check if signal is actionable
    pub fn is_actionable(&self) -> bool {
        matches!(self, SignalType::Long | SignalType::Short | SignalType::Exit)
    }
}

/// Signal metadata
#[derive(Debug, Clone, Default)]
pub struct SignalMetadata {
    /// Index of matched pattern
    pub matched_pattern_idx: Option<usize>,
    /// Market regime at signal time
    pub regime: Option<MarketRegime>,
    /// Number of confirming patterns
    pub confirming_patterns: usize,
    /// Pattern similarity score
    pub similarity: f64,
    /// Historical win rate of matched pattern
    pub historical_win_rate: Option<f64>,
}

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalGeneratorConfig {
    /// Minimum confidence to generate signal
    pub min_confidence: f64,
    /// Minimum strength to generate signal
    pub min_strength: f64,
    /// Default stop loss percentage
    pub default_stop_loss: f64,
    /// Default take profit percentage
    pub default_take_profit: f64,
    /// Risk per trade (Kelly fraction)
    pub risk_per_trade: f64,
    /// Use regime filter
    pub use_regime_filter: bool,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            min_strength: 0.2,
            default_stop_loss: 0.02,
            default_take_profit: 0.04,
            risk_per_trade: 0.02,
            use_regime_filter: true,
        }
    }
}

/// Signal generator using Hopfield Network pattern matching
///
/// # Example
///
/// ```rust
/// use hopfield_trading::trading::signals::{SignalGenerator, SignalGeneratorConfig};
/// use hopfield_trading::hopfield::modern::ModernHopfield;
/// use hopfield_trading::hopfield::HopfieldNetwork;
///
/// let config = SignalGeneratorConfig::default();
/// let mut generator = SignalGenerator::new(config);
///
/// // Create network and store patterns
/// let mut network = ModernHopfield::new(10);
/// let patterns = vec![
///     vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
/// ];
/// network.store_patterns(&patterns);
///
/// // Generate signal
/// let query = vec![0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05];
/// let signal = generator.generate(&network, &query, None);
/// ```
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    config: SignalGeneratorConfig,
    /// Stored labeled patterns for signal generation
    labeled_patterns: Vec<LabeledPattern>,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalGeneratorConfig::default())
    }
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalGeneratorConfig) -> Self {
        Self {
            config,
            labeled_patterns: Vec::new(),
        }
    }

    /// Set labeled patterns for reference
    pub fn set_labeled_patterns(&mut self, patterns: Vec<LabeledPattern>) {
        self.labeled_patterns = patterns;
    }

    /// Generate trading signal from pattern match
    pub fn generate<N: HopfieldNetwork>(
        &self,
        network: &N,
        query: &[f64],
        regime: Option<MarketRegime>,
    ) -> Signal {
        // Get retrieval result
        let result = self.get_retrieval_result(network, query);

        // Determine signal based on match
        let (strength, signal_type) = self.determine_signal(&result, regime);

        // Calculate position size based on confidence and regime
        let position_size = self.calculate_position_size(result.confidence, regime);

        // Calculate risk parameters
        let (stop_loss, take_profit) = self.calculate_risk_params(regime);

        Signal {
            strength,
            signal_type,
            confidence: result.confidence,
            position_size,
            stop_loss: Some(stop_loss),
            take_profit: Some(take_profit),
            metadata: SignalMetadata {
                matched_pattern_idx: result.matched_index,
                regime,
                confirming_patterns: 0,
                similarity: result.confidence,
                historical_win_rate: self.get_historical_win_rate(result.matched_index),
            },
        }
    }

    /// Get retrieval result from network (generic over network type)
    fn get_retrieval_result<N: HopfieldNetwork>(&self, network: &N, query: &[f64]) -> RetrievalResult {
        let pattern = network.retrieve(query);
        let energy = network.energy(&pattern);

        // Calculate confidence based on convergence quality
        let confidence = self.calculate_confidence(&pattern, query);

        RetrievalResult {
            pattern,
            iterations: 0,
            energy,
            converged: true,
            matched_index: self.find_matched_index(query),
            confidence,
        }
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, retrieved: &[f64], query: &[f64]) -> f64 {
        if retrieved.len() != query.len() {
            return 0.0;
        }

        // Cosine similarity
        let dot: f64 = retrieved.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
        let norm_r: f64 = retrieved.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm_q: f64 = query.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if norm_r < 1e-10 || norm_q < 1e-10 {
            return 0.0;
        }

        let similarity = dot / (norm_r * norm_q);

        // Convert to confidence (0 to 1)
        ((similarity + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Find the index of matched pattern in labeled patterns
    fn find_matched_index(&self, query: &[f64]) -> Option<usize> {
        if self.labeled_patterns.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (idx, lp) in self.labeled_patterns.iter().enumerate() {
            if lp.pattern.len() != query.len() {
                continue;
            }

            let sim = self.calculate_confidence(&lp.pattern, query);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }

        if best_sim > 0.5 {
            Some(best_idx)
        } else {
            None
        }
    }

    /// Determine signal type and strength from retrieval result
    fn determine_signal(
        &self,
        result: &RetrievalResult,
        regime: Option<MarketRegime>,
    ) -> (f64, SignalType) {
        // Check confidence threshold
        if result.confidence < self.config.min_confidence {
            return (0.0, SignalType::None);
        }

        // Get label from matched pattern
        let label = result
            .matched_index
            .and_then(|idx| self.labeled_patterns.get(idx))
            .map(|lp| lp.label)
            .unwrap_or(PatternLabel::Hold);

        // Convert label to signal
        let base_strength = label.to_value();

        // Apply regime filter
        let regime_multiplier = if self.config.use_regime_filter {
            regime.map(|r| r.position_bias()).unwrap_or(1.0)
        } else {
            1.0
        };

        let strength = base_strength * result.confidence * regime_multiplier;

        let signal_type = if strength.abs() < self.config.min_strength {
            SignalType::Hold
        } else if strength > 0.0 {
            SignalType::Long
        } else {
            SignalType::Short
        };

        (strength, signal_type)
    }

    /// Calculate position size based on confidence and regime
    fn calculate_position_size(&self, confidence: f64, regime: Option<MarketRegime>) -> f64 {
        let base_size = self.config.risk_per_trade;

        // Adjust for confidence
        let confidence_factor = confidence.powf(2.0);

        // Adjust for regime
        let regime_factor = regime.map(|r| r.risk_multiplier()).unwrap_or(1.0);

        (base_size * confidence_factor * regime_factor).clamp(0.0, 0.1)
    }

    /// Calculate risk parameters (stop loss and take profit)
    fn calculate_risk_params(&self, regime: Option<MarketRegime>) -> (f64, f64) {
        let base_sl = self.config.default_stop_loss;
        let base_tp = self.config.default_take_profit;

        // Adjust for volatility regime
        let vol_multiplier = match regime {
            Some(MarketRegime::HighVolatility) => 1.5,
            Some(MarketRegime::LowVolatility) => 0.7,
            _ => 1.0,
        };

        (base_sl * vol_multiplier, base_tp * vol_multiplier)
    }

    /// Get historical win rate for a pattern index
    fn get_historical_win_rate(&self, idx: Option<usize>) -> Option<f64> {
        idx.and_then(|i| {
            self.labeled_patterns.get(i).map(|lp| {
                match lp.label {
                    PatternLabel::StrongBuy | PatternLabel::StrongSell => 0.65,
                    PatternLabel::Buy | PatternLabel::Sell => 0.55,
                    PatternLabel::Hold => 0.50,
                }
            })
        })
    }

    /// Generate multiple signals for ensemble decision
    pub fn generate_ensemble<N: HopfieldNetwork>(
        &self,
        networks: &[&N],
        query: &[f64],
        regime: Option<MarketRegime>,
    ) -> Signal {
        if networks.is_empty() {
            return Signal {
                strength: 0.0,
                signal_type: SignalType::None,
                confidence: 0.0,
                position_size: 0.0,
                stop_loss: None,
                take_profit: None,
                metadata: SignalMetadata::default(),
            };
        }

        // Collect signals from all networks
        let signals: Vec<Signal> = networks
            .iter()
            .map(|n| self.generate(*n, query, regime))
            .collect();

        // Aggregate signals
        let avg_strength: f64 = signals.iter().map(|s| s.strength).sum::<f64>() / signals.len() as f64;
        let avg_confidence: f64 = signals.iter().map(|s| s.confidence).sum::<f64>() / signals.len() as f64;

        // Determine consensus signal type
        let long_count = signals.iter().filter(|s| s.signal_type == SignalType::Long).count();
        let short_count = signals.iter().filter(|s| s.signal_type == SignalType::Short).count();

        let signal_type = if long_count > signals.len() / 2 {
            SignalType::Long
        } else if short_count > signals.len() / 2 {
            SignalType::Short
        } else {
            SignalType::Hold
        };

        Signal {
            strength: avg_strength,
            signal_type,
            confidence: avg_confidence,
            position_size: self.calculate_position_size(avg_confidence, regime),
            stop_loss: Some(self.config.default_stop_loss),
            take_profit: Some(self.config.default_take_profit),
            metadata: SignalMetadata {
                confirming_patterns: long_count.max(short_count),
                ..Default::default()
            },
        }
    }
}

/// Signal evaluator for backtesting
#[derive(Debug, Clone)]
pub struct SignalEvaluator {
    /// Total signals evaluated
    pub total_signals: usize,
    /// Correct predictions
    pub correct: usize,
    /// Total profit/loss
    pub total_pnl: f64,
    /// Win count
    pub wins: usize,
    /// Loss count
    pub losses: usize,
    /// Maximum drawdown
    pub max_drawdown: f64,
}

impl Default for SignalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalEvaluator {
    /// Create a new evaluator
    pub fn new() -> Self {
        Self {
            total_signals: 0,
            correct: 0,
            total_pnl: 0.0,
            wins: 0,
            losses: 0,
            max_drawdown: 0.0,
        }
    }

    /// Evaluate a signal against actual outcome
    pub fn evaluate(&mut self, signal: &Signal, actual_return: f64) {
        self.total_signals += 1;

        // Check if direction was correct
        let predicted_direction = signal.strength.signum();
        let actual_direction = actual_return.signum();

        if predicted_direction == actual_direction {
            self.correct += 1;
        }

        // Calculate P&L
        let pnl = signal.strength * actual_return * signal.position_size;
        self.total_pnl += pnl;

        if pnl > 0.0 {
            self.wins += 1;
        } else if pnl < 0.0 {
            self.losses += 1;
        }

        // Track drawdown (simplified)
        if self.total_pnl < self.max_drawdown {
            self.max_drawdown = self.total_pnl;
        }
    }

    /// Get accuracy
    pub fn accuracy(&self) -> f64 {
        if self.total_signals == 0 {
            return 0.0;
        }
        self.correct as f64 / self.total_signals as f64
    }

    /// Get win rate
    pub fn win_rate(&self) -> f64 {
        let total = self.wins + self.losses;
        if total == 0 {
            return 0.0;
        }
        self.wins as f64 / total as f64
    }

    /// Get profit factor
    pub fn profit_factor(&self) -> f64 {
        if self.losses == 0 {
            return f64::INFINITY;
        }
        self.wins as f64 / self.losses as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type() {
        assert!(SignalType::Long.is_actionable());
        assert!(SignalType::Short.is_actionable());
        assert!(SignalType::Exit.is_actionable());
        assert!(!SignalType::Hold.is_actionable());
        assert!(!SignalType::None.is_actionable());
    }

    #[test]
    fn test_signal_evaluator() {
        let mut evaluator = SignalEvaluator::new();

        let signal = Signal {
            strength: 0.5,
            signal_type: SignalType::Long,
            confidence: 0.8,
            position_size: 0.02,
            stop_loss: Some(0.02),
            take_profit: Some(0.04),
            metadata: SignalMetadata::default(),
        };

        // Correct prediction
        evaluator.evaluate(&signal, 0.03);
        assert_eq!(evaluator.wins, 1);
        assert_eq!(evaluator.correct, 1);

        // Wrong prediction
        let short_signal = Signal {
            strength: -0.5,
            ..signal.clone()
        };
        evaluator.evaluate(&short_signal, 0.02);
        assert_eq!(evaluator.losses, 1);
    }
}
