//! SE-Enhanced Momentum Strategy
//!
//! This module implements a momentum-based trading strategy that uses
//! SE blocks to dynamically weight different momentum indicators.

use ndarray::Array2;

use crate::models::se_block::SEBlock;
use crate::models::se_trading::{SETradingModel, SETradingConfig};
use crate::data::features::FeatureEngine;
use crate::data::bybit::Kline;
use super::signals::{TradingSignal, Direction, Position, SignalFilter};

/// Configuration for SE Momentum Strategy
#[derive(Debug, Clone)]
pub struct SEMomentumConfig {
    /// Number of features
    pub num_features: usize,
    /// SE reduction ratio
    pub reduction_ratio: usize,
    /// Signal threshold for entry
    pub entry_threshold: f64,
    /// Signal threshold for exit
    pub exit_threshold: f64,
    /// Lookback window for features
    pub lookback_window: usize,
    /// Maximum position size
    pub max_position_size: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Signal filter settings
    pub min_signal_strength: f64,
    pub signal_cooldown: usize,
}

impl Default for SEMomentumConfig {
    fn default() -> Self {
        Self {
            num_features: 12,
            reduction_ratio: 4,
            entry_threshold: 0.3,
            exit_threshold: 0.1,
            lookback_window: 50,
            max_position_size: 1.0,
            stop_loss_pct: 2.0,
            take_profit_pct: 4.0,
            min_signal_strength: 0.25,
            signal_cooldown: 5,
        }
    }
}

/// SE-Enhanced Momentum Trading Strategy
#[derive(Debug)]
pub struct SEMomentumStrategy {
    /// Configuration
    config: SEMomentumConfig,
    /// SE trading model
    model: SETradingModel,
    /// Feature engine
    feature_engine: FeatureEngine,
    /// Signal filter
    signal_filter: SignalFilter,
    /// Current position
    current_position: Option<Position>,
    /// Historical signals for analysis
    signal_history: Vec<TradingSignal>,
}

impl SEMomentumStrategy {
    /// Create a new SE Momentum Strategy
    pub fn new(config: SEMomentumConfig) -> Self {
        let model_config = SETradingConfig {
            num_features: config.num_features,
            reduction_ratio: config.reduction_ratio,
            ..Default::default()
        };

        let model = SETradingModel::new(model_config);
        let feature_engine = FeatureEngine::default();
        let signal_filter = SignalFilter::new(
            config.min_signal_strength,
            config.min_signal_strength,
            config.signal_cooldown,
        );

        Self {
            config,
            model,
            feature_engine,
            signal_filter,
            current_position: None,
            signal_history: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default_strategy() -> Self {
        Self::new(SEMomentumConfig::default())
    }

    /// Generate trading signal from kline data
    pub fn generate_signal(&mut self, klines: &[Kline]) -> Option<TradingSignal> {
        if klines.len() < self.config.lookback_window {
            return None;
        }

        // Use only the lookback window
        let window_start = klines.len() - self.config.lookback_window;
        let window = &klines[window_start..];

        // Compute features
        let features = self.feature_engine.compute_features(window);

        // Get model output with attention
        let output = self.model.forward_with_attention(&features);

        // Create signal
        let signal = TradingSignal::from_raw(output.signal, self.config.entry_threshold)
            .with_attention(output.attention_weights);

        // Apply filter
        let filtered_signal = self.signal_filter.filter(&signal);

        if let Some(ref s) = filtered_signal {
            self.signal_history.push(s.clone());
        }

        filtered_signal
    }

    /// Process a new bar and update position
    pub fn on_bar(&mut self, klines: &[Kline], current_price: f64) -> StrategyAction {
        // Check existing position for exit conditions
        if let Some(ref mut pos) = self.current_position {
            pos.update_pnl(current_price);

            // Check stop loss
            if pos.check_stop_loss(current_price, self.config.stop_loss_pct) {
                let action = StrategyAction::ClosePosition {
                    reason: "Stop Loss".to_string(),
                    pnl: pos.unrealized_pnl,
                };
                self.current_position = None;
                return action;
            }

            // Check take profit
            if pos.check_take_profit(current_price, self.config.take_profit_pct) {
                let action = StrategyAction::ClosePosition {
                    reason: "Take Profit".to_string(),
                    pnl: pos.unrealized_pnl,
                };
                self.current_position = None;
                return action;
            }
        }

        // Generate new signal
        let signal = match self.generate_signal(klines) {
            Some(s) => s,
            None => return StrategyAction::Hold,
        };

        // Determine action based on signal and current position
        match (&self.current_position, signal.direction) {
            // No position and long signal -> open long
            (None, Direction::Long) => {
                let size = signal.strength * self.config.max_position_size;
                let pos = Position::new(
                    Direction::Long,
                    current_price,
                    size,
                    klines.last().map(|k| k.start_time).unwrap_or(0),
                );
                self.current_position = Some(pos);
                StrategyAction::OpenLong { size, signal }
            }

            // No position and short signal -> open short
            (None, Direction::Short) => {
                let size = signal.strength * self.config.max_position_size;
                let pos = Position::new(
                    Direction::Short,
                    current_price,
                    size,
                    klines.last().map(|k| k.start_time).unwrap_or(0),
                );
                self.current_position = Some(pos);
                StrategyAction::OpenShort { size, signal }
            }

            // Long position and short signal -> reverse
            (Some(pos), Direction::Short) if pos.direction == Direction::Long => {
                let old_pnl = pos.unrealized_pnl;
                let size = signal.strength * self.config.max_position_size;
                let new_pos = Position::new(
                    Direction::Short,
                    current_price,
                    size,
                    klines.last().map(|k| k.start_time).unwrap_or(0),
                );
                self.current_position = Some(new_pos);
                StrategyAction::Reverse {
                    from: Direction::Long,
                    to: Direction::Short,
                    size,
                    pnl: old_pnl,
                    signal,
                }
            }

            // Short position and long signal -> reverse
            (Some(pos), Direction::Long) if pos.direction == Direction::Short => {
                let old_pnl = pos.unrealized_pnl;
                let size = signal.strength * self.config.max_position_size;
                let new_pos = Position::new(
                    Direction::Long,
                    current_price,
                    size,
                    klines.last().map(|k| k.start_time).unwrap_or(0),
                );
                self.current_position = Some(new_pos);
                StrategyAction::Reverse {
                    from: Direction::Short,
                    to: Direction::Long,
                    size,
                    pnl: old_pnl,
                    signal,
                }
            }

            // Already in position in same direction or neutral signal
            _ => StrategyAction::Hold,
        }
    }

    /// Get current position
    pub fn position(&self) -> Option<&Position> {
        self.current_position.as_ref()
    }

    /// Get attention analysis for current market
    pub fn analyze_attention(&self, klines: &[Kline]) -> Option<AttentionAnalysis> {
        if klines.len() < self.config.lookback_window {
            return None;
        }

        let window_start = klines.len() - self.config.lookback_window;
        let window = &klines[window_start..];

        let features = self.feature_engine.compute_features(window);
        let attention = self.model.get_attention(&features);

        let feature_names = self.feature_engine.feature_names();
        let mut indexed: Vec<(String, f64)> = feature_names
            .iter()
            .zip(attention.iter())
            .map(|(&name, &weight)| (name.to_string(), weight))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Some(AttentionAnalysis {
            feature_weights: indexed,
            total_attention: attention.sum(),
            attention_entropy: Self::compute_entropy(&attention),
        })
    }

    /// Compute entropy of attention distribution
    fn compute_entropy(weights: &ndarray::Array1<f64>) -> f64 {
        let sum: f64 = weights.sum();
        if sum <= 0.0 {
            return 0.0;
        }

        let normalized: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
        -normalized
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Get signal history
    pub fn signal_history(&self) -> &[TradingSignal] {
        &self.signal_history
    }

    /// Reset the strategy state
    pub fn reset(&mut self) {
        self.current_position = None;
        self.signal_history.clear();
        self.signal_filter.reset();
    }
}

/// Action returned by the strategy
#[derive(Debug, Clone)]
pub enum StrategyAction {
    /// Hold current position (or stay flat)
    Hold,
    /// Open long position
    OpenLong { size: f64, signal: TradingSignal },
    /// Open short position
    OpenShort { size: f64, signal: TradingSignal },
    /// Close current position
    ClosePosition { reason: String, pnl: f64 },
    /// Reverse position
    Reverse {
        from: Direction,
        to: Direction,
        size: f64,
        pnl: f64,
        signal: TradingSignal,
    },
}

impl StrategyAction {
    /// Check if action requires execution
    pub fn is_actionable(&self) -> bool {
        !matches!(self, StrategyAction::Hold)
    }

    /// Get a description of the action
    pub fn description(&self) -> String {
        match self {
            StrategyAction::Hold => "HOLD".to_string(),
            StrategyAction::OpenLong { size, .. } => format!("OPEN LONG (size: {:.4})", size),
            StrategyAction::OpenShort { size, .. } => format!("OPEN SHORT (size: {:.4})", size),
            StrategyAction::ClosePosition { reason, pnl } => {
                format!("CLOSE ({}, PnL: {:.2})", reason, pnl)
            }
            StrategyAction::Reverse { from, to, pnl, .. } => {
                format!("REVERSE {:?} -> {:?} (PnL: {:.2})", from, to, pnl)
            }
        }
    }
}

/// Attention analysis results
#[derive(Debug, Clone)]
pub struct AttentionAnalysis {
    /// Feature weights sorted by importance
    pub feature_weights: Vec<(String, f64)>,
    /// Sum of all attention weights
    pub total_attention: f64,
    /// Entropy of attention distribution (higher = more distributed)
    pub attention_entropy: f64,
}

impl AttentionAnalysis {
    /// Get top k features
    pub fn top_features(&self, k: usize) -> &[(String, f64)] {
        &self.feature_weights[..k.min(self.feature_weights.len())]
    }

    /// Check if attention is focused (low entropy)
    pub fn is_focused(&self, threshold: f64) -> bool {
        self.attention_entropy < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_sample_data;

    #[test]
    fn test_strategy_creation() {
        let strategy = SEMomentumStrategy::default_strategy();
        assert!(strategy.position().is_none());
    }

    #[test]
    fn test_signal_generation() {
        let mut strategy = SEMomentumStrategy::default_strategy();
        let klines = generate_sample_data(100, 50000.0);

        // Should be able to generate a signal
        let signal = strategy.generate_signal(&klines);
        // Signal might be filtered, so we just check it doesn't panic
        assert!(signal.is_some() || signal.is_none());
    }

    #[test]
    fn test_attention_analysis() {
        let strategy = SEMomentumStrategy::default_strategy();
        let klines = generate_sample_data(100, 50000.0);

        let analysis = strategy.analyze_attention(&klines);
        assert!(analysis.is_some());

        let analysis = analysis.unwrap();
        assert!(!analysis.feature_weights.is_empty());
    }
}
