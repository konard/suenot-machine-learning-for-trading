//! Trading strategy implementation.
//!
//! This module provides a trading strategy that uses the CML model
//! for making trading decisions.

use crate::continual::learner::ContinualMetaLearner;
use crate::trading::signals::{Signal, SignalGenerator, SignalConfig};
use crate::data::features::TradingFeatures;
use crate::MarketRegime;

/// Position in the market.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    /// Long position.
    Long(f64),
    /// Short position.
    Short(f64),
    /// Flat (no position).
    Flat,
}

impl Position {
    /// Get position size (negative for short).
    pub fn size(&self) -> f64 {
        match self {
            Position::Long(s) => *s,
            Position::Short(s) => -*s,
            Position::Flat => 0.0,
        }
    }

    /// Check if position is flat.
    pub fn is_flat(&self) -> bool {
        matches!(self, Position::Flat)
    }

    /// Check if position is long.
    pub fn is_long(&self) -> bool {
        matches!(self, Position::Long(_))
    }

    /// Check if position is short.
    pub fn is_short(&self) -> bool {
        matches!(self, Position::Short(_))
    }
}

/// Action to take in trading.
#[derive(Debug, Clone, Copy)]
pub enum TradeAction {
    /// Open a long position.
    OpenLong(f64),
    /// Open a short position.
    OpenShort(f64),
    /// Close current position.
    ClosePosition,
    /// Hold current position.
    Hold,
    /// Increase position size.
    IncreasePosition(f64),
    /// Decrease position size.
    DecreasePosition(f64),
}

/// Configuration for the trading strategy.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Maximum position size.
    pub max_position: f64,
    /// Minimum position size.
    pub min_position: f64,
    /// Position sizing multiplier.
    pub position_multiplier: f64,
    /// Whether to allow short positions.
    pub allow_short: bool,
    /// Number of adaptation samples before trading.
    pub warmup_samples: usize,
    /// Adapt to new regime after N samples.
    pub adaptation_interval: usize,
    /// Signal configuration.
    pub signal_config: SignalConfig,
    /// Risk per trade (as fraction of capital).
    pub risk_per_trade: f64,
    /// Stop loss percentage.
    pub stop_loss: f64,
    /// Take profit percentage.
    pub take_profit: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            max_position: 1.0,
            min_position: 0.1,
            position_multiplier: 1.0,
            allow_short: true,
            warmup_samples: 20,
            adaptation_interval: 50,
            signal_config: SignalConfig::default(),
            risk_per_trade: 0.02,
            stop_loss: 0.05,
            take_profit: 0.10,
        }
    }
}

/// CML-based trading strategy.
pub struct CMLStrategy {
    /// The CML model.
    learner: ContinualMetaLearner,
    /// Signal generator.
    signal_gen: SignalGenerator,
    /// Configuration.
    config: StrategyConfig,
    /// Current position.
    position: Position,
    /// Current regime.
    current_regime: MarketRegime,
    /// Samples since last adaptation.
    samples_since_adaptation: usize,
    /// Entry price for current position.
    entry_price: f64,
    /// Total trades executed.
    total_trades: usize,
    /// Adaptation history.
    adaptation_history: Vec<(usize, MarketRegime)>,
    /// Whether strategy is warmed up.
    is_warmed_up: bool,
}

impl CMLStrategy {
    /// Create a new CML strategy.
    pub fn new(learner: ContinualMetaLearner, config: StrategyConfig) -> Self {
        let signal_gen = SignalGenerator::new(config.signal_config.clone());

        Self {
            learner,
            signal_gen,
            config,
            position: Position::Flat,
            current_regime: MarketRegime::Sideways,
            samples_since_adaptation: 0,
            entry_price: 0.0,
            total_trades: 0,
            adaptation_history: Vec::new(),
            is_warmed_up: false,
        }
    }

    /// Process new data point and get trading action.
    pub fn step(&mut self, features: &TradingFeatures, idx: usize, price: f64) -> TradeAction {
        // Get features for current index
        let feature_vec = match features.get_features(idx) {
            Some(f) => f,
            None => return TradeAction::Hold,
        };

        // Get current regime
        let regime = features.get_regime(idx).unwrap_or(MarketRegime::Sideways);

        // Check if regime changed
        if regime != self.current_regime {
            self.handle_regime_change(regime, idx);
        }

        self.samples_since_adaptation += 1;

        // Check if we need to adapt
        if self.samples_since_adaptation >= self.config.adaptation_interval {
            self.adapt_to_current_data(features, idx);
        }

        // Check warmup
        if !self.is_warmed_up {
            if idx >= self.config.warmup_samples {
                self.is_warmed_up = true;
            } else {
                return TradeAction::Hold;
            }
        }

        // Check stop loss / take profit
        if let Some(action) = self.check_exit_conditions(price) {
            return action;
        }

        // Get model prediction
        let prediction = self.learner.predict(&feature_vec, None);
        let pred_value = prediction.first().copied().unwrap_or(0.0);

        // Generate signal
        let signal = self.signal_gen.generate(pred_value, Some(regime));

        // Convert signal to action
        self.signal_to_action(signal, price)
    }

    /// Handle regime change.
    fn handle_regime_change(&mut self, new_regime: MarketRegime, step: usize) {
        tracing::info!(
            "Regime change: {:?} -> {:?} at step {}",
            self.current_regime,
            new_regime,
            step
        );

        self.adaptation_history.push((step, new_regime));
        self.current_regime = new_regime;
        self.samples_since_adaptation = 0;
    }

    /// Adapt model to current data.
    fn adapt_to_current_data(&mut self, features: &TradingFeatures, current_idx: usize) {
        // Collect recent samples for adaptation
        let start = current_idx.saturating_sub(self.config.adaptation_interval);
        let mut support_data = Vec::new();

        for i in start..current_idx {
            if let Some(feat) = features.get_features(i) {
                // Use next return as target
                if i + 1 < features.returns.len() {
                    let target = vec![features.returns[i + 1].signum()];
                    support_data.push(crate::continual::memory::Experience::new(
                        feat,
                        target,
                        self.current_regime as usize,
                    ));
                }
            }
        }

        if !support_data.is_empty() {
            let _adapted = self.learner.adapt(&support_data);
            self.samples_since_adaptation = 0;
        }
    }

    /// Check exit conditions (stop loss / take profit).
    fn check_exit_conditions(&mut self, current_price: f64) -> Option<TradeAction> {
        if self.position.is_flat() || self.entry_price == 0.0 {
            return None;
        }

        let pnl_pct = match self.position {
            Position::Long(_) => (current_price - self.entry_price) / self.entry_price,
            Position::Short(_) => (self.entry_price - current_price) / self.entry_price,
            Position::Flat => return None,
        };

        // Check stop loss
        if pnl_pct <= -self.config.stop_loss {
            return Some(TradeAction::ClosePosition);
        }

        // Check take profit
        if pnl_pct >= self.config.take_profit {
            return Some(TradeAction::ClosePosition);
        }

        None
    }

    /// Convert signal to trading action.
    fn signal_to_action(&mut self, signal: Signal, price: f64) -> TradeAction {
        let target_size = signal.to_position_size() * self.config.position_multiplier;
        let clamped_size = target_size
            .abs()
            .min(self.config.max_position)
            .max(if target_size.abs() > 0.01 { self.config.min_position } else { 0.0 });

        match (&self.position, signal) {
            // Currently flat
            (Position::Flat, Signal::StrongBuy | Signal::Buy) => {
                self.position = Position::Long(clamped_size);
                self.entry_price = price;
                self.total_trades += 1;
                TradeAction::OpenLong(clamped_size)
            }
            (Position::Flat, Signal::StrongSell | Signal::Sell) if self.config.allow_short => {
                self.position = Position::Short(clamped_size);
                self.entry_price = price;
                self.total_trades += 1;
                TradeAction::OpenShort(clamped_size)
            }
            (Position::Flat, _) => TradeAction::Hold,

            // Currently long
            (Position::Long(_), Signal::StrongSell | Signal::Sell) => {
                self.position = Position::Flat;
                self.entry_price = 0.0;
                TradeAction::ClosePosition
            }
            (Position::Long(current), Signal::StrongBuy) if clamped_size > *current => {
                let increase = clamped_size - *current;
                self.position = Position::Long(clamped_size);
                TradeAction::IncreasePosition(increase)
            }
            (Position::Long(_), _) => TradeAction::Hold,

            // Currently short
            (Position::Short(_), Signal::StrongBuy | Signal::Buy) => {
                self.position = Position::Flat;
                self.entry_price = 0.0;
                TradeAction::ClosePosition
            }
            (Position::Short(current), Signal::StrongSell) if clamped_size > *current => {
                let increase = clamped_size - *current;
                self.position = Position::Short(clamped_size);
                TradeAction::IncreasePosition(increase)
            }
            (Position::Short(_), _) => TradeAction::Hold,
        }
    }

    /// Execute action (update internal state).
    pub fn execute(&mut self, action: TradeAction, price: f64) {
        match action {
            TradeAction::OpenLong(size) => {
                self.position = Position::Long(size);
                self.entry_price = price;
                self.total_trades += 1;
            }
            TradeAction::OpenShort(size) => {
                self.position = Position::Short(size);
                self.entry_price = price;
                self.total_trades += 1;
            }
            TradeAction::ClosePosition => {
                self.position = Position::Flat;
                self.entry_price = 0.0;
            }
            TradeAction::IncreasePosition(delta) => {
                match self.position {
                    Position::Long(s) => self.position = Position::Long(s + delta),
                    Position::Short(s) => self.position = Position::Short(s + delta),
                    Position::Flat => {}
                }
            }
            TradeAction::DecreasePosition(delta) => {
                match self.position {
                    Position::Long(s) => {
                        let new_size = (s - delta).max(0.0);
                        self.position = if new_size > 0.0 {
                            Position::Long(new_size)
                        } else {
                            Position::Flat
                        };
                    }
                    Position::Short(s) => {
                        let new_size = (s - delta).max(0.0);
                        self.position = if new_size > 0.0 {
                            Position::Short(new_size)
                        } else {
                            Position::Flat
                        };
                    }
                    Position::Flat => {}
                }
            }
            TradeAction::Hold => {}
        }
    }

    /// Get current position.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Get current regime.
    pub fn regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get total trades.
    pub fn total_trades(&self) -> usize {
        self.total_trades
    }

    /// Get reference to the learner.
    pub fn learner(&self) -> &ContinualMetaLearner {
        &self.learner
    }

    /// Get mutable reference to the learner.
    pub fn learner_mut(&mut self) -> &mut ContinualMetaLearner {
        &mut self.learner
    }

    /// Get adaptation history.
    pub fn adaptation_history(&self) -> &[(usize, MarketRegime)] {
        &self.adaptation_history
    }

    /// Get configuration.
    pub fn config(&self) -> &StrategyConfig {
        &self.config
    }

    /// Check if strategy is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.is_warmed_up
    }

    /// Reset strategy state.
    pub fn reset(&mut self) {
        self.position = Position::Flat;
        self.current_regime = MarketRegime::Sideways;
        self.samples_since_adaptation = 0;
        self.entry_price = 0.0;
        self.total_trades = 0;
        self.adaptation_history.clear();
        self.is_warmed_up = false;
        self.signal_gen.clear();
    }

    /// Get strategy statistics.
    pub fn stats(&self) -> StrategyStats {
        let signal_stats = self.signal_gen.stats();

        StrategyStats {
            total_trades: self.total_trades,
            adaptations: self.adaptation_history.len(),
            current_position: self.position,
            current_regime: self.current_regime,
            is_warmed_up: self.is_warmed_up,
            buy_signals: signal_stats.buy_signals,
            sell_signals: signal_stats.sell_signals,
        }
    }
}

/// Statistics about the strategy.
#[derive(Debug, Clone)]
pub struct StrategyStats {
    /// Total trades executed.
    pub total_trades: usize,
    /// Number of adaptations.
    pub adaptations: usize,
    /// Current position.
    pub current_position: Position,
    /// Current regime.
    pub current_regime: MarketRegime,
    /// Whether strategy is warmed up.
    pub is_warmed_up: bool,
    /// Number of buy signals generated.
    pub buy_signals: usize,
    /// Number of sell signals generated.
    pub sell_signals: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CMLConfig;

    fn create_test_learner() -> ContinualMetaLearner {
        let config = CMLConfig {
            input_size: 9,
            hidden_size: 16,
            output_size: 1,
            ..Default::default()
        };
        ContinualMetaLearner::new(config)
    }

    #[test]
    fn test_position() {
        assert_eq!(Position::Long(1.0).size(), 1.0);
        assert_eq!(Position::Short(1.0).size(), -1.0);
        assert_eq!(Position::Flat.size(), 0.0);

        assert!(Position::Long(1.0).is_long());
        assert!(Position::Short(1.0).is_short());
        assert!(Position::Flat.is_flat());
    }

    #[test]
    fn test_strategy_creation() {
        let learner = create_test_learner();
        let config = StrategyConfig::default();
        let strategy = CMLStrategy::new(learner, config);

        assert!(strategy.position().is_flat());
        assert!(!strategy.is_warmed_up());
        assert_eq!(strategy.total_trades(), 0);
    }

    #[test]
    fn test_signal_to_action() {
        let learner = create_test_learner();
        let config = StrategyConfig {
            warmup_samples: 0, // Skip warmup for test
            ..Default::default()
        };
        let mut strategy = CMLStrategy::new(learner, config);
        strategy.is_warmed_up = true;

        // Test buy action from flat
        let action = strategy.signal_to_action(Signal::Buy, 100.0);
        assert!(matches!(action, TradeAction::OpenLong(_)));
    }
}
