//! Trading strategy implementation using SNNs

use crate::network::SNNNetwork;
use crate::encoding::DeltaEncoder;
use crate::learning::RewardModulatedSTDP;
use crate::trading::signals::{TradingSignal, SignalStrength, RegimeDetector};
use crate::data::Candle;

/// Trade decision
#[derive(Debug, Clone)]
pub struct TradeDecision {
    /// Trading signal
    pub signal: TradingSignal,
    /// Signal strength
    pub strength: SignalStrength,
    /// Suggested position size (0.0 to 1.0)
    pub position_size: f64,
    /// Stop loss price (if applicable)
    pub stop_loss: Option<f64>,
    /// Take profit price (if applicable)
    pub take_profit: Option<f64>,
    /// Confidence in the decision
    pub confidence: f64,
}

impl TradeDecision {
    /// Create a hold decision
    pub fn hold() -> Self {
        Self {
            signal: TradingSignal::Hold,
            strength: SignalStrength::new(0.0, 0.0),
            position_size: 0.0,
            stop_loss: None,
            take_profit: None,
            confidence: 0.0,
        }
    }

    /// Check if decision suggests taking action
    pub fn is_actionable(&self) -> bool {
        self.signal != TradingSignal::Hold && self.confidence > 0.5
    }
}

/// Common trait for trading strategies
pub trait TradingStrategy: Send + Sync {
    /// Process new market data and generate decision
    fn process(&mut self, candle: &Candle) -> TradeDecision;

    /// Update strategy with trade result (for learning)
    fn update_with_result(&mut self, pnl: f64, risk: f64);

    /// Reset strategy state
    fn reset(&mut self);

    /// Get strategy name
    fn name(&self) -> &str;
}

/// SNN-based trading strategy
pub struct SNNTradingStrategy {
    /// Neural network
    network: SNNNetwork,
    /// Delta encoder for price data
    encoder: DeltaEncoder,
    /// Regime detector
    regime_detector: RegimeDetector,
    /// Learning rule
    learning: Option<RewardModulatedSTDP>,
    /// Previous prices for encoding
    prev_prices: Vec<f64>,
    /// Strategy parameters
    params: StrategyParams,
    /// Performance tracking
    total_pnl: f64,
    trade_count: usize,
}

/// Strategy parameters
#[derive(Debug, Clone)]
pub struct StrategyParams {
    /// Minimum confidence for trading
    pub min_confidence: f64,
    /// Maximum position size
    pub max_position: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Enable learning
    pub learning_enabled: bool,
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_position: 1.0,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.04,
            learning_rate: 0.01,
            learning_enabled: true,
        }
    }
}

impl SNNTradingStrategy {
    /// Create a new SNN trading strategy
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let network = SNNNetwork::builder()
            .input_layer(input_size)
            .hidden_layer(hidden_size)
            .output_layer(2)  // Buy/Sell outputs
            .with_learning(true)
            .build();

        let encoder = DeltaEncoder::for_prices(input_size);
        let regime_detector = RegimeDetector::new(20);

        Self {
            network,
            encoder,
            regime_detector,
            learning: None,
            prev_prices: vec![0.0; input_size],
            params: StrategyParams::default(),
            total_pnl: 0.0,
            trade_count: 0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(mut self, params: StrategyParams) -> Self {
        self.params = params;
        self
    }

    /// Enable reward-modulated learning
    pub fn with_learning(mut self) -> Self {
        let learning = RewardModulatedSTDP::new(
            self.network.input_size(),
            self.network.output_size(),
        );
        self.learning = Some(learning);
        self
    }

    /// Process OHLCV candle into features
    fn extract_features(&self, candle: &Candle) -> Vec<f64> {
        vec![
            candle.return_pct(),
            candle.range() / candle.open,
            candle.body_size() / candle.range().max(0.0001),
            (candle.close - candle.low) / candle.range().max(0.0001),
            candle.volume / 1000.0,  // Normalized volume
        ]
    }

    /// Decode network output to signal
    fn decode_output(&self, buy_spikes: usize, sell_spikes: usize) -> SignalStrength {
        let total = (buy_spikes + sell_spikes) as f64;

        if total < 1.0 {
            return SignalStrength::new(0.0, 0.0);
        }

        let signal = (buy_spikes as f64 - sell_spikes as f64) / total;
        let confidence = total / 100.0;  // More spikes = higher confidence

        SignalStrength::new(signal, confidence.min(1.0))
    }

    /// Calculate position size based on signal and regime
    fn calculate_position(&self, strength: &SignalStrength) -> f64 {
        let regime = self.regime_detector.detect();
        let base_size = strength.effective_signal().abs() * self.params.max_position;

        base_size * regime.position_multiplier()
    }

    /// Get performance statistics
    pub fn stats(&self) -> (f64, usize, f64) {
        let avg_pnl = if self.trade_count > 0 {
            self.total_pnl / self.trade_count as f64
        } else {
            0.0
        };
        (self.total_pnl, self.trade_count, avg_pnl)
    }
}

impl TradingStrategy for SNNTradingStrategy {
    fn process(&mut self, candle: &Candle) -> TradeDecision {
        // Extract features
        let features = self.extract_features(candle);

        // Update regime detector
        self.regime_detector.update(candle.return_pct());

        // Encode as currents
        let currents = self.encoder.process_to_currents(&features, 0.0);

        // Pad or truncate to match input size
        let mut input = vec![0.0; self.network.input_size()];
        for (i, &c) in currents.iter().enumerate() {
            if i < input.len() {
                input[i] = c * 50.0;  // Scale for neuron activation
            }
        }

        // Run network for multiple timesteps
        let spike_trains = self.network.run(&input, 100);

        // Count output spikes
        let buy_spikes = spike_trains.iter().filter(|s| s.get(0) == Some(&true)).count();
        let sell_spikes = spike_trains.iter().filter(|s| s.get(1) == Some(&true)).count();

        // Decode to signal
        let strength = self.decode_output(buy_spikes, sell_spikes);
        let signal = strength.to_signal();

        // Check if actionable
        if !strength.is_actionable(self.params.min_confidence) {
            return TradeDecision::hold();
        }

        // Calculate position and levels
        let position_size = self.calculate_position(&strength);
        let current_price = candle.close;

        let (stop_loss, take_profit) = if signal.is_bullish() {
            (
                Some(current_price * (1.0 - self.params.stop_loss_pct)),
                Some(current_price * (1.0 + self.params.take_profit_pct)),
            )
        } else if signal.is_bearish() {
            (
                Some(current_price * (1.0 + self.params.stop_loss_pct)),
                Some(current_price * (1.0 - self.params.take_profit_pct)),
            )
        } else {
            (None, None)
        };

        TradeDecision {
            signal,
            strength,
            position_size,
            stop_loss,
            take_profit,
            confidence: strength.confidence,
        }
    }

    fn update_with_result(&mut self, pnl: f64, risk: f64) {
        self.total_pnl += pnl;
        self.trade_count += 1;

        // Calculate reward
        let reward = if risk > 0.0 {
            pnl / risk  // Risk-adjusted return
        } else {
            pnl.signum()
        };

        // Apply learning
        if self.params.learning_enabled {
            self.network.learn(reward);
        }
    }

    fn reset(&mut self) {
        self.network.reset();
        self.encoder.reset();
        self.regime_detector.reset();
        self.prev_prices = vec![0.0; self.prev_prices.len()];
    }

    fn name(&self) -> &str {
        "SNN Trading Strategy"
    }
}

/// Simple momentum strategy using SNN for comparison
pub struct MomentumSNNStrategy {
    /// Base strategy
    base: SNNTradingStrategy,
    /// Momentum lookback
    lookback: usize,
    /// Price history
    prices: Vec<f64>,
}

impl MomentumSNNStrategy {
    pub fn new(lookback: usize) -> Self {
        Self {
            base: SNNTradingStrategy::new(lookback, lookback * 2),
            lookback,
            prices: Vec::with_capacity(lookback),
        }
    }
}

impl TradingStrategy for MomentumSNNStrategy {
    fn process(&mut self, candle: &Candle) -> TradeDecision {
        // Update price history
        self.prices.push(candle.close);
        if self.prices.len() > self.lookback {
            self.prices.remove(0);
        }

        // Need enough history
        if self.prices.len() < self.lookback {
            return TradeDecision::hold();
        }

        // Calculate momentum features
        let returns: Vec<f64> = self.prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Use base strategy with momentum features
        self.base.process(candle)
    }

    fn update_with_result(&mut self, pnl: f64, risk: f64) {
        self.base.update_with_result(pnl, risk);
    }

    fn reset(&mut self) {
        self.base.reset();
        self.prices.clear();
    }

    fn name(&self) -> &str {
        "Momentum SNN Strategy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candle(open: f64, close: f64) -> Candle {
        Candle {
            timestamp: 0,
            open,
            high: open.max(close) * 1.01,
            low: open.min(close) * 0.99,
            close,
            volume: 1000.0,
            turnover: 50000.0,
        }
    }

    #[test]
    fn test_strategy_creation() {
        let strategy = SNNTradingStrategy::new(10, 20);
        assert_eq!(strategy.name(), "SNN Trading Strategy");
    }

    #[test]
    fn test_strategy_process() {
        let mut strategy = SNNTradingStrategy::new(5, 10);

        let candle = create_test_candle(100.0, 105.0);
        let decision = strategy.process(&candle);

        // Should return some decision
        assert!(decision.position_size >= 0.0);
    }

    #[test]
    fn test_trade_decision_hold() {
        let decision = TradeDecision::hold();
        assert_eq!(decision.signal, TradingSignal::Hold);
        assert!(!decision.is_actionable());
    }

    #[test]
    fn test_strategy_reset() {
        let mut strategy = SNNTradingStrategy::new(5, 10);

        // Process some data
        let candle = create_test_candle(100.0, 105.0);
        strategy.process(&candle);

        // Reset
        strategy.reset();

        // Stats should be preserved (reset doesn't clear stats)
        let (pnl, trades, _) = strategy.stats();
        assert_eq!(pnl, 0.0);
        assert_eq!(trades, 0);
    }
}
