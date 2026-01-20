//! Adaptive and LLM-based execution strategies.

use crate::data::OrderBook;
use crate::execution::{
    ExecutionAction, ExecutionContext, ExecutionError, LlmAdapter, LlmDecision,
    MarketState, ParentOrder, Side,
};
use crate::strategy::{calculate_limit_price, ExecutionSlice, ExecutionStrategy, StrategyConfig};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Adaptive execution strategy
///
/// Dynamically adjusts execution based on real-time market conditions
/// without requiring an LLM.
#[derive(Debug, Clone)]
pub struct AdaptiveStrategy {
    config: StrategyConfig,
    /// Spread threshold to switch from passive to aggressive (bps)
    spread_threshold_bps: f64,
    /// Imbalance threshold to adjust sizing
    imbalance_threshold: f64,
    /// Volatility regime (tracked)
    current_volatility: f64,
    /// Recent execution performance
    recent_slippage_bps: Vec<f64>,
}

impl AdaptiveStrategy {
    /// Create a new adaptive strategy
    pub fn new() -> Self {
        Self {
            config: StrategyConfig::default(),
            spread_threshold_bps: 10.0,
            imbalance_threshold: 0.3,
            current_volatility: 0.02,
            recent_slippage_bps: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StrategyConfig) -> Self {
        Self {
            config,
            spread_threshold_bps: 10.0,
            imbalance_threshold: 0.3,
            current_volatility: 0.02,
            recent_slippage_bps: Vec::new(),
        }
    }

    /// Set spread threshold
    pub fn with_spread_threshold(mut self, threshold_bps: f64) -> Self {
        self.spread_threshold_bps = threshold_bps;
        self
    }

    /// Update volatility estimate
    pub fn update_volatility(&mut self, volatility: f64) {
        // Exponential moving average
        self.current_volatility = 0.9 * self.current_volatility + 0.1 * volatility;
    }

    /// Record execution slippage
    pub fn record_slippage(&mut self, slippage_bps: f64) {
        self.recent_slippage_bps.push(slippage_bps);
        if self.recent_slippage_bps.len() > 20 {
            self.recent_slippage_bps.remove(0);
        }
    }

    /// Get average recent slippage
    fn average_slippage(&self) -> f64 {
        if self.recent_slippage_bps.is_empty() {
            0.0
        } else {
            self.recent_slippage_bps.iter().sum::<f64>() / self.recent_slippage_bps.len() as f64
        }
    }

    /// Determine execution mode based on market conditions
    fn determine_mode(&self, orderbook: &OrderBook, order: &ParentOrder) -> ExecutionMode {
        let spread_bps = orderbook.spread_bps().unwrap_or(100.0);
        let imbalance = orderbook.imbalance(10);

        // Favorable imbalance for our side?
        let favorable_imbalance = match order.side {
            Side::Buy => imbalance < -self.imbalance_threshold,
            Side::Sell => imbalance > self.imbalance_threshold,
        };

        // Time pressure
        let time_remaining_ratio = order.remaining_time() as f64 / order.time_horizon as f64;
        let urgent = time_remaining_ratio < 0.2 || order.urgency > 0.7;

        // High volatility regime?
        let high_volatility = self.current_volatility > 0.03;

        // Decision logic
        if urgent {
            ExecutionMode::Aggressive
        } else if spread_bps > self.spread_threshold_bps {
            ExecutionMode::Wait
        } else if favorable_imbalance && !high_volatility {
            ExecutionMode::Aggressive
        } else if spread_bps < self.spread_threshold_bps / 2.0 {
            ExecutionMode::Opportunistic
        } else {
            ExecutionMode::Passive
        }
    }

    /// Calculate quantity based on mode
    fn calculate_quantity(
        &self,
        mode: ExecutionMode,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> f64 {
        let remaining = order.remaining_quantity();
        let num_remaining_slices = (order.remaining_time() / self.config.slice_interval_secs).max(1) as f64;
        let base_quantity = remaining / num_remaining_slices;

        let mode_factor = match mode {
            ExecutionMode::Aggressive => 1.5,
            ExecutionMode::Opportunistic => 1.2,
            ExecutionMode::Passive => 0.8,
            ExecutionMode::Wait => 0.0,
        };

        // Adjust for depth
        let depth = match order.side {
            Side::Buy => orderbook.ask_depth(5),
            Side::Sell => orderbook.bid_depth(5),
        };

        let depth_limit = if depth > 0.0 { depth * 0.1 } else { f64::MAX };

        (base_quantity * mode_factor)
            .min(self.config.max_slice_size)
            .max(self.config.min_slice_size)
            .min(remaining)
            .min(depth_limit)
    }
}

impl Default for AdaptiveStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution mode for adaptive strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionMode {
    /// Execute aggressively (market orders or aggressive limits)
    Aggressive,
    /// Execute opportunistically (when conditions are favorable)
    Opportunistic,
    /// Execute passively (use limit orders well inside spread)
    Passive,
    /// Wait for better conditions
    Wait,
}

impl ExecutionStrategy for AdaptiveStrategy {
    fn next_slice(
        &self,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> Result<ExecutionSlice, ExecutionError> {
        let remaining = order.remaining_quantity();

        if remaining <= 0.0 {
            return Ok(ExecutionSlice::market(0.0));
        }

        let mode = self.determine_mode(orderbook, order);

        if mode == ExecutionMode::Wait {
            return Ok(ExecutionSlice::market(0.0));
        }

        let quantity = self.calculate_quantity(mode, order, orderbook);

        let aggressiveness = match mode {
            ExecutionMode::Aggressive => 0.8,
            ExecutionMode::Opportunistic => 0.4,
            ExecutionMode::Passive => -0.3,
            ExecutionMode::Wait => 0.0,
        };

        let slice = if self.config.use_limit_orders && mode != ExecutionMode::Aggressive {
            if let Some(price) = calculate_limit_price(orderbook, order.side, aggressiveness) {
                ExecutionSlice::limit(quantity, price).with_urgency(order.urgency)
            } else {
                ExecutionSlice::market(quantity)
            }
        } else {
            ExecutionSlice::market(quantity)
        };

        Ok(slice)
    }

    fn slice_interval_ms(&self) -> u64 {
        self.config.slice_interval_secs * 1000
    }

    fn name(&self) -> &str {
        "Adaptive"
    }

    fn reset(&mut self) {
        self.recent_slippage_bps.clear();
    }
}

/// LLM-powered execution strategy
///
/// Uses an LLM to make execution decisions based on market conditions
/// and execution context.
pub struct LlmStrategy {
    config: StrategyConfig,
    adapter: Arc<Mutex<LlmAdapter>>,
    last_decision: Option<LlmDecision>,
    fallback: AdaptiveStrategy,
}

impl LlmStrategy {
    /// Create a new LLM strategy
    pub fn new(adapter: LlmAdapter) -> Self {
        Self {
            config: StrategyConfig::default(),
            adapter: Arc::new(Mutex::new(adapter)),
            last_decision: None,
            fallback: AdaptiveStrategy::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(adapter: LlmAdapter, config: StrategyConfig) -> Self {
        Self {
            config: config.clone(),
            adapter: Arc::new(Mutex::new(adapter)),
            last_decision: None,
            fallback: AdaptiveStrategy::with_config(config),
        }
    }

    /// Get the last LLM decision
    pub fn last_decision(&self) -> Option<&LlmDecision> {
        self.last_decision.as_ref()
    }

    /// Build execution context from order
    fn build_context(&self, order: &ParentOrder, orderbook: &OrderBook) -> ExecutionContext {
        ExecutionContext {
            side: order.side,
            total_quantity: order.total_quantity,
            filled_quantity: order.filled_quantity,
            remaining_time: order.remaining_time(),
            target_participation: order.max_participation,
            actual_participation: 0.0, // Would need market volume data
            vwap_slippage_bps: 0.0,    // Would need VWAP tracker
            is_bps: 0.0,               // Would need arrival price tracking
            urgency: order.urgency,
        }
    }

    /// Convert LLM decision to execution slice
    fn decision_to_slice(
        &self,
        decision: &LlmDecision,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> ExecutionSlice {
        let remaining = order.remaining_quantity();

        // Quantity from decision
        let quantity = (remaining * decision.quantity_fraction)
            .min(self.config.max_slice_size)
            .max(self.config.min_slice_size)
            .min(remaining);

        // Price based on aggressiveness
        if self.config.use_limit_orders && decision.aggressiveness < 0.7 {
            if let Some(price) = calculate_limit_price(orderbook, order.side, decision.aggressiveness) {
                return ExecutionSlice::limit(quantity, price).with_urgency(order.urgency);
            }
        }

        ExecutionSlice::market(quantity)
    }
}

impl ExecutionStrategy for LlmStrategy {
    fn next_slice(
        &self,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> Result<ExecutionSlice, ExecutionError> {
        let remaining = order.remaining_quantity();

        if remaining <= 0.0 {
            return Ok(ExecutionSlice::market(0.0));
        }

        // Use last decision if available (would be updated asynchronously in real impl)
        if let Some(ref decision) = self.last_decision {
            match decision.action {
                ExecutionAction::Wait | ExecutionAction::Pause => {
                    return Ok(ExecutionSlice::market(0.0));
                }
                ExecutionAction::Cancel => {
                    return Err(ExecutionError::Strategy("LLM recommended cancel".into()));
                }
                _ => {
                    return Ok(self.decision_to_slice(decision, order, orderbook));
                }
            }
        }

        // Fall back to adaptive strategy
        self.fallback.next_slice(order, orderbook)
    }

    fn slice_interval_ms(&self) -> u64 {
        self.config.slice_interval_secs * 1000
    }

    fn name(&self) -> &str {
        "LLM"
    }

    fn reset(&mut self) {
        self.last_decision = None;
        self.fallback.reset();
    }
}

/// Hybrid strategy that combines multiple strategies
pub struct HybridStrategy {
    strategies: Vec<(Box<dyn ExecutionStrategy>, f64)>, // (strategy, weight)
}

impl HybridStrategy {
    /// Create a new hybrid strategy
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Add a strategy with a weight
    pub fn add_strategy(mut self, strategy: Box<dyn ExecutionStrategy>, weight: f64) -> Self {
        self.strategies.push((strategy, weight));
        self
    }

    /// Normalize weights to sum to 1.0
    fn normalize_weights(&mut self) {
        let sum: f64 = self.strategies.iter().map(|(_, w)| w).sum();
        if sum > 0.0 {
            for (_, w) in &mut self.strategies {
                *w /= sum;
            }
        }
    }
}

impl Default for HybridStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionStrategy for HybridStrategy {
    fn next_slice(
        &self,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> Result<ExecutionSlice, ExecutionError> {
        if self.strategies.is_empty() {
            return Err(ExecutionError::Strategy("No strategies configured".into()));
        }

        // Get slices from all strategies
        let mut total_quantity = 0.0;
        let mut weighted_urgency = 0.0;

        for (strategy, weight) in &self.strategies {
            if let Ok(slice) = strategy.next_slice(order, orderbook) {
                total_quantity += slice.quantity * weight;
                weighted_urgency += slice.urgency * weight;
            }
        }

        let remaining = order.remaining_quantity();
        let quantity = total_quantity.min(remaining);

        if quantity > 0.0 {
            let aggressiveness = weighted_urgency - 0.5; // Convert urgency to aggressiveness
            if let Some(price) = calculate_limit_price(orderbook, order.side, aggressiveness) {
                Ok(ExecutionSlice::limit(quantity, price).with_urgency(weighted_urgency))
            } else {
                Ok(ExecutionSlice::market(quantity))
            }
        } else {
            Ok(ExecutionSlice::market(0.0))
        }
    }

    fn slice_interval_ms(&self) -> u64 {
        // Use minimum interval from all strategies
        self.strategies
            .iter()
            .map(|(s, _)| s.slice_interval_ms())
            .min()
            .unwrap_or(60000)
    }

    fn name(&self) -> &str {
        "Hybrid"
    }

    fn reset(&mut self) {
        for (strategy, _) in &mut self.strategies {
            strategy.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::LlmConfig;

    fn create_test_order() -> ParentOrder {
        let mut order = ParentOrder::new("BTCUSDT".to_string(), Side::Buy, 10.0, 600);
        order.start(50000.0);
        order
    }

    fn create_test_orderbook() -> OrderBook {
        let mut book = OrderBook::new("BTCUSDT".to_string());
        for i in 1..=10 {
            book.update_bid(50000.0 - i as f64 * 1.0, 10.0);
            book.update_ask(50000.0 + i as f64 * 1.0, 10.0);
        }
        book
    }

    #[test]
    fn test_adaptive_strategy() {
        let strategy = AdaptiveStrategy::new();
        let order = create_test_order();
        let book = create_test_orderbook();

        let slice = strategy.next_slice(&order, &book).unwrap();

        assert!(slice.quantity >= 0.0);
        assert!(slice.quantity <= 10.0);
    }

    #[test]
    fn test_adaptive_modes() {
        let strategy = AdaptiveStrategy::new();
        let order = create_test_order();

        // Normal conditions
        let book = create_test_orderbook();
        let mode = strategy.determine_mode(&book, &order);
        assert!(mode != ExecutionMode::Wait);

        // Wide spread should trigger wait
        let mut wide_book = OrderBook::new("BTCUSDT".to_string());
        wide_book.update_bid(49900.0, 10.0); // 200 bps spread
        wide_book.update_ask(50100.0, 10.0);
        let mode = strategy.determine_mode(&wide_book, &order);
        assert_eq!(mode, ExecutionMode::Wait);
    }

    #[test]
    fn test_adaptive_slippage_tracking() {
        let mut strategy = AdaptiveStrategy::new();

        strategy.record_slippage(5.0);
        strategy.record_slippage(10.0);
        strategy.record_slippage(3.0);

        let avg = strategy.average_slippage();
        assert!((avg - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_llm_strategy_fallback() {
        let adapter = LlmAdapter::new(LlmConfig::default()).unwrap();
        let strategy = LlmStrategy::new(adapter);

        let order = create_test_order();
        let book = create_test_orderbook();

        // Without a decision, should fall back to adaptive
        let slice = strategy.next_slice(&order, &book).unwrap();
        assert!(slice.quantity >= 0.0);
    }

    #[test]
    fn test_hybrid_strategy() {
        use crate::strategy::TwapStrategy;

        let hybrid = HybridStrategy::new()
            .add_strategy(Box::new(TwapStrategy::new(60)), 0.5)
            .add_strategy(Box::new(AdaptiveStrategy::new()), 0.5);

        let order = create_test_order();
        let book = create_test_orderbook();

        let slice = hybrid.next_slice(&order, &book).unwrap();
        assert!(slice.quantity >= 0.0);
    }
}
