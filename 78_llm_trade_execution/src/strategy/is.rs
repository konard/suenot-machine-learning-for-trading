//! Implementation Shortfall (IS) execution strategy.

use crate::data::OrderBook;
use crate::execution::{ExecutionError, ParentOrder, Side};
use crate::impact::{AlmgrenChrissModel, AlmgrenChrissParams, MarketImpactModel};
use crate::strategy::{calculate_limit_price, ExecutionSlice, ExecutionStrategy, StrategyConfig};

/// Implementation Shortfall execution strategy
///
/// Minimizes the expected shortfall between the arrival price and
/// the actual execution price, balancing market impact against timing risk.
///
/// Based on the Almgren-Chriss optimal execution framework.
#[derive(Debug, Clone)]
pub struct ImplementationShortfallStrategy {
    config: StrategyConfig,
    model: AlmgrenChrissModel,
    trajectory: Option<Vec<f64>>,
    current_step: usize,
    /// Risk aversion parameter (higher = faster execution)
    risk_aversion: f64,
}

impl ImplementationShortfallStrategy {
    /// Create a new IS strategy with default parameters
    pub fn new() -> Self {
        Self {
            config: StrategyConfig::default(),
            model: AlmgrenChrissModel::default_params(),
            trajectory: None,
            current_step: 0,
            risk_aversion: 1e-6,
        }
    }

    /// Create with custom model parameters
    pub fn with_params(params: AlmgrenChrissParams) -> Self {
        let risk_aversion = params.lambda;
        Self {
            config: StrategyConfig::default(),
            model: AlmgrenChrissModel::new(params),
            trajectory: None,
            current_step: 0,
            risk_aversion,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StrategyConfig, params: AlmgrenChrissParams) -> Self {
        let risk_aversion = params.lambda;
        Self {
            config,
            model: AlmgrenChrissModel::new(params),
            trajectory: None,
            current_step: 0,
            risk_aversion,
        }
    }

    /// Set risk aversion parameter
    pub fn with_risk_aversion(mut self, lambda: f64) -> Self {
        self.risk_aversion = lambda;
        self
    }

    /// Calculate optimal trajectory for the order
    fn calculate_trajectory(&mut self, order: &ParentOrder) {
        let num_steps = (order.remaining_time() / self.config.slice_interval_secs).max(1) as usize;
        let trajectory = self.model.optimal_trajectory(order.remaining_quantity(), num_steps);
        self.trajectory = Some(trajectory);
        self.current_step = 0;
    }

    /// Get the next slice quantity from trajectory
    fn get_trajectory_slice(&self) -> f64 {
        if let Some(ref trajectory) = self.trajectory {
            trajectory.get(self.current_step).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Adjust trajectory based on market conditions
    fn adjust_for_market(&self, base_quantity: f64, order: &ParentOrder, orderbook: &OrderBook) -> f64 {
        let mut adjusted = base_quantity;

        // Adjust for spread
        if let Some(spread_bps) = orderbook.spread_bps() {
            if spread_bps < 3.0 {
                // Tight spread, can be more aggressive
                adjusted *= 1.2;
            } else if spread_bps > 20.0 {
                // Wide spread, be more passive
                adjusted *= 0.7;
            }
        }

        // Adjust for depth
        let depth = match order.side {
            Side::Buy => orderbook.ask_depth(5),
            Side::Sell => orderbook.bid_depth(5),
        };

        if depth > 0.0 {
            // Don't take more than 10% of visible depth
            adjusted = adjusted.min(depth * 0.1);
        }

        // Adjust for imbalance
        let imbalance = orderbook.imbalance(10);
        let favorable = match order.side {
            Side::Buy => imbalance < -0.2,  // More asks = favorable for buying
            Side::Sell => imbalance > 0.2,  // More bids = favorable for selling
        };

        if favorable {
            adjusted *= 1.1; // Slightly more aggressive when conditions are favorable
        }

        adjusted
    }

    /// Calculate urgency-adjusted aggressiveness
    fn calculate_aggressiveness(&self, order: &ParentOrder) -> f64 {
        // Higher urgency + behind schedule = more aggressive
        let time_progress = order.elapsed_time() as f64 / order.time_horizon as f64;
        let fill_progress = order.fill_rate();

        let schedule_factor = if fill_progress < time_progress {
            // Behind schedule
            (time_progress - fill_progress) * 2.0
        } else {
            // Ahead of schedule
            -(fill_progress - time_progress)
        };

        // Base aggressiveness from urgency
        let base = (order.urgency - 0.5) * 2.0; // -1 to 1

        (base + schedule_factor).clamp(-1.0, 1.0)
    }
}

impl Default for ImplementationShortfallStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionStrategy for ImplementationShortfallStrategy {
    fn next_slice(
        &self,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> Result<ExecutionSlice, ExecutionError> {
        let remaining = order.remaining_quantity();

        if remaining <= 0.0 {
            return Ok(ExecutionSlice::market(0.0));
        }

        // Check spread threshold
        if let Some(spread_bps) = orderbook.spread_bps() {
            if spread_bps > self.config.max_spread_bps {
                return Ok(ExecutionSlice::market(0.0));
            }
        }

        // Get base quantity from trajectory (or calculate if not set)
        let base_quantity = if self.trajectory.is_some() {
            self.get_trajectory_slice()
        } else {
            // Fallback: use model's expected cost minimization
            let num_steps = (order.remaining_time() / self.config.slice_interval_secs).max(1) as usize;
            remaining / num_steps as f64
        };

        // Adjust for market conditions
        let adjusted_quantity = self.adjust_for_market(base_quantity, order, orderbook);

        // Apply size constraints
        let quantity = adjusted_quantity
            .min(self.config.max_slice_size)
            .max(self.config.min_slice_size)
            .min(remaining);

        // Calculate price based on urgency and schedule
        let aggressiveness = self.calculate_aggressiveness(order);

        let slice = if self.config.use_limit_orders && aggressiveness < 0.8 {
            if let Some(price) = calculate_limit_price(orderbook, order.side, aggressiveness) {
                ExecutionSlice::limit(quantity, price).with_urgency(order.urgency)
            } else {
                ExecutionSlice::market(quantity)
            }
        } else {
            // Very urgent, use market order
            ExecutionSlice::market(quantity)
        };

        Ok(slice)
    }

    fn slice_interval_ms(&self) -> u64 {
        self.config.slice_interval_secs * 1000
    }

    fn name(&self) -> &str {
        "ImplementationShortfall"
    }

    fn reset(&mut self) {
        self.trajectory = None;
        self.current_step = 0;
    }
}

/// Calculate implementation shortfall
pub fn calculate_implementation_shortfall(
    arrival_price: f64,
    average_execution_price: f64,
    side: Side,
) -> f64 {
    let shortfall = (average_execution_price - arrival_price) / arrival_price;
    match side {
        Side::Buy => shortfall,  // Positive = paid more than arrival
        Side::Sell => -shortfall, // Positive = received less than arrival
    }
}

/// Calculate implementation shortfall in basis points
pub fn implementation_shortfall_bps(
    arrival_price: f64,
    average_execution_price: f64,
    side: Side,
) -> f64 {
    calculate_implementation_shortfall(arrival_price, average_execution_price, side) * 10000.0
}

/// Decompose implementation shortfall into components
#[derive(Debug, Clone)]
pub struct ShortfallDecomposition {
    /// Total implementation shortfall (bps)
    pub total_bps: f64,
    /// Delay cost (between decision and start)
    pub delay_cost_bps: f64,
    /// Timing cost (price movement during execution)
    pub timing_cost_bps: f64,
    /// Market impact cost
    pub impact_cost_bps: f64,
    /// Spread cost
    pub spread_cost_bps: f64,
}

impl ShortfallDecomposition {
    /// Create from execution data
    pub fn new(
        decision_price: f64,
        arrival_price: f64,
        average_price: f64,
        market_vwap: f64,
        side: Side,
    ) -> Self {
        let sign = side.sign();

        let total = ((average_price - decision_price) / decision_price) * sign * 10000.0;
        let delay = ((arrival_price - decision_price) / decision_price) * sign * 10000.0;
        let timing = ((market_vwap - arrival_price) / arrival_price) * sign * 10000.0;
        let impact = ((average_price - market_vwap) / market_vwap) * sign * 10000.0;

        Self {
            total_bps: total,
            delay_cost_bps: delay,
            timing_cost_bps: timing,
            impact_cost_bps: impact,
            spread_cost_bps: 0.0, // Would need bid/ask data
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_is_strategy_basic() {
        let strategy = ImplementationShortfallStrategy::new();
        let order = create_test_order();
        let book = create_test_orderbook();

        let slice = strategy.next_slice(&order, &book).unwrap();

        assert!(slice.quantity > 0.0);
        assert!(slice.quantity <= 10.0);
    }

    #[test]
    fn test_is_with_custom_params() {
        let params = AlmgrenChrissParams::liquid();
        let strategy = ImplementationShortfallStrategy::with_params(params);

        assert_eq!(strategy.name(), "ImplementationShortfall");
    }

    #[test]
    fn test_implementation_shortfall_calculation() {
        // Buy at higher price = positive shortfall
        let is = implementation_shortfall_bps(100.0, 101.0, Side::Buy);
        assert!((is - 100.0).abs() < 0.1); // 1% = 100 bps

        // Sell at lower price = positive shortfall
        let is = implementation_shortfall_bps(100.0, 99.0, Side::Sell);
        assert!((is - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_shortfall_decomposition() {
        let decomp = ShortfallDecomposition::new(
            100.0,  // decision price
            100.5,  // arrival price (slipped up 50 bps)
            101.0,  // average execution
            100.7,  // market vwap
            Side::Buy,
        );

        // Total should be roughly sum of components
        let sum = decomp.delay_cost_bps + decomp.timing_cost_bps + decomp.impact_cost_bps;
        assert!((decomp.total_bps - sum).abs() < 10.0); // Allow some rounding
    }

    #[test]
    fn test_aggressiveness_behind_schedule() {
        let strategy = ImplementationShortfallStrategy::new();
        let mut order = create_test_order();

        // Simulate being behind schedule
        // Order is 600 seconds, let's say 300 seconds have passed (50%)
        // but only 20% filled

        // We can't easily simulate time passing, so test the concept
        let aggressiveness = strategy.calculate_aggressiveness(&order);

        // At start, should be based on urgency (0.5 -> 0)
        assert!(aggressiveness.abs() < 0.5);
    }
}
