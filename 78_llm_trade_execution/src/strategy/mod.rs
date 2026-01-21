//! Execution strategy implementations.
//!
//! This module provides various execution strategies:
//! - TWAP (Time-Weighted Average Price)
//! - VWAP (Volume-Weighted Average Price)
//! - Implementation Shortfall (IS)
//! - Adaptive LLM-based strategy

mod adaptive;
mod is;
mod twap;
mod vwap;

pub use adaptive::{AdaptiveStrategy, LlmStrategy};
pub use is::ImplementationShortfallStrategy;
pub use twap::TwapStrategy;
pub use vwap::{VolumeProfile, VwapStrategy};

use crate::data::OrderBook;
use crate::execution::{ExecutionError, ParentOrder};
use serde::{Deserialize, Serialize};

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Slice interval in seconds
    pub slice_interval_secs: u64,
    /// Minimum slice size
    pub min_slice_size: f64,
    /// Maximum slice size
    pub max_slice_size: f64,
    /// Use limit orders
    pub use_limit_orders: bool,
    /// Limit order offset from mid (bps)
    pub limit_offset_bps: f64,
    /// Maximum spread to execute (bps)
    pub max_spread_bps: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            slice_interval_secs: 60,
            min_slice_size: 0.001,
            max_slice_size: 10.0,
            use_limit_orders: true,
            limit_offset_bps: 2.0,
            max_spread_bps: 50.0,
        }
    }
}

/// Execution slice specification
#[derive(Debug, Clone)]
pub struct ExecutionSlice {
    /// Quantity to execute
    pub quantity: f64,
    /// Limit price (None for market order)
    pub limit_price: Option<f64>,
    /// Time-in-force
    pub time_in_force: TimeInForce,
    /// Urgency level (0-1)
    pub urgency: f64,
}

impl ExecutionSlice {
    /// Create a market order slice
    pub fn market(quantity: f64) -> Self {
        Self {
            quantity,
            limit_price: None,
            time_in_force: TimeInForce::IOC,
            urgency: 1.0,
        }
    }

    /// Create a limit order slice
    pub fn limit(quantity: f64, price: f64) -> Self {
        Self {
            quantity,
            limit_price: Some(price),
            time_in_force: TimeInForce::GTC,
            urgency: 0.5,
        }
    }

    /// Create with specific urgency
    pub fn with_urgency(mut self, urgency: f64) -> Self {
        self.urgency = urgency.clamp(0.0, 1.0);
        self
    }
}

/// Time-in-force options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good till cancelled
    GTC,
    /// Immediate or cancel
    IOC,
    /// Fill or kill
    FOK,
    /// Post only
    PostOnly,
}

/// Execution strategy trait
pub trait ExecutionStrategy: Send + Sync {
    /// Get the next slice to execute
    fn next_slice(
        &self,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> Result<ExecutionSlice, ExecutionError>;

    /// Get the recommended interval between slices (milliseconds)
    fn slice_interval_ms(&self) -> u64;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Check if strategy is complete
    fn is_complete(&self, order: &ParentOrder) -> bool {
        order.remaining_quantity() <= 0.0
    }

    /// Reset strategy state (for new order)
    fn reset(&mut self) {}
}

/// Calculate limit price based on side and aggressiveness
pub fn calculate_limit_price(
    orderbook: &OrderBook,
    side: crate::execution::Side,
    aggressiveness: f64, // -1 (passive) to +1 (aggressive)
) -> Option<f64> {
    let mid = orderbook.mid_price()?;
    let spread = orderbook.spread()?;

    // Calculate offset from mid based on aggressiveness
    // Aggressive: cross the spread, Passive: stay behind best price
    let offset = spread * 0.5 * aggressiveness;

    match side {
        crate::execution::Side::Buy => Some(mid + offset),
        crate::execution::Side::Sell => Some(mid - offset),
    }
}

/// Calculate optimal slice size based on market conditions
pub fn calculate_slice_size(
    remaining: f64,
    remaining_time: f64,
    orderbook: &OrderBook,
    config: &StrategyConfig,
) -> f64 {
    // Base slice: uniform distribution over remaining time
    let base_slice = if remaining_time > 0.0 {
        remaining / (remaining_time / config.slice_interval_secs as f64)
    } else {
        remaining
    };

    // Adjust for order book depth
    let depth = orderbook.bid_depth(5) + orderbook.ask_depth(5);
    let depth_factor = if depth > 0.0 {
        (remaining / depth).min(0.1) // Don't take more than 10% of visible depth
    } else {
        0.1
    };

    // Final slice size
    let slice = (base_slice * (1.0 + depth_factor))
        .min(config.max_slice_size)
        .max(config.min_slice_size)
        .min(remaining);

    slice
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_orderbook() -> OrderBook {
        let mut book = OrderBook::new("TEST".to_string());
        for i in 1..=10 {
            book.update_bid(100.0 - i as f64 * 0.1, 10.0);
            book.update_ask(100.0 + i as f64 * 0.1, 10.0);
        }
        book
    }

    #[test]
    fn test_execution_slice_market() {
        let slice = ExecutionSlice::market(1.0);
        assert!(slice.limit_price.is_none());
        assert_eq!(slice.time_in_force, TimeInForce::IOC);
    }

    #[test]
    fn test_execution_slice_limit() {
        let slice = ExecutionSlice::limit(1.0, 100.0);
        assert_eq!(slice.limit_price, Some(100.0));
        assert_eq!(slice.time_in_force, TimeInForce::GTC);
    }

    #[test]
    fn test_calculate_limit_price() {
        let book = create_test_orderbook();

        // Aggressive buy should be above mid
        let aggressive_buy = calculate_limit_price(
            &book,
            crate::execution::Side::Buy,
            0.8,
        )
        .unwrap();

        let mid = book.mid_price().unwrap();
        assert!(aggressive_buy > mid);

        // Passive buy should be below mid
        let passive_buy = calculate_limit_price(
            &book,
            crate::execution::Side::Buy,
            -0.8,
        )
        .unwrap();

        assert!(passive_buy < mid);
    }

    #[test]
    fn test_calculate_slice_size() {
        let book = create_test_orderbook();
        let config = StrategyConfig::default();

        let slice = calculate_slice_size(10.0, 600.0, &book, &config);

        assert!(slice > 0.0);
        assert!(slice <= config.max_slice_size);
        assert!(slice >= config.min_slice_size);
    }
}
