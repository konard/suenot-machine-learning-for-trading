//! TWAP (Time-Weighted Average Price) execution strategy.

use crate::data::OrderBook;
use crate::execution::{ExecutionError, ParentOrder, Side};
use crate::strategy::{calculate_limit_price, ExecutionSlice, ExecutionStrategy, StrategyConfig};

/// TWAP execution strategy
///
/// Divides the order into equal-sized slices executed at regular intervals.
/// This is the simplest execution algorithm, suitable when:
/// - Market impact is expected to be uniform over time
/// - Volume profile is relatively flat
/// - Simplicity is preferred over optimization
#[derive(Debug, Clone)]
pub struct TwapStrategy {
    config: StrategyConfig,
    /// Number of slices per time period
    slices_executed: u32,
}

impl TwapStrategy {
    /// Create a new TWAP strategy with default config
    pub fn new(slice_interval_secs: u64) -> Self {
        Self {
            config: StrategyConfig {
                slice_interval_secs,
                ..Default::default()
            },
            slices_executed: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StrategyConfig) -> Self {
        Self {
            config,
            slices_executed: 0,
        }
    }

    /// Calculate the number of slices for the order
    fn calculate_num_slices(&self, order: &ParentOrder) -> u32 {
        let remaining_time = order.remaining_time();
        (remaining_time / self.config.slice_interval_secs).max(1) as u32
    }

    /// Calculate slice quantity
    fn calculate_slice_quantity(&self, order: &ParentOrder) -> f64 {
        let num_slices = self.calculate_num_slices(order);
        let remaining = order.remaining_quantity();

        // Uniform distribution
        let slice = remaining / num_slices as f64;

        // Apply size constraints
        slice
            .min(self.config.max_slice_size)
            .max(self.config.min_slice_size)
            .min(remaining)
    }
}

impl ExecutionStrategy for TwapStrategy {
    fn next_slice(
        &self,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> Result<ExecutionSlice, ExecutionError> {
        let remaining = order.remaining_quantity();

        if remaining <= 0.0 {
            return Ok(ExecutionSlice::market(0.0));
        }

        // Check spread
        if let Some(spread_bps) = orderbook.spread_bps() {
            if spread_bps > self.config.max_spread_bps {
                // Spread too wide, wait
                return Ok(ExecutionSlice::market(0.0));
            }
        }

        let quantity = self.calculate_slice_quantity(order);

        let slice = if self.config.use_limit_orders {
            // Use limit order slightly inside the spread
            let aggressiveness = order.urgency * 0.5; // Scale urgency
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
        "TWAP"
    }

    fn reset(&mut self) {
        self.slices_executed = 0;
    }
}

/// TWAP with randomization to avoid detection
#[derive(Debug, Clone)]
pub struct RandomizedTwapStrategy {
    base: TwapStrategy,
    /// Size randomization range (fraction)
    size_jitter: f64,
    /// Time randomization range (fraction)
    time_jitter: f64,
}

impl RandomizedTwapStrategy {
    /// Create a new randomized TWAP strategy
    pub fn new(slice_interval_secs: u64, size_jitter: f64, time_jitter: f64) -> Self {
        Self {
            base: TwapStrategy::new(slice_interval_secs),
            size_jitter: size_jitter.clamp(0.0, 0.5),
            time_jitter: time_jitter.clamp(0.0, 0.5),
        }
    }

    /// Apply jitter to a value
    fn apply_jitter(&self, value: f64, jitter: f64) -> f64 {
        let random = rand::random::<f64>(); // 0 to 1
        let adjustment = 1.0 + jitter * (2.0 * random - 1.0); // 1-jitter to 1+jitter
        value * adjustment
    }
}

impl ExecutionStrategy for RandomizedTwapStrategy {
    fn next_slice(
        &self,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> Result<ExecutionSlice, ExecutionError> {
        let mut slice = self.base.next_slice(order, orderbook)?;

        // Apply size jitter
        slice.quantity = self.apply_jitter(slice.quantity, self.size_jitter);
        slice.quantity = slice.quantity.min(order.remaining_quantity());

        Ok(slice)
    }

    fn slice_interval_ms(&self) -> u64 {
        let base_interval = self.base.slice_interval_ms();
        self.apply_jitter(base_interval as f64, self.time_jitter) as u64
    }

    fn name(&self) -> &str {
        "RandomizedTWAP"
    }

    fn reset(&mut self) {
        self.base.reset();
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
            book.update_bid(50000.0 - i as f64 * 1.0, 1.0);
            book.update_ask(50000.0 + i as f64 * 1.0, 1.0);
        }
        book
    }

    #[test]
    fn test_twap_slice_quantity() {
        let strategy = TwapStrategy::new(60); // 60 second slices
        let order = create_test_order();
        let book = create_test_orderbook();

        let slice = strategy.next_slice(&order, &book).unwrap();

        // 600 seconds / 60 = 10 slices, 10.0 qty / 10 = 1.0 per slice
        assert!(slice.quantity > 0.0);
        assert!(slice.quantity <= 10.0);
    }

    #[test]
    fn test_twap_handles_remaining() {
        let strategy = TwapStrategy::new(60);
        let mut order = create_test_order();
        let book = create_test_orderbook();

        // Fill most of the order
        order.record_fill(9.5, 50000.0);

        let slice = strategy.next_slice(&order, &book).unwrap();

        // Should not exceed remaining
        assert!(slice.quantity <= 0.5);
    }

    #[test]
    fn test_twap_respects_spread_limit() {
        let strategy = TwapStrategy::with_config(StrategyConfig {
            max_spread_bps: 1.0, // Very tight
            ..Default::default()
        });
        let order = create_test_order();
        let book = create_test_orderbook(); // Has wider spread

        let slice = strategy.next_slice(&order, &book).unwrap();

        // Should skip due to wide spread
        assert_eq!(slice.quantity, 0.0);
    }

    #[test]
    fn test_randomized_twap() {
        let strategy = RandomizedTwapStrategy::new(60, 0.2, 0.2);
        let order = create_test_order();
        let book = create_test_orderbook();

        // Run multiple times to test randomization
        let mut quantities = Vec::new();
        for _ in 0..10 {
            let slice = strategy.next_slice(&order, &book).unwrap();
            quantities.push(slice.quantity);
        }

        // Should have some variation
        let min = quantities.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = quantities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // With 20% jitter, we should see some variation
        assert!(max > min || quantities.len() < 3);
    }

    #[test]
    fn test_strategy_name() {
        let twap = TwapStrategy::new(60);
        assert_eq!(twap.name(), "TWAP");

        let randomized = RandomizedTwapStrategy::new(60, 0.1, 0.1);
        assert_eq!(randomized.name(), "RandomizedTWAP");
    }
}
