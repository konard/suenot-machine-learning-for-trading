//! VWAP (Volume-Weighted Average Price) execution strategy.

use crate::data::OrderBook;
use crate::execution::{ExecutionError, ParentOrder};
use crate::strategy::{calculate_limit_price, ExecutionSlice, ExecutionStrategy, StrategyConfig};

/// Intraday volume profile
#[derive(Debug, Clone)]
pub struct VolumeProfile {
    /// Volume weights for each period (should sum to 1.0)
    pub weights: Vec<f64>,
    /// Period duration in minutes
    pub period_minutes: u32,
}

impl VolumeProfile {
    /// Create a uniform volume profile
    pub fn uniform(num_periods: usize) -> Self {
        let weight = 1.0 / num_periods as f64;
        Self {
            weights: vec![weight; num_periods],
            period_minutes: 60,
        }
    }

    /// Create a typical equity market U-shaped profile
    /// Higher volume at market open and close
    pub fn equity_u_shape() -> Self {
        // 6.5 hour trading day, 13 30-minute periods
        let weights = vec![
            0.12, // 9:30 - 10:00 (high)
            0.08, // 10:00 - 10:30
            0.07, // 10:30 - 11:00
            0.06, // 11:00 - 11:30
            0.05, // 11:30 - 12:00
            0.05, // 12:00 - 12:30 (lunch, low)
            0.05, // 12:30 - 13:00
            0.06, // 13:00 - 13:30
            0.07, // 13:30 - 14:00
            0.08, // 14:00 - 14:30
            0.09, // 14:30 - 15:00
            0.10, // 15:00 - 15:30
            0.12, // 15:30 - 16:00 (high)
        ];
        Self {
            weights,
            period_minutes: 30,
        }
    }

    /// Create a crypto 24h profile (relatively flat with slight variations)
    pub fn crypto_24h() -> Self {
        // 24 1-hour periods
        let weights = vec![
            0.035, // 00:00 UTC (low, Asia night)
            0.035,
            0.040, // 02:00 UTC (Asia morning)
            0.045,
            0.050,
            0.050, // 05:00 UTC (Asia day)
            0.045,
            0.045,
            0.050, // 08:00 UTC (Europe open)
            0.055,
            0.055,
            0.050,
            0.045, // 12:00 UTC
            0.045,
            0.050, // 14:00 UTC (US pre-market)
            0.055,
            0.060, // 16:00 UTC (US market hours)
            0.055,
            0.050,
            0.045,
            0.040, // 20:00 UTC (US evening)
            0.035,
            0.030, // 22:00 UTC (low)
            0.030,
        ];
        Self {
            weights,
            period_minutes: 60,
        }
    }

    /// Get the weight for a specific period index
    pub fn get_weight(&self, period: usize) -> f64 {
        self.weights.get(period).copied().unwrap_or(0.0)
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }
}

/// VWAP execution strategy
///
/// Executes proportionally to expected volume profile to minimize
/// deviation from the market VWAP.
#[derive(Debug, Clone)]
pub struct VwapStrategy {
    config: StrategyConfig,
    profile: VolumeProfile,
    current_period: usize,
    period_executed: f64,
}

impl VwapStrategy {
    /// Create a new VWAP strategy with uniform profile
    pub fn new(num_periods: usize) -> Self {
        Self {
            config: StrategyConfig::default(),
            profile: VolumeProfile::uniform(num_periods),
            current_period: 0,
            period_executed: 0.0,
        }
    }

    /// Create with custom volume profile
    pub fn with_profile(profile: VolumeProfile) -> Self {
        Self {
            config: StrategyConfig::default(),
            profile,
            current_period: 0,
            period_executed: 0.0,
        }
    }

    /// Create with configuration
    pub fn with_config(config: StrategyConfig, profile: VolumeProfile) -> Self {
        Self {
            config,
            profile,
            current_period: 0,
            period_executed: 0.0,
        }
    }

    /// Create for equity markets
    pub fn equity() -> Self {
        Self::with_profile(VolumeProfile::equity_u_shape())
    }

    /// Create for crypto markets
    pub fn crypto() -> Self {
        Self::with_profile(VolumeProfile::crypto_24h())
    }

    /// Calculate target quantity for current period
    fn calculate_period_target(&self, order: &ParentOrder) -> f64 {
        let weight = self.profile.get_weight(self.current_period);
        order.total_quantity * weight
    }

    /// Calculate remaining target for current period
    fn calculate_remaining_target(&self, order: &ParentOrder) -> f64 {
        let period_target = self.calculate_period_target(order);
        (period_target - self.period_executed).max(0.0)
    }

    /// Update period based on elapsed time
    fn update_period(&mut self, order: &ParentOrder) {
        let elapsed = order.elapsed_time();
        let period_duration = self.config.slice_interval_secs * (self.profile.weights.len() as u64);
        let new_period = ((elapsed as f64 / period_duration as f64) * self.profile.weights.len() as f64) as usize;

        if new_period != self.current_period && new_period < self.profile.weights.len() {
            self.current_period = new_period;
            self.period_executed = 0.0;
        }
    }
}

impl ExecutionStrategy for VwapStrategy {
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
                return Ok(ExecutionSlice::market(0.0));
            }
        }

        // Calculate quantity based on volume profile
        let period_target = self.calculate_remaining_target(order);

        // Scale down if ahead of schedule, scale up if behind
        let time_progress = order.elapsed_time() as f64 / order.time_horizon as f64;
        let fill_progress = order.fill_rate();

        let adjustment = if fill_progress < time_progress {
            1.2 // Behind schedule, increase
        } else if fill_progress > time_progress + 0.1 {
            0.8 // Ahead of schedule, decrease
        } else {
            1.0
        };

        let quantity = (period_target * adjustment)
            .min(self.config.max_slice_size)
            .max(self.config.min_slice_size)
            .min(remaining);

        let slice = if self.config.use_limit_orders {
            let aggressiveness = if fill_progress < time_progress {
                order.urgency * 0.7 // More aggressive when behind
            } else {
                order.urgency * 0.3 // More passive when ahead
            };

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
        "VWAP"
    }

    fn reset(&mut self) {
        self.current_period = 0;
        self.period_executed = 0.0;
    }
}

/// Real-time VWAP tracker
#[derive(Debug, Clone, Default)]
pub struct VwapTracker {
    /// Cumulative price * volume
    cumulative_pv: f64,
    /// Cumulative volume
    cumulative_volume: f64,
    /// Number of trades
    trade_count: u64,
}

impl VwapTracker {
    /// Create a new VWAP tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a trade to the tracker
    pub fn add_trade(&mut self, price: f64, volume: f64) {
        self.cumulative_pv += price * volume;
        self.cumulative_volume += volume;
        self.trade_count += 1;
    }

    /// Get the current VWAP
    pub fn vwap(&self) -> Option<f64> {
        if self.cumulative_volume > 0.0 {
            Some(self.cumulative_pv / self.cumulative_volume)
        } else {
            None
        }
    }

    /// Get the total volume
    pub fn total_volume(&self) -> f64 {
        self.cumulative_volume
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.cumulative_pv = 0.0;
        self.cumulative_volume = 0.0;
        self.trade_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::Side;

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
    fn test_volume_profile_uniform() {
        let profile = VolumeProfile::uniform(10);
        assert_eq!(profile.weights.len(), 10);

        let sum: f64 = profile.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_volume_profile_equity() {
        let profile = VolumeProfile::equity_u_shape();

        // First and last periods should have higher volume
        let first = profile.get_weight(0);
        let middle = profile.get_weight(6);
        let last = profile.get_weight(12);

        assert!(first > middle);
        assert!(last > middle);
    }

    #[test]
    fn test_vwap_strategy() {
        let strategy = VwapStrategy::new(10);
        let order = create_test_order();
        let book = create_test_orderbook();

        let slice = strategy.next_slice(&order, &book).unwrap();

        assert!(slice.quantity > 0.0);
        assert!(slice.quantity <= 10.0);
    }

    #[test]
    fn test_vwap_tracker() {
        let mut tracker = VwapTracker::new();

        tracker.add_trade(100.0, 10.0); // 1000 PV
        tracker.add_trade(102.0, 5.0);  // 510 PV

        let vwap = tracker.vwap().unwrap();
        // Expected: (1000 + 510) / 15 = 100.67
        assert!((vwap - 100.67).abs() < 0.01);

        assert_eq!(tracker.total_volume(), 15.0);
    }

    #[test]
    fn test_equity_vwap() {
        let strategy = VwapStrategy::equity();
        assert_eq!(strategy.name(), "VWAP");
        assert_eq!(strategy.profile.weights.len(), 13);
    }

    #[test]
    fn test_crypto_vwap() {
        let strategy = VwapStrategy::crypto();
        assert_eq!(strategy.profile.weights.len(), 24);
    }
}
