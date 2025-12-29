//! # Order Flow Imbalance (OFI)
//!
//! Implementation of Order Flow Imbalance based on Cont et al. (2014):
//! "The Price Impact of Order Book Events"
//!
//! OFI measures the net order flow pressure by tracking changes in the order book.

use crate::data::orderbook::OrderBook;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

/// Order Flow Imbalance calculator
#[derive(Debug)]
pub struct OrderFlowCalculator {
    /// Previous order book for comparison
    prev_book: Option<OrderBookState>,
    /// Rolling OFI values
    ofi_history: VecDeque<OfiPoint>,
    /// Maximum history size
    max_history: usize,
    /// Cumulative OFI
    cumulative_ofi: f64,
}

/// Simplified order book state for OFI calculation
#[derive(Debug, Clone)]
struct OrderBookState {
    timestamp: DateTime<Utc>,
    bid_price: f64,
    bid_size: f64,
    ask_price: f64,
    ask_size: f64,
}

/// Single OFI data point
#[derive(Debug, Clone)]
pub struct OfiPoint {
    pub timestamp: DateTime<Utc>,
    pub ofi: f64,
    pub delta_bid: f64,
    pub delta_ask: f64,
}

impl Default for OrderFlowCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderFlowCalculator {
    /// Create a new OFI calculator
    pub fn new() -> Self {
        Self::with_history_size(10000)
    }

    /// Create with custom history size
    pub fn with_history_size(max_history: usize) -> Self {
        Self {
            prev_book: None,
            ofi_history: VecDeque::with_capacity(max_history),
            max_history,
            cumulative_ofi: 0.0,
        }
    }

    /// Calculate OFI from a new order book snapshot
    ///
    /// Returns the OFI value (positive = buy pressure, negative = sell pressure)
    pub fn update(&mut self, book: &OrderBook) -> Option<f64> {
        let current = OrderBookState {
            timestamp: book.timestamp,
            bid_price: book.best_bid()?.price,
            bid_size: book.best_bid()?.size,
            ask_price: book.best_ask()?.price,
            ask_size: book.best_ask()?.size,
        };

        let ofi = if let Some(prev) = &self.prev_book {
            let (delta_bid, delta_ask) = self.calculate_deltas(prev, &current);
            let ofi = delta_bid + delta_ask;

            // Store in history
            let point = OfiPoint {
                timestamp: current.timestamp,
                ofi,
                delta_bid,
                delta_ask,
            };

            self.ofi_history.push_back(point);
            if self.ofi_history.len() > self.max_history {
                self.ofi_history.pop_front();
            }

            // Update cumulative
            self.cumulative_ofi += ofi;

            Some(ofi)
        } else {
            None
        };

        self.prev_book = Some(current);
        ofi
    }

    /// Calculate OFI between two order book states (static method)
    pub fn calculate_ofi(prev: &OrderBook, current: &OrderBook) -> Option<f64> {
        let prev_state = OrderBookState {
            timestamp: prev.timestamp,
            bid_price: prev.best_bid()?.price,
            bid_size: prev.best_bid()?.size,
            ask_price: prev.best_ask()?.price,
            ask_size: prev.best_ask()?.size,
        };

        let current_state = OrderBookState {
            timestamp: current.timestamp,
            bid_price: current.best_bid()?.price,
            bid_size: current.best_bid()?.size,
            ask_price: current.best_ask()?.price,
            ask_size: current.best_ask()?.size,
        };

        let calc = Self::new();
        let (delta_bid, delta_ask) = calc.calculate_deltas(&prev_state, &current_state);
        Some(delta_bid + delta_ask)
    }

    /// Calculate bid and ask deltas according to Cont et al. (2014)
    fn calculate_deltas(&self, prev: &OrderBookState, current: &OrderBookState) -> (f64, f64) {
        // Bid side changes
        let delta_bid = if current.bid_price > prev.bid_price {
            // Bid price improved - aggressive buying
            current.bid_size
        } else if current.bid_price == prev.bid_price {
            // Price unchanged - net change in size
            current.bid_size - prev.bid_size
        } else {
            // Bid price dropped - buying pressure decreased
            -prev.bid_size
        };

        // Ask side changes (note: signs are inverted)
        let delta_ask = if current.ask_price < prev.ask_price {
            // Ask price dropped - aggressive selling
            -current.ask_size
        } else if current.ask_price == prev.ask_price {
            // Price unchanged - net change in size (inverted)
            -(current.ask_size - prev.ask_size)
        } else {
            // Ask price increased - selling pressure decreased
            prev.ask_size
        };

        (delta_bid, delta_ask)
    }

    /// Get cumulative OFI
    pub fn cumulative(&self) -> f64 {
        self.cumulative_ofi
    }

    /// Get OFI sum over last N updates
    pub fn sum_last_n(&self, n: usize) -> f64 {
        self.ofi_history
            .iter()
            .rev()
            .take(n)
            .map(|p| p.ofi)
            .sum()
    }

    /// Get OFI sum over a time window
    pub fn sum_window(&self, window_seconds: i64) -> f64 {
        if self.ofi_history.is_empty() {
            return 0.0;
        }

        let latest = self.ofi_history.back().unwrap().timestamp;
        let cutoff = latest - chrono::Duration::seconds(window_seconds);

        self.ofi_history
            .iter()
            .filter(|p| p.timestamp >= cutoff)
            .map(|p| p.ofi)
            .sum()
    }

    /// Get OFI for 1 minute window
    pub fn ofi_1min(&self) -> f64 {
        self.sum_window(60)
    }

    /// Get OFI for 5 minute window
    pub fn ofi_5min(&self) -> f64 {
        self.sum_window(300)
    }

    /// Get OFI for 15 minute window
    pub fn ofi_15min(&self) -> f64 {
        self.sum_window(900)
    }

    /// Calculate Z-score of current OFI
    pub fn z_score(&self, window: usize) -> Option<f64> {
        if self.ofi_history.len() < window {
            return None;
        }

        let values: Vec<f64> = self.ofi_history
            .iter()
            .rev()
            .take(window)
            .map(|p| p.ofi)
            .collect();

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / values.len() as f64;
        let std = variance.sqrt();

        if std > 0.0 {
            let current = values.first()?;
            Some((current - mean) / std)
        } else {
            Some(0.0)
        }
    }

    /// Get history length
    pub fn history_len(&self) -> usize {
        self.ofi_history.len()
    }

    /// Get recent OFI values
    pub fn recent(&self, n: usize) -> Vec<&OfiPoint> {
        self.ofi_history.iter().rev().take(n).collect()
    }

    /// Reset the calculator
    pub fn reset(&mut self) {
        self.prev_book = None;
        self.ofi_history.clear();
        self.cumulative_ofi = 0.0;
    }

    /// Get statistics about the OFI distribution
    pub fn statistics(&self) -> OfiStatistics {
        if self.ofi_history.is_empty() {
            return OfiStatistics::default();
        }

        let values: Vec<f64> = self.ofi_history.iter().map(|p| p.ofi).collect();
        let n = values.len() as f64;

        let sum: f64 = values.iter().sum();
        let mean = sum / n;

        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let positive_count = values.iter().filter(|&&x| x > 0.0).count();
        let negative_count = values.iter().filter(|&&x| x < 0.0).count();

        OfiStatistics {
            count: values.len(),
            mean,
            std,
            min,
            max,
            cumulative: self.cumulative_ofi,
            positive_ratio: positive_count as f64 / n,
            negative_ratio: negative_count as f64 / n,
        }
    }
}

/// Statistics about OFI distribution
#[derive(Debug, Clone, Default)]
pub struct OfiStatistics {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub cumulative: f64,
    pub positive_ratio: f64,
    pub negative_ratio: f64,
}

/// Multi-level OFI calculator (considers multiple order book levels)
#[derive(Debug)]
pub struct MultiLevelOfiCalculator {
    /// Calculators for each level
    level_calculators: Vec<OrderFlowCalculator>,
    /// Weights for each level
    weights: Vec<f64>,
}

impl MultiLevelOfiCalculator {
    /// Create a new multi-level calculator with specified number of levels
    pub fn new(levels: usize) -> Self {
        let mut weights = Vec::with_capacity(levels);
        let mut calculators = Vec::with_capacity(levels);

        // Exponentially decaying weights
        for i in 0..levels {
            weights.push(1.0 / (1.0 + i as f64));
            calculators.push(OrderFlowCalculator::new());
        }

        // Normalize weights
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        Self {
            level_calculators: calculators,
            weights,
        }
    }

    /// Calculate weighted OFI across multiple levels
    pub fn update(&mut self, book: &OrderBook) -> f64 {
        let mut weighted_ofi = 0.0;

        for (i, (calc, weight)) in self
            .level_calculators
            .iter_mut()
            .zip(self.weights.iter())
            .enumerate()
        {
            // Create a book view with only level i as the best level
            if let Some(level_book) = self.create_level_book(book, i) {
                if let Some(ofi) = calc.update(&level_book) {
                    weighted_ofi += ofi * weight;
                }
            }
        }

        weighted_ofi
    }

    fn create_level_book(&self, book: &OrderBook, level: usize) -> Option<OrderBook> {
        if level >= book.bids.len() || level >= book.asks.len() {
            return None;
        }

        let bids = vec![book.bids[level].clone()];
        let asks = vec![book.asks[level].clone()];

        Some(OrderBook::new(
            book.symbol.clone(),
            book.timestamp,
            bids,
            asks,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::orderbook::OrderBookLevel;

    fn create_book(bid_price: f64, bid_size: f64, ask_price: f64, ask_size: f64) -> OrderBook {
        let bids = vec![OrderBookLevel::new(bid_price, bid_size, 1)];
        let asks = vec![OrderBookLevel::new(ask_price, ask_size, 1)];
        OrderBook::new("BTCUSDT".to_string(), Utc::now(), bids, asks)
    }

    #[test]
    fn test_ofi_bid_price_increase() {
        let mut calc = OrderFlowCalculator::new();

        // Initial book
        let book1 = create_book(100.0, 10.0, 101.0, 10.0);
        calc.update(&book1);

        // Bid price increases - aggressive buying
        let book2 = create_book(100.5, 15.0, 101.0, 10.0);
        let ofi = calc.update(&book2).unwrap();

        // OFI should be positive (bid size at new level)
        assert!(ofi > 0.0);
        assert!((ofi - 15.0).abs() < 0.001); // delta_bid = 15, delta_ask = 0
    }

    #[test]
    fn test_ofi_bid_price_decrease() {
        let mut calc = OrderFlowCalculator::new();

        let book1 = create_book(100.0, 10.0, 101.0, 10.0);
        calc.update(&book1);

        // Bid price decreases - buying pressure dropped
        let book2 = create_book(99.5, 20.0, 101.0, 10.0);
        let ofi = calc.update(&book2).unwrap();

        // OFI should be negative
        assert!(ofi < 0.0);
        assert!((ofi - (-10.0)).abs() < 0.001); // delta_bid = -10 (old size)
    }

    #[test]
    fn test_ofi_ask_price_decrease() {
        let mut calc = OrderFlowCalculator::new();

        let book1 = create_book(100.0, 10.0, 101.0, 10.0);
        calc.update(&book1);

        // Ask price decreases - aggressive selling
        let book2 = create_book(100.0, 10.0, 100.5, 8.0);
        let ofi = calc.update(&book2).unwrap();

        // OFI should be negative
        assert!(ofi < 0.0);
        assert!((ofi - (-8.0)).abs() < 0.001); // delta_ask = -8
    }

    #[test]
    fn test_ofi_size_change_only() {
        let mut calc = OrderFlowCalculator::new();

        let book1 = create_book(100.0, 10.0, 101.0, 10.0);
        calc.update(&book1);

        // Only sizes change
        let book2 = create_book(100.0, 15.0, 101.0, 8.0);
        let ofi = calc.update(&book2).unwrap();

        // delta_bid = 15 - 10 = 5
        // delta_ask = -(8 - 10) = 2
        // OFI = 5 + 2 = 7
        assert!((ofi - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_ofi_cumulative() {
        let mut calc = OrderFlowCalculator::new();

        let book1 = create_book(100.0, 10.0, 101.0, 10.0);
        calc.update(&book1);

        let book2 = create_book(100.0, 15.0, 101.0, 10.0);
        calc.update(&book2);

        let book3 = create_book(100.0, 20.0, 101.0, 10.0);
        calc.update(&book3);

        // Cumulative should be sum of all OFIs
        assert!((calc.cumulative() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_static_calculate_ofi() {
        let book1 = create_book(100.0, 10.0, 101.0, 10.0);
        let book2 = create_book(100.5, 15.0, 101.0, 10.0);

        let ofi = OrderFlowCalculator::calculate_ofi(&book1, &book2).unwrap();
        assert!((ofi - 15.0).abs() < 0.001);
    }
}
