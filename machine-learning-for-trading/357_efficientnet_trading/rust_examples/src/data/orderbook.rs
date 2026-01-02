//! Order book data structures

use serde::{Deserialize, Serialize};

/// Single order book level (price + quantity)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

impl OrderBookLevel {
    pub fn new(price: f64, quantity: f64) -> Self {
        Self { price, quantity }
    }

    /// Get notional value
    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Create an empty order book
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            timestamp: 0,
            bids: Vec::new(),
            asks: Vec::new(),
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread (best ask - best bid)
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Get total bid volume
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Get total ask volume
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.iter().map(|l| l.quantity).sum()
    }

    /// Get volume imbalance (-1 to 1, positive means more bids)
    pub fn volume_imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        let total = bid_vol + ask_vol;

        if total == 0.0 {
            0.0
        } else {
            (bid_vol - ask_vol) / total
        }
    }

    /// Get bid volume within percentage of best bid
    pub fn bid_volume_within(&self, pct: f64) -> f64 {
        let best = match self.best_bid() {
            Some(p) => p,
            None => return 0.0,
        };
        let threshold = best * (1.0 - pct / 100.0);

        self.bids
            .iter()
            .filter(|l| l.price >= threshold)
            .map(|l| l.quantity)
            .sum()
    }

    /// Get ask volume within percentage of best ask
    pub fn ask_volume_within(&self, pct: f64) -> f64 {
        let best = match self.best_ask() {
            Some(p) => p,
            None => return 0.0,
        };
        let threshold = best * (1.0 + pct / 100.0);

        self.asks
            .iter()
            .filter(|l| l.price <= threshold)
            .map(|l| l.quantity)
            .sum()
    }

    /// Calculate VWAP for bids
    pub fn bid_vwap(&self, depth: usize) -> Option<f64> {
        let levels: Vec<_> = self.bids.iter().take(depth).collect();
        if levels.is_empty() {
            return None;
        }

        let total_qty: f64 = levels.iter().map(|l| l.quantity).sum();
        if total_qty == 0.0 {
            return None;
        }

        let vwap: f64 = levels.iter().map(|l| l.price * l.quantity).sum::<f64>() / total_qty;
        Some(vwap)
    }

    /// Calculate VWAP for asks
    pub fn ask_vwap(&self, depth: usize) -> Option<f64> {
        let levels: Vec<_> = self.asks.iter().take(depth).collect();
        if levels.is_empty() {
            return None;
        }

        let total_qty: f64 = levels.iter().map(|l| l.quantity).sum();
        if total_qty == 0.0 {
            return None;
        }

        let vwap: f64 = levels.iter().map(|l| l.price * l.quantity).sum::<f64>() / total_qty;
        Some(vwap)
    }

    /// Get price levels as vectors for visualization
    pub fn to_price_levels(&self, depth: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let bid_prices: Vec<f64> = self.bids.iter().take(depth).map(|l| l.price).collect();
        let bid_quantities: Vec<f64> = self.bids.iter().take(depth).map(|l| l.quantity).collect();
        let ask_prices: Vec<f64> = self.asks.iter().take(depth).map(|l| l.price).collect();
        let ask_quantities: Vec<f64> = self.asks.iter().take(depth).map(|l| l.quantity).collect();

        (bid_prices, bid_quantities, ask_prices, ask_quantities)
    }
}

/// Collection of order book snapshots over time
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub snapshots: Vec<OrderBook>,
}

impl OrderBookSnapshot {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    pub fn push(&mut self, ob: OrderBook) {
        self.snapshots.push(ob);
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Convert to matrix format for heatmap visualization
    /// Returns (price_levels, time_indices, bid_volumes, ask_volumes)
    pub fn to_heatmap_data(
        &self,
        price_levels: usize,
    ) -> (Vec<f64>, Vec<u64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        if self.snapshots.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }

        // Find price range across all snapshots
        let mut min_price = f64::MAX;
        let mut max_price = f64::MIN;

        for snapshot in &self.snapshots {
            if let Some(bid) = snapshot.bids.last() {
                min_price = min_price.min(bid.price);
            }
            if let Some(ask) = snapshot.asks.last() {
                max_price = max_price.max(ask.price);
            }
        }

        let price_step = (max_price - min_price) / price_levels as f64;
        let prices: Vec<f64> = (0..price_levels)
            .map(|i| min_price + i as f64 * price_step)
            .collect();

        let times: Vec<u64> = self.snapshots.iter().map(|s| s.timestamp).collect();

        let mut bid_volumes = vec![vec![0.0; self.snapshots.len()]; price_levels];
        let mut ask_volumes = vec![vec![0.0; self.snapshots.len()]; price_levels];

        for (t_idx, snapshot) in self.snapshots.iter().enumerate() {
            for bid in &snapshot.bids {
                let p_idx = ((bid.price - min_price) / price_step).floor() as usize;
                if p_idx < price_levels {
                    bid_volumes[p_idx][t_idx] += bid.quantity;
                }
            }
            for ask in &snapshot.asks {
                let p_idx = ((ask.price - min_price) / price_step).floor() as usize;
                if p_idx < price_levels {
                    ask_volumes[p_idx][t_idx] += ask.quantity;
                }
            }
        }

        (prices, times, bid_volumes, ask_volumes)
    }
}

impl Default for OrderBookSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_orderbook() -> OrderBook {
        OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 1234567890,
            bids: vec![
                OrderBookLevel::new(100.0, 10.0),
                OrderBookLevel::new(99.0, 20.0),
                OrderBookLevel::new(98.0, 30.0),
            ],
            asks: vec![
                OrderBookLevel::new(101.0, 15.0),
                OrderBookLevel::new(102.0, 25.0),
                OrderBookLevel::new(103.0, 35.0),
            ],
        }
    }

    #[test]
    fn test_best_prices() {
        let ob = sample_orderbook();
        assert_eq!(ob.best_bid(), Some(100.0));
        assert_eq!(ob.best_ask(), Some(101.0));
    }

    #[test]
    fn test_mid_price() {
        let ob = sample_orderbook();
        assert_eq!(ob.mid_price(), Some(100.5));
    }

    #[test]
    fn test_spread() {
        let ob = sample_orderbook();
        assert_eq!(ob.spread(), Some(1.0));
    }

    #[test]
    fn test_volume_imbalance() {
        let ob = sample_orderbook();
        let bid_vol = 10.0 + 20.0 + 30.0; // 60
        let ask_vol = 15.0 + 25.0 + 35.0; // 75
        let expected = (bid_vol - ask_vol) / (bid_vol + ask_vol);
        assert!((ob.volume_imbalance() - expected).abs() < 0.001);
    }
}
