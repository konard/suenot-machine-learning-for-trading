//! Microstructure Features for Trading ML
//!
//! This library computes market microstructure features from orderbook and trade data,
//! designed for real-time use with Bybit cryptocurrency market data.

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ─── Data Structures ───────────────────────────────────────────────────────────

/// A single level in the order book (price, quantity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// A snapshot of the order book.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: u64,
}

/// A single trade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub price: f64,
    pub quantity: f64,
    pub is_buyer_maker: bool, // true = sell-initiated, false = buy-initiated
    pub timestamp: u64,
}

/// Computed microstructure features.
#[derive(Debug, Clone, Serialize)]
pub struct MicrostructureFeatures {
    pub absolute_spread: f64,
    pub relative_spread: f64,
    pub effective_spread: f64,
    pub midprice: f64,
    pub order_imbalance: f64,
    pub trade_imbalance: f64,
    pub kyles_lambda: f64,
    pub amihud_illiquidity: f64,
    pub vpin: f64,
}

// ─── Microstructure Calculator ─────────────────────────────────────────────────

/// Main calculator for microstructure features.
pub struct MicrostructureCalculator {
    /// Small constant to prevent division by zero.
    pub epsilon: f64,
    /// Number of depth levels to use for order imbalance.
    pub depth_levels: usize,
    /// Volume per bucket for VPIN calculation.
    pub vpin_bucket_volume: f64,
    /// Number of buckets in VPIN rolling window.
    pub vpin_window: usize,
}

impl Default for MicrostructureCalculator {
    fn default() -> Self {
        Self {
            epsilon: 1e-10,
            depth_levels: 5,
            vpin_bucket_volume: 1.0,
            vpin_window: 50,
        }
    }
}

impl MicrostructureCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the number of depth levels for order imbalance.
    pub fn with_depth_levels(mut self, levels: usize) -> Self {
        self.depth_levels = levels;
        self
    }

    /// Configure VPIN parameters.
    pub fn with_vpin_params(mut self, bucket_volume: f64, window: usize) -> Self {
        self.vpin_bucket_volume = bucket_volume;
        self.vpin_window = window;
        self
    }

    // ─── Spread Calculations ───────────────────────────────────────────────

    /// Compute the midprice from the best bid and ask.
    pub fn midprice(book: &OrderBook) -> f64 {
        if book.bids.is_empty() || book.asks.is_empty() {
            return 0.0;
        }
        (book.bids[0].price + book.asks[0].price) / 2.0
    }

    /// Absolute spread: ask - bid.
    pub fn absolute_spread(book: &OrderBook) -> f64 {
        if book.bids.is_empty() || book.asks.is_empty() {
            return 0.0;
        }
        book.asks[0].price - book.bids[0].price
    }

    /// Relative (proportional) spread: (ask - bid) / midprice.
    pub fn relative_spread(book: &OrderBook) -> f64 {
        let mid = Self::midprice(book);
        if mid <= 0.0 {
            return 0.0;
        }
        Self::absolute_spread(book) / mid
    }

    /// Effective spread: 2 * |trade_price - midprice|.
    pub fn effective_spread(trade_price: f64, midprice: f64) -> f64 {
        2.0 * (trade_price - midprice).abs()
    }

    /// Average effective spread over a series of trades.
    pub fn avg_effective_spread(trades: &[Trade], midprice: f64) -> f64 {
        if trades.is_empty() {
            return 0.0;
        }
        let sum: f64 = trades
            .iter()
            .map(|t| Self::effective_spread(t.price, midprice))
            .sum();
        sum / trades.len() as f64
    }

    // ─── Order Imbalance ───────────────────────────────────────────────────

    /// Order imbalance from the order book: (bid_vol - ask_vol) / (bid_vol + ask_vol).
    /// Uses up to `depth_levels` levels on each side.
    pub fn order_imbalance(&self, book: &OrderBook) -> f64 {
        let bid_vol: f64 = book
            .bids
            .iter()
            .take(self.depth_levels)
            .map(|l| l.quantity)
            .sum();
        let ask_vol: f64 = book
            .asks
            .iter()
            .take(self.depth_levels)
            .map(|l| l.quantity)
            .sum();
        let total = bid_vol + ask_vol;
        if total <= self.epsilon {
            return 0.0;
        }
        (bid_vol - ask_vol) / total
    }

    /// Trade imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol).
    pub fn trade_imbalance(&self, trades: &[Trade]) -> f64 {
        let mut buy_vol = 0.0;
        let mut sell_vol = 0.0;
        for t in trades {
            if t.is_buyer_maker {
                sell_vol += t.quantity;
            } else {
                buy_vol += t.quantity;
            }
        }
        let total = buy_vol + sell_vol;
        if total <= self.epsilon {
            return 0.0;
        }
        (buy_vol - sell_vol) / total
    }

    // ─── Kyle's Lambda ─────────────────────────────────────────────────────

    /// Estimate Kyle's lambda via OLS regression of price changes on signed order flow.
    ///
    /// Takes parallel slices of price changes and signed order flows.
    /// Returns the regression coefficient (lambda).
    pub fn kyles_lambda(price_changes: &[f64], order_flows: &[f64]) -> f64 {
        let n = price_changes.len().min(order_flows.len());
        if n < 2 {
            return 0.0;
        }

        let mean_of = order_flows[..n].iter().sum::<f64>() / n as f64;
        let mean_dp = price_changes[..n].iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var = 0.0;
        for i in 0..n {
            let of_dev = order_flows[i] - mean_of;
            let dp_dev = price_changes[i] - mean_dp;
            cov += of_dev * dp_dev;
            var += of_dev * of_dev;
        }

        if var.abs() < 1e-15 {
            return 0.0;
        }

        cov / var
    }

    /// Compute Kyle's lambda from a series of trades by grouping into intervals.
    ///
    /// `interval_ms` defines the grouping interval in milliseconds.
    pub fn kyles_lambda_from_trades(trades: &[Trade], interval_ms: u64) -> f64 {
        if trades.len() < 2 || interval_ms == 0 {
            return 0.0;
        }

        let start = trades[0].timestamp;
        let end = trades.last().unwrap().timestamp;
        if end <= start {
            return 0.0;
        }

        // Group trades into intervals
        let mut intervals: Vec<(f64, f64)> = Vec::new(); // (price_change, signed_flow)
        let mut interval_start = start;
        let mut first_price = trades[0].price;
        let mut last_price = trades[0].price;
        let mut signed_flow = 0.0;

        for trade in trades {
            if trade.timestamp >= interval_start + interval_ms {
                let price_change = last_price - first_price;
                intervals.push((price_change, signed_flow));

                interval_start += interval_ms;
                first_price = trade.price;
                signed_flow = 0.0;
            }
            last_price = trade.price;
            let sign = if trade.is_buyer_maker { -1.0 } else { 1.0 };
            signed_flow += sign * trade.quantity;
        }
        // Push the last interval
        let price_change = last_price - first_price;
        intervals.push((price_change, signed_flow));

        if intervals.len() < 2 {
            return 0.0;
        }

        let price_changes: Vec<f64> = intervals.iter().map(|x| x.0).collect();
        let flows: Vec<f64> = intervals.iter().map(|x| x.1).collect();

        Self::kyles_lambda(&price_changes, &flows)
    }

    // ─── Amihud Illiquidity ────────────────────────────────────────────────

    /// Compute the Amihud illiquidity ratio: mean(|return| / volume).
    ///
    /// Takes parallel slices of absolute returns and volumes.
    pub fn amihud_illiquidity(abs_returns: &[f64], volumes: &[f64], epsilon: f64) -> f64 {
        let n = abs_returns.len().min(volumes.len());
        if n == 0 {
            return 0.0;
        }

        let sum: f64 = (0..n)
            .map(|i| abs_returns[i] / (volumes[i] + epsilon))
            .sum();

        sum / n as f64
    }

    /// Compute Amihud illiquidity from a series of trades grouped by interval.
    pub fn amihud_from_trades(&self, trades: &[Trade], interval_ms: u64) -> f64 {
        if trades.len() < 2 || interval_ms == 0 {
            return 0.0;
        }

        let start = trades[0].timestamp;
        let mut intervals: Vec<(f64, f64)> = Vec::new(); // (abs_return, volume)
        let mut interval_start = start;
        let mut first_price = trades[0].price;
        let mut last_price = trades[0].price;
        let mut volume = 0.0;

        for trade in trades {
            if trade.timestamp >= interval_start + interval_ms {
                if first_price > 0.0 {
                    let ret = ((last_price / first_price) - 1.0).abs();
                    intervals.push((ret, volume));
                }
                interval_start += interval_ms;
                first_price = trade.price;
                volume = 0.0;
            }
            last_price = trade.price;
            volume += trade.quantity;
        }
        // Last interval
        if first_price > 0.0 {
            let ret = ((last_price / first_price) - 1.0).abs();
            intervals.push((ret, volume));
        }

        if intervals.is_empty() {
            return 0.0;
        }

        let abs_returns: Vec<f64> = intervals.iter().map(|x| x.0).collect();
        let volumes: Vec<f64> = intervals.iter().map(|x| x.1).collect();

        Self::amihud_illiquidity(&abs_returns, &volumes, self.epsilon)
    }

    // ─── VPIN ──────────────────────────────────────────────────────────────

    /// Compute VPIN (Volume-Synchronized Probability of Informed Trading).
    ///
    /// Groups trades into volume buckets and computes the average absolute
    /// imbalance across a rolling window of buckets.
    pub fn vpin(&self, trades: &[Trade]) -> f64 {
        if trades.is_empty() || self.vpin_bucket_volume <= 0.0 {
            return 0.0;
        }

        // Build volume buckets
        let mut buckets: Vec<(f64, f64)> = Vec::new(); // (buy_vol, sell_vol) per bucket
        let mut buy_vol = 0.0;
        let mut sell_vol = 0.0;
        let mut bucket_vol = 0.0;

        for trade in trades {
            let vol = trade.quantity;
            if trade.is_buyer_maker {
                sell_vol += vol;
            } else {
                buy_vol += vol;
            }
            bucket_vol += vol;

            if bucket_vol >= self.vpin_bucket_volume {
                buckets.push((buy_vol, sell_vol));
                buy_vol = 0.0;
                sell_vol = 0.0;
                bucket_vol = 0.0;
            }
        }

        // If we have leftover volume, include it as a partial bucket
        if bucket_vol > 0.0 {
            buckets.push((buy_vol, sell_vol));
        }

        if buckets.is_empty() {
            return 0.0;
        }

        // Use the last `vpin_window` buckets
        let start = if buckets.len() > self.vpin_window {
            buckets.len() - self.vpin_window
        } else {
            0
        };
        let window = &buckets[start..];
        let n = window.len() as f64;

        let sum_abs_imbalance: f64 = window
            .iter()
            .map(|(bv, sv)| (sv - bv).abs())
            .sum();

        sum_abs_imbalance / (n * self.vpin_bucket_volume)
    }

    // ─── All Features ──────────────────────────────────────────────────────

    /// Compute all microstructure features from an order book and recent trades.
    pub fn compute_all(&self, book: &OrderBook, trades: &[Trade], interval_ms: u64) -> MicrostructureFeatures {
        let midprice = Self::midprice(book);

        MicrostructureFeatures {
            absolute_spread: Self::absolute_spread(book),
            relative_spread: Self::relative_spread(book),
            effective_spread: Self::avg_effective_spread(trades, midprice),
            midprice,
            order_imbalance: self.order_imbalance(book),
            trade_imbalance: self.trade_imbalance(trades),
            kyles_lambda: Self::kyles_lambda_from_trades(trades, interval_ms),
            amihud_illiquidity: self.amihud_from_trades(trades, interval_ms),
            vpin: self.vpin(trades),
        }
    }
}

// ─── Bybit API Integration ────────────────────────────────────────────────────

/// Bybit API response structures.
pub mod bybit {
    use super::*;

    #[derive(Debug, Deserialize)]
    pub struct BybitResponse<T> {
        #[serde(rename = "retCode")]
        pub ret_code: i32,
        #[serde(rename = "retMsg")]
        pub ret_msg: String,
        pub result: T,
    }

    #[derive(Debug, Deserialize)]
    pub struct OrderBookResult {
        pub s: String,
        pub b: Vec<Vec<String>>, // bids: [price, qty]
        pub a: Vec<Vec<String>>, // asks: [price, qty]
        pub ts: u64,
    }

    #[derive(Debug, Deserialize)]
    pub struct TradeResult {
        pub list: Vec<TradeEntry>,
    }

    #[derive(Debug, Deserialize)]
    pub struct TradeEntry {
        #[serde(rename = "execId")]
        pub exec_id: String,
        pub symbol: String,
        pub price: String,
        pub size: String,
        pub side: String,
        pub time: String,
    }

    /// Fetch the order book for a given symbol from Bybit.
    pub fn fetch_orderbook(symbol: &str, depth: u32) -> Result<OrderBook> {
        let url = format!(
            "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
            symbol, depth
        );
        let resp: BybitResponse<OrderBookResult> =
            reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let bids = resp
            .result
            .b
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks = resp
            .result
            .a
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            bids,
            asks,
            timestamp: resp.result.ts,
        })
    }

    /// Fetch recent trades for a given symbol from Bybit.
    pub fn fetch_recent_trades(symbol: &str, limit: u32) -> Result<Vec<Trade>> {
        let url = format!(
            "https://api.bybit.com/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            symbol, limit
        );
        let resp: BybitResponse<TradeResult> =
            reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let trades = resp
            .result
            .list
            .iter()
            .map(|entry| Trade {
                price: entry.price.parse().unwrap_or(0.0),
                quantity: entry.size.parse().unwrap_or(0.0),
                is_buyer_maker: entry.side == "Sell",
                timestamp: entry.time.parse().unwrap_or(0),
            })
            .collect();

        Ok(trades)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_orderbook() -> OrderBook {
        OrderBook {
            bids: vec![
                OrderBookLevel { price: 100.0, quantity: 10.0 },
                OrderBookLevel { price: 99.0, quantity: 20.0 },
                OrderBookLevel { price: 98.0, quantity: 30.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, quantity: 5.0 },
                OrderBookLevel { price: 102.0, quantity: 15.0 },
                OrderBookLevel { price: 103.0, quantity: 25.0 },
            ],
            timestamp: 1000,
        }
    }

    fn sample_trades() -> Vec<Trade> {
        vec![
            Trade { price: 100.5, quantity: 1.0, is_buyer_maker: false, timestamp: 1000 },
            Trade { price: 100.6, quantity: 2.0, is_buyer_maker: false, timestamp: 1100 },
            Trade { price: 100.4, quantity: 1.5, is_buyer_maker: true, timestamp: 1200 },
            Trade { price: 100.3, quantity: 3.0, is_buyer_maker: true, timestamp: 1300 },
            Trade { price: 100.7, quantity: 0.5, is_buyer_maker: false, timestamp: 1400 },
            Trade { price: 100.8, quantity: 1.0, is_buyer_maker: false, timestamp: 2100 },
            Trade { price: 100.2, quantity: 2.0, is_buyer_maker: true, timestamp: 2200 },
            Trade { price: 100.9, quantity: 1.5, is_buyer_maker: false, timestamp: 2300 },
        ]
    }

    #[test]
    fn test_spread_calculations() {
        let book = sample_orderbook();
        let abs_spread = MicrostructureCalculator::absolute_spread(&book);
        assert!((abs_spread - 1.0).abs() < 1e-10, "Absolute spread should be 1.0");

        let rel_spread = MicrostructureCalculator::relative_spread(&book);
        let expected_rel = 1.0 / 100.5;
        assert!(
            (rel_spread - expected_rel).abs() < 1e-10,
            "Relative spread should be {}, got {}",
            expected_rel,
            rel_spread
        );
    }

    #[test]
    fn test_effective_spread() {
        let eff = MicrostructureCalculator::effective_spread(100.8, 100.5);
        assert!((eff - 0.6).abs() < 1e-10, "Effective spread should be 0.6, got {}", eff);
    }

    #[test]
    fn test_order_imbalance() {
        let book = sample_orderbook();
        let calc = MicrostructureCalculator::new().with_depth_levels(3);
        let oi = calc.order_imbalance(&book);

        // bid_vol = 10 + 20 + 30 = 60, ask_vol = 5 + 15 + 25 = 45
        // OI = (60 - 45) / (60 + 45) = 15 / 105
        let expected = 15.0 / 105.0;
        assert!(
            (oi - expected).abs() < 1e-10,
            "Order imbalance should be {}, got {}",
            expected,
            oi
        );
    }

    #[test]
    fn test_trade_imbalance() {
        let trades = sample_trades();
        let calc = MicrostructureCalculator::new();
        let ti = calc.trade_imbalance(&trades);

        // buy: 1.0 + 2.0 + 0.5 + 1.0 + 1.5 = 6.0
        // sell: 1.5 + 3.0 + 2.0 = 6.5
        // TI = (6.0 - 6.5) / (6.0 + 6.5) = -0.5 / 12.5 = -0.04
        let expected = -0.5 / 12.5;
        assert!(
            (ti - expected).abs() < 1e-10,
            "Trade imbalance should be {}, got {}",
            expected,
            ti
        );
    }

    #[test]
    fn test_kyles_lambda() {
        // Simple test: price changes = 2 * order_flow + noise
        let order_flows = vec![1.0, -2.0, 3.0, -1.0, 2.0, -3.0, 1.5, -0.5];
        let price_changes: Vec<f64> = order_flows.iter().map(|x| 2.0 * x).collect();

        let lambda = MicrostructureCalculator::kyles_lambda(&price_changes, &order_flows);
        assert!(
            (lambda - 2.0).abs() < 1e-10,
            "Kyle's lambda should be 2.0, got {}",
            lambda
        );
    }

    #[test]
    fn test_amihud_illiquidity() {
        let abs_returns = vec![0.01, 0.02, 0.005, 0.015];
        let volumes = vec![100.0, 200.0, 50.0, 150.0];

        let illiq = MicrostructureCalculator::amihud_illiquidity(&abs_returns, &volumes, 1e-10);
        let expected = (0.01 / 100.0 + 0.02 / 200.0 + 0.005 / 50.0 + 0.015 / 150.0) / 4.0;
        assert!(
            (illiq - expected).abs() < 1e-12,
            "Amihud should be {}, got {}",
            expected,
            illiq
        );
    }

    #[test]
    fn test_vpin() {
        // Create trades that fill exactly 5 buckets of size 2.0
        let trades = vec![
            Trade { price: 100.0, quantity: 1.0, is_buyer_maker: false, timestamp: 1 },
            Trade { price: 100.0, quantity: 1.0, is_buyer_maker: false, timestamp: 2 },
            // bucket 1: buy=2.0, sell=0.0, |imbalance|=2.0
            Trade { price: 100.0, quantity: 1.0, is_buyer_maker: true, timestamp: 3 },
            Trade { price: 100.0, quantity: 1.0, is_buyer_maker: true, timestamp: 4 },
            // bucket 2: buy=0.0, sell=2.0, |imbalance|=2.0
            Trade { price: 100.0, quantity: 1.0, is_buyer_maker: false, timestamp: 5 },
            Trade { price: 100.0, quantity: 1.0, is_buyer_maker: true, timestamp: 6 },
            // bucket 3: buy=1.0, sell=1.0, |imbalance|=0.0
        ];

        let calc = MicrostructureCalculator::new().with_vpin_params(2.0, 50);
        let vpin = calc.vpin(&trades);

        // 3 buckets: imbalances = |0-2|=2.0, |2-0|=2.0, |0|=0.0
        // VPIN = (2 + 2 + 0) / (3 * 2) = 4/6 = 0.6667
        let expected = 4.0 / 6.0;
        assert!(
            (vpin - expected).abs() < 1e-10,
            "VPIN should be {}, got {}",
            expected,
            vpin
        );
    }

    #[test]
    fn test_empty_inputs() {
        let empty_book = OrderBook {
            bids: vec![],
            asks: vec![],
            timestamp: 0,
        };
        let calc = MicrostructureCalculator::new();

        assert_eq!(MicrostructureCalculator::absolute_spread(&empty_book), 0.0);
        assert_eq!(MicrostructureCalculator::relative_spread(&empty_book), 0.0);
        assert_eq!(MicrostructureCalculator::midprice(&empty_book), 0.0);
        assert_eq!(calc.order_imbalance(&empty_book), 0.0);
        assert_eq!(calc.trade_imbalance(&[]), 0.0);
        assert_eq!(MicrostructureCalculator::kyles_lambda(&[], &[]), 0.0);
        assert_eq!(MicrostructureCalculator::amihud_illiquidity(&[], &[], 1e-10), 0.0);
        assert_eq!(calc.vpin(&[]), 0.0);
    }
}
