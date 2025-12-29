//! # Trade Data Structures
//!
//! Structures for representing trade (time & sales) data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single trade execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trading symbol
    pub symbol: String,
    /// Timestamp of the trade
    pub timestamp: DateTime<Utc>,
    /// Execution price
    pub price: f64,
    /// Trade size (quantity)
    pub size: f64,
    /// True if buyer was the maker (seller was aggressor)
    pub is_buyer_maker: bool,
    /// Trade ID
    pub trade_id: String,
}

impl Trade {
    /// Create a new trade
    pub fn new(
        symbol: String,
        timestamp: DateTime<Utc>,
        price: f64,
        size: f64,
        is_buyer_maker: bool,
        trade_id: String,
    ) -> Self {
        Self {
            symbol,
            timestamp,
            price,
            size,
            is_buyer_maker,
            trade_id,
        }
    }

    /// Get the trade side (buy or sell from taker's perspective)
    pub fn side(&self) -> TradeSide {
        if self.is_buyer_maker {
            TradeSide::Sell // Seller was aggressor
        } else {
            TradeSide::Buy // Buyer was aggressor
        }
    }

    /// Get the notional value of the trade
    pub fn notional(&self) -> f64 {
        self.price * self.size
    }

    /// Check if this is a buy trade (from taker perspective)
    pub fn is_buy(&self) -> bool {
        !self.is_buyer_maker
    }

    /// Check if this is a sell trade (from taker perspective)
    pub fn is_sell(&self) -> bool {
        self.is_buyer_maker
    }
}

/// Trade side from taker's perspective
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl TradeSide {
    /// Get the sign for calculations (+1 for buy, -1 for sell)
    pub fn sign(&self) -> f64 {
        match self {
            TradeSide::Buy => 1.0,
            TradeSide::Sell => -1.0,
        }
    }
}

/// Aggregated trade statistics over a time window
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradeStats {
    /// Number of trades
    pub count: usize,
    /// Total volume
    pub volume: f64,
    /// Total notional value
    pub notional: f64,
    /// Buy volume
    pub buy_volume: f64,
    /// Sell volume
    pub sell_volume: f64,
    /// Number of buy trades
    pub buy_count: usize,
    /// Number of sell trades
    pub sell_count: usize,
    /// Volume-weighted average price
    pub vwap: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// First price
    pub open: f64,
    /// Last price
    pub close: f64,
    /// Average trade size
    pub avg_size: f64,
    /// Largest trade size
    pub max_size: f64,
}

impl TradeStats {
    /// Create trade statistics from a slice of trades
    pub fn from_trades(trades: &[Trade]) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        let mut stats = Self {
            count: trades.len(),
            high: f64::MIN,
            low: f64::MAX,
            open: trades.first().map(|t| t.price).unwrap_or(0.0),
            close: trades.last().map(|t| t.price).unwrap_or(0.0),
            ..Default::default()
        };

        let mut total_notional = 0.0;

        for trade in trades {
            stats.volume += trade.size;
            total_notional += trade.notional();

            if trade.is_buy() {
                stats.buy_volume += trade.size;
                stats.buy_count += 1;
            } else {
                stats.sell_volume += trade.size;
                stats.sell_count += 1;
            }

            if trade.price > stats.high {
                stats.high = trade.price;
            }
            if trade.price < stats.low {
                stats.low = trade.price;
            }
            if trade.size > stats.max_size {
                stats.max_size = trade.size;
            }
        }

        stats.notional = total_notional;
        stats.vwap = if stats.volume > 0.0 {
            total_notional / stats.volume
        } else {
            0.0
        };
        stats.avg_size = if stats.count > 0 {
            stats.volume / stats.count as f64
        } else {
            0.0
        };

        stats
    }

    /// Calculate trade imbalance
    ///
    /// Returns (buy_volume - sell_volume) / total_volume
    /// Range: [-1, 1], positive = more buying
    pub fn trade_imbalance(&self) -> f64 {
        if self.volume > 0.0 {
            (self.buy_volume - self.sell_volume) / self.volume
        } else {
            0.0
        }
    }

    /// Calculate trade count imbalance
    pub fn count_imbalance(&self) -> f64 {
        let total = self.buy_count + self.sell_count;
        if total > 0 {
            (self.buy_count as f64 - self.sell_count as f64) / total as f64
        } else {
            0.0
        }
    }

    /// Calculate price range as percentage
    pub fn range_pct(&self) -> f64 {
        if self.low > 0.0 {
            (self.high - self.low) / self.low * 100.0
        } else {
            0.0
        }
    }

    /// Calculate return (close - open) / open
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open * 100.0
        } else {
            0.0
        }
    }
}

/// Trade classifier using Lee-Ready algorithm
pub struct TradeClassifier {
    /// Previous trade price for tick test
    prev_price: Option<f64>,
    /// Previous tick direction
    prev_tick: f64,
}

impl TradeClassifier {
    pub fn new() -> Self {
        Self {
            prev_price: None,
            prev_tick: 1.0,
        }
    }

    /// Classify a trade as buy or sell using quote and tick test
    ///
    /// Lee-Ready algorithm:
    /// 1. If price > midquote -> buy
    /// 2. If price < midquote -> sell
    /// 3. If price = midquote -> use tick test
    pub fn classify(&mut self, price: f64, mid_price: f64) -> TradeSide {
        let side = if price > mid_price {
            TradeSide::Buy
        } else if price < mid_price {
            TradeSide::Sell
        } else {
            // Tick test
            if let Some(prev) = self.prev_price {
                if price > prev {
                    self.prev_tick = 1.0;
                } else if price < prev {
                    self.prev_tick = -1.0;
                }
                // If equal, use previous tick
            }

            if self.prev_tick > 0.0 {
                TradeSide::Buy
            } else {
                TradeSide::Sell
            }
        };

        self.prev_price = Some(price);
        side
    }
}

impl Default for TradeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Volume bucket for VPIN calculation
#[derive(Debug, Clone, Default)]
pub struct VolumeBucket {
    /// Target bucket volume
    pub target_volume: f64,
    /// Current filled volume
    pub filled_volume: f64,
    /// Buy volume in bucket
    pub buy_volume: f64,
    /// Sell volume in bucket
    pub sell_volume: f64,
    /// Start timestamp
    pub start_time: Option<DateTime<Utc>>,
    /// End timestamp
    pub end_time: Option<DateTime<Utc>>,
}

impl VolumeBucket {
    /// Create a new volume bucket
    pub fn new(target_volume: f64) -> Self {
        Self {
            target_volume,
            ..Default::default()
        }
    }

    /// Check if bucket is complete
    pub fn is_complete(&self) -> bool {
        self.filled_volume >= self.target_volume
    }

    /// Get remaining volume to fill
    pub fn remaining(&self) -> f64 {
        (self.target_volume - self.filled_volume).max(0.0)
    }

    /// Add a trade to the bucket
    ///
    /// Returns the volume that spilled over (if any)
    pub fn add_trade(&mut self, trade: &Trade) -> f64 {
        if self.start_time.is_none() {
            self.start_time = Some(trade.timestamp);
        }
        self.end_time = Some(trade.timestamp);

        let remaining = self.remaining();
        let fill = trade.size.min(remaining);
        let spillover = trade.size - fill;

        self.filled_volume += fill;

        // Pro-rate the buy/sell based on fill amount
        let ratio = fill / trade.size;
        if trade.is_buy() {
            self.buy_volume += trade.size * ratio;
        } else {
            self.sell_volume += trade.size * ratio;
        }

        spillover
    }

    /// Calculate imbalance for this bucket
    pub fn imbalance(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total > 0.0 {
            (self.buy_volume - self.sell_volume).abs() / total
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_side() {
        let buy_trade = Trade::new(
            "BTCUSDT".to_string(),
            Utc::now(),
            100.0,
            1.0,
            false, // buyer was taker
            "1".to_string(),
        );
        assert_eq!(buy_trade.side(), TradeSide::Buy);
        assert!(buy_trade.is_buy());

        let sell_trade = Trade::new(
            "BTCUSDT".to_string(),
            Utc::now(),
            100.0,
            1.0,
            true, // buyer was maker
            "2".to_string(),
        );
        assert_eq!(sell_trade.side(), TradeSide::Sell);
        assert!(sell_trade.is_sell());
    }

    #[test]
    fn test_trade_stats() {
        let trades = vec![
            Trade::new("BTC".to_string(), Utc::now(), 100.0, 1.0, false, "1".to_string()),
            Trade::new("BTC".to_string(), Utc::now(), 101.0, 2.0, true, "2".to_string()),
            Trade::new("BTC".to_string(), Utc::now(), 102.0, 1.5, false, "3".to_string()),
        ];

        let stats = TradeStats::from_trades(&trades);

        assert_eq!(stats.count, 3);
        assert!((stats.volume - 4.5).abs() < 0.001);
        assert!((stats.buy_volume - 2.5).abs() < 0.001);
        assert!((stats.sell_volume - 2.0).abs() < 0.001);
        assert!((stats.open - 100.0).abs() < 0.001);
        assert!((stats.close - 102.0).abs() < 0.001);
        assert!((stats.high - 102.0).abs() < 0.001);
        assert!((stats.low - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_volume_bucket() {
        let mut bucket = VolumeBucket::new(10.0);

        let trade1 = Trade::new("BTC".to_string(), Utc::now(), 100.0, 4.0, false, "1".to_string());
        let spillover = bucket.add_trade(&trade1);
        assert!((spillover - 0.0).abs() < 0.001);
        assert!(!bucket.is_complete());

        let trade2 = Trade::new("BTC".to_string(), Utc::now(), 100.0, 8.0, true, "2".to_string());
        let spillover = bucket.add_trade(&trade2);
        assert!((spillover - 2.0).abs() < 0.001);
        assert!(bucket.is_complete());
    }

    #[test]
    fn test_trade_classifier() {
        let mut classifier = TradeClassifier::new();

        // Above mid -> buy
        assert_eq!(classifier.classify(100.5, 100.0), TradeSide::Buy);

        // Below mid -> sell
        assert_eq!(classifier.classify(99.5, 100.0), TradeSide::Sell);

        // At mid with uptick -> buy
        classifier.classify(101.0, 100.0); // Set uptick
        assert_eq!(classifier.classify(100.0, 100.0), TradeSide::Buy);
    }
}
