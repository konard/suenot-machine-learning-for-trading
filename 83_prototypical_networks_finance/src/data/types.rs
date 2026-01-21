//! Data types for market data structures
//!
//! This module defines the core data structures used to represent
//! market data from various sources like Bybit.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Candlestick/Kline data representing OHLCV for a time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp of the kline
    pub timestamp: DateTime<Utc>,
    /// Opening price
    pub open: f64,
    /// Highest price during the period
    pub high: f64,
    /// Lowest price during the period
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Quote volume (volume * price)
    pub quote_volume: f64,
    /// Number of trades
    pub trade_count: Option<u64>,
}

impl Kline {
    /// Create a new Kline instance
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            quote_volume: volume * close,
            trade_count: None,
        }
    }

    /// Calculate the return (close/open - 1)
    pub fn returns(&self) -> f64 {
        if self.open > 0.0 {
            self.close / self.open - 1.0
        } else {
            0.0
        }
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body (close - open)
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate upper wick
    pub fn upper_wick(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower wick
    pub fn lower_wick(&self) -> f64 {
        self.close.min(self.open) - self.low
    }
}

/// Single level in an order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price at this level
    pub price: f64,
    /// Quantity available at this price
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
    /// Trading symbol
    pub symbol: String,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Calculate bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate spread as percentage of mid price
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                let mid = (bid + ask) / 2.0;
                if mid > 0.0 {
                    Some((ask - bid) / mid)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Calculate order book imbalance
    /// Positive values indicate more bid volume (bullish)
    /// Negative values indicate more ask volume (bearish)
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();
        let total = bid_volume + ask_volume;

        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }
}

/// Ticker data representing current market state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Timestamp of the ticker
    pub timestamp: DateTime<Utc>,
    /// Trading symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h high price
    pub high_24h: f64,
    /// 24h low price
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h quote volume
    pub quote_volume_24h: f64,
    /// 24h price change percentage
    pub price_change_pct_24h: f64,
    /// Best bid price
    pub bid_price: Option<f64>,
    /// Best ask price
    pub ask_price: Option<f64>,
}

/// Individual trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Timestamp of the trade
    pub timestamp: DateTime<Utc>,
    /// Trading symbol
    pub symbol: String,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub quantity: f64,
    /// Whether the buyer was the maker
    pub is_buyer_maker: bool,
}

impl Trade {
    /// Get the side of the trade from taker's perspective
    pub fn taker_side(&self) -> TradeSide {
        if self.is_buyer_maker {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        }
    }
}

/// Trade side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Funding rate for perpetual futures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    /// Timestamp when funding rate was recorded
    pub timestamp: DateTime<Utc>,
    /// Trading symbol
    pub symbol: String,
    /// Funding rate (positive means longs pay shorts)
    pub funding_rate: f64,
    /// Next funding time
    pub funding_time: DateTime<Utc>,
}

impl FundingRate {
    /// Check if funding rate indicates bullish sentiment
    /// High positive funding = overleveraged longs = potentially bearish
    /// High negative funding = overleveraged shorts = potentially bullish
    pub fn sentiment_signal(&self) -> f64 {
        // Inverse relationship: high funding = bearish, low funding = bullish
        -self.funding_rate * 100.0
    }
}

/// Open interest data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenInterest {
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
    /// Trading symbol
    pub symbol: String,
    /// Total open interest in contracts
    pub open_interest: f64,
    /// Open interest value in quote currency
    pub open_interest_value: f64,
}

impl OpenInterest {
    /// Calculate 24h change if previous value is provided
    pub fn change_pct(&self, previous: &OpenInterest) -> f64 {
        if previous.open_interest > 0.0 {
            (self.open_interest / previous.open_interest - 1.0) * 100.0
        } else {
            0.0
        }
    }
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong bullish trend
    StrongUptrend,
    /// Weak bullish trend
    WeakUptrend,
    /// Sideways/consolidation
    Sideways,
    /// Weak bearish trend
    WeakDowntrend,
    /// Strong bearish trend (crash)
    StrongDowntrend,
}

impl MarketRegime {
    /// Get all possible regimes
    pub fn all() -> Vec<MarketRegime> {
        vec![
            MarketRegime::StrongUptrend,
            MarketRegime::WeakUptrend,
            MarketRegime::Sideways,
            MarketRegime::WeakDowntrend,
            MarketRegime::StrongDowntrend,
        ]
    }

    /// Get the number of regimes
    pub fn count() -> usize {
        5
    }

    /// Convert to class index
    pub fn to_index(&self) -> usize {
        match self {
            MarketRegime::StrongUptrend => 0,
            MarketRegime::WeakUptrend => 1,
            MarketRegime::Sideways => 2,
            MarketRegime::WeakDowntrend => 3,
            MarketRegime::StrongDowntrend => 4,
        }
    }

    /// Convert from class index
    pub fn from_index(index: usize) -> Option<MarketRegime> {
        match index {
            0 => Some(MarketRegime::StrongUptrend),
            1 => Some(MarketRegime::WeakUptrend),
            2 => Some(MarketRegime::Sideways),
            3 => Some(MarketRegime::WeakDowntrend),
            4 => Some(MarketRegime::StrongDowntrend),
            _ => None,
        }
    }

    /// Get trading bias for this regime
    pub fn trading_bias(&self) -> TradingBias {
        match self {
            MarketRegime::StrongUptrend => TradingBias::StrongLong,
            MarketRegime::WeakUptrend => TradingBias::WeakLong,
            MarketRegime::Sideways => TradingBias::Neutral,
            MarketRegime::WeakDowntrend => TradingBias::WeakShort,
            MarketRegime::StrongDowntrend => TradingBias::StrongShort,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            MarketRegime::StrongUptrend => "Strong Uptrend",
            MarketRegime::WeakUptrend => "Weak Uptrend",
            MarketRegime::Sideways => "Sideways",
            MarketRegime::WeakDowntrend => "Weak Downtrend",
            MarketRegime::StrongDowntrend => "Strong Downtrend",
        }
    }
}

/// Trading bias based on market regime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingBias {
    StrongLong,
    WeakLong,
    Neutral,
    WeakShort,
    StrongShort,
}

impl TradingBias {
    /// Get position sizing multiplier (0.0 to 1.0)
    pub fn position_multiplier(&self) -> f64 {
        match self {
            TradingBias::StrongLong => 1.0,
            TradingBias::WeakLong => 0.5,
            TradingBias::Neutral => 0.0,
            TradingBias::WeakShort => -0.5,
            TradingBias::StrongShort => -1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_kline_calculations() {
        let kline = Kline::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
        );

        assert!((kline.returns() - 0.05).abs() < 1e-10);
        assert!((kline.range() - 15.0).abs() < 1e-10);
        assert!((kline.body() - 5.0).abs() < 1e-10);
        assert!(kline.is_bullish());
        assert!((kline.upper_wick() - 5.0).abs() < 1e-10);
        assert!((kline.lower_wick() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_order_book() {
        let order_book = OrderBook {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel { price: 99.0, quantity: 10.0 },
                OrderBookLevel { price: 98.0, quantity: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, quantity: 15.0 },
                OrderBookLevel { price: 102.0, quantity: 25.0 },
            ],
        };

        assert_eq!(order_book.best_bid(), Some(99.0));
        assert_eq!(order_book.best_ask(), Some(101.0));
        assert_eq!(order_book.spread(), Some(2.0));
        assert_eq!(order_book.mid_price(), Some(100.0));

        // Imbalance: (10 - 15) / (10 + 15) = -5/25 = -0.2
        let imbalance = order_book.imbalance(1);
        assert!((imbalance - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_market_regime() {
        assert_eq!(MarketRegime::count(), 5);
        assert_eq!(MarketRegime::StrongUptrend.to_index(), 0);
        assert_eq!(MarketRegime::from_index(0), Some(MarketRegime::StrongUptrend));
        assert_eq!(MarketRegime::from_index(10), None);
    }
}
