//! Data module for Bybit API and market data processing
//!
//! This module provides functionality to fetch and process
//! cryptocurrency market data from Bybit exchange.

mod bybit;
mod events;
mod features;

pub use bybit::{BybitClient, BybitConfig, BybitError};
pub use events::{Event, EventType, EventStream};
pub use features::{FeatureEngine, MarketFeatures};

use serde::{Deserialize, Serialize};
use ndarray::Array1;

/// Event features for TGN processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFeatures {
    /// Price at event time
    pub price: f64,
    /// Volume/size of the event
    pub volume: f64,
    /// Buy (1) or Sell (-1) side
    pub side: f64,
    /// Price change percentage
    pub price_change: f64,
    /// Volume relative to average
    pub volume_ratio: f64,
    /// Order book imbalance
    pub imbalance: f64,
    /// Spread at event time
    pub spread: f64,
    /// Volatility estimate
    pub volatility: f64,
}

impl Default for EventFeatures {
    fn default() -> Self {
        Self {
            price: 0.0,
            volume: 0.0,
            side: 0.0,
            price_change: 0.0,
            volume_ratio: 1.0,
            imbalance: 0.0,
            spread: 0.0,
            volatility: 0.0,
        }
    }
}

impl EventFeatures {
    /// Convert to vector for model input
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.price.ln().max(-10.0).min(20.0),  // Log-normalized price
            self.volume.ln().max(-10.0).min(20.0), // Log-normalized volume
            self.side,
            self.price_change,
            self.volume_ratio.ln(),
            self.imbalance,
            self.spread.ln().max(-20.0).min(0.0),
            self.volatility,
        ])
    }

    /// Create from trade data
    pub fn from_trade(price: f64, size: f64, is_buy: bool) -> Self {
        Self {
            price,
            volume: size,
            side: if is_buy { 1.0 } else { -1.0 },
            ..Default::default()
        }
    }
}

/// OHLCV Kline/Candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Open timestamp (milliseconds)
    pub timestamp: u64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Calculate return from open to close
    pub fn return_oc(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate range as percentage of close
    pub fn range_pct(&self) -> f64 {
        if self.close > 0.0 {
            self.range() / self.close
        } else {
            0.0
        }
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate typical price
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Real-time ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// Price change 24h percentage
    pub price_change_24h: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl Ticker {
    /// Calculate bid-ask spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Calculate spread as percentage
    pub fn spread_pct(&self) -> f64 {
        if self.bid_price > 0.0 {
            self.spread() / self.bid_price
        } else {
            0.0
        }
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Size/quantity at this level
    pub size: f64,
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: u64,
}

impl OrderBook {
    /// Create new order book
    pub fn new(symbol: &str, bids: Vec<OrderBookLevel>, asks: Vec<OrderBookLevel>) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: 0,
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

    /// Calculate spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
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
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(levels).map(|l| l.size).sum();
        let ask_volume: f64 = self.asks.iter().take(levels).map(|l| l.size).sum();

        let total = bid_volume + ask_volume;
        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }

    /// Calculate VWAP for bids
    pub fn bid_vwap(&self, levels: usize) -> Option<f64> {
        let levels: Vec<_> = self.bids.iter().take(levels).collect();
        if levels.is_empty() {
            return None;
        }

        let total_value: f64 = levels.iter().map(|l| l.price * l.size).sum();
        let total_volume: f64 = levels.iter().map(|l| l.size).sum();

        if total_volume > 0.0 {
            Some(total_value / total_volume)
        } else {
            None
        }
    }

    /// Calculate VWAP for asks
    pub fn ask_vwap(&self, levels: usize) -> Option<f64> {
        let levels: Vec<_> = self.asks.iter().take(levels).collect();
        if levels.is_empty() {
            return None;
        }

        let total_value: f64 = levels.iter().map(|l| l.price * l.size).sum();
        let total_volume: f64 = levels.iter().map(|l| l.size).sum();

        if total_volume > 0.0 {
            Some(total_value / total_volume)
        } else {
            None
        }
    }
}

/// Funding rate data for perpetual contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    /// Symbol
    pub symbol: String,
    /// Funding rate
    pub funding_rate: f64,
    /// Funding rate timestamp
    pub funding_rate_timestamp: u64,
    /// Next funding time
    pub next_funding_time: u64,
}

/// Open interest data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenInterest {
    /// Symbol
    pub symbol: String,
    /// Open interest value
    pub open_interest: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Price
    pub price: f64,
    /// Size/Quantity
    pub size: f64,
    /// Side: "Buy" or "Sell"
    pub side: String,
    /// Timestamp
    pub timestamp: u64,
}

impl Trade {
    /// Check if buy trade
    pub fn is_buy(&self) -> bool {
        self.side == "Buy"
    }

    /// Get signed size (positive for buy, negative for sell)
    pub fn signed_size(&self) -> f64 {
        if self.is_buy() {
            self.size
        } else {
            -self.size
        }
    }

    /// Convert to event features
    pub fn to_features(&self) -> EventFeatures {
        EventFeatures::from_trade(self.price, self.size, self.is_buy())
    }
}

/// Market data snapshot for a symbol
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Symbol
    pub symbol: String,
    /// Current ticker
    pub ticker: Option<Ticker>,
    /// Current order book
    pub orderbook: Option<OrderBook>,
    /// Recent klines
    pub klines: Vec<Kline>,
    /// Recent trades
    pub trades: Vec<Trade>,
    /// Funding rate (for perps)
    pub funding: Option<FundingRate>,
    /// Open interest (for perps)
    pub open_interest: Option<OpenInterest>,
    /// Timestamp
    pub timestamp: u64,
}

impl MarketSnapshot {
    /// Create new empty snapshot
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            ticker: None,
            orderbook: None,
            klines: Vec::new(),
            trades: Vec::new(),
            funding: None,
            open_interest: None,
            timestamp: 0,
        }
    }

    /// Get current price
    pub fn price(&self) -> Option<f64> {
        self.ticker.as_ref().map(|t| t.last_price)
    }

    /// Calculate VWAP from recent trades
    pub fn vwap(&self) -> Option<f64> {
        if self.trades.is_empty() {
            return None;
        }

        let mut total_value = 0.0;
        let mut total_volume = 0.0;

        for trade in &self.trades {
            total_value += trade.price * trade.size;
            total_volume += trade.size;
        }

        if total_volume > 0.0 {
            Some(total_value / total_volume)
        } else {
            None
        }
    }

    /// Calculate buy/sell imbalance from recent trades
    pub fn trade_imbalance(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }

        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        for trade in &self.trades {
            if trade.is_buy() {
                buy_volume += trade.size;
            } else {
                sell_volume += trade.size;
            }
        }

        let total = buy_volume + sell_volume;
        if total > 0.0 {
            (buy_volume - sell_volume) / total
        } else {
            0.0
        }
    }

    /// Extract event features from snapshot
    pub fn to_features(&self) -> EventFeatures {
        let price = self.price().unwrap_or(0.0);
        let volume = self.trades.iter().map(|t| t.size).sum();
        let imbalance = self.trade_imbalance();
        let spread = self.orderbook.as_ref().and_then(|ob| ob.spread()).unwrap_or(0.0);

        // Calculate volatility from recent klines
        let volatility = if self.klines.len() >= 2 {
            let returns: Vec<f64> = self.klines.windows(2)
                .map(|w| w[1].return_oc())
                .collect();
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        EventFeatures {
            price,
            volume,
            side: if imbalance > 0.0 { 1.0 } else { -1.0 },
            price_change: 0.0,
            volume_ratio: 1.0,
            imbalance,
            spread,
            volatility,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline() {
        let kline = Kline {
            timestamp: 1000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert_eq!(kline.return_oc(), 0.05);
        assert_eq!(kline.range(), 15.0);
        assert!(kline.is_bullish());
    }

    #[test]
    fn test_ticker() {
        let ticker = Ticker {
            symbol: "BTCUSDT".to_string(),
            last_price: 50000.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            volume_24h: 10000.0,
            turnover_24h: 500000000.0,
            price_change_24h: 2.0,
            bid_price: 49995.0,
            ask_price: 50005.0,
            timestamp: 1000,
        };

        assert_eq!(ticker.spread(), 10.0);
        assert_eq!(ticker.mid_price(), 50000.0);
    }

    #[test]
    fn test_orderbook_imbalance() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel { price: 49999.0, size: 10.0 },
                OrderBookLevel { price: 49998.0, size: 5.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 50001.0, size: 8.0 },
                OrderBookLevel { price: 50002.0, size: 2.0 },
            ],
            timestamp: 1000,
        };

        let imbalance = ob.imbalance(2);
        assert!(imbalance > 0.0); // More bid volume
    }

    #[test]
    fn test_event_features() {
        let features = EventFeatures::from_trade(50000.0, 1.5, true);
        let vector = features.to_vector();
        assert_eq!(vector.len(), 8);
    }
}
