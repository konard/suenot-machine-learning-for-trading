//! Bybit API response structures.

use serde::{Deserialize, Serialize};

/// Generic Bybit API response wrapper.
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
    pub time: u64,
}

/// Kline (candlestick) response result.
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<KlineData>,
}

/// Individual kline data from API.
/// Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
#[derive(Debug, Deserialize)]
pub struct KlineData(
    pub String, // startTime
    pub String, // open
    pub String, // high
    pub String, // low
    pub String, // close
    pub String, // volume
    pub String, // turnover
);

/// Ticker information.
#[derive(Debug, Deserialize, Serialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

/// Tickers result.
#[derive(Debug, Deserialize)]
pub struct TickersResult {
    pub category: String,
    pub list: Vec<TickerInfo>,
}

/// Instrument info (trading pair details).
#[derive(Debug, Deserialize, Serialize)]
pub struct InstrumentInfo {
    pub symbol: String,
    #[serde(rename = "baseCoin")]
    pub base_coin: String,
    #[serde(rename = "quoteCoin")]
    pub quote_coin: String,
    pub status: String,
    #[serde(rename = "lotSizeFilter")]
    pub lot_size_filter: LotSizeFilter,
    #[serde(rename = "priceFilter")]
    pub price_filter: PriceFilter,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LotSizeFilter {
    #[serde(rename = "basePrecision")]
    pub base_precision: String,
    #[serde(rename = "quotePrecision")]
    pub quote_precision: String,
    #[serde(rename = "minOrderQty")]
    pub min_order_qty: String,
    #[serde(rename = "maxOrderQty")]
    pub max_order_qty: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PriceFilter {
    #[serde(rename = "tickSize")]
    pub tick_size: String,
}

/// Instruments result.
#[derive(Debug, Deserialize)]
pub struct InstrumentsResult {
    pub category: String,
    pub list: Vec<InstrumentInfo>,
}
