//! Data loading from Yahoo Finance and Bybit.

use crate::backtest::PriceBar;
use crate::error::{RagError, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Market data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamps: Vec<DateTime<Utc>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl MarketData {
    /// Create new market data.
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            timestamps: Vec::new(),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Vec::new(),
        }
    }

    /// Add a data point.
    pub fn add_bar(
        &mut self,
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) {
        self.timestamps.push(timestamp);
        self.open.push(open);
        self.high.push(high);
        self.low.push(low);
        self.close.push(close);
        self.volume.push(volume);
    }

    /// Get number of bars.
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Convert to price bars.
    pub fn to_price_bars(&self) -> Vec<PriceBar> {
        (0..self.len())
            .map(|i| {
                PriceBar::new(
                    self.timestamps[i],
                    self.open[i],
                    self.high[i],
                    self.low[i],
                    self.close[i],
                    self.volume[i],
                )
            })
            .collect()
    }

    /// Get returns.
    pub fn returns(&self) -> Vec<f64> {
        if self.close.len() < 2 {
            return Vec::new();
        }

        self.close
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
}

/// Yahoo Finance data loader.
pub struct YahooFinanceLoader {
    base_url: String,
}

impl YahooFinanceLoader {
    /// Create a new Yahoo Finance loader.
    pub fn new() -> Self {
        Self {
            base_url: "https://query1.finance.yahoo.com/v8/finance/chart".to_string(),
        }
    }

    /// Fetch historical data for a symbol.
    /// Note: This is a simplified implementation. In production,
    /// you would use a proper Yahoo Finance API client.
    pub async fn fetch(
        &self,
        symbol: &str,
        period: &str,
        interval: &str,
    ) -> Result<MarketData> {
        let url = format!(
            "{}/{}?period1=0&period2=9999999999&interval={}&range={}",
            self.base_url, symbol, interval, period
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .await
            .map_err(|e| RagError::DataFetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(RagError::DataFetch(format!(
                "Failed to fetch data: {}",
                response.status()
            )));
        }

        let body: String = response
            .text()
            .await
            .map_err(|e: reqwest::Error| RagError::DataFetch(e.to_string()))?;

        self.parse_response(symbol, &body)
    }

    /// Parse Yahoo Finance response.
    fn parse_response(&self, symbol: &str, body: &str) -> Result<MarketData> {
        let json: serde_json::Value =
            serde_json::from_str(body).map_err(|e| RagError::Parse(e.to_string()))?;

        let result = json["chart"]["result"]
            .get(0)
            .ok_or_else(|| RagError::Parse("No data in response".to_string()))?;

        let timestamps = result["timestamp"]
            .as_array()
            .ok_or_else(|| RagError::Parse("No timestamps".to_string()))?;

        let quotes = &result["indicators"]["quote"][0];

        let mut data = MarketData::new(symbol);

        for (i, ts) in timestamps.iter().enumerate() {
            let timestamp = ts.as_i64().unwrap_or(0);
            let dt = Utc.timestamp_opt(timestamp, 0).single().unwrap_or(Utc::now());

            let open = quotes["open"][i].as_f64().unwrap_or(0.0);
            let high = quotes["high"][i].as_f64().unwrap_or(0.0);
            let low = quotes["low"][i].as_f64().unwrap_or(0.0);
            let close = quotes["close"][i].as_f64().unwrap_or(0.0);
            let volume = quotes["volume"][i].as_f64().unwrap_or(0.0);

            if close > 0.0 {
                data.add_bar(dt, open, high, low, close, volume);
            }
        }

        Ok(data)
    }
}

impl Default for YahooFinanceLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Bybit API response structures.
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Bybit data loader for cryptocurrency data.
pub struct BybitLoader {
    base_url: String,
}

impl BybitLoader {
    /// Create a new Bybit loader.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data.
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<MarketData> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| RagError::DataFetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(RagError::DataFetch(format!(
                "Bybit API error: {}",
                response.status()
            )));
        }

        let body: BybitResponse = response
            .json()
            .await
            .map_err(|e: reqwest::Error| RagError::Parse(e.to_string()))?;

        if body.ret_code != 0 {
            return Err(RagError::DataFetch(format!(
                "Bybit error: {}",
                body.ret_msg
            )));
        }

        self.parse_klines(symbol, &body.result.list)
    }

    /// Parse Bybit kline data.
    fn parse_klines(&self, symbol: &str, klines: &[Vec<String>]) -> Result<MarketData> {
        let mut data = MarketData::new(symbol);

        // Bybit returns data in reverse chronological order
        for kline in klines.iter().rev() {
            if kline.len() < 6 {
                continue;
            }

            let timestamp_ms: i64 = kline[0].parse().unwrap_or(0);
            let dt = Utc.timestamp_millis_opt(timestamp_ms).single().unwrap_or(Utc::now());

            let open: f64 = kline[1].parse().unwrap_or(0.0);
            let high: f64 = kline[2].parse().unwrap_or(0.0);
            let low: f64 = kline[3].parse().unwrap_or(0.0);
            let close: f64 = kline[4].parse().unwrap_or(0.0);
            let volume: f64 = kline[5].parse().unwrap_or(0.0);

            if close > 0.0 {
                data.add_bar(dt, open, high, low, close, volume);
            }
        }

        Ok(data)
    }

    /// Fetch ticker information.
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| RagError::DataFetch(e.to_string()))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e: reqwest::Error| RagError::Parse(e.to_string()))?;

        let ticker = body["result"]["list"]
            .get(0)
            .ok_or_else(|| RagError::Parse("No ticker data".to_string()))?;

        Ok(TickerInfo {
            symbol: symbol.to_string(),
            last_price: ticker["lastPrice"]
                .as_str()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            high_24h: ticker["highPrice24h"]
                .as_str()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            low_24h: ticker["lowPrice24h"]
                .as_str()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            volume_24h: ticker["volume24h"]
                .as_str()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
            price_change_24h: ticker["price24hPcnt"]
                .as_str()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0),
        })
    }
}

impl Default for BybitLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Ticker information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    pub last_price: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub volume_24h: f64,
    pub price_change_24h: f64,
}

/// Generate mock market data for testing.
pub fn generate_mock_data(symbol: &str, num_bars: usize, start_price: f64) -> MarketData {
    use std::f64::consts::PI;

    let mut data = MarketData::new(symbol);
    let mut price = start_price;

    for i in 0..num_bars {
        let ts = Utc::now() - chrono::Duration::days((num_bars - i) as i64);

        // Add some randomness with a trend and seasonality
        let trend = 0.001 * (i as f64);
        let seasonality = 0.02 * (2.0 * PI * i as f64 / 20.0).sin();
        let noise = 0.01 * ((i * 7) % 13) as f64 / 13.0 - 0.005;

        price *= 1.0 + trend + seasonality + noise;

        let volatility = price * 0.02;
        let open = price;
        let high = price + volatility * ((i * 3) % 10) as f64 / 10.0;
        let low = price - volatility * ((i * 5) % 10) as f64 / 10.0;
        let close = price + volatility * (((i * 11) % 20) as f64 / 20.0 - 0.5);
        let volume = 1_000_000.0 * (1.0 + ((i * 7) % 5) as f64 / 5.0);

        data.add_bar(ts, open, high, low, close, volume);
    }

    data
}

/// Unified data loader that can fetch from multiple sources.
pub struct DataLoader {
    yahoo: YahooFinanceLoader,
    bybit: BybitLoader,
}

impl DataLoader {
    /// Create a new data loader.
    pub fn new() -> Self {
        Self {
            yahoo: YahooFinanceLoader::new(),
            bybit: BybitLoader::new(),
        }
    }

    /// Fetch stock data from Yahoo Finance.
    pub async fn fetch_stock(
        &self,
        symbol: &str,
        period: &str,
        interval: &str,
    ) -> Result<MarketData> {
        self.yahoo.fetch(symbol, period, interval).await
    }

    /// Fetch crypto data from Bybit.
    pub async fn fetch_crypto(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<MarketData> {
        self.bybit.fetch_klines(symbol, interval, limit).await
    }

    /// Get mock data for testing.
    pub fn mock_data(&self, symbol: &str, num_bars: usize, start_price: f64) -> MarketData {
        generate_mock_data(symbol, num_bars, start_price)
    }
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data() {
        let mut data = MarketData::new("TEST");
        assert!(data.is_empty());

        data.add_bar(Utc::now(), 100.0, 105.0, 95.0, 102.0, 1000.0);
        assert_eq!(data.len(), 1);

        let bars = data.to_price_bars();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].close, 102.0);
    }

    #[test]
    fn test_returns() {
        let mut data = MarketData::new("TEST");
        data.add_bar(Utc::now(), 100.0, 105.0, 95.0, 100.0, 1000.0);
        data.add_bar(Utc::now(), 100.0, 105.0, 95.0, 110.0, 1000.0);
        data.add_bar(Utc::now(), 110.0, 115.0, 105.0, 105.0, 1000.0);

        let returns = data.returns();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 0.001); // 10% return
    }

    #[test]
    fn test_mock_data() {
        let data = generate_mock_data("AAPL", 100, 150.0);
        assert_eq!(data.len(), 100);
        assert!(data.close.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_data_loader() {
        let loader = DataLoader::new();
        let data = loader.mock_data("BTCUSDT", 50, 45000.0);
        assert_eq!(data.len(), 50);
    }
}
