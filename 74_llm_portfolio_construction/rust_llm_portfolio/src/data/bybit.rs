//! Bybit API client for fetching cryptocurrency market data

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OHLCV {
    /// Convert timestamp to DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp).unwrap_or_default()
    }

    /// Calculate return from previous candle
    pub fn return_from(&self, prev: &OHLCV) -> f64 {
        if prev.close == 0.0 {
            0.0
        } else {
            (self.close - prev.close) / prev.close
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline result from Bybit API
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Ticker result from Bybit API
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerData>,
}

/// Individual ticker data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TickerData {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
}

/// Bybit API client
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
    timeout: Duration,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Fetch kline/candlestick data
    pub fn fetch_klines(&self, symbol: &str, interval: &str, limit: u32) -> Result<Vec<OHLCV>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .timeout(self.timeout)
            .send()?;

        let data: BybitResponse<KlineResult> = response.json()?;

        if data.ret_code != 0 {
            return Err(anyhow!("API error: {}", data.ret_msg));
        }

        // Parse klines (Bybit returns in reverse chronological order)
        let mut klines: Vec<OHLCV> = data
            .result
            .list
            .iter()
            .filter_map(|k| {
                if k.len() >= 6 {
                    Some(OHLCV {
                        timestamp: k[0].parse().unwrap_or(0),
                        open: k[1].parse().unwrap_or(0.0),
                        high: k[2].parse().unwrap_or(0.0),
                        low: k[3].parse().unwrap_or(0.0),
                        close: k[4].parse().unwrap_or(0.0),
                        volume: k[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Reverse to chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Fetch historical klines for multiple days
    pub fn fetch_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        days: u32,
    ) -> Result<Vec<OHLCV>> {
        let interval_minutes: u32 = match interval {
            "1" => 1,
            "5" => 5,
            "15" => 15,
            "30" => 30,
            "60" => 60,
            "240" => 240,
            "D" => 1440,
            "W" => 10080,
            _ => 1440,
        };

        let total_candles = (days * 24 * 60) / interval_minutes;
        let mut all_klines: Vec<OHLCV> = Vec::new();
        let mut end_time = Utc::now().timestamp_millis();

        while all_klines.len() < total_candles as usize {
            let batch_size = 1000.min(total_candles as usize - all_klines.len()) as u32;

            let url = format!("{}/v5/market/kline", self.base_url);

            let response = self
                .client
                .get(&url)
                .query(&[
                    ("category", "linear"),
                    ("symbol", symbol),
                    ("interval", interval),
                    ("limit", &batch_size.to_string()),
                    ("end", &end_time.to_string()),
                ])
                .timeout(self.timeout)
                .send()?;

            let data: BybitResponse<KlineResult> = response.json()?;

            if data.ret_code != 0 {
                return Err(anyhow!("API error: {}", data.ret_msg));
            }

            if data.result.list.is_empty() {
                break;
            }

            for k in data.result.list.iter().rev() {
                if k.len() >= 6 {
                    all_klines.push(OHLCV {
                        timestamp: k[0].parse().unwrap_or(0),
                        open: k[1].parse().unwrap_or(0.0),
                        high: k[2].parse().unwrap_or(0.0),
                        low: k[3].parse().unwrap_or(0.0),
                        close: k[4].parse().unwrap_or(0.0),
                        volume: k[5].parse().unwrap_or(0.0),
                    });
                }
            }

            // Update end time for next batch
            if let Some(last) = data.result.list.last() {
                if !last.is_empty() {
                    end_time = last[0].parse::<i64>().unwrap_or(end_time) - 1;
                }
            }

            // Rate limiting
            thread::sleep(Duration::from_millis(100));
        }

        // Sort and deduplicate
        all_klines.sort_by_key(|k| k.timestamp);
        all_klines.dedup_by_key(|k| k.timestamp);

        // Return requested number
        let start_idx = all_klines.len().saturating_sub(total_candles as usize);
        Ok(all_klines[start_idx..].to_vec())
    }

    /// Fetch current ticker data
    pub fn fetch_ticker(&self, symbol: &str) -> Result<TickerData> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .timeout(self.timeout)
            .send()?;

        let data: BybitResponse<TickerResult> = response.json()?;

        if data.ret_code != 0 {
            return Err(anyhow!("API error: {}", data.ret_msg));
        }

        data.result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Ticker not found for {}", symbol))
    }

    /// Fetch tickers for multiple symbols
    pub fn fetch_multiple_tickers(&self, symbols: &[&str]) -> Result<HashMap<String, TickerData>> {
        let mut tickers = HashMap::new();

        for symbol in symbols {
            match self.fetch_ticker(symbol) {
                Ok(ticker) => {
                    tickers.insert(symbol.to_string(), ticker);
                }
                Err(e) => {
                    log::warn!("Failed to fetch ticker for {}: {}", symbol, e);
                }
            }
            thread::sleep(Duration::from_millis(50));
        }

        Ok(tickers)
    }

    /// Calculate returns from OHLCV data
    pub fn calculate_returns(klines: &[OHLCV]) -> Vec<f64> {
        if klines.len() < 2 {
            return vec![];
        }

        klines
            .windows(2)
            .map(|w| w[1].return_from(&w[0]))
            .collect()
    }

    /// Calculate volatility from OHLCV data
    pub fn calculate_volatility(klines: &[OHLCV], annualize: bool) -> f64 {
        let returns = Self::calculate_returns(klines);
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        if annualize {
            std * (365.0_f64).sqrt() // Crypto trades 365 days
        } else {
            std
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_return() {
        let prev = OHLCV {
            timestamp: 0,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 100.0,
            volume: 1000.0,
        };

        let current = OHLCV {
            timestamp: 1,
            open: 100.0,
            high: 110.0,
            low: 98.0,
            close: 105.0,
            volume: 1200.0,
        };

        let ret = current.return_from(&prev);
        assert!((ret - 0.05).abs() < 0.0001);
    }

    #[test]
    fn test_calculate_returns() {
        let klines = vec![
            OHLCV {
                timestamp: 0,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 100.0,
                volume: 1000.0,
            },
            OHLCV {
                timestamp: 1,
                open: 100.0,
                high: 110.0,
                low: 98.0,
                close: 110.0,
                volume: 1200.0,
            },
            OHLCV {
                timestamp: 2,
                open: 110.0,
                high: 115.0,
                low: 105.0,
                close: 105.0,
                volume: 1100.0,
            },
        ];

        let returns = BybitClient::calculate_returns(&klines);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.10).abs() < 0.0001);
    }
}
