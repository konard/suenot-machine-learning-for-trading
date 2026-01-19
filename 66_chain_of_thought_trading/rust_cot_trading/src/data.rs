//! Data Loaders for Market Data
//!
//! Support for loading data from Yahoo Finance, Bybit, and mock sources.

use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use crate::error::{Error, Result};

/// OHLCV (Open, High, Low, Close, Volume) bar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
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
}

/// Trait for data loaders.
#[async_trait]
pub trait DataLoader: Send + Sync {
    /// Load OHLCV data for a symbol.
    async fn load(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        interval: &str,
    ) -> Result<Vec<OHLCV>>;

    /// Get latest OHLCV data.
    async fn get_latest(&self, symbol: &str, limit: usize) -> Result<Vec<OHLCV>>;
}

/// Yahoo Finance data loader.
pub struct YahooLoader {
    client: reqwest::Client,
}

impl YahooLoader {
    /// Create a new Yahoo Finance loader.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for YahooLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataLoader for YahooLoader {
    async fn load(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        interval: &str,
    ) -> Result<Vec<OHLCV>> {
        let period1 = start.timestamp();
        let period2 = end.timestamp();

        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}",
            symbol, period1, period2, interval
        );

        let response = self.client.get(&url)
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .await
            .map_err(|e| Error::DataLoadError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(Error::DataLoadError(format!(
                "Yahoo API returned status: {}",
                response.status()
            )));
        }

        let data: serde_json::Value = response.json().await?;

        // Parse Yahoo Finance response
        let result = &data["chart"]["result"][0];
        let timestamps = result["timestamp"].as_array()
            .ok_or_else(|| Error::DataLoadError("No timestamps in response".to_string()))?;
        let indicators = &result["indicators"]["quote"][0];

        let opens = indicators["open"].as_array();
        let highs = indicators["high"].as_array();
        let lows = indicators["low"].as_array();
        let closes = indicators["close"].as_array();
        let volumes = indicators["volume"].as_array();

        if opens.is_none() || highs.is_none() || lows.is_none() ||
           closes.is_none() || volumes.is_none() {
            return Err(Error::DataLoadError("Missing OHLCV data in response".to_string()));
        }

        let opens = opens.unwrap();
        let highs = highs.unwrap();
        let lows = lows.unwrap();
        let closes = closes.unwrap();
        let volumes = volumes.unwrap();

        let mut bars = Vec::with_capacity(timestamps.len());

        for i in 0..timestamps.len() {
            let ts = timestamps[i].as_i64().unwrap_or(0);
            let timestamp = DateTime::from_timestamp(ts, 0)
                .unwrap_or_else(Utc::now);

            let open = opens[i].as_f64().unwrap_or(0.0);
            let high = highs[i].as_f64().unwrap_or(0.0);
            let low = lows[i].as_f64().unwrap_or(0.0);
            let close = closes[i].as_f64().unwrap_or(0.0);
            let volume = volumes[i].as_f64().unwrap_or(0.0);

            // Skip invalid bars
            if open > 0.0 && close > 0.0 {
                bars.push(OHLCV {
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume,
                });
            }
        }

        Ok(bars)
    }

    async fn get_latest(&self, symbol: &str, limit: usize) -> Result<Vec<OHLCV>> {
        let end = Utc::now();
        let start = end - Duration::days((limit * 2) as i64);

        let mut bars = self.load(symbol, start, end, "1d").await?;

        // Return only the last 'limit' bars
        if bars.len() > limit {
            bars = bars.split_off(bars.len() - limit);
        }

        Ok(bars)
    }
}

/// Bybit cryptocurrency data loader.
pub struct BybitLoader {
    client: reqwest::Client,
    base_url: String,
}

impl BybitLoader {
    /// Create a new Bybit loader.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }
}

impl Default for BybitLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataLoader for BybitLoader {
    async fn load(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        interval: &str,
    ) -> Result<Vec<OHLCV>> {
        // Convert interval to Bybit format
        let bybit_interval = match interval {
            "1m" => "1",
            "5m" => "5",
            "15m" => "15",
            "1h" => "60",
            "4h" => "240",
            "1d" | "1D" => "D",
            "1w" | "1W" => "W",
            _ => "D",
        };

        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=200",
            self.base_url, symbol, bybit_interval, start_ms, end_ms
        );

        let response = self.client.get(&url)
            .send()
            .await
            .map_err(|e| Error::DataLoadError(e.to_string()))?;

        let data: serde_json::Value = response.json().await?;

        if data["retCode"].as_i64() != Some(0) {
            let msg = data["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(Error::DataLoadError(format!("Bybit API error: {}", msg)));
        }

        let klines = data["result"]["list"].as_array()
            .ok_or_else(|| Error::DataLoadError("No klines in response".to_string()))?;

        let mut bars = Vec::with_capacity(klines.len());

        for kline in klines {
            let arr = kline.as_array()
                .ok_or_else(|| Error::DataLoadError("Invalid kline format".to_string()))?;

            let ts_ms = arr[0].as_str()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);

            let timestamp = DateTime::from_timestamp_millis(ts_ms)
                .unwrap_or_else(Utc::now);

            let open = arr[1].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let high = arr[2].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let low = arr[3].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let close = arr[4].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let volume = arr[5].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);

            bars.push(OHLCV {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        // Bybit returns data newest first, so reverse
        bars.reverse();

        Ok(bars)
    }

    async fn get_latest(&self, symbol: &str, limit: usize) -> Result<Vec<OHLCV>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval=D&limit={}",
            self.base_url, symbol, limit.min(200)
        );

        let response = self.client.get(&url)
            .send()
            .await
            .map_err(|e| Error::DataLoadError(e.to_string()))?;

        let data: serde_json::Value = response.json().await?;

        if data["retCode"].as_i64() != Some(0) {
            let msg = data["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(Error::DataLoadError(format!("Bybit API error: {}", msg)));
        }

        let klines = data["result"]["list"].as_array()
            .ok_or_else(|| Error::DataLoadError("No klines in response".to_string()))?;

        let mut bars = Vec::with_capacity(klines.len());

        for kline in klines {
            let arr = kline.as_array()
                .ok_or_else(|| Error::DataLoadError("Invalid kline format".to_string()))?;

            let ts_ms = arr[0].as_str()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);

            let timestamp = DateTime::from_timestamp_millis(ts_ms)
                .unwrap_or_else(Utc::now);

            let open = arr[1].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let high = arr[2].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let low = arr[3].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let close = arr[4].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let volume = arr[5].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);

            bars.push(OHLCV {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        bars.reverse();
        Ok(bars)
    }
}

/// Mock data loader for testing.
pub struct MockLoader {
    seed: u64,
}

impl MockLoader {
    /// Create a new mock loader.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for MockLoader {
    fn default() -> Self {
        Self::new(42)
    }
}

#[async_trait]
impl DataLoader for MockLoader {
    async fn load(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        interval: &str,
    ) -> Result<Vec<OHLCV>> {
        use rand::{Rng, SeedableRng};

        // Calculate number of bars based on interval
        let duration = end - start;
        let hours_per_bar = match interval {
            "1m" => 1.0 / 60.0,
            "5m" => 5.0 / 60.0,
            "15m" => 0.25,
            "1h" => 1.0,
            "4h" => 4.0,
            "1d" => 24.0,
            "1w" => 168.0,
            _ => 24.0,
        };

        let total_hours = duration.num_hours() as f64;
        let num_bars = (total_hours / hours_per_bar) as usize;

        if num_bars == 0 {
            return Ok(Vec::new());
        }

        // Use symbol hash for different patterns
        let symbol_seed: u64 = symbol.bytes().map(|b| b as u64).sum();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + symbol_seed);

        let base_price = if symbol.contains("BTC") {
            40000.0
        } else if symbol.contains("ETH") {
            2500.0
        } else {
            100.0
        };

        let volatility = rng.gen_range(0.01..0.03);
        let drift = rng.gen_range(-0.0005..0.0005);

        let mut bars = Vec::with_capacity(num_bars);
        let mut price = base_price;
        let mut current_time = start;
        let bar_duration = Duration::hours((hours_per_bar * 60.0) as i64)
            .max(Duration::minutes(1));

        for _ in 0..num_bars {
            let ret = rng.gen_range(-volatility..volatility) + drift;
            let open = price;
            price *= 1.0 + ret;
            let close = price;

            let high = open.max(close) * (1.0 + rng.gen_range(0.0..volatility * 0.5));
            let low = open.min(close) * (1.0 - rng.gen_range(0.0..volatility * 0.5));
            let volume = 1_000_000.0 * rng.gen_range(0.5..2.0) * (1.0 + ret.abs() * 10.0);

            bars.push(OHLCV {
                timestamp: current_time,
                open,
                high,
                low,
                close,
                volume,
            });

            current_time = current_time + bar_duration;
        }

        Ok(bars)
    }

    async fn get_latest(&self, symbol: &str, limit: usize) -> Result<Vec<OHLCV>> {
        let end = Utc::now();
        let start = end - Duration::days((limit + 10) as i64);

        let mut bars = self.load(symbol, start, end, "1d").await?;

        if bars.len() > limit {
            bars = bars.split_off(bars.len() - limit);
        }

        Ok(bars)
    }
}

/// Add technical indicators to OHLCV data.
pub fn add_indicators(bars: &[OHLCV]) -> Vec<BarWithIndicators> {
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let mut result = Vec::with_capacity(bars.len());

    for (i, bar) in bars.iter().enumerate() {
        let sma_20 = if i >= 20 {
            closes[i.saturating_sub(19)..=i].iter().sum::<f64>() / 20.0
        } else {
            bar.close
        };

        let sma_50 = if i >= 50 {
            closes[i.saturating_sub(49)..=i].iter().sum::<f64>() / 50.0
        } else {
            bar.close
        };

        // Simple RSI calculation
        let rsi = if i >= 14 {
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (i - 13)..=i {
                if j > 0 {
                    let change = closes[j] - closes[j - 1];
                    if change > 0.0 {
                        gains += change;
                    } else {
                        losses += change.abs();
                    }
                }
            }
            if losses == 0.0 {
                100.0
            } else {
                let rs = gains / losses;
                100.0 - (100.0 / (1.0 + rs))
            }
        } else {
            50.0
        };

        // Simple MACD
        let ema_12 = calculate_ema(&closes[..=i], 12);
        let ema_26 = calculate_ema(&closes[..=i], 26);
        let macd = ema_12 - ema_26;

        // Simple ATR
        let atr = if i >= 14 {
            let mut tr_sum = 0.0;
            for j in (i - 13)..=i {
                if j > 0 {
                    let tr = (bars[j].high - bars[j].low)
                        .max((bars[j].high - closes[j - 1]).abs())
                        .max((bars[j].low - closes[j - 1]).abs());
                    tr_sum += tr;
                }
            }
            tr_sum / 14.0
        } else {
            bar.high - bar.low
        };

        result.push(BarWithIndicators {
            bar: bar.clone(),
            sma_20,
            sma_50,
            rsi,
            macd,
            macd_signal: macd * 0.9, // Simplified
            atr,
        });
    }

    result
}

fn calculate_ema(prices: &[f64], period: usize) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = prices[0];

    for price in prices.iter().skip(1) {
        ema = (price - ema) * multiplier + ema;
    }

    ema
}

/// OHLCV bar with technical indicators.
#[derive(Debug, Clone)]
pub struct BarWithIndicators {
    /// Base OHLCV data
    pub bar: OHLCV,
    /// 20-period SMA
    pub sma_20: f64,
    /// 50-period SMA
    pub sma_50: f64,
    /// RSI (14-period)
    pub rsi: f64,
    /// MACD line
    pub macd: f64,
    /// MACD signal line
    pub macd_signal: f64,
    /// Average True Range (14-period)
    pub atr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_loader() {
        let loader = MockLoader::new(42);
        let end = Utc::now();
        let start = end - Duration::days(30);

        let bars = loader.load("AAPL", start, end, "1d").await.unwrap();

        assert!(!bars.is_empty());
        for bar in &bars {
            assert!(bar.high >= bar.low);
            assert!(bar.high >= bar.open);
            assert!(bar.high >= bar.close);
            assert!(bar.low <= bar.open);
            assert!(bar.low <= bar.close);
        }
    }

    #[tokio::test]
    async fn test_indicators() {
        let loader = MockLoader::new(42);
        let bars = loader.get_latest("AAPL", 100).await.unwrap();

        let with_indicators = add_indicators(&bars);

        assert_eq!(with_indicators.len(), bars.len());

        // Check last bar has reasonable indicators
        let last = with_indicators.last().unwrap();
        assert!(last.rsi >= 0.0 && last.rsi <= 100.0);
        assert!(last.atr > 0.0);
    }
}
