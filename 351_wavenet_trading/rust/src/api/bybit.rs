//! Bybit API клиент для получения рыночных данных

use crate::types::Candle;
use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;

/// Интервалы свечей
#[derive(Debug, Clone, Copy)]
pub enum Interval {
    Min1,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour4,
    Day1,
    Week1,
}

impl Interval {
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour4 => "240",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
        }
    }

    pub fn minutes(&self) -> i64 {
        match self {
            Interval::Min1 => 1,
            Interval::Min5 => 5,
            Interval::Min15 => 15,
            Interval::Min30 => 30,
            Interval::Hour1 => 60,
            Interval::Hour4 => 240,
            Interval::Day1 => 1440,
            Interval::Week1 => 10080,
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "1m" => Some(Interval::Min1),
            "5m" => Some(Interval::Min5),
            "15m" => Some(Interval::Min15),
            "30m" => Some(Interval::Min30),
            "1h" => Some(Interval::Hour1),
            "4h" => Some(Interval::Hour4),
            "1d" => Some(Interval::Day1),
            "1w" => Some(Interval::Week1),
            _ => None,
        }
    }
}

/// Ответ от Bybit API
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
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

/// Клиент для работы с Bybit API
#[derive(Debug, Clone)]
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Получить исторические свечи
    ///
    /// # Arguments
    /// * `symbol` - Торговая пара (например, "BTCUSDT")
    /// * `interval` - Интервал свечей
    /// * `limit` - Количество свечей (макс. 1000)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        limit: u32,
    ) -> Result<Vec<Candle>> {
        let limit = limit.min(1000);

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval.as_str(),
            limit
        );

        let response: BybitResponse = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request to Bybit")?
            .json()
            .await
            .context("Failed to parse Bybit response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let candles = response
            .result
            .list
            .into_iter()
            .filter_map(|item| self.parse_candle(&item))
            .collect::<Vec<_>>();

        // Bybit возвращает свечи в обратном порядке (новые первыми)
        let mut candles = candles;
        candles.reverse();

        Ok(candles)
    }

    /// Получить свечи за период
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::new();
        let mut current_end = end;

        loop {
            let url = format!(
                "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit=1000&end={}",
                self.base_url,
                symbol,
                interval.as_str(),
                current_end.timestamp_millis()
            );

            let response: BybitResponse = self
                .client
                .get(&url)
                .send()
                .await
                .context("Failed to send request")?
                .json()
                .await
                .context("Failed to parse response")?;

            if response.ret_code != 0 {
                anyhow::bail!("Bybit API error: {}", response.ret_msg);
            }

            let candles: Vec<Candle> = response
                .result
                .list
                .into_iter()
                .filter_map(|item| self.parse_candle(&item))
                .collect();

            if candles.is_empty() {
                break;
            }

            let oldest_timestamp = candles.last().map(|c| c.timestamp).unwrap();

            all_candles.extend(candles);

            if oldest_timestamp <= start {
                break;
            }

            current_end = oldest_timestamp;

            // Пауза между запросами
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Фильтруем по диапазону и сортируем
        all_candles.retain(|c| c.timestamp >= start && c.timestamp <= end);
        all_candles.sort_by_key(|c| c.timestamp);
        all_candles.dedup_by_key(|c| c.timestamp);

        Ok(all_candles)
    }

    /// Получить список доступных торговых пар
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        let url = format!("{}/v5/market/instruments-info?category=spot", self.base_url);

        #[derive(Deserialize)]
        struct InstrumentsResponse {
            #[serde(rename = "retCode")]
            ret_code: i32,
            result: InstrumentsResult,
        }

        #[derive(Deserialize)]
        struct InstrumentsResult {
            list: Vec<Instrument>,
        }

        #[derive(Deserialize)]
        struct Instrument {
            symbol: String,
        }

        let response: InstrumentsResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("Failed to get instruments");
        }

        Ok(response.result.list.into_iter().map(|i| i.symbol).collect())
    }

    /// Получить текущий тикер
    pub async fn get_ticker(&self, symbol: &str) -> Result<f64> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        #[derive(Deserialize)]
        struct TickerResponse {
            #[serde(rename = "retCode")]
            ret_code: i32,
            result: TickerResult,
        }

        #[derive(Deserialize)]
        struct TickerResult {
            list: Vec<Ticker>,
        }

        #[derive(Deserialize)]
        struct Ticker {
            #[serde(rename = "lastPrice")]
            last_price: String,
        }

        let response: TickerResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("Failed to get ticker");
        }

        let price = response
            .result
            .list
            .first()
            .ok_or_else(|| anyhow::anyhow!("No ticker data"))?
            .last_price
            .parse()
            .context("Failed to parse price")?;

        Ok(price)
    }

    fn parse_candle(&self, item: &[String]) -> Option<Candle> {
        if item.len() < 6 {
            return None;
        }

        let timestamp_ms: i64 = item[0].parse().ok()?;
        let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

        Some(Candle {
            timestamp,
            open: item[1].parse().ok()?,
            high: item[2].parse().ok()?,
            low: item[3].parse().ok()?,
            close: item[4].parse().ok()?,
            volume: item[5].parse().ok()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_klines() {
        let client = BybitClient::new();
        let candles = client.get_klines("BTCUSDT", Interval::Hour1, 10).await;
        assert!(candles.is_ok());
        let candles = candles.unwrap();
        assert!(!candles.is_empty());
    }
}
