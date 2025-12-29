//! Клиент Bybit API

use super::error::ApiError;
use super::types::*;
use reqwest::Client;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Базовый URL для Bybit API
const BYBIT_BASE_URL: &str = "https://api.bybit.com";
const BYBIT_TESTNET_URL: &str = "https://api-testnet.bybit.com";

/// Клиент Bybit API
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Создать новый клиент для mainnet
    pub fn new() -> Self {
        Self::with_base_url(BYBIT_BASE_URL)
    }

    /// Создать клиент для testnet
    pub fn testnet() -> Self {
        Self::with_base_url(BYBIT_TESTNET_URL)
    }

    /// Создать клиент с произвольным базовым URL
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.into(),
        }
    }

    /// Получить исторические свечи (klines)
    ///
    /// # Arguments
    /// * `symbol` - Торговая пара (например, "BTCUSDT")
    /// * `interval` - Интервал свечи
    /// * `limit` - Количество свечей (макс. 1000)
    /// * `start` - Начальное время (Unix timestamp в мс)
    /// * `end` - Конечное время (Unix timestamp в мс)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        limit: Option<u32>,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Result<Vec<Candle>, ApiError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = vec![
            ("category", "spot".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.as_str().to_string()),
        ];

        if let Some(l) = limit {
            params.push(("limit", l.min(1000).to_string()));
        }
        if let Some(s) = start {
            params.push(("start", s.to_string()));
        }
        if let Some(e) = end {
            params.push(("end", e.to_string()));
        }

        debug!("Fetching klines for {} with interval {}", symbol, interval);

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let data: BybitResponse<KlineResult> = response.json().await?;

        if data.ret_code != 0 {
            return Err(ApiError::bybit(data.ret_code, data.ret_msg));
        }

        let candles: Vec<Candle> = data.result.list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Candle {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                        turnover: row[6].parse().unwrap_or(0.0),
                    })
                } else {
                    warn!("Invalid kline row: {:?}", row);
                    None
                }
            })
            .collect();

        // Bybit возвращает в обратном порядке, сортируем по времени
        let mut candles = candles;
        candles.sort_by_key(|c| c.timestamp);

        info!("Fetched {} candles for {}", candles.len(), symbol);

        Ok(candles)
    }

    /// Получить все свечи за указанный период
    ///
    /// Автоматически делает несколько запросов если период больше лимита.
    pub async fn get_all_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start: i64,
        end: i64,
    ) -> Result<Vec<Candle>, ApiError> {
        let mut all_candles = Vec::new();
        let mut current_start = start;
        let interval_ms = interval.seconds() as i64 * 1000;

        while current_start < end {
            let candles = self.get_klines(
                symbol,
                interval,
                Some(1000),
                Some(current_start),
                Some(end),
            ).await?;

            if candles.is_empty() {
                break;
            }

            let last_timestamp = candles.last().map(|c| c.timestamp).unwrap_or(end);
            all_candles.extend(candles);

            // Переходим к следующему периоду
            current_start = last_timestamp + interval_ms;

            // Небольшая задержка чтобы не превысить rate limit
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Удаляем дубликаты
        all_candles.sort_by_key(|c| c.timestamp);
        all_candles.dedup_by_key(|c| c.timestamp);

        Ok(all_candles)
    }

    /// Получить книгу ордеров
    pub async fn get_order_book(
        &self,
        symbol: &str,
        limit: Option<u32>,
    ) -> Result<OrderBook, ApiError> {
        let url = format!("{}/v5/market/orderbook", self.base_url);

        let limit = limit.unwrap_or(50).min(500);
        let params = [
            ("category", "spot"),
            ("symbol", symbol),
            ("limit", &limit.to_string()),
        ];

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;

        let ret_code = data["retCode"].as_i64().unwrap_or(-1) as i32;
        if ret_code != 0 {
            let msg = data["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(ApiError::bybit(ret_code, msg));
        }

        let result = &data["result"];
        let timestamp = result["ts"].as_i64().unwrap_or(0);

        let parse_levels = |arr: &serde_json::Value| -> Vec<OrderBookLevel> {
            arr.as_array()
                .map(|levels| {
                    levels.iter().filter_map(|level| {
                        let arr = level.as_array()?;
                        if arr.len() >= 2 {
                            Some(OrderBookLevel {
                                price: arr[0].as_str()?.parse().ok()?,
                                size: arr[1].as_str()?.parse().ok()?,
                            })
                        } else {
                            None
                        }
                    }).collect()
                })
                .unwrap_or_default()
        };

        Ok(OrderBook {
            symbol: symbol.to_string(),
            timestamp,
            bids: parse_levels(&result["b"]),
            asks: parse_levels(&result["a"]),
        })
    }

    /// Получить информацию о тикере
    pub async fn get_ticker(&self, symbol: &str) -> Result<serde_json::Value, ApiError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let params = [
            ("category", "spot"),
            ("symbol", symbol),
        ];

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;

        let ret_code = data["retCode"].as_i64().unwrap_or(-1) as i32;
        if ret_code != 0 {
            let msg = data["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(ApiError::bybit(ret_code, msg));
        }

        Ok(data["result"]["list"][0].clone())
    }

    /// Получить последние сделки
    pub async fn get_recent_trades(
        &self,
        symbol: &str,
        limit: Option<u32>,
    ) -> Result<Vec<Trade>, ApiError> {
        let url = format!("{}/v5/market/recent-trade", self.base_url);

        let limit = limit.unwrap_or(60).min(1000);
        let params = [
            ("category", "spot"),
            ("symbol", symbol),
            ("limit", &limit.to_string()),
        ];

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;

        let ret_code = data["retCode"].as_i64().unwrap_or(-1) as i32;
        if ret_code != 0 {
            let msg = data["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(ApiError::bybit(ret_code, msg));
        }

        let trades = data["result"]["list"]
            .as_array()
            .map(|arr| {
                arr.iter().filter_map(|t| {
                    Some(Trade {
                        timestamp: t["time"].as_str()?.parse().ok()?,
                        symbol: symbol.to_string(),
                        side: t["side"].as_str()?.to_string(),
                        price: t["price"].as_str()?.parse().ok()?,
                        size: t["size"].as_str()?.parse().ok()?,
                        trade_id: t["execId"].as_str()?.to_string(),
                    })
                }).collect()
            })
            .unwrap_or_default();

        Ok(trades)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert!(client.base_url.contains("bybit.com"));

        let testnet = BybitClient::testnet();
        assert!(testnet.base_url.contains("testnet"));
    }
}
