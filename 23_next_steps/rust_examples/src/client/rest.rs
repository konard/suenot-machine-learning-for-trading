//! REST API клиент для Bybit

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use reqwest::Client;
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::*;
use crate::data::{Interval, Kline};

type HmacSha256 = Hmac<Sha256>;

/// Клиент для работы с Bybit API
#[derive(Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    api_secret: Option<String>,
}

impl BybitClient {
    /// Создать клиент для mainnet
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
        }
    }

    /// Создать клиент для testnet
    pub fn new_testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
        }
    }

    /// Установить API ключи для приватных эндпоинтов
    pub fn with_credentials(mut self, api_key: &str, api_secret: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self.api_secret = Some(api_secret.to_string());
        self
    }

    /// Создать клиент из переменных окружения
    pub fn from_env() -> Result<Self> {
        dotenv::dotenv().ok();

        let use_testnet = std::env::var("BYBIT_TESTNET")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(true);

        let mut client = if use_testnet {
            Self::new_testnet()
        } else {
            Self::new()
        };

        if let (Ok(key), Ok(secret)) = (
            std::env::var("BYBIT_API_KEY"),
            std::env::var("BYBIT_API_SECRET"),
        ) {
            client = client.with_credentials(&key, &secret);
        }

        Ok(client)
    }

    /// Получить текущий timestamp в миллисекундах
    fn get_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64
    }

    /// Создать подпись для приватных запросов
    fn sign(&self, timestamp: u64, params: &str) -> Result<String> {
        let api_key = self
            .api_key
            .as_ref()
            .context("API key not set")?;
        let api_secret = self
            .api_secret
            .as_ref()
            .context("API secret not set")?;

        let recv_window = 5000;
        let sign_str = format!("{}{}{}{}", timestamp, api_key, recv_window, params);

        let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
            .context("Invalid API secret")?;
        mac.update(sign_str.as_bytes());

        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    /// Получить информацию о тикере
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: ApiResponse<TickerResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        response
            .result
            .list
            .into_iter()
            .next()
            .context("Ticker not found")
    }

    /// Получить исторические свечи (klines)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        limit: Option<u32>,
    ) -> Result<Vec<Kline>> {
        let limit = limit.unwrap_or(200).min(1000);
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: ApiResponse<KlineResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse for chronological order
        Ok(klines.into_iter().rev().collect())
    }

    /// Получить исторические свечи с указанием временного диапазона
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url, symbol, interval, start_time, end_time
        );

        let response: ApiResponse<KlineResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(klines.into_iter().rev().collect())
    }

    /// Получить стакан заявок
    pub async fn get_orderbook(&self, symbol: &str, limit: Option<u32>) -> Result<OrderBookResult> {
        let limit = limit.unwrap_or(25).min(500);
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: ApiResponse<OrderBookResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        Ok(response.result)
    }

    /// Разместить ордер (требует API ключи)
    pub async fn place_order(
        &self,
        symbol: &str,
        side: Side,
        order_type: OrderType,
        qty: f64,
        price: Option<f64>,
    ) -> Result<OrderResult> {
        let timestamp = Self::get_timestamp();

        let order = OrderRequest {
            category: "spot".to_string(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            order_type: order_type.to_string(),
            qty: qty.to_string(),
            price: price.map(|p| p.to_string()),
            time_in_force: if order_type == OrderType::Market {
                "IOC".to_string()
            } else {
                "GTC".to_string()
            },
        };

        let body = serde_json::to_string(&order)?;
        let signature = self.sign(timestamp, &body)?;

        let response: ApiResponse<OrderResult> = self
            .client
            .post(&format!("{}/v5/order/create", self.base_url))
            .header("X-BAPI-API-KEY", self.api_key.as_ref().unwrap())
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-SIGN", signature)
            .header("X-BAPI-RECV-WINDOW", "5000")
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        Ok(response.result)
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

    #[tokio::test]
    async fn test_get_ticker() {
        let client = BybitClient::new();
        let result = client.get_ticker("BTCUSDT").await;
        assert!(result.is_ok());
    }
}
