//! Bybit API client implementation

use super::types::*;
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::Client;
use sha2::Sha256;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Bybit API base URLs
const MAINNET_URL: &str = "https://api.bybit.com";
const TESTNET_URL: &str = "https://api-testnet.bybit.com";

/// Bybit API client
#[derive(Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    api_secret: Option<String>,
}

impl BybitClient {
    /// Create a new Bybit client for public endpoints
    pub fn new(api_key: Option<String>, api_secret: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: MAINNET_URL.to_string(),
            api_key,
            api_secret,
        }
    }

    /// Create a client for testnet
    pub fn testnet(api_key: Option<String>, api_secret: Option<String>) -> Self {
        let mut client = Self::new(api_key, api_secret);
        client.base_url = TESTNET_URL.to_string();
        client
    }

    /// Generate signature for authenticated requests
    fn sign(&self, timestamp: i64, params: &str) -> Option<String> {
        if let (Some(key), Some(secret)) = (&self.api_key, &self.api_secret) {
            let recv_window = 5000;
            let sign_str = format!("{}{}{}{}", timestamp, key, recv_window, params);

            let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
                .expect("HMAC can take key of any size");
            mac.update(sign_str.as_bytes());
            let result = mac.finalize();

            Some(hex::encode(result.into_bytes()))
        } else {
            None
        }
    }

    /// Get klines (candlestick) data for a symbol
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, ApiError> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        debug!("Fetching klines for {} ({})", symbol, interval);

        let response: BybitResponse<KlinesResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or(ApiError::ApiError {
            code: -1,
            message: "No result in response".to_string(),
        })?;

        let klines: Vec<Kline> = result
            .list
            .iter()
            .filter_map(|k| {
                if k.len() >= 7 {
                    Some(Kline {
                        symbol: symbol.to_string(),
                        open_time: k[0].parse().unwrap_or(0),
                        open: k[1].parse().unwrap_or(0.0),
                        high: k[2].parse().unwrap_or(0.0),
                        low: k[3].parse().unwrap_or(0.0),
                        close: k[4].parse().unwrap_or(0.0),
                        volume: k[5].parse().unwrap_or(0.0),
                        turnover: k[6].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        let mut klines = klines;
        klines.reverse();

        info!("Fetched {} klines for {}", klines.len(), symbol);
        Ok(klines)
    }

    /// Get klines for multiple symbols
    pub async fn get_klines_batch(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> Result<HashMap<String, Vec<Kline>>, ApiError> {
        let mut results = HashMap::new();

        for symbol in symbols {
            match self.get_klines(symbol, interval, limit).await {
                Ok(klines) => {
                    results.insert(symbol.to_string(), klines);
                }
                Err(e) => {
                    warn!("Failed to fetch klines for {}: {}", symbol, e);
                }
            }

            // Small delay to avoid rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(results)
    }

    /// Get 24h ticker for all symbols
    pub async fn get_tickers(&self, category: &str) -> Result<Vec<Ticker>, ApiError> {
        let url = format!(
            "{}/v5/market/tickers?category={}",
            self.base_url, category
        );

        debug!("Fetching tickers for category {}", category);

        let response: BybitResponse<TickersResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or(ApiError::ApiError {
            code: -1,
            message: "No result in response".to_string(),
        })?;

        let tickers: Vec<Ticker> = result
            .list
            .iter()
            .map(|t| Ticker {
                symbol: t.symbol.clone(),
                last_price: t.last_price.parse().unwrap_or(0.0),
                high_24h: t.high_price_24h.parse().unwrap_or(0.0),
                low_24h: t.low_price_24h.parse().unwrap_or(0.0),
                volume_24h: t.volume_24h.parse().unwrap_or(0.0),
                turnover_24h: t.turnover_24h.parse().unwrap_or(0.0),
                price_change_24h: t.price_24h_pcnt.parse().unwrap_or(0.0),
                bid_price: t.bid1_price.parse().unwrap_or(0.0),
                ask_price: t.ask1_price.parse().unwrap_or(0.0),
            })
            .collect();

        info!("Fetched {} tickers", tickers.len());
        Ok(tickers)
    }

    /// Get ticker for a specific symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, ApiError> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: BybitResponse<TickersResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or(ApiError::ApiError {
            code: -1,
            message: "No result in response".to_string(),
        })?;

        let item = result.list.first().ok_or(ApiError::InvalidSymbol(symbol.to_string()))?;

        Ok(Ticker {
            symbol: item.symbol.clone(),
            last_price: item.last_price.parse().unwrap_or(0.0),
            high_24h: item.high_price_24h.parse().unwrap_or(0.0),
            low_24h: item.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: item.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: item.turnover_24h.parse().unwrap_or(0.0),
            price_change_24h: item.price_24h_pcnt.parse().unwrap_or(0.0),
            bid_price: item.bid1_price.parse().unwrap_or(0.0),
            ask_price: item.ask1_price.parse().unwrap_or(0.0),
        })
    }

    /// Get order book for a symbol
    pub async fn get_orderbook(&self, symbol: &str, limit: u32) -> Result<OrderBook, ApiError> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        debug!("Fetching orderbook for {}", symbol);

        let response: BybitResponse<OrderBookResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or(ApiError::ApiError {
            code: -1,
            message: "No result in response".to_string(),
        })?;

        let bids: Vec<OrderBookLevel> = result
            .b
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .a
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: result.s,
            bids,
            asks,
            timestamp: result.ts,
        })
    }

    /// Get current server time
    pub async fn get_server_time(&self) -> Result<i64, ApiError> {
        let url = format!("{}/v5/market/time", self.base_url);

        #[derive(Deserialize)]
        struct TimeResult {
            #[serde(rename = "timeSecond")]
            time_second: String,
            #[serde(rename = "timeNano")]
            time_nano: String,
        }

        let response: BybitResponse<TimeResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or(ApiError::ApiError {
            code: -1,
            message: "No result in response".to_string(),
        })?;

        Ok(result.time_second.parse().unwrap_or(0) * 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_server_time() {
        let client = BybitClient::new(None, None);
        let time = client.get_server_time().await;
        assert!(time.is_ok());
        assert!(time.unwrap() > 0);
    }
}
