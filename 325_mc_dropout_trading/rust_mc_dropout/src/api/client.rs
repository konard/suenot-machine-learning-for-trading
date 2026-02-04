//! Bybit API client implementation

use super::types::*;
use anyhow::{anyhow, Result};
use reqwest::Client;

/// Base URL for Bybit API
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Client for interacting with Bybit API
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create a new Bybit client for testnet
    pub fn testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Get ticker for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response: BybitResponse<TickerResponse> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!("API error: {}", response.ret_msg));
        }

        let item = response.result.list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No ticker found for {}", symbol))?;

        Ok(Ticker {
            symbol: item.symbol,
            last_price: item.last_price.parse().unwrap_or(0.0),
            bid_price: item.bid_price.parse().unwrap_or(0.0),
            ask_price: item.ask_price.parse().unwrap_or(0.0),
            volume_24h: item.volume_24h.parse().unwrap_or(0.0),
            price_change_24h: item.price_change_24h.parse().unwrap_or(0.0),
            high_24h: item.high_24h.parse().unwrap_or(0.0),
            low_24h: item.low_24h.parse().unwrap_or(0.0),
            timestamp: chrono::Utc::now().timestamp_millis(),
        })
    }

    /// Get tickers for multiple symbols
    pub async fn get_tickers(&self, symbols: &[&str]) -> Result<Vec<Ticker>> {
        let mut tickers = Vec::new();
        for symbol in symbols {
            match self.get_ticker(symbol).await {
                Ok(ticker) => tickers.push(ticker),
                Err(e) => log::warn!("Failed to fetch ticker for {}: {}", symbol, e),
            }
        }
        Ok(tickers)
    }

    /// Get klines (candlestick data)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse<KlineResponse> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!("API error: {}", response.ret_msg));
        }

        let klines: Vec<Kline> = response.result.list
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Kline {
                        start_time: item[0].parse().unwrap_or(0),
                        open: item[1].parse().unwrap_or(0.0),
                        high: item[2].parse().unwrap_or(0.0),
                        low: item[3].parse().unwrap_or(0.0),
                        close: item[4].parse().unwrap_or(0.0),
                        volume: item[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(klines)
    }

    /// Get order book
    pub async fn get_orderbook(&self, symbol: &str, limit: u32) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: BybitResponse<OrderBookResponse> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!("API error: {}", response.ret_msg));
        }

        let bids: Vec<OrderBookLevel> = response.result.b
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 2 {
                    Some(OrderBookLevel {
                        price: item[0].parse().unwrap_or(0.0),
                        size: item[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = response.result.a
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 2 {
                    Some(OrderBookLevel {
                        price: item[0].parse().unwrap_or(0.0),
                        size: item[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: response.result.s,
            bids,
            asks,
            timestamp: response.result.ts,
        })
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
        let ticker = result.unwrap();
        assert_eq!(ticker.symbol, "BTCUSDT");
        assert!(ticker.last_price > 0.0);
    }
}
