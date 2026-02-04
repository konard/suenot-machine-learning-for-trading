//! Bybit API client for fetching market data

use crate::api::types::*;
use anyhow::{anyhow, Result};
use reqwest::Client;
use std::collections::HashMap;

/// Bybit API endpoints
const BYBIT_API_URL: &str = "https://api.bybit.com";
const BYBIT_TESTNET_URL: &str = "https://api-testnet.bybit.com";

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create new client for mainnet
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create new client for testnet
    pub fn testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_TESTNET_URL.to_string(),
        }
    }

    /// Fetch kline (OHLCV) data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let api_response: ApiResponse<KlineResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!("API error: {}", api_response.ret_msg));
        }

        let klines: Vec<Kline> = api_response
            .result
            .list
            .iter()
            .map(|item| Kline {
                start_time: item[0].parse().unwrap_or(0),
                open: item[1].parse().unwrap_or(0.0),
                high: item[2].parse().unwrap_or(0.0),
                low: item[3].parse().unwrap_or(0.0),
                close: item[4].parse().unwrap_or(0.0),
                volume: item[5].parse().unwrap_or(0.0),
                turnover: item[6].parse().unwrap_or(0.0),
            })
            .collect();

        // Reverse to get chronological order (Bybit returns newest first)
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch tickers for multiple symbols
    pub async fn get_tickers(&self) -> Result<Vec<Ticker>> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "spot")])
            .send()
            .await?;

        let api_response: ApiResponse<TickerResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!("API error: {}", api_response.ret_msg));
        }

        let tickers: Vec<Ticker> = api_response
            .result
            .list
            .iter()
            .map(|item| Ticker {
                symbol: item.symbol.clone(),
                last_price: item.last_price.parse().unwrap_or(0.0),
                bid_price: item.bid_price.parse().unwrap_or(0.0),
                ask_price: item.ask_price.parse().unwrap_or(0.0),
                high_24h: item.high_24h.parse().unwrap_or(0.0),
                low_24h: item.low_24h.parse().unwrap_or(0.0),
                volume_24h: item.volume_24h.parse().unwrap_or(0.0),
                turnover_24h: item.turnover_24h.parse().unwrap_or(0.0),
                price_change_24h: item.price_change_24h.parse().unwrap_or(0.0),
                timestamp: api_response.time,
            })
            .collect();

        Ok(tickers)
    }

    /// Fetch tickers filtered by symbols
    pub async fn get_tickers_filtered(&self, symbols: &[&str]) -> Result<Vec<Ticker>> {
        let all_tickers = self.get_tickers().await?;

        let filtered: Vec<Ticker> = all_tickers
            .into_iter()
            .filter(|t| symbols.contains(&t.symbol.as_str()))
            .collect();

        Ok(filtered)
    }

    /// Fetch order book
    pub async fn get_orderbook(&self, symbol: &str, limit: u32) -> Result<OrderBook> {
        let url = format!("{}/v5/market/orderbook", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let api_response: ApiResponse<OrderBookResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!("API error: {}", api_response.ret_msg));
        }

        let result = api_response.result;

        let bids: Vec<PriceLevel> = result
            .b
            .iter()
            .map(|level| PriceLevel {
                price: level[0].parse().unwrap_or(0.0),
                size: level[1].parse().unwrap_or(0.0),
            })
            .collect();

        let asks: Vec<PriceLevel> = result
            .a
            .iter()
            .map(|level| PriceLevel {
                price: level[0].parse().unwrap_or(0.0),
                size: level[1].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(OrderBook {
            symbol: result.s,
            timestamp: result.ts,
            bids,
            asks,
        })
    }

    /// Fetch recent trades
    pub async fn get_recent_trades(&self, symbol: &str, limit: u32) -> Result<Vec<Trade>> {
        let url = format!("{}/v5/market/recent-trade", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        #[derive(serde::Deserialize)]
        struct TradeResult {
            list: Vec<TradeData>,
        }

        #[derive(serde::Deserialize)]
        struct TradeData {
            #[serde(rename = "execId")]
            id: String,
            symbol: String,
            price: String,
            size: String,
            side: String,
            time: String,
        }

        let api_response: ApiResponse<TradeResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!("API error: {}", api_response.ret_msg));
        }

        let trades: Vec<Trade> = api_response
            .result
            .list
            .iter()
            .map(|item| Trade {
                id: item.id.clone(),
                symbol: item.symbol.clone(),
                price: item.price.parse().unwrap_or(0.0),
                size: item.size.parse().unwrap_or(0.0),
                side: item.side.clone(),
                timestamp: item.time.parse().unwrap_or(0),
            })
            .collect();

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

    #[tokio::test]
    async fn test_get_tickers() {
        let client = BybitClient::new();
        let result = client.get_tickers().await;
        assert!(result.is_ok());
    }
}
