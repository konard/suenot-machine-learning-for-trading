//! Bybit WebSocket client for real-time market data
//!
//! Provides streaming market data via WebSocket connection.

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Bybit WebSocket URL for spot market
const BYBIT_WS_URL: &str = "wss://stream.bybit.com/v5/public/spot";

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WsMessage {
    Ticker(TickerUpdate),
    Orderbook(OrderbookUpdate),
    Trade(TradeUpdate),
    Pong(PongMessage),
}

/// Ticker update from WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerUpdate {
    pub topic: String,
    #[serde(rename = "type")]
    pub update_type: String,
    pub ts: i64,
    pub data: TickerData,
}

/// Ticker data payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_change_24h: String,
}

/// Orderbook update from WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookUpdate {
    pub topic: String,
    #[serde(rename = "type")]
    pub update_type: String,
    pub ts: i64,
    pub data: OrderbookData,
}

/// Orderbook data payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookData {
    pub s: String,  // symbol
    pub b: Vec<Vec<String>>,  // bids [price, size]
    pub a: Vec<Vec<String>>,  // asks [price, size]
    pub u: u64,  // update id
    pub seq: u64,  // sequence
}

/// Trade update from WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeUpdate {
    pub topic: String,
    #[serde(rename = "type")]
    pub update_type: String,
    pub ts: i64,
    pub data: Vec<TradeData>,
}

/// Trade data payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    #[serde(rename = "T")]
    pub timestamp: i64,
    pub s: String,  // symbol
    #[serde(rename = "S")]
    pub side: String,  // Buy or Sell
    pub v: String,  // volume
    pub p: String,  // price
    pub i: String,  // trade id
    #[serde(rename = "BT")]
    pub is_block_trade: bool,
}

/// Pong response from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PongMessage {
    pub success: bool,
    pub ret_msg: String,
    pub op: String,
}

/// Subscription request
#[derive(Debug, Serialize)]
struct SubscribeRequest {
    op: String,
    args: Vec<String>,
}

/// Bybit WebSocket client
pub struct BybitWebSocket {
    url: String,
    subscriptions: Arc<RwLock<Vec<String>>>,
}

impl Default for BybitWebSocket {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitWebSocket {
    /// Create a new WebSocket client
    pub fn new() -> Self {
        Self {
            url: BYBIT_WS_URL.to_string(),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create with custom URL
    pub fn with_url(url: &str) -> Self {
        Self {
            url: url.to_string(),
            subscriptions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Subscribe to ticker updates for symbols
    pub async fn subscribe_tickers(&self, symbols: &[&str]) -> Result<mpsc::Receiver<TickerUpdate>> {
        let (tx, rx) = mpsc::channel(1000);
        let topics: Vec<String> = symbols.iter().map(|s| format!("tickers.{}", s)).collect();

        self.connect_and_subscribe(topics, move |msg| {
            if let WsMessage::Ticker(update) = msg {
                let _ = tx.try_send(update);
            }
        })
        .await?;

        Ok(rx)
    }

    /// Subscribe to orderbook updates for symbols
    pub async fn subscribe_orderbook(&self, symbols: &[&str], depth: u32) -> Result<mpsc::Receiver<OrderbookUpdate>> {
        let (tx, rx) = mpsc::channel(1000);
        let topics: Vec<String> = symbols
            .iter()
            .map(|s| format!("orderbook.{}.{}", depth, s))
            .collect();

        self.connect_and_subscribe(topics, move |msg| {
            if let WsMessage::Orderbook(update) = msg {
                let _ = tx.try_send(update);
            }
        })
        .await?;

        Ok(rx)
    }

    /// Subscribe to trade updates for symbols
    pub async fn subscribe_trades(&self, symbols: &[&str]) -> Result<mpsc::Receiver<TradeUpdate>> {
        let (tx, rx) = mpsc::channel(1000);
        let topics: Vec<String> = symbols.iter().map(|s| format!("publicTrade.{}", s)).collect();

        self.connect_and_subscribe(topics, move |msg| {
            if let WsMessage::Trade(update) = msg {
                let _ = tx.try_send(update);
            }
        })
        .await?;

        Ok(rx)
    }

    /// Internal method to connect and subscribe
    async fn connect_and_subscribe<F>(&self, topics: Vec<String>, handler: F) -> Result<()>
    where
        F: Fn(WsMessage) + Send + 'static,
    {
        let url = url::Url::parse(&self.url)?;
        let (ws_stream, _) = connect_async(url)
            .await
            .context("Failed to connect to WebSocket")?;

        let (mut write, mut read) = ws_stream.split();

        // Send subscription request
        let subscribe_msg = SubscribeRequest {
            op: "subscribe".to_string(),
            args: topics.clone(),
        };
        let msg_text = serde_json::to_string(&subscribe_msg)?;
        write.send(Message::Text(msg_text)).await?;

        // Store subscriptions
        {
            let mut subs = self.subscriptions.write().await;
            subs.extend(topics);
        }

        // Spawn message handler
        tokio::spawn(async move {
            while let Some(msg_result) = read.next().await {
                match msg_result {
                    Ok(Message::Text(text)) => {
                        if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                            handler(ws_msg);
                        }
                    }
                    Ok(Message::Ping(data)) => {
                        let _ = write.send(Message::Pong(data)).await;
                    }
                    Ok(Message::Close(_)) => break,
                    Err(e) => {
                        log::error!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }
}

/// Parsed real-time ticker for easier consumption
#[derive(Debug, Clone)]
pub struct RealtimeTicker {
    pub symbol: String,
    pub last_price: f64,
    pub price_change_24h: f64,
    pub volume_24h: f64,
    pub timestamp: i64,
}

impl TryFrom<TickerUpdate> for RealtimeTicker {
    type Error = anyhow::Error;

    fn try_from(update: TickerUpdate) -> Result<Self> {
        Ok(Self {
            symbol: update.data.symbol,
            last_price: update.data.last_price.parse()?,
            price_change_24h: update.data.price_change_24h.parse()?,
            volume_24h: update.data.volume_24h.parse()?,
            timestamp: update.ts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_creation() {
        let ws = BybitWebSocket::new();
        assert_eq!(ws.url, BYBIT_WS_URL);
    }
}
