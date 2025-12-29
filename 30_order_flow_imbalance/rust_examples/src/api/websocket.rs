//! # Bybit WebSocket Client
//!
//! Real-time market data streaming from Bybit exchange.

use anyhow::Result;
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::endpoints;
use crate::data::orderbook::{OrderBook, OrderBookLevel};
use crate::data::trade::Trade;

/// WebSocket message types
#[derive(Debug, Clone)]
pub enum WsMessage {
    OrderBook(OrderBook),
    Trade(Trade),
    Connected,
    Disconnected,
    Error(String),
}

/// Bybit WebSocket client for real-time data
pub struct BybitWebSocket {
    testnet: bool,
    sender: broadcast::Sender<WsMessage>,
    is_connected: Arc<RwLock<bool>>,
}

impl BybitWebSocket {
    /// Create a new WebSocket client
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1000);
        Self {
            testnet: false,
            sender,
            is_connected: Arc::new(RwLock::new(false)),
        }
    }

    /// Create a testnet WebSocket client
    pub fn testnet() -> Self {
        let (sender, _) = broadcast::channel(1000);
        Self {
            testnet: true,
            sender,
            is_connected: Arc::new(RwLock::new(false)),
        }
    }

    /// Subscribe to receive messages
    pub fn subscribe(&self) -> broadcast::Receiver<WsMessage> {
        self.sender.subscribe()
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        *self.is_connected.read().await
    }

    /// Connect and subscribe to orderbook updates
    pub async fn connect_orderbook(&self, symbol: &str, depth: u32) -> Result<()> {
        let url = if self.testnet {
            endpoints::TESTNET_WS
        } else {
            endpoints::MAINNET_WS
        };

        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Set connected status
        {
            let mut connected = self.is_connected.write().await;
            *connected = true;
        }
        let _ = self.sender.send(WsMessage::Connected);

        // Subscribe to orderbook
        let subscribe_msg = SubscribeMessage {
            op: "subscribe".to_string(),
            args: vec![format!("orderbook.{}.{}", depth, symbol)],
        };
        let msg = serde_json::to_string(&subscribe_msg)?;
        write.send(Message::Text(msg)).await?;

        let sender = self.sender.clone();
        let is_connected = self.is_connected.clone();
        let symbol_owned = symbol.to_string();

        // Spawn message handler
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Ok(data) = serde_json::from_str::<WsOrderBookData>(&text) {
                            if let Some(ob) = parse_ws_orderbook(&symbol_owned, &data) {
                                let _ = sender.send(WsMessage::OrderBook(ob));
                            }
                        }
                    }
                    Ok(Message::Ping(data)) => {
                        // Handle ping - connection is still alive
                        log::debug!("Received ping: {:?}", data);
                    }
                    Ok(Message::Close(_)) => {
                        let mut connected = is_connected.write().await;
                        *connected = false;
                        let _ = sender.send(WsMessage::Disconnected);
                        break;
                    }
                    Err(e) => {
                        let _ = sender.send(WsMessage::Error(e.to_string()));
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }

    /// Connect and subscribe to trade updates
    pub async fn connect_trades(&self, symbol: &str) -> Result<()> {
        let url = if self.testnet {
            endpoints::TESTNET_WS
        } else {
            endpoints::MAINNET_WS
        };

        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Set connected status
        {
            let mut connected = self.is_connected.write().await;
            *connected = true;
        }
        let _ = self.sender.send(WsMessage::Connected);

        // Subscribe to trades
        let subscribe_msg = SubscribeMessage {
            op: "subscribe".to_string(),
            args: vec![format!("publicTrade.{}", symbol)],
        };
        let msg = serde_json::to_string(&subscribe_msg)?;
        write.send(Message::Text(msg)).await?;

        let sender = self.sender.clone();
        let is_connected = self.is_connected.clone();
        let symbol_owned = symbol.to_string();

        // Spawn message handler
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        if let Ok(data) = serde_json::from_str::<WsTradeData>(&text) {
                            for trade in parse_ws_trades(&symbol_owned, &data) {
                                let _ = sender.send(WsMessage::Trade(trade));
                            }
                        }
                    }
                    Ok(Message::Close(_)) => {
                        let mut connected = is_connected.write().await;
                        *connected = false;
                        let _ = sender.send(WsMessage::Disconnected);
                        break;
                    }
                    Err(e) => {
                        let _ = sender.send(WsMessage::Error(e.to_string()));
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }
}

impl Default for BybitWebSocket {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WebSocket Message Types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct SubscribeMessage {
    op: String,
    args: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WsOrderBookData {
    topic: Option<String>,
    #[serde(rename = "type")]
    msg_type: Option<String>,
    ts: Option<u64>,
    data: Option<WsOrderBookPayload>,
}

#[derive(Debug, Deserialize)]
struct WsOrderBookPayload {
    s: String,
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
    u: u64,
}

#[derive(Debug, Deserialize)]
struct WsTradeData {
    topic: Option<String>,
    data: Option<Vec<WsTradePayload>>,
}

#[derive(Debug, Deserialize)]
struct WsTradePayload {
    #[serde(rename = "T")]
    timestamp: u64,
    s: String,
    #[serde(rename = "S")]
    side: String,
    v: String, // size
    p: String, // price
    i: String, // trade id
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parsing Functions
// ═══════════════════════════════════════════════════════════════════════════════

fn parse_ws_orderbook(symbol: &str, data: &WsOrderBookData) -> Option<OrderBook> {
    let payload = data.data.as_ref()?;
    let timestamp = data
        .ts
        .and_then(|ts| DateTime::from_timestamp_millis(ts as i64))
        .unwrap_or_else(Utc::now);

    let bids: Vec<OrderBookLevel> = payload
        .b
        .iter()
        .enumerate()
        .filter_map(|(i, level)| {
            let price = level.first()?.parse::<f64>().ok()?;
            let size = level.get(1)?.parse::<f64>().ok()?;
            Some(OrderBookLevel::new(price, size, i + 1))
        })
        .collect();

    let asks: Vec<OrderBookLevel> = payload
        .a
        .iter()
        .enumerate()
        .filter_map(|(i, level)| {
            let price = level.first()?.parse::<f64>().ok()?;
            let size = level.get(1)?.parse::<f64>().ok()?;
            Some(OrderBookLevel::new(price, size, i + 1))
        })
        .collect();

    Some(OrderBook::new(symbol.to_string(), timestamp, bids, asks))
}

fn parse_ws_trades(symbol: &str, data: &WsTradeData) -> Vec<Trade> {
    data.data
        .as_ref()
        .map(|trades| {
            trades
                .iter()
                .filter_map(|t| {
                    let timestamp = DateTime::from_timestamp_millis(t.timestamp as i64)?;
                    let price = t.p.parse::<f64>().ok()?;
                    let size = t.v.parse::<f64>().ok()?;
                    let is_buyer_maker = t.side == "Sell";

                    Some(Trade::new(
                        symbol.to_string(),
                        timestamp,
                        price,
                        size,
                        is_buyer_maker,
                        t.i.clone(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_client_creation() {
        let ws = BybitWebSocket::new();
        assert!(!ws.testnet);

        let ws_testnet = BybitWebSocket::testnet();
        assert!(ws_testnet.testnet);
    }
}
