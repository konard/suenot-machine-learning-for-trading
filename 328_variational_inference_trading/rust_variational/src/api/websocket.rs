//! Bybit WebSocket client
//!
//! Provides real-time market data streaming from Bybit.

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

const BYBIT_WS_URL: &str = "wss://stream.bybit.com/v5/public/spot";

/// WebSocket subscription message
#[derive(Debug, Serialize)]
struct SubscribeMessage {
    op: String,
    args: Vec<String>,
}

/// WebSocket response
#[derive(Debug, Deserialize)]
pub struct WsResponse {
    pub topic: Option<String>,
    pub data: Option<serde_json::Value>,
    pub ts: Option<i64>,
}

/// Bybit WebSocket client
pub struct BybitWebSocket {
    url: String,
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
        }
    }

    /// Connect and subscribe to kline updates
    pub async fn subscribe_klines<F>(
        &self,
        symbols: &[&str],
        interval: &str,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(WsResponse),
    {
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Build subscription topics
        let topics: Vec<String> = symbols
            .iter()
            .map(|s| format!("kline.{}.{}", interval, s))
            .collect();

        let subscribe_msg = SubscribeMessage {
            op: "subscribe".to_string(),
            args: topics,
        };

        let msg = serde_json::to_string(&subscribe_msg)?;
        write.send(Message::Text(msg)).await?;

        // Read messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(response) = serde_json::from_str::<WsResponse>(&text) {
                        callback(response);
                    }
                }
                Ok(Message::Ping(data)) => {
                    write.send(Message::Pong(data)).await?;
                }
                Err(e) => {
                    log::error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Connect and subscribe to orderbook updates
    pub async fn subscribe_orderbook<F>(
        &self,
        symbols: &[&str],
        depth: u32,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(WsResponse),
    {
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Build subscription topics
        let topics: Vec<String> = symbols
            .iter()
            .map(|s| format!("orderbook.{}.{}", depth, s))
            .collect();

        let subscribe_msg = SubscribeMessage {
            op: "subscribe".to_string(),
            args: topics,
        };

        let msg = serde_json::to_string(&subscribe_msg)?;
        write.send(Message::Text(msg)).await?;

        // Read messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(response) = serde_json::from_str::<WsResponse>(&text) {
                        callback(response);
                    }
                }
                Ok(Message::Ping(data)) => {
                    write.send(Message::Pong(data)).await?;
                }
                Err(e) => {
                    log::error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }
}
