//! Bybit WebSocket client for real-time data streaming
//!
//! Provides streaming access to klines, trades, and order book updates.

use crate::data::{Candle, OrderBook, OrderBookLevel, Trade, TradeSide};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

/// WebSocket stream types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Kline/candlestick stream
    Kline { symbol: String, interval: String },
    /// Trade stream
    Trade { symbol: String },
    /// Order book stream
    OrderBook { symbol: String, depth: usize },
    /// Ticker stream
    Ticker { symbol: String },
}

impl StreamType {
    /// Get the subscription topic string
    pub fn topic(&self) -> String {
        match self {
            StreamType::Kline { symbol, interval } => format!("kline.{}.{}", interval, symbol),
            StreamType::Trade { symbol } => format!("publicTrade.{}", symbol),
            StreamType::OrderBook { symbol, depth } => format!("orderbook.{}.{}", depth, symbol),
            StreamType::Ticker { symbol } => format!("tickers.{}", symbol),
        }
    }
}

/// WebSocket message types
#[derive(Debug, Clone)]
pub enum WebSocketMessage {
    /// New candle data
    Candle(Candle),
    /// New trade
    Trade(Trade),
    /// Order book update
    OrderBook(OrderBook),
    /// Ticker update
    Ticker(TickerUpdate),
    /// Connection status
    Connected,
    /// Disconnection
    Disconnected,
    /// Error
    Error(String),
}

/// Ticker update data
#[derive(Debug, Clone)]
pub struct TickerUpdate {
    pub symbol: String,
    pub last_price: f64,
    pub price_24h_pcnt: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub volume_24h: f64,
    pub timestamp: u64,
}

/// Bybit WebSocket client
pub struct BybitWebSocket {
    url: String,
    subscriptions: Vec<StreamType>,
    sender: Option<mpsc::Sender<WebSocketMessage>>,
}

impl BybitWebSocket {
    /// Create a new WebSocket client for mainnet
    pub fn new() -> Self {
        Self {
            url: "wss://stream.bybit.com/v5/public/linear".to_string(),
            subscriptions: Vec::new(),
            sender: None,
        }
    }

    /// Create a new WebSocket client for testnet
    pub fn testnet() -> Self {
        Self {
            url: "wss://stream-testnet.bybit.com/v5/public/linear".to_string(),
            subscriptions: Vec::new(),
            sender: None,
        }
    }

    /// Subscribe to a stream
    pub fn subscribe(&mut self, stream: StreamType) -> &mut Self {
        if !self.subscriptions.contains(&stream) {
            self.subscriptions.push(stream);
        }
        self
    }

    /// Subscribe to kline stream
    pub fn subscribe_kline(&mut self, symbol: &str, interval: &str) -> &mut Self {
        self.subscribe(StreamType::Kline {
            symbol: symbol.to_string(),
            interval: interval.to_string(),
        })
    }

    /// Subscribe to trade stream
    pub fn subscribe_trades(&mut self, symbol: &str) -> &mut Self {
        self.subscribe(StreamType::Trade {
            symbol: symbol.to_string(),
        })
    }

    /// Subscribe to order book stream
    pub fn subscribe_orderbook(&mut self, symbol: &str, depth: usize) -> &mut Self {
        self.subscribe(StreamType::OrderBook {
            symbol: symbol.to_string(),
            depth,
        })
    }

    /// Connect and start streaming
    ///
    /// Returns a receiver channel for messages
    pub async fn connect(&mut self) -> anyhow::Result<mpsc::Receiver<WebSocketMessage>> {
        let (tx, rx) = mpsc::channel(1000);
        self.sender = Some(tx.clone());

        let url = self.url.clone();
        let subscriptions = self.subscriptions.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::run_connection(url, subscriptions, tx).await {
                error!("WebSocket connection error: {}", e);
            }
        });

        Ok(rx)
    }

    /// Internal connection handler
    async fn run_connection(
        url: String,
        subscriptions: Vec<StreamType>,
        tx: mpsc::Sender<WebSocketMessage>,
    ) -> anyhow::Result<()> {
        info!("Connecting to {}", url);

        let (ws_stream, _) = connect_async(&url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Send subscription message
        let topics: Vec<String> = subscriptions.iter().map(|s| s.topic()).collect();
        let sub_msg = SubscribeMessage {
            op: "subscribe".to_string(),
            args: topics,
        };

        let sub_json = serde_json::to_string(&sub_msg)?;
        write.send(Message::Text(sub_json)).await?;

        tx.send(WebSocketMessage::Connected).await.ok();
        info!("WebSocket connected and subscribed");

        // Ping task
        let ping_tx = tx.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(20));
            loop {
                interval.tick().await;
                debug!("Sending ping");
            }
        });

        // Read messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = Self::handle_message(&text, &tx).await {
                        warn!("Error handling message: {}", e);
                    }
                }
                Ok(Message::Ping(data)) => {
                    write.send(Message::Pong(data)).await.ok();
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket closed by server");
                    tx.send(WebSocketMessage::Disconnected).await.ok();
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    tx.send(WebSocketMessage::Error(e.to_string())).await.ok();
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Handle incoming WebSocket message
    async fn handle_message(
        text: &str,
        tx: &mpsc::Sender<WebSocketMessage>,
    ) -> anyhow::Result<()> {
        let json: serde_json::Value = serde_json::from_str(text)?;

        // Check for subscription confirmation
        if json.get("op").and_then(|v| v.as_str()) == Some("subscribe") {
            debug!("Subscription confirmed: {:?}", json.get("args"));
            return Ok(());
        }

        // Check for pong
        if json.get("op").and_then(|v| v.as_str()) == Some("pong") {
            return Ok(());
        }

        // Parse topic
        let topic = json.get("topic").and_then(|v| v.as_str()).unwrap_or("");

        if topic.starts_with("kline.") {
            Self::handle_kline_message(&json, tx).await?;
        } else if topic.starts_with("publicTrade.") {
            Self::handle_trade_message(&json, tx).await?;
        } else if topic.starts_with("orderbook.") {
            Self::handle_orderbook_message(&json, tx).await?;
        } else if topic.starts_with("tickers.") {
            Self::handle_ticker_message(&json, tx).await?;
        }

        Ok(())
    }

    async fn handle_kline_message(
        json: &serde_json::Value,
        tx: &mpsc::Sender<WebSocketMessage>,
    ) -> anyhow::Result<()> {
        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
            for item in data {
                let candle = Candle {
                    timestamp: item
                        .get("start")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    open: item
                        .get("open")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0),
                    high: item
                        .get("high")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0),
                    low: item
                        .get("low")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0),
                    close: item
                        .get("close")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0),
                    volume: item
                        .get("volume")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0),
                };

                tx.send(WebSocketMessage::Candle(candle)).await.ok();
            }
        }

        Ok(())
    }

    async fn handle_trade_message(
        json: &serde_json::Value,
        tx: &mpsc::Sender<WebSocketMessage>,
    ) -> anyhow::Result<()> {
        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
            for item in data {
                let trade = Trade {
                    id: item
                        .get("i")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    symbol: item
                        .get("s")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    price: item
                        .get("p")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0),
                    quantity: item
                        .get("v")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0),
                    side: if item.get("S").and_then(|v| v.as_str()) == Some("Buy") {
                        TradeSide::Buy
                    } else {
                        TradeSide::Sell
                    },
                    timestamp: item.get("T").and_then(|v| v.as_u64()).unwrap_or(0),
                };

                tx.send(WebSocketMessage::Trade(trade)).await.ok();
            }
        }

        Ok(())
    }

    async fn handle_orderbook_message(
        json: &serde_json::Value,
        tx: &mpsc::Sender<WebSocketMessage>,
    ) -> anyhow::Result<()> {
        if let Some(data) = json.get("data") {
            let symbol = data
                .get("s")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let timestamp = json.get("ts").and_then(|v| v.as_u64()).unwrap_or(0);

            let bids: Vec<OrderBookLevel> = data
                .get("b")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| {
                            let arr = item.as_array()?;
                            Some(OrderBookLevel {
                                price: arr.first()?.as_str()?.parse().ok()?,
                                quantity: arr.get(1)?.as_str()?.parse().ok()?,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            let asks: Vec<OrderBookLevel> = data
                .get("a")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| {
                            let arr = item.as_array()?;
                            Some(OrderBookLevel {
                                price: arr.first()?.as_str()?.parse().ok()?,
                                quantity: arr.get(1)?.as_str()?.parse().ok()?,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            let orderbook = OrderBook {
                symbol,
                timestamp,
                bids,
                asks,
            };

            tx.send(WebSocketMessage::OrderBook(orderbook)).await.ok();
        }

        Ok(())
    }

    async fn handle_ticker_message(
        json: &serde_json::Value,
        tx: &mpsc::Sender<WebSocketMessage>,
    ) -> anyhow::Result<()> {
        if let Some(data) = json.get("data") {
            let ticker = TickerUpdate {
                symbol: data
                    .get("symbol")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                last_price: data
                    .get("lastPrice")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                price_24h_pcnt: data
                    .get("price24hPcnt")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                high_24h: data
                    .get("highPrice24h")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                low_24h: data
                    .get("lowPrice24h")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                volume_24h: data
                    .get("volume24h")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0),
                timestamp: json.get("ts").and_then(|v| v.as_u64()).unwrap_or(0),
            };

            tx.send(WebSocketMessage::Ticker(ticker)).await.ok();
        }

        Ok(())
    }
}

impl Default for BybitWebSocket {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize)]
struct SubscribeMessage {
    op: String,
    args: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_type_topic() {
        let kline = StreamType::Kline {
            symbol: "BTCUSDT".to_string(),
            interval: "5".to_string(),
        };
        assert_eq!(kline.topic(), "kline.5.BTCUSDT");

        let trade = StreamType::Trade {
            symbol: "ETHUSDT".to_string(),
        };
        assert_eq!(trade.topic(), "publicTrade.ETHUSDT");

        let orderbook = StreamType::OrderBook {
            symbol: "BTCUSDT".to_string(),
            depth: 50,
        };
        assert_eq!(orderbook.topic(), "orderbook.50.BTCUSDT");
    }
}
