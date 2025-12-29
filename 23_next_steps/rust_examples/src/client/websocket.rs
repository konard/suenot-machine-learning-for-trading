//! WebSocket клиент для Bybit

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error};

use super::types::WsMessage;

/// WebSocket клиент для Bybit
pub struct BybitWebSocket {
    url: String,
}

/// Тип данных для подписки
#[derive(Debug, Clone)]
pub enum SubscriptionType {
    /// Тикер (последняя цена)
    Ticker(String),
    /// Свечи
    Kline { symbol: String, interval: String },
    /// Стакан заявок
    OrderBook { symbol: String, depth: u32 },
    /// Сделки
    Trade(String),
}

impl SubscriptionType {
    fn to_topic(&self) -> String {
        match self {
            SubscriptionType::Ticker(symbol) => format!("tickers.{}", symbol),
            SubscriptionType::Kline { symbol, interval } => {
                format!("kline.{}.{}", interval, symbol)
            }
            SubscriptionType::OrderBook { symbol, depth } => {
                format!("orderbook.{}.{}", depth, symbol)
            }
            SubscriptionType::Trade(symbol) => format!("publicTrade.{}", symbol),
        }
    }
}

impl BybitWebSocket {
    /// Создать WebSocket клиент для mainnet
    pub fn new() -> Self {
        Self {
            url: "wss://stream.bybit.com/v5/public/spot".to_string(),
        }
    }

    /// Создать WebSocket клиент для testnet
    pub fn new_testnet() -> Self {
        Self {
            url: "wss://stream-testnet.bybit.com/v5/public/spot".to_string(),
        }
    }

    /// Подписаться на обновления и получать сообщения через канал
    pub async fn subscribe(
        &self,
        subscriptions: Vec<SubscriptionType>,
    ) -> Result<mpsc::Receiver<WsMessage>> {
        let (tx, rx) = mpsc::channel(100);
        let url = self.url.clone();

        let topics: Vec<String> = subscriptions.iter().map(|s| s.to_topic()).collect();

        tokio::spawn(async move {
            if let Err(e) = Self::run_websocket(url, topics, tx).await {
                error!("WebSocket error: {}", e);
            }
        });

        Ok(rx)
    }

    async fn run_websocket(
        url: String,
        topics: Vec<String>,
        tx: mpsc::Sender<WsMessage>,
    ) -> Result<()> {
        let (ws_stream, _) = connect_async(&url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Отправляем подписку
        let subscribe_msg = json!({
            "op": "subscribe",
            "args": topics
        });

        write.send(Message::Text(subscribe_msg.to_string())).await?;
        info!("Subscribed to topics: {:?}", topics);

        // Читаем сообщения
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    // Пропускаем pong и subscription confirmations
                    if text.contains("\"op\"") {
                        continue;
                    }

                    if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                        if tx.send(ws_msg).await.is_err() {
                            warn!("Receiver dropped, stopping WebSocket");
                            break;
                        }
                    }
                }
                Ok(Message::Ping(data)) => {
                    write.send(Message::Pong(data)).await?;
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket closed by server");
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Подписаться на тикер и получать обновления цены
    pub async fn subscribe_ticker(&self, symbol: &str) -> Result<mpsc::Receiver<WsMessage>> {
        self.subscribe(vec![SubscriptionType::Ticker(symbol.to_string())])
            .await
    }

    /// Подписаться на свечи
    pub async fn subscribe_kline(
        &self,
        symbol: &str,
        interval: &str,
    ) -> Result<mpsc::Receiver<WsMessage>> {
        self.subscribe(vec![SubscriptionType::Kline {
            symbol: symbol.to_string(),
            interval: interval.to_string(),
        }])
        .await
    }

    /// Подписаться на стакан заявок
    pub async fn subscribe_orderbook(
        &self,
        symbol: &str,
        depth: u32,
    ) -> Result<mpsc::Receiver<WsMessage>> {
        self.subscribe(vec![SubscriptionType::OrderBook {
            symbol: symbol.to_string(),
            depth,
        }])
        .await
    }
}

impl Default for BybitWebSocket {
    fn default() -> Self {
        Self::new()
    }
}
