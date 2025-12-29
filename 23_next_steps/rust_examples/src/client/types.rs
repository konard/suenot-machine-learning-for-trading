//! Типы данных для Bybit API

use serde::{Deserialize, Serialize};

/// Ответ API Bybit
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
    pub time: u64,
}

/// Информация о тикере
#[derive(Debug, Clone, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

/// Результат запроса тикеров
#[derive(Debug, Deserialize)]
pub struct TickerResult {
    pub category: String,
    pub list: Vec<TickerInfo>,
}

/// Результат запроса свечей
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub category: String,
    pub symbol: String,
    pub list: Vec<Vec<String>>,
}

/// Уровень стакана заявок
#[derive(Debug, Clone, Deserialize)]
pub struct OrderBookLevel {
    pub price: String,
    pub size: String,
}

/// Результат запроса стакана
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    pub s: String,  // symbol
    pub b: Vec<Vec<String>>,  // bids
    pub a: Vec<Vec<String>>,  // asks
    pub ts: u64,  // timestamp
    pub u: u64,   // update id
}

/// Информация о балансе
#[derive(Debug, Clone, Deserialize)]
pub struct BalanceInfo {
    #[serde(rename = "accountType")]
    pub account_type: String,
    #[serde(rename = "totalEquity")]
    pub total_equity: String,
    #[serde(rename = "totalWalletBalance")]
    pub total_wallet_balance: String,
}

/// Параметры ордера
#[derive(Debug, Clone, Serialize)]
pub struct OrderRequest {
    pub category: String,
    pub symbol: String,
    pub side: String,
    #[serde(rename = "orderType")]
    pub order_type: String,
    pub qty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<String>,
    #[serde(rename = "timeInForce")]
    pub time_in_force: String,
}

/// Результат создания ордера
#[derive(Debug, Deserialize)]
pub struct OrderResult {
    #[serde(rename = "orderId")]
    pub order_id: String,
    #[serde(rename = "orderLinkId")]
    pub order_link_id: String,
}

/// Сторона ордера
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Buy => write!(f, "Buy"),
            Side::Sell => write!(f, "Sell"),
        }
    }
}

/// Тип ордера
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Market => write!(f, "Market"),
            OrderType::Limit => write!(f, "Limit"),
        }
    }
}

/// WebSocket сообщение
#[derive(Debug, Deserialize)]
pub struct WsMessage {
    pub topic: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub ts: u64,
    pub data: serde_json::Value,
}

/// Тик данные из WebSocket
#[derive(Debug, Clone, Deserialize)]
pub struct WsTicker {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
}
