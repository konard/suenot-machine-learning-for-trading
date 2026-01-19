//! API Module
//!
//! Provides clients for external APIs including OpenAI and Bybit.

use crate::error::{Error, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// OpenAI API client for LLM interactions
pub struct OpenAIClient {
    api_key: String,
    client: Client,
    model: String,
    base_url: String,
}

impl OpenAIClient {
    /// Create new OpenAI client
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
            model: "gpt-4o-mini".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Set custom base URL (for compatible APIs)
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Complete a chat message
    pub async fn chat_completion(&self, messages: &[ChatMessage]) -> Result<String> {
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            temperature: 0.1,
            max_tokens: Some(1000),
        };

        let response = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::ApiError(format!(
                "OpenAI API error ({}): {}",
                status, error_text
            )));
        }

        let response: ChatCompletionResponse = response.json().await?;

        response.choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| Error::ParseError("No response from OpenAI".to_string()))
    }
}

/// Chat message for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// Create a system message
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }

    /// Create a user message
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    /// Create an assistant message
    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

/// Bybit API client for cryptocurrency trading
pub struct BybitClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    api_secret: Option<String>,
}

impl BybitClient {
    /// Create new Bybit client (public endpoints only)
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
        }
    }

    /// Create with API credentials for authenticated endpoints
    pub fn with_credentials(api_key: &str, api_secret: &str) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            base_url: "https://api.bybit.com".to_string(),
            api_key: Some(api_key.to_string()),
            api_secret: Some(api_secret.to_string()),
        }
    }

    /// Use testnet
    pub fn testnet(mut self) -> Self {
        self.base_url = "https://api-testnet.bybit.com".to_string();
        self
    }

    /// Get server time
    pub async fn server_time(&self) -> Result<i64> {
        let url = format!("{}/v5/market/time", self.base_url);
        let response: serde_json::Value = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        response["result"]["timeSecond"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| Error::ParseError("Failed to parse server time".to_string()))
    }

    /// Get ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: serde_json::Value = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        let ret_code = response["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            let msg = response["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(Error::ApiError(format!("Bybit API error: {}", msg)));
        }

        let list = response["result"]["list"]
            .as_array()
            .ok_or_else(|| Error::ParseError("No ticker data".to_string()))?;

        let ticker = list.first()
            .ok_or_else(|| Error::ParseError("Empty ticker list".to_string()))?;

        Ok(TickerInfo {
            symbol: ticker["symbol"].as_str().unwrap_or("").to_string(),
            last_price: ticker["lastPrice"].as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            bid_price: ticker["bid1Price"].as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            ask_price: ticker["ask1Price"].as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            volume_24h: ticker["volume24h"].as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            price_change_24h: ticker["price24hPcnt"].as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
        })
    }

    /// Get orderbook
    pub async fn get_orderbook(&self, symbol: &str, limit: u32) -> Result<Orderbook> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: serde_json::Value = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        let ret_code = response["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            let msg = response["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(Error::ApiError(format!("Bybit API error: {}", msg)));
        }

        let result = &response["result"];

        let parse_levels = |key: &str| -> Vec<OrderbookLevel> {
            result[key]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|level| {
                            let arr = level.as_array()?;
                            Some(OrderbookLevel {
                                price: arr.get(0)?.as_str()?.parse().ok()?,
                                quantity: arr.get(1)?.as_str()?.parse().ok()?,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        Ok(Orderbook {
            symbol: result["s"].as_str().unwrap_or("").to_string(),
            bids: parse_levels("b"),
            asks: parse_levels("a"),
            timestamp: result["ts"].as_i64().unwrap_or(0),
        })
    }

    /// Get recent trades
    pub async fn get_recent_trades(&self, symbol: &str, limit: u32) -> Result<Vec<TradeRecord>> {
        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: serde_json::Value = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        let ret_code = response["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            let msg = response["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(Error::ApiError(format!("Bybit API error: {}", msg)));
        }

        let list = response["result"]["list"]
            .as_array()
            .ok_or_else(|| Error::ParseError("No trade data".to_string()))?;

        let trades = list
            .iter()
            .filter_map(|t| {
                Some(TradeRecord {
                    exec_id: t["execId"].as_str()?.to_string(),
                    symbol: t["symbol"].as_str()?.to_string(),
                    price: t["price"].as_str()?.parse().ok()?,
                    size: t["size"].as_str()?.parse().ok()?,
                    side: t["side"].as_str()?.to_string(),
                    time: t["time"].as_str()?.parse().ok()?,
                })
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

/// Ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub price_change_24h: f64,
}

/// Orderbook data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orderbook {
    pub symbol: String,
    pub bids: Vec<OrderbookLevel>,
    pub asks: Vec<OrderbookLevel>,
    pub timestamp: i64,
}

/// Single orderbook level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub exec_id: String,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub side: String,
    pub time: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");

        let msg = ChatMessage::user("Hello");
        assert_eq!(msg.role, "user");

        let msg = ChatMessage::assistant("Hi there!");
        assert_eq!(msg.role, "assistant");
    }

    #[test]
    fn test_bybit_client_creation() {
        let client = BybitClient::new();
        assert!(client.api_key.is_none());

        let client = BybitClient::with_credentials("key", "secret");
        assert!(client.api_key.is_some());
    }
}
