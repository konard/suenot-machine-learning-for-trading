//! Bybit API client for fetching cryptocurrency market data
//!
//! This module provides functionality to:
//! - Fetch market announcements and news
//! - Get trading pair information
//! - Retrieve market data for analysis

use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Errors that can occur when interacting with Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Authentication failed: {0}")]
    AuthError(String),
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: Option<T>,
    pub time: Option<u64>,
}

/// Announcement from Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Announcement {
    pub title: String,
    pub description: String,
    #[serde(rename = "dateTimestamp")]
    pub date_timestamp: u64,
    pub url: String,
    #[serde(rename = "type")]
    pub announcement_type: Option<AnnouncementType>,
    pub tags: Vec<String>,
}

/// Types of announcements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnouncementType {
    pub key: String,
    pub title: String,
}

/// Announcements list response
#[derive(Debug, Deserialize)]
pub struct AnnouncementList {
    pub total: u32,
    pub list: Vec<Announcement>,
}

/// Trading symbol information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub symbol: String,
    #[serde(rename = "baseCoin")]
    pub base_coin: String,
    #[serde(rename = "quoteCoin")]
    pub quote_coin: String,
    pub status: String,
}

/// Symbols list response
#[derive(Debug, Deserialize)]
pub struct SymbolsList {
    pub category: String,
    pub list: Vec<Symbol>,
}

/// Market ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "prevPrice24h")]
    pub prev_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

/// Ticker list response
#[derive(Debug, Deserialize)]
pub struct TickerList {
    pub category: String,
    pub list: Vec<Ticker>,
}

/// Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub start_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Kline list response
#[derive(Debug, Deserialize)]
pub struct KlineList {
    pub category: String,
    pub symbol: String,
    pub list: Vec<Vec<String>>,
}

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    api_secret: Option<String>,
}

impl BybitClient {
    /// Create a new Bybit client without authentication (public endpoints only)
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
        }
    }

    /// Create a new Bybit client with API authentication
    pub fn with_auth(api_key: String, api_secret: String) -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
            api_key: Some(api_key),
            api_secret: Some(api_secret),
        }
    }

    /// Use testnet instead of mainnet
    pub fn use_testnet(mut self) -> Self {
        self.base_url = "https://api-testnet.bybit.com".to_string();
        self
    }

    /// Generate signature for authenticated requests
    fn sign(&self, timestamp: u64, params: &str) -> Result<String, BybitError> {
        let api_key = self
            .api_key
            .as_ref()
            .ok_or_else(|| BybitError::AuthError("API key not set".to_string()))?;
        let api_secret = self
            .api_secret
            .as_ref()
            .ok_or_else(|| BybitError::AuthError("API secret not set".to_string()))?;

        let recv_window = "5000";
        let sign_str = format!("{}{}{}{}", timestamp, api_key, recv_window, params);

        let mut mac =
            Hmac::<Sha256>::new_from_slice(api_secret.as_bytes()).expect("HMAC can take any size");
        mac.update(sign_str.as_bytes());

        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    /// Get current timestamp in milliseconds
    fn timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64
    }

    /// Fetch announcements from Bybit
    ///
    /// # Arguments
    /// * `locale` - Language code (e.g., "en-US", "ru-RU")
    /// * `limit` - Number of announcements to fetch (max 50)
    ///
    /// # Example
    /// ```no_run
    /// use topic_modeling::api::bybit::BybitClient;
    ///
    /// let client = BybitClient::new();
    /// let announcements = client.get_announcements("en-US", 20).unwrap();
    /// for ann in announcements {
    ///     println!("{}: {}", ann.title, ann.description);
    /// }
    /// ```
    pub fn get_announcements(
        &self,
        locale: &str,
        limit: u32,
    ) -> Result<Vec<Announcement>, BybitError> {
        let url = format!(
            "{}/v5/announcements/index?locale={}&limit={}",
            self.base_url, locale, limit
        );

        let response = self.client.get(&url).send()?;
        let api_response: ApiResponse<AnnouncementList> = response.json()?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        Ok(api_response
            .result
            .map(|r| r.list)
            .unwrap_or_else(Vec::new))
    }

    /// Fetch trading symbols/pairs
    ///
    /// # Arguments
    /// * `category` - Market category: "spot", "linear", "inverse", "option"
    ///
    /// # Example
    /// ```no_run
    /// use topic_modeling::api::bybit::BybitClient;
    ///
    /// let client = BybitClient::new();
    /// let symbols = client.get_symbols("spot").unwrap();
    /// for symbol in symbols {
    ///     println!("{}: {} / {}", symbol.symbol, symbol.base_coin, symbol.quote_coin);
    /// }
    /// ```
    pub fn get_symbols(&self, category: &str) -> Result<Vec<Symbol>, BybitError> {
        let url = format!(
            "{}/v5/market/instruments-info?category={}",
            self.base_url, category
        );

        let response = self.client.get(&url).send()?;
        let api_response: ApiResponse<SymbolsList> = response.json()?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        Ok(api_response
            .result
            .map(|r| r.list)
            .unwrap_or_else(Vec::new))
    }

    /// Fetch market tickers
    ///
    /// # Arguments
    /// * `category` - Market category: "spot", "linear", "inverse", "option"
    /// * `symbol` - Optional specific symbol (e.g., "BTCUSDT")
    pub fn get_tickers(
        &self,
        category: &str,
        symbol: Option<&str>,
    ) -> Result<Vec<Ticker>, BybitError> {
        let mut url = format!("{}/v5/market/tickers?category={}", self.base_url, category);

        if let Some(s) = symbol {
            url.push_str(&format!("&symbol={}", s));
        }

        let response = self.client.get(&url).send()?;
        let api_response: ApiResponse<TickerList> = response.json()?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        Ok(api_response
            .result
            .map(|r| r.list)
            .unwrap_or_else(Vec::new))
    }

    /// Fetch historical kline (candlestick) data
    ///
    /// # Arguments
    /// * `category` - Market category: "spot", "linear", "inverse"
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval: "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Example
    /// ```no_run
    /// use topic_modeling::api::bybit::BybitClient;
    ///
    /// let client = BybitClient::new();
    /// let klines = client.get_klines("spot", "BTCUSDT", "60", 100).unwrap();
    /// for kline in klines {
    ///     println!("Open: {}, Close: {}, Volume: {}", kline.open, kline.close, kline.volume);
    /// }
    /// ```
    pub fn get_klines(
        &self,
        category: &str,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category={}&symbol={}&interval={}&limit={}",
            self.base_url, category, symbol, interval, limit
        );

        let response = self.client.get(&url).send()?;
        let api_response: ApiResponse<KlineList> = response.json()?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let klines = api_response
            .result
            .map(|r| {
                r.list
                    .into_iter()
                    .filter_map(|k| {
                        if k.len() >= 7 {
                            Some(Kline {
                                start_time: k[0].parse().unwrap_or(0),
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
                    .collect()
            })
            .unwrap_or_else(Vec::new);

        Ok(klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Document structure for topic modeling
/// Combines announcements and market context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDocument {
    /// Unique document ID
    pub id: String,
    /// Document title
    pub title: String,
    /// Document content (cleaned text)
    pub content: String,
    /// Document timestamp
    pub timestamp: u64,
    /// Associated symbols (if any)
    pub symbols: Vec<String>,
    /// Document type (announcement, analysis, etc.)
    pub doc_type: String,
    /// Market context at the time
    pub market_context: Option<MarketContext>,
}

/// Market context at a specific point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    /// BTC price at the time
    pub btc_price: f64,
    /// BTC 24h change percentage
    pub btc_change_24h: f64,
    /// Total market volume
    pub total_volume: f64,
}

impl MarketDocument {
    /// Create a new document from an announcement
    pub fn from_announcement(announcement: &Announcement) -> Self {
        // Extract symbols mentioned in the title/description
        let text = format!("{} {}", announcement.title, announcement.description);
        let symbols = extract_crypto_symbols(&text);

        Self {
            id: format!("ann_{}", announcement.date_timestamp),
            title: announcement.title.clone(),
            content: announcement.description.clone(),
            timestamp: announcement.date_timestamp,
            symbols,
            doc_type: announcement
                .announcement_type
                .as_ref()
                .map(|t| t.key.clone())
                .unwrap_or_else(|| "general".to_string()),
            market_context: None,
        }
    }

    /// Add market context to the document
    pub fn with_market_context(mut self, context: MarketContext) -> Self {
        self.market_context = Some(context);
        self
    }

    /// Get full text for topic modeling
    pub fn full_text(&self) -> String {
        format!("{} {}", self.title, self.content)
    }
}

/// Extract cryptocurrency symbol mentions from text
fn extract_crypto_symbols(text: &str) -> Vec<String> {
    let common_symbols = [
        "BTC", "ETH", "USDT", "BNB", "XRP", "ADA", "DOGE", "SOL", "DOT", "MATIC", "SHIB", "LTC",
        "TRX", "AVAX", "LINK", "ATOM", "UNI", "XMR", "ETC", "XLM", "BCH", "ALGO", "VET", "FIL",
        "ICP", "HBAR", "NEAR", "APE", "QNT", "AAVE", "GRT", "FTM", "SAND", "MANA", "THETA", "AXS",
        "EOS", "EGLD", "FLOW", "CHZ", "XTZ", "MKR", "ZEC", "KLAY", "NEO", "CAKE", "BTT", "IOTA",
    ];

    let text_upper = text.to_uppercase();
    common_symbols
        .iter()
        .filter(|&&symbol| {
            // Check if symbol appears as a word boundary
            let pattern = format!(r"\b{}\b", symbol);
            regex::Regex::new(&pattern)
                .map(|re| re.is_match(&text_upper))
                .unwrap_or(false)
        })
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_crypto_symbols() {
        let text = "Bitcoin (BTC) and Ethereum (ETH) are leading cryptocurrencies";
        let symbols = extract_crypto_symbols(text);
        assert!(symbols.contains(&"BTC".to_string()));
        assert!(symbols.contains(&"ETH".to_string()));
    }

    #[test]
    fn test_market_document_from_announcement() {
        let announcement = Announcement {
            title: "BTC Trading Update".to_string(),
            description: "New features for BTC and ETH trading".to_string(),
            date_timestamp: 1700000000000,
            url: "https://bybit.com/announcement".to_string(),
            announcement_type: None,
            tags: vec![],
        };

        let doc = MarketDocument::from_announcement(&announcement);
        assert!(doc.symbols.contains(&"BTC".to_string()));
        assert!(doc.symbols.contains(&"ETH".to_string()));
    }
}
