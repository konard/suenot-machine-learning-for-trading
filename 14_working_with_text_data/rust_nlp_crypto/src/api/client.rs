//! HTTP клиент для Bybit API

use crate::models::{Announcement, BybitConfig, Kline};
use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Клиент для работы с Bybit API
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    config: BybitConfig,
}

impl BybitClient {
    /// Создать новый клиент с настройками по умолчанию
    pub fn new() -> Self {
        Self::with_config(BybitConfig::default())
    }

    /// Создать клиент с пользовательской конфигурацией
    pub fn with_config(config: BybitConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Получить анонсы Bybit
    pub async fn get_announcements(&self, limit: usize) -> Result<Vec<Announcement>> {
        let url = format!("{}/v5/announcements/index", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("locale", "en-US"), ("limit", &limit.to_string())])
            .send()
            .await
            .context("Failed to fetch announcements")?;

        let api_response: BybitAnnouncementResponse = response
            .json()
            .await
            .context("Failed to parse announcements response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} - {}",
                api_response.ret_code,
                api_response.ret_msg
            );
        }

        let announcements = api_response
            .result
            .list
            .into_iter()
            .map(|item| item.into())
            .collect();

        Ok(announcements)
    }

    /// Получить свечи (klines) для символа
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.config.base_url);

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
            .await
            .context("Failed to fetch klines")?;

        let api_response: BybitKlineResponse = response
            .json()
            .await
            .context("Failed to parse klines response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} - {}",
                api_response.ret_code,
                api_response.ret_msg
            );
        }

        let klines = api_response
            .result
            .list
            .into_iter()
            .filter_map(|item| parse_kline_item(&item))
            .collect();

        Ok(klines)
    }

    /// Получить текущую цену символа
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!("{}/v5/market/tickers", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "spot"), ("symbol", symbol)])
            .send()
            .await
            .context("Failed to fetch ticker")?;

        let api_response: BybitTickerResponse = response
            .json()
            .await
            .context("Failed to parse ticker response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} - {}",
                api_response.ret_code,
                api_response.ret_msg
            );
        }

        api_response
            .result
            .list
            .into_iter()
            .next()
            .map(|item| TickerInfo {
                symbol: item.symbol,
                last_price: item.last_price.parse().unwrap_or(0.0),
                price_24h_pcnt: item.price_24h_pcnt.parse().unwrap_or(0.0),
                volume_24h: item.volume_24h.parse().unwrap_or(0.0),
                turnover_24h: item.turnover_24h.parse().unwrap_or(0.0),
            })
            .context("No ticker data found")
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Информация о тикере
#[derive(Debug, Clone)]
pub struct TickerInfo {
    pub symbol: String,
    pub last_price: f64,
    pub price_24h_pcnt: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
}

// ============= API Response Types =============

#[derive(Debug, Deserialize)]
struct BybitAnnouncementResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: AnnouncementResult,
}

#[derive(Debug, Deserialize)]
struct AnnouncementResult {
    list: Vec<AnnouncementItem>,
}

#[derive(Debug, Deserialize)]
struct AnnouncementItem {
    title: String,
    description: String,
    #[serde(rename = "type")]
    type_info: TypeInfo,
    #[serde(rename = "dateTimestamp")]
    date_timestamp: i64,
    url: String,
}

#[derive(Debug, Deserialize)]
struct TypeInfo {
    key: String,
}

impl From<AnnouncementItem> for Announcement {
    fn from(item: AnnouncementItem) -> Self {
        use crate::models::AnnouncementType;

        let announcement_type = match item.type_info.key.as_str() {
            "new_crypto" | "new_listing" => AnnouncementType::NewListing,
            "delisting" => AnnouncementType::Delisting,
            "product_update" => AnnouncementType::ProductUpdate,
            "maintenance" => AnnouncementType::Maintenance,
            "activities" | "promo" => AnnouncementType::Promotion,
            _ => AnnouncementType::Other,
        };

        // Извлечение символов криптовалют из заголовка
        let symbols = extract_crypto_symbols(&item.title);

        Announcement {
            id: format!("{}", item.date_timestamp),
            title: item.title,
            description: item.description,
            announcement_type,
            publish_time: Utc.timestamp_millis_opt(item.date_timestamp).unwrap(),
            symbols,
            url: Some(item.url),
        }
    }
}

/// Извлечение символов криптовалют из текста
fn extract_crypto_symbols(text: &str) -> Vec<String> {
    let common_symbols = [
        "BTC", "ETH", "USDT", "USDC", "BNB", "XRP", "ADA", "DOGE", "SOL", "DOT", "MATIC", "SHIB",
        "LTC", "TRX", "AVAX", "LINK", "ATOM", "UNI", "XLM", "ETC", "APT", "ARB", "OP", "SUI",
    ];

    let upper_text = text.to_uppercase();
    common_symbols
        .iter()
        .filter(|s| upper_text.contains(*s))
        .map(|s| s.to_string())
        .collect()
}

#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: KlineResult,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

fn parse_kline_item(item: &[String]) -> Option<Kline> {
    if item.len() < 7 {
        return None;
    }

    let open_time_ms: i64 = item[0].parse().ok()?;
    let open: f64 = item[1].parse().ok()?;
    let high: f64 = item[2].parse().ok()?;
    let low: f64 = item[3].parse().ok()?;
    let close: f64 = item[4].parse().ok()?;
    let volume: f64 = item[5].parse().ok()?;
    let turnover: f64 = item[6].parse().ok()?;

    Some(Kline {
        open_time: Utc.timestamp_millis_opt(open_time_ms).unwrap(),
        open,
        high,
        low,
        close,
        volume,
        turnover,
    })
}

#[derive(Debug, Deserialize)]
struct BybitTickerResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: TickerResult,
}

#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
struct TickerItem {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_crypto_symbols() {
        let text = "New listing: BTC and ETH pairs available now!";
        let symbols = extract_crypto_symbols(text);
        assert!(symbols.contains(&"BTC".to_string()));
        assert!(symbols.contains(&"ETH".to_string()));
    }

    #[test]
    fn test_default_config() {
        let config = BybitConfig::default();
        assert_eq!(config.base_url, "https://api.bybit.com");
        assert!(config.api_key.is_none());
    }
}
