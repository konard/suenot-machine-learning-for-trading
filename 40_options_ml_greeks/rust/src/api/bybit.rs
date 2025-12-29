//! # Bybit API Client
//!
//! Клиент для работы с Bybit Options API.
//!
//! ## Возможности
//!
//! - Получение рыночных данных (свечи, тикеры)
//! - Получение опционных цепочек
//! - Исполнение сделок (требует API ключи)
//!
//! ## Пример
//!
//! ```rust,no_run
//! use options_greeks_ml::api::bybit::BybitClient;
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = BybitClient::new_public();
//!     let candles = client.get_klines("BTCUSDT", "1h", 100).await.unwrap();
//! }
//! ```

use super::{ApiError, ApiResult};
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

/// Базовые URL для Bybit API
const MAINNET_URL: &str = "https://api.bybit.com";
const TESTNET_URL: &str = "https://api-testnet.bybit.com";

/// Bybit API клиент
#[derive(Clone)]
pub struct BybitClient {
    /// HTTP клиент
    client: Client,
    /// Базовый URL
    base_url: String,
    /// API ключ (опционально)
    api_key: Option<String>,
    /// API секрет (опционально)
    api_secret: Option<String>,
    /// Время ожидания приёма окна (мс)
    recv_window: u64,
}

impl BybitClient {
    /// Создать публичный клиент (только чтение)
    pub fn new_public() -> Self {
        Self {
            client: Client::new(),
            base_url: MAINNET_URL.to_string(),
            api_key: None,
            api_secret: None,
            recv_window: 5000,
        }
    }

    /// Создать клиент с API ключами
    pub fn new(api_key: String, api_secret: String) -> Self {
        Self {
            client: Client::new(),
            base_url: MAINNET_URL.to_string(),
            api_key: Some(api_key),
            api_secret: Some(api_secret),
            recv_window: 5000,
        }
    }

    /// Создать клиент для тестнета
    pub fn new_testnet(api_key: String, api_secret: String) -> Self {
        Self {
            client: Client::new(),
            base_url: TESTNET_URL.to_string(),
            api_key: Some(api_key),
            api_secret: Some(api_secret),
            recv_window: 5000,
        }
    }

    /// Получить текущий timestamp
    fn timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Создать подпись для запроса
    fn sign(&self, params: &str) -> String {
        let secret = self.api_secret.as_ref().expect("API secret required");
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(params.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Выполнить публичный GET запрос
    async fn get_public<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        params: &[(&str, &str)],
    ) -> ApiResult<T> {
        let url = format!("{}{}", self.base_url, endpoint);

        let response = self
            .client
            .get(&url)
            .query(params)
            .send()
            .await?;

        let body: BybitResponse<T> = response.json().await?;

        if body.ret_code != 0 {
            return Err(ApiError::ApiResponse {
                code: body.ret_code,
                message: body.ret_msg,
            });
        }

        body.result.ok_or_else(|| ApiError::ApiResponse {
            code: -1,
            message: "Empty result".to_string(),
        })
    }

    /// Получить свечи (OHLCV)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> ApiResult<Vec<BybitKline>> {
        let limit_str = limit.to_string();
        let params = [
            ("category", "linear"),
            ("symbol", symbol),
            ("interval", interval),
            ("limit", &limit_str),
        ];

        let response: KlineResponse = self.get_public("/v5/market/kline", &params).await?;
        Ok(response.list)
    }

    /// Получить текущий тикер
    pub async fn get_ticker(&self, symbol: &str) -> ApiResult<BybitTicker> {
        let params = [("category", "linear"), ("symbol", symbol)];

        let response: TickerResponse = self.get_public("/v5/market/tickers", &params).await?;

        response
            .list
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::ApiResponse {
                code: -1,
                message: "Ticker not found".to_string(),
            })
    }

    /// Получить опционную цепочку
    pub async fn get_options_chain(
        &self,
        base_coin: &str,
    ) -> ApiResult<Vec<BybitOption>> {
        let params = [("category", "option"), ("baseCoin", base_coin)];

        let response: OptionsResponse = self.get_public("/v5/market/tickers", &params).await?;
        Ok(response.list)
    }

    /// Получить опционный тикер
    pub async fn get_option_ticker(&self, symbol: &str) -> ApiResult<BybitOption> {
        let params = [("category", "option"), ("symbol", symbol)];

        let response: OptionsResponse = self.get_public("/v5/market/tickers", &params).await?;

        response
            .list
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::ApiResponse {
                code: -1,
                message: "Option not found".to_string(),
            })
    }

    /// Получить инструменты (опционы)
    pub async fn get_option_instruments(
        &self,
        base_coin: &str,
    ) -> ApiResult<Vec<BybitInstrument>> {
        let params = [("category", "option"), ("baseCoin", base_coin)];

        let response: InstrumentsResponse =
            self.get_public("/v5/market/instruments-info", &params).await?;
        Ok(response.list)
    }

    /// Получить историческую волатильность
    pub async fn get_historical_volatility(
        &self,
        base_coin: &str,
        period: u32,
    ) -> ApiResult<Vec<BybitHistoricalVol>> {
        let period_str = period.to_string();
        let params = [
            ("category", "option"),
            ("baseCoin", base_coin),
            ("period", &period_str),
        ];

        let response: HistoricalVolResponse =
            self.get_public("/v5/market/historical-volatility", &params).await?;
        Ok(response.list)
    }
}

// ============= Response Types =============

#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
}

#[derive(Debug, Deserialize)]
struct KlineResponse {
    list: Vec<BybitKline>,
}

#[derive(Debug, Deserialize)]
struct TickerResponse {
    list: Vec<BybitTicker>,
}

#[derive(Debug, Deserialize)]
struct OptionsResponse {
    list: Vec<BybitOption>,
}

#[derive(Debug, Deserialize)]
struct InstrumentsResponse {
    list: Vec<BybitInstrument>,
}

#[derive(Debug, Deserialize)]
struct HistoricalVolResponse {
    list: Vec<BybitHistoricalVol>,
}

// ============= Data Types =============

/// Свеча Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitKline {
    /// Timestamp открытия (мс)
    #[serde(rename = "0")]
    pub open_time: String,
    /// Цена открытия
    #[serde(rename = "1")]
    pub open: String,
    /// Максимум
    #[serde(rename = "2")]
    pub high: String,
    /// Минимум
    #[serde(rename = "3")]
    pub low: String,
    /// Цена закрытия
    #[serde(rename = "4")]
    pub close: String,
    /// Объём
    #[serde(rename = "5")]
    pub volume: String,
    /// Turnover
    #[serde(rename = "6")]
    pub turnover: String,
}

impl BybitKline {
    /// Конвертировать в наш тип Candle
    pub fn to_candle(&self) -> crate::models::Candle {
        use chrono::TimeZone;

        let timestamp_ms: i64 = self.open_time.parse().unwrap_or(0);
        let timestamp = Utc.timestamp_millis_opt(timestamp_ms).unwrap();

        crate::models::Candle {
            timestamp,
            open: self.open.parse().unwrap_or(0.0),
            high: self.high.parse().unwrap_or(0.0),
            low: self.low.parse().unwrap_or(0.0),
            close: self.close.parse().unwrap_or(0.0),
            volume: self.volume.parse().unwrap_or(0.0),
        }
    }
}

/// Тикер Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitTicker {
    /// Символ
    pub symbol: String,
    /// Последняя цена
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    /// Изменение за 24ч (%)
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    /// Максимум за 24ч
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    /// Минимум за 24ч
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    /// Объём за 24ч
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    /// Bid цена
    #[serde(rename = "bid1Price")]
    pub bid_price: String,
    /// Ask цена
    #[serde(rename = "ask1Price")]
    pub ask_price: String,
}

/// Опцион Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitOption {
    /// Символ
    pub symbol: String,
    /// Bid цена
    #[serde(rename = "bid1Price")]
    pub bid_price: String,
    /// Bid размер
    #[serde(rename = "bid1Size")]
    pub bid_size: String,
    /// Ask цена
    #[serde(rename = "ask1Price")]
    pub ask_price: String,
    /// Ask размер
    #[serde(rename = "ask1Size")]
    pub ask_size: String,
    /// Последняя цена
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    /// Mark price
    #[serde(rename = "markPrice")]
    pub mark_price: String,
    /// Подразумеваемая волатильность
    #[serde(rename = "markIv")]
    pub mark_iv: String,
    /// Дельта
    pub delta: String,
    /// Гамма
    pub gamma: String,
    /// Тета
    pub theta: String,
    /// Вега
    pub vega: String,
    /// Открытый интерес
    #[serde(rename = "openInterest")]
    pub open_interest: String,
    /// Объём за 24ч
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
}

impl BybitOption {
    /// Получить IV как число
    pub fn iv(&self) -> f64 {
        self.mark_iv.parse().unwrap_or(0.0)
    }

    /// Получить дельту как число
    pub fn delta_f64(&self) -> f64 {
        self.delta.parse().unwrap_or(0.0)
    }

    /// Получить гамму как число
    pub fn gamma_f64(&self) -> f64 {
        self.gamma.parse().unwrap_or(0.0)
    }

    /// Получить тету как число
    pub fn theta_f64(&self) -> f64 {
        self.theta.parse().unwrap_or(0.0)
    }

    /// Получить вегу как число
    pub fn vega_f64(&self) -> f64 {
        self.vega.parse().unwrap_or(0.0)
    }

    /// Получить mid price
    pub fn mid_price(&self) -> f64 {
        let bid: f64 = self.bid_price.parse().unwrap_or(0.0);
        let ask: f64 = self.ask_price.parse().unwrap_or(0.0);
        (bid + ask) / 2.0
    }
}

/// Инструмент Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitInstrument {
    /// Символ
    pub symbol: String,
    /// Базовая монета
    #[serde(rename = "baseCoin")]
    pub base_coin: String,
    /// Статус
    pub status: String,
    /// Дата поставки
    #[serde(rename = "deliveryTime")]
    pub delivery_time: String,
    /// Тип опциона (Call/Put)
    #[serde(rename = "optionsType")]
    pub options_type: String,
}

impl BybitInstrument {
    /// Это call опцион?
    pub fn is_call(&self) -> bool {
        self.options_type == "Call"
    }

    /// Это put опцион?
    pub fn is_put(&self) -> bool {
        self.options_type == "Put"
    }

    /// Парсинг символа для получения параметров опциона
    /// Формат: BTC-31DEC24-100000-C
    pub fn parse_symbol(&self) -> Option<ParsedOptionSymbol> {
        let parts: Vec<&str> = self.symbol.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        let base = parts[0].to_string();
        let expiry = parts[1].to_string();
        let strike: f64 = parts[2].parse().ok()?;
        let option_type = match parts[3] {
            "C" => crate::greeks::OptionType::Call,
            "P" => crate::greeks::OptionType::Put,
            _ => return None,
        };

        Some(ParsedOptionSymbol {
            base,
            expiry,
            strike,
            option_type,
        })
    }
}

/// Распарсенный символ опциона
#[derive(Debug, Clone)]
pub struct ParsedOptionSymbol {
    pub base: String,
    pub expiry: String,
    pub strike: f64,
    pub option_type: crate::greeks::OptionType,
}

/// Историческая волатильность Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitHistoricalVol {
    /// Период (дни)
    pub period: u32,
    /// Значение волатильности
    pub value: String,
    /// Timestamp
    pub time: String,
}

impl BybitHistoricalVol {
    /// Получить волатильность как число
    pub fn volatility(&self) -> f64 {
        self.value.parse().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_option_symbol() {
        let instrument = BybitInstrument {
            symbol: "BTC-31DEC24-100000-C".to_string(),
            base_coin: "BTC".to_string(),
            status: "Trading".to_string(),
            delivery_time: "1735632000000".to_string(),
            options_type: "Call".to_string(),
        };

        let parsed = instrument.parse_symbol().unwrap();
        assert_eq!(parsed.base, "BTC");
        assert_eq!(parsed.strike, 100000.0);
        assert!(matches!(parsed.option_type, crate::greeks::OptionType::Call));
    }

    #[tokio::test]
    #[ignore] // Требует сетевого подключения
    async fn test_get_ticker() {
        let client = BybitClient::new_public();
        let ticker = client.get_ticker("BTCUSDT").await;
        assert!(ticker.is_ok());
    }
}
