//! # Bybit API Client
//!
//! Модуль для работы с публичным API биржи Bybit.
//! Поддерживает получение OHLCV данных, стакана заявок и тикеров.

use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Ошибки при работе с Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),
}

/// Результат операции с Bybit API
pub type Result<T> = std::result::Result<T, BybitError>;

/// Свеча (OHLCV данные)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Время открытия (Unix timestamp в миллисекундах)
    pub open_time: i64,
    /// Цена открытия
    pub open: f64,
    /// Максимальная цена
    pub high: f64,
    /// Минимальная цена
    pub low: f64,
    /// Цена закрытия
    pub close: f64,
    /// Объем в базовой валюте
    pub volume: f64,
    /// Объем в котируемой валюте
    pub turnover: f64,
}

impl Kline {
    /// Возвращает время открытия как DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.open_time).unwrap()
    }

    /// Возвращает изменение цены (close - open)
    pub fn price_change(&self) -> f64 {
        self.close - self.open
    }

    /// Возвращает процентное изменение цены
    pub fn price_change_percent(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open * 100.0
        }
    }

    /// Возвращает размах (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Возвращает типичную цену (high + low + close) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Проверяет, является ли свеча бычьей
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Уровень в стакане заявок
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Цена
    pub price: f64,
    /// Размер
    pub size: f64,
}

/// Стакан заявок
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Символ
    pub symbol: String,
    /// Заявки на покупку (bids)
    pub bids: Vec<OrderBookLevel>,
    /// Заявки на продажу (asks)
    pub asks: Vec<OrderBookLevel>,
    /// Время обновления
    pub timestamp: i64,
}

impl OrderBook {
    /// Возвращает лучшую цену покупки
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Возвращает лучшую цену продажи
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Возвращает спред
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Возвращает процентный спред
    pub fn spread_percent(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if bid > 0.0 => Some((ask - bid) / bid * 100.0),
            _ => None,
        }
    }

    /// Возвращает глубину заявок на покупку до определенного уровня
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.size).sum()
    }

    /// Возвращает глубину заявок на продажу до определенного уровня
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.size).sum()
    }

    /// Возвращает дисбаланс стакана (bid_depth - ask_depth) / (bid_depth + ask_depth)
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_depth = self.bid_depth(levels);
        let ask_depth = self.ask_depth(levels);
        let total = bid_depth + ask_depth;
        if total == 0.0 {
            0.0
        } else {
            (bid_depth - ask_depth) / total
        }
    }
}

/// Тикер (текущее состояние рынка)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Символ
    pub symbol: String,
    /// Последняя цена
    pub last_price: f64,
    /// Цена 24ч назад
    pub prev_price_24h: f64,
    /// Изменение цены за 24ч (процент)
    pub price_24h_pcnt: f64,
    /// Максимум за 24ч
    pub high_price_24h: f64,
    /// Минимум за 24ч
    pub low_price_24h: f64,
    /// Объем за 24ч
    pub volume_24h: f64,
    /// Оборот за 24ч
    pub turnover_24h: f64,
    /// Цена открытого интереса
    pub open_interest: Option<f64>,
}

/// Ответ от API Bybit
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Результат запроса свечей
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Результат запроса стакана
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String,
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
    ts: i64,
}

/// Результат запроса тикеров
#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
    list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
struct TickerData {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "prevPrice24h")]
    prev_price_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
    #[serde(rename = "openInterest")]
    open_interest: Option<String>,
}

/// Клиент для работы с Bybit API
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Создает новый клиент с дефолтными настройками
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Создает клиент с кастомным URL (для тестнета)
    pub fn with_testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Преобразует интервал в формат Bybit
    fn parse_interval(&self, interval: &str) -> Result<&str> {
        match interval {
            "1m" | "1" => Ok("1"),
            "3m" | "3" => Ok("3"),
            "5m" | "5" => Ok("5"),
            "15m" | "15" => Ok("15"),
            "30m" | "30" => Ok("30"),
            "1h" | "60" => Ok("60"),
            "2h" | "120" => Ok("120"),
            "4h" | "240" => Ok("240"),
            "6h" | "360" => Ok("360"),
            "12h" | "720" => Ok("720"),
            "1d" | "D" => Ok("D"),
            "1w" | "W" => Ok("W"),
            "1M" | "M" => Ok("M"),
            _ => Err(BybitError::InvalidInterval(interval.to_string())),
        }
    }

    /// Получает исторические свечи (OHLCV)
    ///
    /// # Аргументы
    /// * `symbol` - Торговая пара (например, "BTCUSDT")
    /// * `interval` - Интервал свечей ("1m", "5m", "15m", "1h", "4h", "1d" и т.д.)
    /// * `limit` - Количество свечей (максимум 1000)
    ///
    /// # Пример
    /// ```no_run
    /// use crypto_autoencoders::BybitClient;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let client = BybitClient::new();
    ///     let klines = client.get_klines("BTCUSDT", "1h", 100).await.unwrap();
    ///     println!("Получено {} свечей", klines.len());
    /// }
    /// ```
    pub async fn get_klines(&self, symbol: &str, interval: &str, limit: u32) -> Result<Vec<Kline>> {
        let interval = self.parse_interval(interval)?;
        let limit = limit.min(1000);

        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = HashMap::new();
        params.insert("category", "spot");
        params.insert("symbol", symbol);
        params.insert("interval", interval);

        let limit_str = limit.to_string();
        params.insert("limit", &limit_str);

        let response: ApiResponse<KlineResult> =
            self.client.get(&url).query(&params).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Kline {
                        open_time: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit возвращает данные в обратном порядке (от новых к старым)
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Получает исторические свечи с указанием временного диапазона
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Kline>> {
        let interval = self.parse_interval(interval)?;

        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = HashMap::new();
        params.insert("category", "spot");
        params.insert("symbol", symbol);
        params.insert("interval", interval);

        let start_str = start_time.to_string();
        let end_str = end_time.to_string();
        params.insert("start", &start_str);
        params.insert("end", &end_str);
        params.insert("limit", "1000");

        let response: ApiResponse<KlineResult> =
            self.client.get(&url).query(&params).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Kline {
                        open_time: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Получает стакан заявок
    ///
    /// # Аргументы
    /// * `symbol` - Торговая пара
    /// * `limit` - Глубина стакана (максимум 200)
    pub async fn get_orderbook(&self, symbol: &str, limit: u32) -> Result<OrderBook> {
        let url = format!("{}/v5/market/orderbook", self.base_url);
        let limit = limit.min(200);

        let mut params = HashMap::new();
        params.insert("category", "spot");
        params.insert("symbol", symbol);

        let limit_str = limit.to_string();
        params.insert("limit", &limit_str);

        let response: ApiResponse<OrderBookResult> =
            self.client.get(&url).query(&params).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let bids: Vec<OrderBookLevel> = response
            .result
            .b
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 2 {
                    Some(OrderBookLevel {
                        price: item[0].parse().ok()?,
                        size: item[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = response
            .result
            .a
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 2 {
                    Some(OrderBookLevel {
                        price: item[0].parse().ok()?,
                        size: item[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: response.result.s,
            bids,
            asks,
            timestamp: response.result.ts,
        })
    }

    /// Получает тикер для символа
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let mut params = HashMap::new();
        params.insert("category", "spot");
        params.insert("symbol", symbol);

        let response: ApiResponse<TickerResult> =
            self.client.get(&url).query(&params).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let ticker_data = response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::ParseError("No ticker data".to_string()))?;

        Ok(Ticker {
            symbol: ticker_data.symbol,
            last_price: ticker_data.last_price.parse().unwrap_or(0.0),
            prev_price_24h: ticker_data.prev_price_24h.parse().unwrap_or(0.0),
            price_24h_pcnt: ticker_data.price_24h_pcnt.parse().unwrap_or(0.0),
            high_price_24h: ticker_data.high_price_24h.parse().unwrap_or(0.0),
            low_price_24h: ticker_data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: ticker_data.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: ticker_data.turnover_24h.parse().unwrap_or(0.0),
            open_interest: ticker_data.open_interest.and_then(|s| s.parse().ok()),
        })
    }

    /// Получает тикеры для нескольких символов
    pub async fn get_tickers(&self) -> Result<Vec<Ticker>> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let mut params = HashMap::new();
        params.insert("category", "spot");

        let response: ApiResponse<TickerResult> =
            self.client.get(&url).query(&params).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let tickers: Vec<Ticker> = response
            .result
            .list
            .into_iter()
            .map(|ticker_data| Ticker {
                symbol: ticker_data.symbol,
                last_price: ticker_data.last_price.parse().unwrap_or(0.0),
                prev_price_24h: ticker_data.prev_price_24h.parse().unwrap_or(0.0),
                price_24h_pcnt: ticker_data.price_24h_pcnt.parse().unwrap_or(0.0),
                high_price_24h: ticker_data.high_price_24h.parse().unwrap_or(0.0),
                low_price_24h: ticker_data.low_price_24h.parse().unwrap_or(0.0),
                volume_24h: ticker_data.volume_24h.parse().unwrap_or(0.0),
                turnover_24h: ticker_data.turnover_24h.parse().unwrap_or(0.0),
                open_interest: ticker_data.open_interest.and_then(|s| s.parse().ok()),
            })
            .collect();

        Ok(tickers)
    }

    /// Получает список доступных торговых пар
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        let tickers = self.get_tickers().await?;
        Ok(tickers.into_iter().map(|t| t.symbol).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_methods() {
        let kline = Kline {
            open_time: 1700000000000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert_eq!(kline.price_change(), 5.0);
        assert_eq!(kline.price_change_percent(), 5.0);
        assert_eq!(kline.range(), 15.0);
        assert!((kline.typical_price() - 103.333).abs() < 0.01);
        assert!(kline.is_bullish());
    }

    #[test]
    fn test_orderbook_methods() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 100.0,
                    size: 10.0,
                },
                OrderBookLevel {
                    price: 99.0,
                    size: 20.0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 101.0,
                    size: 15.0,
                },
                OrderBookLevel {
                    price: 102.0,
                    size: 25.0,
                },
            ],
            timestamp: 1700000000000,
        };

        assert_eq!(orderbook.best_bid(), Some(100.0));
        assert_eq!(orderbook.best_ask(), Some(101.0));
        assert_eq!(orderbook.spread(), Some(1.0));
        assert_eq!(orderbook.bid_depth(2), 30.0);
        assert_eq!(orderbook.ask_depth(2), 40.0);
    }
}
