//! Типы ответов API Bybit

use serde::{Deserialize, Serialize};

/// Общая обёртка ответа API Bybit V5
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitResponse<T> {
    /// Код возврата (0 = успех)
    pub ret_code: i32,
    /// Сообщение
    pub ret_msg: String,
    /// Полезная нагрузка
    pub result: T,
    /// Время ответа
    pub time: i64,
}

impl<T> BybitResponse<T> {
    /// Проверить, успешен ли ответ
    pub fn is_ok(&self) -> bool {
        self.ret_code == 0
    }
}

/// Ответ для свечей (klines)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KlineResult {
    /// Символ
    pub symbol: String,
    /// Категория
    pub category: String,
    /// Список свечей
    pub list: Vec<KlineData>,
}

/// Данные одной свечи от Bybit
/// Формат: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
#[derive(Debug, Deserialize)]
pub struct KlineData(
    pub String, // start_time
    pub String, // open
    pub String, // high
    pub String, // low
    pub String, // close
    pub String, // volume
    pub String, // turnover
);

impl KlineData {
    /// Временная метка в миллисекундах
    pub fn timestamp(&self) -> i64 {
        self.0.parse().unwrap_or(0)
    }

    /// Цена открытия
    pub fn open(&self) -> f64 {
        self.1.parse().unwrap_or(0.0)
    }

    /// Максимальная цена
    pub fn high(&self) -> f64 {
        self.2.parse().unwrap_or(0.0)
    }

    /// Минимальная цена
    pub fn low(&self) -> f64 {
        self.3.parse().unwrap_or(0.0)
    }

    /// Цена закрытия
    pub fn close(&self) -> f64 {
        self.4.parse().unwrap_or(0.0)
    }

    /// Объём
    pub fn volume(&self) -> f64 {
        self.5.parse().unwrap_or(0.0)
    }

    /// Оборот
    pub fn turnover(&self) -> f64 {
        self.6.parse().unwrap_or(0.0)
    }
}

/// Ответ для тикеров
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerResult {
    /// Категория
    pub category: String,
    /// Список тикеров
    pub list: Vec<TickerData>,
}

/// Данные тикера от Bybit
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerData {
    /// Символ
    pub symbol: String,
    /// Последняя цена
    #[serde(default)]
    pub last_price: String,
    /// Цена bid
    #[serde(default)]
    pub bid1_price: String,
    /// Объём bid
    #[serde(default)]
    pub bid1_size: String,
    /// Цена ask
    #[serde(default)]
    pub ask1_price: String,
    /// Объём ask
    #[serde(default)]
    pub ask1_size: String,
    /// Изменение за 24ч
    #[serde(default)]
    pub price_24h_pcnt: String,
    /// Максимум за 24ч
    #[serde(default)]
    pub high_price_24h: String,
    /// Минимум за 24ч
    #[serde(default)]
    pub low_price_24h: String,
    /// Объём за 24ч
    #[serde(default)]
    pub volume_24h: String,
    /// Оборот за 24ч
    #[serde(default)]
    pub turnover_24h: String,
}

/// Ответ для стакана заявок
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderBookResult {
    /// Символ
    #[serde(rename = "s")]
    pub symbol: String,
    /// Bids (покупка) - [[price, size], ...]
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    /// Asks (продажа) - [[price, size], ...]
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
    /// Временная метка
    #[serde(rename = "ts")]
    pub timestamp: i64,
    /// Update ID
    #[serde(rename = "u")]
    pub update_id: i64,
}

/// Ответ для списка инструментов
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstrumentsResult {
    /// Категория
    pub category: String,
    /// Список инструментов
    pub list: Vec<InstrumentInfo>,
    /// Курсор для пагинации
    pub next_page_cursor: Option<String>,
}

/// Информация об инструменте
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstrumentInfo {
    /// Символ
    pub symbol: String,
    /// Базовая валюта
    pub base_coin: String,
    /// Котируемая валюта
    pub quote_coin: String,
    /// Статус
    pub status: String,
}
