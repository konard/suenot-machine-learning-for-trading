//! Клиент для работы с Bybit API
//!
//! Этот модуль предоставляет функции для загрузки исторических данных
//! с криптовалютной биржи Bybit.

use super::types::{Candle, PriceSeries};
use anyhow::{Context, Result};
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde::Deserialize;
use std::collections::HashMap;

/// Базовый URL для Bybit API v5
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Клиент для Bybit API
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

/// Ответ API Bybit
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
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

/// Информация о тикере
#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
    list: Vec<TickerInfo>,
}

/// Информация о тикере
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct TickerInfo {
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

impl BybitClient {
    /// Создать новый клиент
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Создать клиент с кастомным URL (для тестирования)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Получить исторические свечи (klines)
    ///
    /// # Аргументы
    /// * `symbol` - Символ торговой пары (например, "BTCUSDT")
    /// * `interval` - Временной интервал ("1", "5", "15", "60", "240", "D", "W")
    /// * `start` - Начальная дата
    /// * `end` - Конечная дата
    /// * `limit` - Максимальное количество свечей (по умолчанию 200, макс 1000)
    ///
    /// # Интервалы
    /// - "1" = 1 минута
    /// - "5" = 5 минут
    /// - "15" = 15 минут
    /// - "60" = 1 час
    /// - "240" = 4 часа
    /// - "D" = 1 день
    /// - "W" = 1 неделя
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: Option<u32>,
    ) -> Result<PriceSeries> {
        let mut params = HashMap::new();
        params.insert("category", "spot".to_string());
        params.insert("symbol", symbol.to_string());
        params.insert("interval", interval.to_string());

        if let Some(start_time) = start {
            params.insert("start", (start_time.timestamp_millis()).to_string());
        }

        if let Some(end_time) = end {
            params.insert("end", (end_time.timestamp_millis()).to_string());
        }

        params.insert("limit", limit.unwrap_or(200).to_string());

        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await
            .context("Failed to send request to Bybit API")?;

        let response_text = response.text().await?;

        let bybit_response: BybitResponse<KlineResult> =
            serde_json::from_str(&response_text).context("Failed to parse Bybit response")?;

        if bybit_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} (code: {})",
                bybit_response.ret_msg,
                bybit_response.ret_code
            );
        }

        let mut series = PriceSeries::new(symbol.to_string(), interval.to_string());

        // Bybit возвращает данные в обратном порядке (новые первые)
        for kline in bybit_response.result.list.iter().rev() {
            if kline.len() >= 6 {
                let timestamp_ms: i64 = kline[0].parse().unwrap_or(0);
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).unwrap();

                let candle = Candle {
                    timestamp,
                    open: kline[1].parse().unwrap_or(0.0),
                    high: kline[2].parse().unwrap_or(0.0),
                    low: kline[3].parse().unwrap_or(0.0),
                    close: kline[4].parse().unwrap_or(0.0),
                    volume: kline[5].parse().unwrap_or(0.0),
                    turnover: kline.get(6).and_then(|v| v.parse().ok()),
                };

                series.push(candle);
            }
        }

        Ok(series)
    }

    /// Получить все свечи за период (с пагинацией)
    ///
    /// Bybit ограничивает количество свечей в одном запросе (1000),
    /// эта функция автоматически делает несколько запросов.
    pub async fn get_all_klines(
        &self,
        symbol: &str,
        interval: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<PriceSeries> {
        let mut all_series = PriceSeries::new(symbol.to_string(), interval.to_string());
        let mut current_end = end;

        let interval_duration = match interval {
            "1" => Duration::minutes(1),
            "5" => Duration::minutes(5),
            "15" => Duration::minutes(15),
            "60" => Duration::hours(1),
            "240" => Duration::hours(4),
            "D" => Duration::days(1),
            "W" => Duration::weeks(1),
            _ => Duration::hours(1),
        };

        loop {
            let series = self
                .get_klines(symbol, interval, Some(start), Some(current_end), Some(1000))
                .await?;

            if series.is_empty() {
                break;
            }

            let oldest_timestamp = series.first().unwrap().timestamp;

            // Добавляем свечи (в обратном порядке, чтобы сохранить хронологию)
            for candle in series.candles {
                all_series.candles.insert(0, candle);
            }

            // Проверяем, достигли ли начала
            if oldest_timestamp <= start {
                break;
            }

            // Следующий запрос с более ранней датой
            current_end = oldest_timestamp - interval_duration;

            // Небольшая задержка для соблюдения rate limits
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Сортируем по времени
        all_series.sort_by_time();

        // Удаляем дубликаты
        all_series.candles.dedup_by(|a, b| a.timestamp == b.timestamp);

        Ok(all_series)
    }

    /// Получить текущие тикеры для нескольких символов
    pub async fn get_tickers(&self, symbols: &[&str]) -> Result<HashMap<String, TickerInfo>> {
        let mut result = HashMap::new();

        // Bybit не поддерживает массовый запрос тикеров, делаем по одному
        // Но можно запросить все тикеры и отфильтровать
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "spot")])
            .send()
            .await
            .context("Failed to send request to Bybit API")?;

        let bybit_response: BybitResponse<TickerResult> = response
            .json()
            .await
            .context("Failed to parse Bybit response")?;

        if bybit_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} (code: {})",
                bybit_response.ret_msg,
                bybit_response.ret_code
            );
        }

        for ticker in bybit_response.result.list {
            if symbols.contains(&ticker.symbol.as_str()) {
                result.insert(ticker.symbol.clone(), ticker);
            }
        }

        Ok(result)
    }

    /// Получить 24-часовое изменение цены
    pub async fn get_24h_change(&self, symbol: &str) -> Result<f64> {
        let tickers = self.get_tickers(&[symbol]).await?;

        tickers
            .get(symbol)
            .and_then(|t| t.price_24h_pcnt.parse::<f64>().ok())
            .ok_or_else(|| anyhow::anyhow!("Symbol {} not found", symbol))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Список популярных криптовалютных пар на Bybit
pub const CRYPTO_UNIVERSE: &[&str] = &[
    "BTCUSDT",    // Bitcoin
    "ETHUSDT",    // Ethereum
    "SOLUSDT",    // Solana
    "BNBUSDT",    // Binance Coin
    "XRPUSDT",    // Ripple
    "ADAUSDT",    // Cardano
    "AVAXUSDT",   // Avalanche
    "DOTUSDT",    // Polkadot
    "MATICUSDT",  // Polygon
    "LINKUSDT",   // Chainlink
    "ATOMUSDT",   // Cosmos
    "LTCUSDT",    // Litecoin
    "UNIUSDT",    // Uniswap
    "NEARUSDT",   // Near Protocol
    "APTUSDT",    // Aptos
    "ARBUSDT",    // Arbitrum
    "OPUSDT",     // Optimism
    "INJUSDT",    // Injective
    "SUIUSDT",    // Sui
    "SEIUSDT",    // Sei
];

/// Получить рекомендуемую вселенную активов для моментум стратегии
pub fn get_momentum_universe() -> Vec<&'static str> {
    // Отбираем наиболее ликвидные активы
    CRYPTO_UNIVERSE[..12].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_universe() {
        let universe = get_momentum_universe();
        assert!(!universe.is_empty());
        assert!(universe.contains(&"BTCUSDT"));
        assert!(universe.contains(&"ETHUSDT"));
    }
}
