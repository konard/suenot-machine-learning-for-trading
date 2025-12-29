//! Клиент для работы с API Bybit V5

use reqwest::Client;
use tracing::{debug, info, warn};

use super::error::ApiError;
use super::response::*;
use super::{Category, Interval, BYBIT_API_URL};
use crate::data::{Kline, OrderBook, OrderBookLevel, Ticker};

/// Клиент API Bybit
#[derive(Debug, Clone)]
pub struct BybitClient {
    /// HTTP клиент
    client: Client,
    /// Базовый URL
    base_url: String,
    /// Категория по умолчанию
    default_category: Category,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Создать новый клиент с настройками по умолчанию
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: BYBIT_API_URL.to_string(),
            default_category: Category::Linear,
        }
    }

    /// Создать клиент для testnet
    pub fn testnet() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: super::BYBIT_TESTNET_URL.to_string(),
            default_category: Category::Linear,
        }
    }

    /// Установить базовый URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Установить категорию по умолчанию
    pub fn with_category(mut self, category: Category) -> Self {
        self.default_category = category;
        self
    }

    /// Получить свечи (klines)
    ///
    /// # Аргументы
    /// - `symbol` - Символ торговой пары (например, "BTCUSDT")
    /// - `interval` - Интервал свечи
    /// - `limit` - Количество свечей (макс. 1000)
    ///
    /// # Пример
    /// ```rust,no_run
    /// use alpha_factors::{BybitClient, api::Interval};
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let client = BybitClient::new();
    ///     let klines = client.get_klines_with_interval("BTCUSDT", Interval::Hour1, 100).await?;
    ///     println!("Последняя свеча: close = {}", klines.last().unwrap().close);
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_klines_with_interval(
        &self,
        symbol: &str,
        interval: Interval,
        limit: u32,
    ) -> Result<Vec<Kline>, ApiError> {
        self.get_klines(symbol, interval.as_str(), limit).await
    }

    /// Получить свечи с интервалом в виде строки
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, ApiError> {
        let limit = limit.min(1000);

        let url = format!("{}/v5/market/kline", self.base_url);

        debug!(symbol, interval, limit, "Fetching klines");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let data: BybitResponse<KlineResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        // Bybit возвращает свечи в обратном порядке (новые первыми)
        let mut klines: Vec<Kline> = data
            .result
            .list
            .into_iter()
            .map(|k| {
                Kline::new(
                    k.timestamp(),
                    symbol,
                    interval,
                    k.open(),
                    k.high(),
                    k.low(),
                    k.close(),
                    k.volume(),
                    k.turnover(),
                )
            })
            .collect();

        // Сортируем по времени (старые первыми)
        klines.reverse();

        info!(symbol, count = klines.len(), "Fetched klines");

        Ok(klines)
    }

    /// Получить свечи за указанный период времени
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Kline>, ApiError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        debug!(symbol, ?interval, start_time, end_time, "Fetching klines range");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("symbol", symbol),
                ("interval", interval.as_str()),
                ("start", &start_time.to_string()),
                ("end", &end_time.to_string()),
                ("limit", "1000"),
            ])
            .send()
            .await?;

        let data: BybitResponse<KlineResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let mut klines: Vec<Kline> = data
            .result
            .list
            .into_iter()
            .map(|k| {
                Kline::new(
                    k.timestamp(),
                    symbol,
                    interval.as_str(),
                    k.open(),
                    k.high(),
                    k.low(),
                    k.close(),
                    k.volume(),
                    k.turnover(),
                )
            })
            .collect();

        klines.reverse();

        Ok(klines)
    }

    /// Получить тикер для символа
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, ApiError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        debug!(symbol, "Fetching ticker");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("symbol", symbol),
            ])
            .send()
            .await?;

        let data: BybitResponse<TickerResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let ticker_data = data
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::no_data(symbol))?;

        Ok(Ticker {
            symbol: ticker_data.symbol,
            last_price: ticker_data.last_price.parse().unwrap_or(0.0),
            bid_price: ticker_data.bid1_price.parse().unwrap_or(0.0),
            bid_size: ticker_data.bid1_size.parse().unwrap_or(0.0),
            ask_price: ticker_data.ask1_price.parse().unwrap_or(0.0),
            ask_size: ticker_data.ask1_size.parse().unwrap_or(0.0),
            price_change_24h: 0.0, // Вычисляется из процента
            price_change_percent_24h: ticker_data.price_24h_pcnt.parse().unwrap_or(0.0) * 100.0,
            high_24h: ticker_data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: ticker_data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: ticker_data.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: ticker_data.turnover_24h.parse().unwrap_or(0.0),
            timestamp: data.time,
        })
    }

    /// Получить список всех тикеров
    pub async fn get_all_tickers(&self) -> Result<Vec<Ticker>, ApiError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        debug!("Fetching all tickers");

        let response = self
            .client
            .get(&url)
            .query(&[("category", self.default_category.as_str())])
            .send()
            .await?;

        let data: BybitResponse<TickerResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let tickers: Vec<Ticker> = data
            .result
            .list
            .into_iter()
            .map(|t| Ticker {
                symbol: t.symbol,
                last_price: t.last_price.parse().unwrap_or(0.0),
                bid_price: t.bid1_price.parse().unwrap_or(0.0),
                bid_size: t.bid1_size.parse().unwrap_or(0.0),
                ask_price: t.ask1_price.parse().unwrap_or(0.0),
                ask_size: t.ask1_size.parse().unwrap_or(0.0),
                price_change_24h: 0.0,
                price_change_percent_24h: t.price_24h_pcnt.parse().unwrap_or(0.0) * 100.0,
                high_24h: t.high_price_24h.parse().unwrap_or(0.0),
                low_24h: t.low_price_24h.parse().unwrap_or(0.0),
                volume_24h: t.volume_24h.parse().unwrap_or(0.0),
                turnover_24h: t.turnover_24h.parse().unwrap_or(0.0),
                timestamp: data.time,
            })
            .collect();

        info!(count = tickers.len(), "Fetched all tickers");

        Ok(tickers)
    }

    /// Получить стакан заявок
    pub async fn get_orderbook(&self, symbol: &str, depth: u32) -> Result<OrderBook, ApiError> {
        let depth = depth.min(500);
        let url = format!("{}/v5/market/orderbook", self.base_url);

        debug!(symbol, depth, "Fetching orderbook");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("symbol", symbol),
                ("limit", &depth.to_string()),
            ])
            .send()
            .await?;

        let data: BybitResponse<OrderBookResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let result = data.result;

        let bids: Vec<OrderBookLevel> = result
            .bids
            .into_iter()
            .map(|[price, size]| {
                OrderBookLevel::new(
                    price.parse().unwrap_or(0.0),
                    size.parse().unwrap_or(0.0),
                )
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .asks
            .into_iter()
            .map(|[price, size]| {
                OrderBookLevel::new(
                    price.parse().unwrap_or(0.0),
                    size.parse().unwrap_or(0.0),
                )
            })
            .collect();

        Ok(OrderBook::new(symbol, bids, asks, result.timestamp))
    }

    /// Получить список доступных инструментов
    pub async fn get_instruments(&self) -> Result<Vec<InstrumentInfo>, ApiError> {
        let url = format!("{}/v5/market/instruments-info", self.base_url);

        debug!("Fetching instruments");

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", self.default_category.as_str()),
                ("limit", "1000"),
            ])
            .send()
            .await?;

        let data: BybitResponse<InstrumentsResult> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let instruments = data.result.list;

        info!(count = instruments.len(), "Fetched instruments");

        Ok(instruments)
    }

    /// Получить текущее серверное время
    pub async fn get_server_time(&self) -> Result<i64, ApiError> {
        let url = format!("{}/v5/market/time", self.base_url);

        let response = self.client.get(&url).send().await?;

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct TimeResponse {
            time_second: String,
            time_nano: String,
        }

        let data: BybitResponse<TimeResponse> = response.json().await?;

        if !data.is_ok() {
            return Err(ApiError::bybit_error(data.ret_code, data.ret_msg));
        }

        let time_ms = data.result.time_second.parse::<i64>().unwrap_or(0) * 1000;

        Ok(time_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_URL);
    }

    #[test]
    fn test_testnet_client() {
        let client = BybitClient::testnet();
        assert_eq!(client.base_url, super::super::BYBIT_TESTNET_URL);
    }

    #[test]
    fn test_with_category() {
        let client = BybitClient::new().with_category(Category::Spot);
        assert_eq!(client.default_category, Category::Spot);
    }
}
