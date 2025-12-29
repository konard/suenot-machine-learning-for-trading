//! Клиент для работы с Bybit API
//!
//! Этот модуль предоставляет простой интерфейс для получения
//! рыночных данных с биржи Bybit.

use super::types::{BybitError, BybitKlineResponse, Candle, Interval};
use log::{debug, info};

/// Базовый URL API Bybit
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Клиент для работы с Bybit API
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Создаёт новый клиент Bybit
    ///
    /// # Пример
    ///
    /// ```rust
    /// use crypto_rnn::data::BybitClient;
    ///
    /// let client = BybitClient::new();
    /// ```
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Создаёт клиент с пользовательским URL
    ///
    /// Полезно для тестирования или использования тестовой сети
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Получает исторические свечи (klines) для символа
    ///
    /// # Аргументы
    ///
    /// * `symbol` - Торговая пара (например, "BTCUSDT")
    /// * `interval` - Интервал свечей (например, "1h", "4h", "1d")
    /// * `limit` - Количество свечей (максимум 1000)
    ///
    /// # Пример
    ///
    /// ```rust,no_run
    /// use crypto_rnn::data::BybitClient;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let client = BybitClient::new();
    ///     let candles = client.get_klines("BTCUSDT", "1h", 100).await.unwrap();
    ///     println!("Последняя цена: {}", candles.last().unwrap().close);
    /// }
    /// ```
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Candle>, BybitError> {
        let interval_enum = Interval::from_str(interval)?;
        let limit = limit.min(1000); // Bybit ограничивает до 1000

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.to_api_string(),
            limit
        );

        debug!("Запрос к Bybit API: {}", url);

        let response = self.client.get(&url).send().await?;
        let data: BybitKlineResponse = response.json().await?;

        if data.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: data.ret_code,
                message: data.ret_msg,
            });
        }

        let candles = self.parse_klines(&data.result.list)?;
        info!(
            "Получено {} свечей для {} ({})",
            candles.len(),
            symbol,
            interval
        );

        Ok(candles)
    }

    /// Получает свечи за определённый период времени
    ///
    /// # Аргументы
    ///
    /// * `symbol` - Торговая пара
    /// * `interval` - Интервал свечей
    /// * `start_time` - Начало периода (Unix timestamp в миллисекундах)
    /// * `end_time` - Конец периода (Unix timestamp в миллисекундах)
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Candle>, BybitError> {
        let interval_enum = Interval::from_str(interval)?;

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.to_api_string(),
            start_time,
            end_time
        );

        debug!("Запрос к Bybit API: {}", url);

        let response = self.client.get(&url).send().await?;
        let data: BybitKlineResponse = response.json().await?;

        if data.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: data.ret_code,
                message: data.ret_msg,
            });
        }

        self.parse_klines(&data.result.list)
    }

    /// Получает большое количество исторических свечей
    ///
    /// Автоматически делает несколько запросов для получения
    /// более 1000 свечей.
    ///
    /// # Аргументы
    ///
    /// * `symbol` - Торговая пара
    /// * `interval` - Интервал свечей
    /// * `total_count` - Общее количество свечей
    pub async fn get_klines_bulk(
        &self,
        symbol: &str,
        interval: &str,
        total_count: u32,
    ) -> Result<Vec<Candle>, BybitError> {
        let interval_enum = Interval::from_str(interval)?;
        let mut all_candles = Vec::with_capacity(total_count as usize);
        let mut end_time: Option<i64> = None;

        let batches = (total_count as f64 / 1000.0).ceil() as u32;

        for batch in 0..batches {
            let remaining = total_count - (batch * 1000);
            let limit = remaining.min(1000);

            let url = if let Some(end) = end_time {
                format!(
                    "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}&end={}",
                    self.base_url,
                    symbol.to_uppercase(),
                    interval_enum.to_api_string(),
                    limit,
                    end
                )
            } else {
                format!(
                    "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
                    self.base_url,
                    symbol.to_uppercase(),
                    interval_enum.to_api_string(),
                    limit
                )
            };

            debug!("Пакетный запрос {}/{}: {}", batch + 1, batches, url);

            let response = self.client.get(&url).send().await?;
            let data: BybitKlineResponse = response.json().await?;

            if data.ret_code != 0 {
                return Err(BybitError::ApiError {
                    code: data.ret_code,
                    message: data.ret_msg,
                });
            }

            let candles = self.parse_klines(&data.result.list)?;

            if candles.is_empty() {
                break;
            }

            // Bybit возвращает данные от новых к старым
            // Находим самую раннюю свечу для следующего запроса
            if let Some(oldest) = candles.last() {
                end_time = Some(oldest.timestamp - 1);
            }

            all_candles.extend(candles);

            // Небольшая задержка между запросами
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Сортируем по времени (от старых к новым)
        all_candles.sort_by_key(|c| c.timestamp);

        info!(
            "Всего получено {} свечей для {} ({})",
            all_candles.len(),
            symbol,
            interval
        );

        Ok(all_candles)
    }

    /// Получает список доступных торговых пар
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!("{}/v5/market/instruments-info?category=spot", self.base_url);

        let response = self.client.get(&url).send().await?;
        let data: serde_json::Value = response.json().await?;

        let symbols: Vec<String> = data["result"]["list"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|item| item["symbol"].as_str().map(|s| s.to_string()))
            .collect();

        Ok(symbols)
    }

    /// Парсит сырые данные свечей из API
    fn parse_klines(&self, raw: &[Vec<String>]) -> Result<Vec<Candle>, BybitError> {
        let candles: Vec<Candle> = raw
            .iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Candle::new(
                        row[0].parse().unwrap_or(0),
                        row[1].parse().unwrap_or(0.0),
                        row[2].parse().unwrap_or(0.0),
                        row[3].parse().unwrap_or(0.0),
                        row[4].parse().unwrap_or(0.0),
                        row[5].parse().unwrap_or(0.0),
                        row[6].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        if candles.is_empty() && !raw.is_empty() {
            return Err(BybitError::NoData);
        }

        Ok(candles)
    }
}

/// Синхронный клиент для простых скриптов
pub struct BybitClientSync {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl Default for BybitClientSync {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClientSync {
    /// Создаёт новый синхронный клиент
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Получает свечи синхронно
    pub fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Candle>, BybitError> {
        let interval_enum = Interval::from_str(interval)?;
        let limit = limit.min(1000);

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.to_api_string(),
            limit
        );

        let response = self.client.get(&url).send()?;
        let data: BybitKlineResponse = response.json()?;

        if data.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: data.ret_code,
                message: data.ret_msg,
            });
        }

        self.parse_klines(&data.result.list)
    }

    fn parse_klines(&self, raw: &[Vec<String>]) -> Result<Vec<Candle>, BybitError> {
        let candles: Vec<Candle> = raw
            .iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Candle::new(
                        row[0].parse().unwrap_or(0),
                        row[1].parse().unwrap_or(0.0),
                        row[2].parse().unwrap_or(0.0),
                        row[3].parse().unwrap_or(0.0),
                        row[4].parse().unwrap_or(0.0),
                        row[5].parse().unwrap_or(0.0),
                        row[6].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok(candles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_URL);
    }

    #[tokio::test]
    async fn test_custom_base_url() {
        let client = BybitClient::with_base_url("https://api-testnet.bybit.com");
        assert_eq!(client.base_url, "https://api-testnet.bybit.com");
    }
}
