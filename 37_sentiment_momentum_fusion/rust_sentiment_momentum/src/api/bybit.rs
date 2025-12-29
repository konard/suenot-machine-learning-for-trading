//! Bybit API клиент
//!
//! Клиент для получения рыночных данных с криптовалютной биржи Bybit.
//! Использует Bybit API v5.

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Deserialize;
use tracing::{info, warn};

use super::types::{Kline, KlineData};

/// Базовый URL Bybit API
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Базовый URL Bybit Testnet
const BYBIT_TESTNET_BASE: &str = "https://api-testnet.bybit.com";

/// Обёртка ответа Bybit API
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Результат запроса kline
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

/// Клиент Bybit API
#[derive(Debug, Clone)]
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
    /// Создание нового клиента Bybit
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Создание клиента с произвольным базовым URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Создание клиента для тестовой сети
    pub fn testnet() -> Self {
        Self::with_base_url(BYBIT_TESTNET_BASE)
    }

    /// Получение данных свечей (kline/OHLCV)
    ///
    /// # Аргументы
    ///
    /// * `symbol` - Символ торговой пары (например, "BTCUSDT")
    /// * `interval` - Интервал свечей: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    /// * `start_time` - Начальная временная метка в миллисекундах (опционально)
    /// * `end_time` - Конечная временная метка в миллисекундах (опционально)
    /// * `limit` - Количество записей (макс. 1000, по умолчанию 200)
    ///
    /// # Возвращает
    ///
    /// Вектор свечей OHLCV
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: Option<i64>,
        end_time: Option<i64>,
        limit: Option<u32>,
    ) -> Result<Vec<Kline>> {
        let mut params = vec![
            ("category", "linear".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.unwrap_or(200).min(1000).to_string()),
        ];

        if let Some(start) = start_time {
            params.push(("start", start.to_string()));
        }
        if let Some(end) = end_time {
            params.push(("end", end.to_string()));
        }

        let url = format!("{}/v5/market/kline", self.base_url);

        info!(
            "Получение свечей для {} с интервалом {} с Bybit",
            symbol, interval
        );

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json::<BybitResponse<KlineResult>>()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "Ошибка Bybit API: {} (код: {})",
                response.ret_msg,
                response.ret_code
            ));
        }

        let data: Vec<Kline> = response
            .result
            .list
            .iter()
            .filter_map(|kline| {
                if kline.len() >= 7 {
                    Some(Kline::new(
                        kline[0].parse().ok()?,
                        kline[1].parse().ok()?,
                        kline[2].parse().ok()?,
                        kline[3].parse().ok()?,
                        kline[4].parse().ok()?,
                        kline[5].parse().ok()?,
                        kline[6].parse().ok()?,
                    ))
                } else {
                    warn!("Некорректные данные свечи: {:?}", kline);
                    None
                }
            })
            .collect();

        // Bybit возвращает данные в порядке убывания, разворачиваем
        let mut data = data;
        data.reverse();

        info!("Получено {} свечей для {}", data.len(), symbol);
        Ok(data)
    }

    /// Получение исторических данных с пагинацией
    ///
    /// Этот метод обрабатывает пагинацию для получения более 1000 записей.
    pub async fn fetch_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<KlineData> {
        let mut all_data: Vec<Kline> = Vec::new();
        let mut current_end = end_time;
        let limit = 1000u32;

        let interval_ms = Self::interval_to_ms(interval)?;

        loop {
            let data = self
                .fetch_klines(symbol, interval, Some(start_time), Some(current_end), Some(limit))
                .await?;

            if data.is_empty() {
                break;
            }

            let oldest_timestamp = data.first().map(|d| d.timestamp).unwrap_or(start_time);

            // Добавляем новые данные (старше) к существующим
            let mut new_data = data;
            new_data.extend(all_data);
            all_data = new_data;

            // Проверяем, достигли ли начала
            if oldest_timestamp <= start_time {
                break;
            }

            // Сдвигаем временное окно для получения более старых данных
            current_end = oldest_timestamp - interval_ms;

            // Небольшая задержка для избежания rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Удаляем дубликаты и сортируем по времени
        all_data.sort_by_key(|d| d.timestamp);
        all_data.dedup_by_key(|d| d.timestamp);

        // Фильтруем по запрошенному диапазону
        all_data.retain(|d| d.timestamp >= start_time && d.timestamp <= end_time);

        info!(
            "Получено {} свечей для {} за период {} - {}",
            all_data.len(),
            symbol,
            start_time,
            end_time
        );

        Ok(KlineData::new(
            symbol.to_string(),
            interval.to_string(),
            all_data,
        ))
    }

    /// Получение списка доступных торговых пар
    pub async fn get_symbols(&self, category: &str) -> Result<Vec<String>> {
        let url = format!("{}/v5/market/instruments-info", self.base_url);

        let response: serde_json::Value = self
            .client
            .get(&url)
            .query(&[("category", category)])
            .send()
            .await?
            .json()
            .await?;

        let symbols: Vec<String> = response["result"]["list"]
            .as_array()
            .ok_or_else(|| anyhow!("Неверный формат ответа"))?
            .iter()
            .filter_map(|item| item["symbol"].as_str().map(String::from))
            .collect();

        Ok(symbols)
    }

    /// Получение текущей цены
    pub async fn get_ticker(&self, symbol: &str) -> Result<f64> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response: serde_json::Value = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await?
            .json()
            .await?;

        let price = response["result"]["list"][0]["lastPrice"]
            .as_str()
            .ok_or_else(|| anyhow!("Не удалось получить цену"))?
            .parse::<f64>()?;

        Ok(price)
    }

    /// Преобразование интервала в миллисекунды
    fn interval_to_ms(interval: &str) -> Result<i64> {
        let ms = match interval {
            "1" => 60_000,
            "3" => 3 * 60_000,
            "5" => 5 * 60_000,
            "15" => 15 * 60_000,
            "30" => 30 * 60_000,
            "60" => 60 * 60_000,
            "120" => 120 * 60_000,
            "240" => 240 * 60_000,
            "360" => 360 * 60_000,
            "720" => 720 * 60_000,
            "D" => 24 * 60 * 60_000,
            "W" => 7 * 24 * 60 * 60_000,
            "M" => 30 * 24 * 60 * 60_000,
            _ => return Err(anyhow!("Неверный интервал: {}", interval)),
        };
        Ok(ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_to_ms() {
        assert_eq!(BybitClient::interval_to_ms("1").unwrap(), 60_000);
        assert_eq!(BybitClient::interval_to_ms("60").unwrap(), 3_600_000);
        assert_eq!(BybitClient::interval_to_ms("D").unwrap(), 86_400_000);
    }

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert!(client.base_url.contains("bybit.com"));

        let testnet = BybitClient::testnet();
        assert!(testnet.base_url.contains("testnet"));
    }
}
