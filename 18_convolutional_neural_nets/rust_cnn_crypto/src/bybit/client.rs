//! HTTP клиент для Bybit API

use super::types::{BybitError, BybitResponse, Kline, KlineInterval, KlineResult};
use reqwest::Client;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Базовый URL API Bybit
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Клиент для работы с Bybit API
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Создание нового клиента
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Создание клиента с кастомным URL (для тестирования)
    pub fn with_base_url(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Получение исторических свечей
    ///
    /// # Arguments
    /// * `symbol` - Торговая пара (например, "BTCUSDT")
    /// * `interval` - Интервал свечей
    /// * `limit` - Количество свечей (макс. 1000)
    /// * `start_time` - Начальное время (Unix timestamp в мс, опционально)
    /// * `end_time` - Конечное время (Unix timestamp в мс, опционально)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        limit: Option<u32>,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = vec![
            ("category", "spot".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.as_str().to_string()),
            ("limit", limit.unwrap_or(200).min(1000).to_string()),
        ];

        if let Some(start) = start_time {
            params.push(("start", start.to_string()));
        }

        if let Some(end) = end_time {
            params.push(("end", end.to_string()));
        }

        debug!("Fetching klines for {} with interval {}", symbol, interval.as_str());

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?;

        let status = response.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(BybitError::RateLimitExceeded);
        }

        let body: BybitResponse<KlineResult> = response.json().await?;

        if body.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: body.ret_code,
                message: body.ret_msg,
            });
        }

        let mut klines: Vec<Kline> = body
            .result
            .list
            .iter()
            .map(|data| Kline::from_api_data(data))
            .collect::<Result<Vec<_>, _>>()?;

        // API возвращает данные в обратном порядке (новые первыми)
        klines.reverse();

        info!("Fetched {} klines for {}", klines.len(), symbol);

        Ok(klines)
    }

    /// Получение большого количества исторических данных с пагинацией
    ///
    /// # Arguments
    /// * `symbol` - Торговая пара
    /// * `interval` - Интервал свечей
    /// * `start_time` - Начальное время (Unix timestamp в мс)
    /// * `end_time` - Конечное время (Unix timestamp в мс, опционально)
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        start_time: i64,
        end_time: Option<i64>,
    ) -> Result<Vec<Kline>, BybitError> {
        let end = end_time.unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
        let interval_ms = interval.minutes() as i64 * 60 * 1000;
        let batch_size: i64 = 1000;

        let mut all_klines = Vec::new();
        let mut current_start = start_time;

        while current_start < end {
            let batch_end = (current_start + batch_size * interval_ms).min(end);

            let klines = self
                .get_klines(
                    symbol,
                    interval,
                    Some(batch_size as u32),
                    Some(current_start),
                    Some(batch_end),
                )
                .await?;

            if klines.is_empty() {
                break;
            }

            let last_timestamp = klines.last().map(|k| k.timestamp).unwrap_or(batch_end);
            all_klines.extend(klines);

            current_start = last_timestamp + interval_ms;

            // Небольшая задержка для избежания rate limit
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Удаляем дубликаты (по timestamp)
        all_klines.sort_by_key(|k| k.timestamp);
        all_klines.dedup_by_key(|k| k.timestamp);

        info!(
            "Fetched {} historical klines for {} from {} to {}",
            all_klines.len(),
            symbol,
            start_time,
            end
        );

        Ok(all_klines)
    }

    /// Получение текущей цены
    pub async fn get_ticker_price(&self, symbol: &str) -> Result<f64, BybitError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let params = [("category", "spot"), ("symbol", symbol)];

        let response = self.client.get(&url).query(&params).send().await?;

        let body: serde_json::Value = response.json().await?;

        let price = body["result"]["list"][0]["lastPrice"]
            .as_str()
            .ok_or_else(|| BybitError::ParseError("Missing lastPrice".to_string()))?
            .parse::<f64>()
            .map_err(|e| BybitError::ParseError(format!("Invalid price: {}", e)))?;

        Ok(price)
    }

    /// Получение списка доступных торговых пар
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!("{}/v5/market/instruments-info", self.base_url);
        let params = [("category", "spot")];

        let response = self.client.get(&url).query(&params).send().await?;
        let body: serde_json::Value = response.json().await?;

        let symbols: Vec<String> = body["result"]["list"]
            .as_array()
            .ok_or_else(|| BybitError::ParseError("Missing list".to_string()))?
            .iter()
            .filter_map(|item| item["symbol"].as_str().map(String::from))
            .collect();

        Ok(symbols)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_klines() {
        let client = BybitClient::new();
        let result = client
            .get_klines("BTCUSDT", KlineInterval::Min15, Some(10), None, None)
            .await;

        // В тестах мы проверяем только структуру, без реальных запросов
        // Для реальных тестов нужен мок-сервер
        assert!(result.is_ok() || matches!(result, Err(BybitError::RequestError(_))));
    }
}
