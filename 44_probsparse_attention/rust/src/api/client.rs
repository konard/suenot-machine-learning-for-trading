//! HTTP клиент для Bybit API

use super::types::{ApiResponse, BybitError, Kline, KlinesResult};
use reqwest::Client;
use std::time::Duration;
use tracing::{debug, info};

/// Базовый URL для Bybit API v5
const BASE_URL: &str = "https://api.bybit.com";

/// Клиент для работы с Bybit API
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Создаёт новый клиент
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BASE_URL.to_string(),
        }
    }

    /// Создаёт клиент с кастомным URL (для тестирования)
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

    /// Получает исторические свечи
    ///
    /// # Arguments
    ///
    /// * `symbol` - Торговая пара (например, "BTCUSDT")
    /// * `interval` - Интервал свечей ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
    /// * `limit` - Количество свечей (макс. 1000)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use informer_probsparse::BybitClient;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let client = BybitClient::new();
    ///     let klines = client.get_klines("BTCUSDT", "60", 100).await.unwrap();
    ///     println!("Fetched {} klines", klines.len());
    /// }
    /// ```
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>, BybitError> {
        self.get_klines_with_time(symbol, interval, limit, None, None).await
    }

    /// Получает исторические свечи с указанием временного диапазона
    pub async fn get_klines_with_time(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        start: Option<u64>,
        end: Option<u64>,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = vec![
            ("category", "linear".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.min(1000).to_string()),
        ];

        if let Some(s) = start {
            params.push(("start", s.to_string()));
        }
        if let Some(e) = end {
            params.push(("end", e.to_string()));
        }

        debug!("Fetching klines for {} interval={} limit={}", symbol, interval, limit);

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

        let api_response: ApiResponse<KlinesResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let result = api_response
            .result
            .ok_or_else(|| BybitError::ParseError("No result in response".to_string()))?;

        let mut klines: Vec<Kline> = result
            .list
            .iter()
            .map(|arr| Kline::from_bybit_array(arr))
            .collect::<Result<Vec<_>, _>>()?;

        // Bybit возвращает данные в обратном порядке (новые первые)
        klines.reverse();

        info!("Fetched {} klines for {}", klines.len(), symbol);

        Ok(klines)
    }

    /// Получает данные для нескольких символов
    pub async fn get_multi_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<Vec<(String, Vec<Kline>)>, BybitError> {
        let mut results = Vec::new();

        // Последовательно для избежания rate limits
        for symbol in symbols {
            let klines = self.get_klines(symbol, interval, limit).await?;
            results.push((symbol.to_string(), klines));

            // Небольшая задержка между запросами
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(results)
    }

    /// Получает больше свечей с пагинацией
    ///
    /// Bybit возвращает максимум 1000 свечей за раз,
    /// эта функция делает несколько запросов для получения большего количества
    pub async fn get_klines_extended(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut all_klines = Vec::new();
        let mut remaining = limit;
        let mut end_time: Option<u64> = None;

        while remaining > 0 {
            let fetch_limit = remaining.min(1000);

            let klines = self.get_klines_with_time(
                symbol,
                interval,
                fetch_limit,
                None,
                end_time,
            ).await?;

            if klines.is_empty() {
                break;
            }

            // Обновляем end_time для следующего запроса
            if let Some(first) = klines.first() {
                end_time = Some(first.timestamp - 1);
            }

            remaining -= klines.len();
            all_klines.extend(klines);

            // Rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Сортируем по времени
        all_klines.sort_by_key(|k| k.timestamp);

        Ok(all_klines)
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

    // Integration tests - uncomment to run against real API
    /*
    #[tokio::test]
    async fn test_get_klines() {
        let client = BybitClient::new();
        let klines = client.get_klines("BTCUSDT", "60", 100).await.unwrap();

        assert!(!klines.is_empty());
        assert!(klines.len() <= 100);

        // Check data is valid
        for k in &klines {
            assert!(k.close > 0.0);
            assert!(k.volume >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_get_multi_klines() {
        let client = BybitClient::new();
        let symbols = &["BTCUSDT", "ETHUSDT"];
        let results = client.get_multi_klines(symbols, "60", 50).await.unwrap();

        assert_eq!(results.len(), 2);
        for (symbol, klines) in &results {
            assert!(!klines.is_empty());
            println!("Got {} klines for {}", klines.len(), symbol);
        }
    }
    */
}
