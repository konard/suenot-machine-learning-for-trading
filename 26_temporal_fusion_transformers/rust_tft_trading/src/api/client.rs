//! Bybit API Client
//!
//! Клиент для работы с публичным API биржи Bybit.

use super::types::*;
use reqwest::Client;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

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
    /// Создает новый клиент для основной сети
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Создает клиент для тестовой сети
    pub fn with_testnet() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Создает клиент с кастомным URL
    pub fn with_url(base_url: &str) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: base_url.to_string(),
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
    /// use tft_trading::BybitClient;
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

        let response: ApiResponse<KlineResult> = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

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

        // Bybit возвращает данные в обратном хронологическом порядке
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Получает исторические свечи за определенный временной диапазон
    ///
    /// # Аргументы
    /// * `symbol` - Торговая пара
    /// * `interval` - Интервал свечей
    /// * `start_time` - Начальное время (Unix timestamp в миллисекундах)
    /// * `end_time` - Конечное время (Unix timestamp в миллисекундах)
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

        let response: ApiResponse<KlineResult> = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

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

    /// Получает большой объем исторических данных с пагинацией
    ///
    /// # Аргументы
    /// * `symbol` - Торговая пара
    /// * `interval` - Интервал свечей
    /// * `total_candles` - Общее количество свечей для загрузки
    /// * `delay_ms` - Задержка между запросами (для избежания rate limit)
    pub async fn get_klines_paginated(
        &self,
        symbol: &str,
        interval: &str,
        total_candles: usize,
        delay_ms: u64,
    ) -> Result<Vec<Kline>> {
        let mut all_klines = Vec::new();
        let mut end_time: Option<i64> = None;

        let interval_ms = self.interval_to_ms(interval)?;

        while all_klines.len() < total_candles {
            let batch_size = (total_candles - all_klines.len()).min(1000) as u32;

            let klines = if let Some(end) = end_time {
                // Получаем данные до определенного времени
                let start = end - (batch_size as i64 * interval_ms);
                self.get_klines_range(symbol, interval, start, end).await?
            } else {
                // Первый запрос - последние свечи
                self.get_klines(symbol, interval, batch_size).await?
            };

            if klines.is_empty() {
                break;
            }

            // Запоминаем время для следующей итерации
            end_time = Some(klines.first().unwrap().open_time - 1);

            // Добавляем в начало (т.к. идем назад во времени)
            let mut new_klines = klines;
            new_klines.extend(all_klines);
            all_klines = new_klines;

            // Задержка между запросами
            if delay_ms > 0 {
                sleep(Duration::from_millis(delay_ms)).await;
            }

            log::debug!(
                "Loaded {} candles for {}, total: {}",
                batch_size,
                symbol,
                all_klines.len()
            );
        }

        // Обрезаем до нужного количества
        if all_klines.len() > total_candles {
            all_klines = all_klines[all_klines.len() - total_candles..].to_vec();
        }

        Ok(all_klines)
    }

    /// Преобразует интервал в миллисекунды
    fn interval_to_ms(&self, interval: &str) -> Result<i64> {
        let ms = match interval {
            "1m" | "1" => 60_000,
            "3m" | "3" => 180_000,
            "5m" | "5" => 300_000,
            "15m" | "15" => 900_000,
            "30m" | "30" => 1_800_000,
            "1h" | "60" => 3_600_000,
            "2h" | "120" => 7_200_000,
            "4h" | "240" => 14_400_000,
            "6h" | "360" => 21_600_000,
            "12h" | "720" => 43_200_000,
            "1d" | "D" => 86_400_000,
            "1w" | "W" => 604_800_000,
            _ => return Err(BybitError::InvalidInterval(interval.to_string())),
        };
        Ok(ms)
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

        let response: ApiResponse<OrderBookResult> = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

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

    /// Получает тикер для одного символа
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let mut params = HashMap::new();
        params.insert("category", "spot");
        params.insert("symbol", symbol);

        let response: ApiResponse<TickerResult> = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

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

        Ok(parse_ticker_data(ticker_data))
    }

    /// Получает тикеры для всех торговых пар
    pub async fn get_all_tickers(&self) -> Result<Vec<Ticker>> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let mut params = HashMap::new();
        params.insert("category", "spot");

        let response: ApiResponse<TickerResult> = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

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
            .map(parse_ticker_data)
            .collect();

        Ok(tickers)
    }

    /// Получает список всех доступных торговых пар
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        let tickers = self.get_all_tickers().await?;
        Ok(tickers.into_iter().map(|t| t.symbol).collect())
    }

    /// Получает топ торговых пар по объему
    pub async fn get_top_symbols_by_volume(&self, count: usize) -> Result<Vec<String>> {
        let mut tickers = self.get_all_tickers().await?;

        // Сортируем по объему торгов (по убыванию)
        tickers.sort_by(|a, b| {
            b.volume_24h
                .partial_cmp(&a.volume_24h)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Берем только USDT пары
        let top_symbols: Vec<String> = tickers
            .into_iter()
            .filter(|t| t.symbol.ends_with("USDT"))
            .take(count)
            .map(|t| t.symbol)
            .collect();

        Ok(top_symbols)
    }
}

/// Вспомогательная функция для парсинга данных тикера
fn parse_ticker_data(data: TickerData) -> Ticker {
    Ticker {
        symbol: data.symbol,
        last_price: data.last_price.parse().unwrap_or(0.0),
        prev_price_24h: data.prev_price_24h.parse().unwrap_or(0.0),
        price_24h_pcnt: data.price_24h_pcnt.parse().unwrap_or(0.0),
        high_price_24h: data.high_price_24h.parse().unwrap_or(0.0),
        low_price_24h: data.low_price_24h.parse().unwrap_or(0.0),
        volume_24h: data.volume_24h.parse().unwrap_or(0.0),
        turnover_24h: data.turnover_24h.parse().unwrap_or(0.0),
        open_interest: data.open_interest.and_then(|s| s.parse().ok()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_parsing() {
        let client = BybitClient::new();

        assert_eq!(client.parse_interval("1m").unwrap(), "1");
        assert_eq!(client.parse_interval("1h").unwrap(), "60");
        assert_eq!(client.parse_interval("1d").unwrap(), "D");
        assert!(client.parse_interval("invalid").is_err());
    }

    #[test]
    fn test_interval_to_ms() {
        let client = BybitClient::new();

        assert_eq!(client.interval_to_ms("1m").unwrap(), 60_000);
        assert_eq!(client.interval_to_ms("1h").unwrap(), 3_600_000);
        assert_eq!(client.interval_to_ms("1d").unwrap(), 86_400_000);
    }
}
