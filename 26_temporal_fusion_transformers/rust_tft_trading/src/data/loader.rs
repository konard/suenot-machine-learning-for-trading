//! Data Loader
//!
//! Модуль для загрузки и подготовки данных из различных источников.

use crate::api::{BybitClient, Kline};
use super::{Dataset, FeatureExtractor, Features, TimeSeriesDataset, TimeSeriesDatasetConfig};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Конфигурация загрузчика данных
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    /// Символ торговой пары
    pub symbol: String,

    /// Интервал свечей
    pub interval: String,

    /// Количество свечей для загрузки
    pub num_candles: usize,

    /// Длина encoder context
    pub encoder_length: usize,

    /// Длина prediction horizon
    pub prediction_length: usize,

    /// Использовать ли тестовую сеть Bybit
    pub use_testnet: bool,

    /// Задержка между запросами (ms)
    pub request_delay_ms: u64,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            interval: "1h".to_string(),
            num_candles: 5000,
            encoder_length: 168,
            prediction_length: 24,
            use_testnet: false,
            request_delay_ms: 100,
        }
    }
}

impl DataLoaderConfig {
    /// Создает конфигурацию для Bitcoin
    pub fn btc_hourly() -> Self {
        Self::default()
    }

    /// Создает конфигурацию для Ethereum
    pub fn eth_hourly() -> Self {
        Self {
            symbol: "ETHUSDT".to_string(),
            ..Default::default()
        }
    }

    /// Создает конфигурацию для дневных данных
    pub fn daily(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            interval: "1d".to_string(),
            num_candles: 1000,
            encoder_length: 30, // 30 дней
            prediction_length: 7, // 7 дней
            ..Default::default()
        }
    }

    /// Создает конфигурацию для 4-часовых данных
    pub fn four_hourly(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            interval: "4h".to_string(),
            num_candles: 3000,
            encoder_length: 42, // 7 дней (6 свечей в день)
            prediction_length: 6, // 24 часа
            ..Default::default()
        }
    }
}

/// Загрузчик данных
pub struct DataLoader {
    /// Конфигурация
    config: DataLoaderConfig,

    /// Экстрактор признаков
    feature_extractor: FeatureExtractor,

    /// Клиент Bybit (опционально)
    client: Option<BybitClient>,
}

impl DataLoader {
    /// Создает новый загрузчик с конфигурацией по умолчанию
    pub fn new() -> Self {
        Self::with_config(DataLoaderConfig::default())
    }

    /// Создает загрузчик с заданной конфигурацией
    pub fn with_config(config: DataLoaderConfig) -> Self {
        let client = if config.use_testnet {
            Some(BybitClient::with_testnet())
        } else {
            Some(BybitClient::new())
        };

        Self {
            config,
            feature_extractor: FeatureExtractor::default(),
            client,
        }
    }

    /// Устанавливает экстрактор признаков
    pub fn with_feature_extractor(mut self, extractor: FeatureExtractor) -> Self {
        self.feature_extractor = extractor;
        self
    }

    /// Загружает данные с Bybit и создает dataset
    pub async fn load_from_bybit(&self) -> Result<Dataset> {
        let client = self.client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Bybit client not initialized"))?;

        log::info!(
            "Loading {} candles for {} ({})",
            self.config.num_candles,
            self.config.symbol,
            self.config.interval
        );

        // Загружаем свечи
        let klines = client
            .get_klines_paginated(
                &self.config.symbol,
                &self.config.interval,
                self.config.num_candles,
                self.config.request_delay_ms,
            )
            .await?;

        log::info!("Loaded {} candles", klines.len());

        // Создаем dataset
        let dataset = self.prepare_dataset(&klines)?;

        log::info!("Created dataset with {} samples", dataset.len());

        Ok(dataset)
    }

    /// Загружает данные из CSV файла
    pub fn load_from_csv<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let mut reader = csv::Reader::from_path(path)?;

        let mut klines = Vec::new();

        for result in reader.records() {
            let record = result?;

            // Ожидаемый формат: timestamp,open,high,low,close,volume,turnover
            if record.len() >= 7 {
                let kline = Kline {
                    open_time: record[0].parse()?,
                    open: record[1].parse()?,
                    high: record[2].parse()?,
                    low: record[3].parse()?,
                    close: record[4].parse()?,
                    volume: record[5].parse()?,
                    turnover: record[6].parse()?,
                };
                klines.push(kline);
            }
        }

        log::info!("Loaded {} candles from CSV", klines.len());

        self.prepare_dataset(&klines)
    }

    /// Подготавливает dataset из свечей
    pub fn prepare_dataset(&self, klines: &[Kline]) -> Result<Dataset> {
        if klines.is_empty() {
            return Err(anyhow::anyhow!("No klines provided"));
        }

        // Извлекаем признаки
        let mut features = self.feature_extractor.extract(klines);

        // Нормализуем признаки
        features.normalize();

        // Создаем конфигурацию dataset
        let dataset_config = TimeSeriesDatasetConfig {
            encoder_length: self.config.encoder_length,
            prediction_length: self.config.prediction_length,
            target_idx: 4, // returns
            known_future_indices: vec![20, 21, 22, 23], // временные признаки
            static_indices: vec![],
            step: 1,
        };

        // Создаем TimeSeriesDataset
        let ts_dataset = TimeSeriesDataset::new(features, dataset_config);

        // Конвертируем в Dataset
        Ok(ts_dataset.to_dataset())
    }

    /// Загружает данные для нескольких символов
    pub async fn load_multiple_symbols(&self, symbols: &[&str]) -> Result<Vec<Dataset>> {
        let client = self.client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Bybit client not initialized"))?;

        let mut datasets = Vec::new();

        for symbol in symbols {
            log::info!("Loading data for {}", symbol);

            let klines = client
                .get_klines_paginated(
                    symbol,
                    &self.config.interval,
                    self.config.num_candles,
                    self.config.request_delay_ms,
                )
                .await?;

            if klines.len() >= self.config.encoder_length + self.config.prediction_length {
                let dataset = self.prepare_dataset(&klines)?;
                datasets.push(dataset);
                log::info!("Created dataset for {} with {} samples", symbol, dataset.len());
            } else {
                log::warn!(
                    "Not enough data for {}: {} candles (need {})",
                    symbol,
                    klines.len(),
                    self.config.encoder_length + self.config.prediction_length
                );
            }
        }

        Ok(datasets)
    }

    /// Сохраняет свечи в CSV
    pub fn save_klines_to_csv<P: AsRef<Path>>(&self, klines: &[Kline], path: P) -> Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        // Заголовок
        writer.write_record(&[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ])?;

        for kline in klines {
            writer.write_record(&[
                kline.open_time.to_string(),
                kline.open.to_string(),
                kline.high.to_string(),
                kline.low.to_string(),
                kline.close.to_string(),
                kline.volume.to_string(),
                kline.turnover.to_string(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Возвращает конфигурацию
    pub fn config(&self) -> &DataLoaderConfig {
        &self.config
    }
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_presets() {
        let btc = DataLoaderConfig::btc_hourly();
        assert_eq!(btc.symbol, "BTCUSDT");
        assert_eq!(btc.interval, "1h");

        let eth = DataLoaderConfig::eth_hourly();
        assert_eq!(eth.symbol, "ETHUSDT");

        let daily = DataLoaderConfig::daily("SOLUSDT");
        assert_eq!(daily.symbol, "SOLUSDT");
        assert_eq!(daily.interval, "1d");
    }
}
