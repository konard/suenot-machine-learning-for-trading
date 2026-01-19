//! Data loading utilities
//!
//! Functions for loading and preprocessing trading data.

use crate::api::{BybitClient, KlineData, KlineInterval, MarketData};
use std::path::Path;

/// Data loader error
#[derive(Debug, thiserror::Error)]
pub enum DataLoaderError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("CSV parse error: {0}")]
    CsvError(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Insufficient data: need {needed}, got {got}")]
    InsufficientData { needed: usize, got: usize },
}

/// Data loader for various sources
pub struct DataLoader {
    bybit_client: Option<BybitClient>,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DataLoader {
    /// Create a new data loader
    pub fn new() -> Self {
        Self {
            bybit_client: Some(BybitClient::new()),
        }
    }

    /// Create a data loader without API clients
    pub fn offline() -> Self {
        Self {
            bybit_client: None,
        }
    }

    /// Fetch data from Bybit
    pub async fn fetch_bybit(
        &self,
        symbol: &str,
        interval: KlineInterval,
        limit: u32,
    ) -> Result<MarketData, DataLoaderError> {
        let client = self
            .bybit_client
            .as_ref()
            .ok_or_else(|| DataLoaderError::ApiError("Bybit client not initialized".to_string()))?;

        client
            .get_klines(symbol, interval, limit, None, None)
            .await
            .map_err(|e| DataLoaderError::ApiError(e.to_string()))
    }

    /// Load data from CSV file
    ///
    /// Expected CSV format: timestamp,open,high,low,close,volume,turnover
    pub fn load_csv<P: AsRef<Path>>(&self, path: P) -> Result<MarketData, DataLoaderError> {
        let content = std::fs::read_to_string(&path)?;
        let mut klines = Vec::new();

        for (idx, line) in content.lines().enumerate() {
            // Skip header
            if idx == 0 && line.contains("timestamp") {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 6 {
                continue;
            }

            let kline = KlineData {
                timestamp: parts[0]
                    .parse()
                    .map_err(|e| DataLoaderError::CsvError(format!("Invalid timestamp: {}", e)))?,
                open: parts[1]
                    .parse()
                    .map_err(|e| DataLoaderError::CsvError(format!("Invalid open: {}", e)))?,
                high: parts[2]
                    .parse()
                    .map_err(|e| DataLoaderError::CsvError(format!("Invalid high: {}", e)))?,
                low: parts[3]
                    .parse()
                    .map_err(|e| DataLoaderError::CsvError(format!("Invalid low: {}", e)))?,
                close: parts[4]
                    .parse()
                    .map_err(|e| DataLoaderError::CsvError(format!("Invalid close: {}", e)))?,
                volume: parts[5]
                    .parse()
                    .map_err(|e| DataLoaderError::CsvError(format!("Invalid volume: {}", e)))?,
                turnover: parts.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            };

            klines.push(kline);
        }

        let filename = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        Ok(MarketData::new(
            filename.to_string(),
            "unknown".to_string(),
            klines,
        ))
    }

    /// Generate synthetic data for testing
    pub fn generate_synthetic(
        &self,
        n_samples: usize,
        seed: u64,
    ) -> MarketData {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        let mut klines = Vec::with_capacity(n_samples);
        let mut price = 100.0;
        let base_volume = 1000.0;

        for i in 0..n_samples {
            // Random walk with momentum
            let momentum = if i > 10 {
                let recent_return = (price - klines[i - 10].close) / klines[i - 10].close;
                recent_return * 0.1 // Small momentum factor
            } else {
                0.0
            };

            let noise: f64 = rng.gen_range(-0.02..0.02);
            let trend: f64 = 0.0001; // Small upward bias
            let return_val = noise + momentum + trend;

            let open = price;
            price *= 1.0 + return_val;

            let volatility = rng.gen_range(0.005..0.02);
            let high = open.max(price) * (1.0 + volatility);
            let low = open.min(price) * (1.0 - volatility);

            let volume = base_volume * (1.0 + rng.gen_range(-0.5..1.0));

            klines.push(KlineData {
                timestamp: i as i64 * 60000, // 1 minute intervals
                open,
                high,
                low,
                close: price,
                volume,
                turnover: price * volume,
            });
        }

        MarketData::new(
            "SYNTHETIC".to_string(),
            "1m".to_string(),
            klines,
        )
    }

    /// Validate that data has sufficient samples
    pub fn validate_data(&self, data: &MarketData, min_samples: usize) -> Result<(), DataLoaderError> {
        if data.len() < min_samples {
            return Err(DataLoaderError::InsufficientData {
                needed: min_samples,
                got: data.len(),
            });
        }
        Ok(())
    }

    /// Save data to CSV
    pub fn save_csv<P: AsRef<Path>>(
        &self,
        data: &MarketData,
        path: P,
    ) -> Result<(), DataLoaderError> {
        let mut content = String::from("timestamp,open,high,low,close,volume,turnover\n");

        for kline in &data.klines {
            content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                kline.timestamp,
                kline.open,
                kline.high,
                kline.low,
                kline.close,
                kline.volume,
                kline.turnover
            ));
        }

        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let loader = DataLoader::offline();
        let data = loader.generate_synthetic(1000, 42);

        assert_eq!(data.len(), 1000);
        assert_eq!(data.symbol, "SYNTHETIC");

        // Check data validity
        for kline in &data.klines {
            assert!(kline.high >= kline.low);
            assert!(kline.high >= kline.open);
            assert!(kline.high >= kline.close);
            assert!(kline.low <= kline.open);
            assert!(kline.low <= kline.close);
            assert!(kline.volume > 0.0);
        }
    }

    #[test]
    fn test_validate_data() {
        let loader = DataLoader::offline();
        let data = loader.generate_synthetic(100, 42);

        assert!(loader.validate_data(&data, 100).is_ok());
        assert!(loader.validate_data(&data, 101).is_err());
    }
}
