//! Data loading utilities

use crate::api::{BybitClient, Kline};
use crate::data::features::{Features, compute_features};
use anyhow::Result;

/// Data loader for fetching and processing market data
pub struct DataLoader {
    client: BybitClient,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new() -> Self {
        Self {
            client: BybitClient::new(),
        }
    }

    /// Fetch and compute features for a symbol
    pub async fn load_symbol(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Features>> {
        let klines = self.client.fetch_klines(symbol, interval, limit, None, None).await?;
        let features = compute_features(&klines);
        Ok(features)
    }

    /// Fetch raw klines
    pub async fn load_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let klines = self.client.fetch_klines(symbol, interval, limit, None, None).await?;
        Ok(klines)
    }

    /// Load multiple symbols
    pub async fn load_multiple(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> Result<Vec<(String, Vec<Features>)>> {
        let mut results = Vec::new();

        for symbol in symbols {
            let features = self.load_symbol(symbol, interval, limit).await?;
            results.push((symbol.to_string(), features));
        }

        Ok(results)
    }
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate sample data for testing
pub fn generate_sample_data(n_samples: usize) -> Vec<Features> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n_samples)
        .map(|_| Features {
            log_return: rng.gen_range(-0.05..0.05),
            return_5: rng.gen_range(-0.1..0.1),
            return_10: rng.gen_range(-0.15..0.15),
            volume_ratio: rng.gen_range(0.5..2.0),
            volume_change: rng.gen_range(-0.5..0.5),
            volatility_5: rng.gen_range(0.01..0.05),
            volatility_20: rng.gen_range(0.01..0.03),
            rsi_14: rng.gen_range(0.2..0.8),
            rsi_7: rng.gen_range(0.2..0.8),
            macd: rng.gen_range(-0.01..0.01),
            macd_signal: rng.gen_range(-0.01..0.01),
            macd_hist: rng.gen_range(-0.005..0.005),
            bb_position: rng.gen_range(0.0..1.0),
            atr_ratio: rng.gen_range(0.01..0.05),
            ma_10: rng.gen_range(0.95..1.05),
            ma_20: rng.gen_range(0.95..1.05),
        })
        .collect()
}
