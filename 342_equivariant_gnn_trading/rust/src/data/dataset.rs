//! Dataset Module
//!
//! Provides structures for organizing market data for training and inference.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::candle::Candle;

/// A single data point for training/inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp
    pub timestamp: u64,

    /// Asset symbol
    pub symbol: String,

    /// Feature vector
    pub features: Vec<f64>,

    /// Target label (optional, for training)
    pub label: Option<i32>,

    /// Future return (optional, for training)
    pub future_return: Option<f64>,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(
        timestamp: u64,
        symbol: String,
        features: Vec<f64>,
        label: Option<i32>,
        future_return: Option<f64>,
    ) -> Self {
        Self {
            timestamp,
            symbol,
            features,
            label,
            future_return,
        }
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.features.len()
    }
}

/// Market dataset containing multiple assets
#[derive(Debug, Clone)]
pub struct MarketDataset {
    /// Data organized by timestamp, then by symbol
    data: HashMap<u64, HashMap<String, DataPoint>>,

    /// Ordered list of timestamps
    timestamps: Vec<u64>,

    /// List of symbols
    symbols: Vec<String>,

    /// Lookback window for feature extraction
    lookback: usize,

    /// Prediction horizon
    horizon: usize,
}

impl MarketDataset {
    /// Create an empty dataset
    pub fn new(lookback: usize, horizon: usize) -> Self {
        Self {
            data: HashMap::new(),
            timestamps: Vec::new(),
            symbols: Vec::new(),
            lookback,
            horizon,
        }
    }

    /// Create dataset from multi-asset candle data
    pub fn from_candles(
        candles_map: &HashMap<String, Vec<Candle>>,
        lookback: usize,
        horizon: usize,
    ) -> Self {
        let mut dataset = Self::new(lookback, horizon);

        // Get all unique timestamps
        let mut all_timestamps: Vec<u64> = candles_map
            .values()
            .flat_map(|candles| candles.iter().map(|c| c.timestamp))
            .collect();
        all_timestamps.sort();
        all_timestamps.dedup();

        dataset.timestamps = all_timestamps;
        dataset.symbols = candles_map.keys().cloned().collect();
        dataset.symbols.sort();

        // Build data map
        for (symbol, candles) in candles_map {
            for candle in candles {
                dataset
                    .data
                    .entry(candle.timestamp)
                    .or_insert_with(HashMap::new)
                    .insert(
                        symbol.clone(),
                        DataPoint::new(
                            candle.timestamp,
                            symbol.clone(),
                            vec![
                                candle.return_pct(),
                                candle.range_pct(),
                                candle.close_position(),
                                candle.volume,
                            ],
                            None,
                            None,
                        ),
                    );
            }
        }

        dataset
    }

    /// Get data point at specific timestamp and symbol
    pub fn get(&self, timestamp: u64, symbol: &str) -> Option<&DataPoint> {
        self.data
            .get(&timestamp)
            .and_then(|m| m.get(symbol))
    }

    /// Get all data points at a specific timestamp
    pub fn get_at_timestamp(&self, timestamp: u64) -> Option<&HashMap<String, DataPoint>> {
        self.data.get(&timestamp)
    }

    /// Get number of timestamps
    pub fn num_timestamps(&self) -> usize {
        self.timestamps.len()
    }

    /// Get number of symbols
    pub fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Get ordered timestamps
    pub fn timestamps(&self) -> &[u64] {
        &self.timestamps
    }

    /// Get symbols
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Get lookback window
    pub fn lookback(&self) -> usize {
        self.lookback
    }

    /// Get prediction horizon
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// Calculate correlation matrix between assets at a given time window
    pub fn correlation_matrix(&self, end_idx: usize) -> Vec<Vec<f64>> {
        let n = self.symbols.len();
        let start_idx = end_idx.saturating_sub(self.lookback);

        // Collect returns for each asset
        let mut returns: Vec<Vec<f64>> = vec![Vec::new(); n];

        for t in start_idx..=end_idx {
            if t >= self.timestamps.len() {
                break;
            }
            let timestamp = self.timestamps[t];
            if let Some(data_at_t) = self.get_at_timestamp(timestamp) {
                for (i, symbol) in self.symbols.iter().enumerate() {
                    if let Some(dp) = data_at_t.get(symbol) {
                        if !dp.features.is_empty() {
                            returns[i].push(dp.features[0]); // First feature is return
                        }
                    }
                }
            }
        }

        // Compute correlation matrix
        let mut corr_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    corr_matrix[i][j] = 1.0;
                } else if !returns[i].is_empty() && !returns[j].is_empty() {
                    corr_matrix[i][j] = correlation(&returns[i], &returns[j]);
                }
            }
        }

        corr_matrix
    }

    /// Get a batch of data points for training
    pub fn get_batch(&self, indices: &[usize]) -> Vec<HashMap<String, DataPoint>> {
        indices
            .iter()
            .filter_map(|&idx| {
                if idx < self.timestamps.len() {
                    self.get_at_timestamp(self.timestamps[idx]).cloned()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Iterate over time windows
    pub fn windows(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.timestamps
            .iter()
            .enumerate()
            .skip(self.lookback)
            .map(|(i, &ts)| (i, ts))
    }
}

/// Calculate Pearson correlation coefficient
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x: f64 = x.iter().take(n).sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().take(n).sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom.abs() < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_point_creation() {
        let dp = DataPoint::new(
            1700000000000,
            "BTCUSDT".to_string(),
            vec![0.01, 0.02, 0.5, 1000.0],
            Some(1),
            Some(0.02),
        );

        assert_eq!(dp.feature_dim(), 4);
        assert_eq!(dp.label, Some(1));
    }

    #[test]
    fn test_dataset_from_candles() {
        let mut candles_map = HashMap::new();

        candles_map.insert(
            "BTCUSDT".to_string(),
            vec![
                Candle::new(1000, 100.0, 105.0, 98.0, 103.0, 1000.0, 100000.0),
                Candle::new(2000, 103.0, 108.0, 101.0, 106.0, 1100.0, 115000.0),
            ],
        );

        candles_map.insert(
            "ETHUSDT".to_string(),
            vec![
                Candle::new(1000, 2000.0, 2100.0, 1950.0, 2050.0, 500.0, 1000000.0),
                Candle::new(2000, 2050.0, 2150.0, 2000.0, 2100.0, 550.0, 1100000.0),
            ],
        );

        let dataset = MarketDataset::from_candles(&candles_map, 10, 1);

        assert_eq!(dataset.num_symbols(), 2);
        assert_eq!(dataset.num_timestamps(), 2);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);

        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = correlation(&x, &z);
        assert!((corr_neg - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_empty_dataset() {
        let dataset = MarketDataset::new(10, 1);
        assert_eq!(dataset.num_timestamps(), 0);
        assert_eq!(dataset.num_symbols(), 0);
    }
}
