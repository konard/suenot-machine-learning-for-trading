//! Data loading utilities for market data.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::path::Path;

/// Data loader for market data from various sources.
pub struct DataLoader {
    /// Cached dataframe
    data: Option<DataFrame>,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DataLoader {
    /// Create a new data loader.
    pub fn new() -> Self {
        Self { data: None }
    }

    /// Load data from a CSV file.
    pub fn load_csv(&mut self, path: impl AsRef<Path>) -> Result<&DataFrame, PolarsError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        self.data = Some(df);
        Ok(self.data.as_ref().unwrap())
    }

    /// Create dataframe from kline data.
    pub fn from_klines(&mut self, klines: &[crate::api::Kline]) -> Result<&DataFrame, PolarsError> {
        let start_times: Vec<i64> = klines.iter().map(|k| k.start_time as i64).collect();
        let opens: Vec<f64> = klines.iter().map(|k| k.open).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        let df = df![
            "timestamp" => start_times,
            "open" => opens,
            "high" => highs,
            "low" => lows,
            "close" => closes,
            "volume" => volumes,
        ]?;

        self.data = Some(df);
        Ok(self.data.as_ref().unwrap())
    }

    /// Get the loaded dataframe.
    pub fn get_data(&self) -> Option<&DataFrame> {
        self.data.as_ref()
    }

    /// Extract price array from loaded data.
    pub fn get_prices(&self, column: &str) -> Option<Array1<f64>> {
        self.data.as_ref().and_then(|df| {
            df.column(column)
                .ok()
                .and_then(|col| col.f64().ok())
                .map(|ca| {
                    Array1::from_vec(
                        ca.iter()
                            .map(|v| v.unwrap_or(0.0))
                            .collect(),
                    )
                })
        })
    }

    /// Convert dataframe to feature matrix.
    pub fn to_feature_matrix(&self, columns: &[&str]) -> Option<Array2<f64>> {
        let df = self.data.as_ref()?;
        let n_rows = df.height();
        let n_cols = columns.len();

        let mut matrix = Array2::zeros((n_rows, n_cols));

        for (j, col_name) in columns.iter().enumerate() {
            if let Ok(col) = df.column(col_name) {
                if let Ok(ca) = col.f64() {
                    for (i, val) in ca.iter().enumerate() {
                        matrix[[i, j]] = val.unwrap_or(0.0);
                    }
                }
            }
        }

        Some(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let loader = DataLoader::new();
        assert!(loader.data.is_none());
    }

    #[test]
    fn test_from_klines() {
        use crate::api::Kline;

        let klines = vec![
            Kline {
                start_time: 1000,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 102.0,
                volume: 1000.0,
                turnover: 100000.0,
            },
            Kline {
                start_time: 2000,
                open: 102.0,
                high: 108.0,
                low: 100.0,
                close: 106.0,
                volume: 1200.0,
                turnover: 120000.0,
            },
        ];

        let mut loader = DataLoader::new();
        let df = loader.from_klines(&klines).unwrap();
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 6);
    }
}
