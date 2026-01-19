//! Data loading utilities

use crate::api::KlineData;
use ndarray::{Array1, Array2};

/// OHLCV data structure
#[derive(Debug, Clone)]
pub struct OHLCV {
    /// Timestamps
    pub timestamps: Vec<i64>,
    /// Open prices
    pub open: Array1<f64>,
    /// High prices
    pub high: Array1<f64>,
    /// Low prices
    pub low: Array1<f64>,
    /// Close prices
    pub close: Array1<f64>,
    /// Trading volumes
    pub volume: Array1<f64>,
}

impl OHLCV {
    /// Create OHLCV from kline data
    pub fn from_klines(klines: &[KlineData]) -> Self {
        let n = klines.len();

        let mut timestamps = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for kline in klines {
            timestamps.push(kline.timestamp);
            open.push(kline.open);
            high.push(kline.high);
            low.push(kline.low);
            close.push(kline.close);
            volume.push(kline.volume);
        }

        Self {
            timestamps,
            open: Array1::from_vec(open),
            high: Array1::from_vec(high),
            low: Array1::from_vec(low),
            close: Array1::from_vec(close),
            volume: Array1::from_vec(volume),
        }
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Check if data is empty
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Convert to 2D array [n_samples, 5] for OHLCV
    pub fn to_array(&self) -> Array2<f64> {
        let n = self.len();
        let mut arr = Array2::zeros((n, 5));

        for i in 0..n {
            arr[[i, 0]] = self.open[i];
            arr[[i, 1]] = self.high[i];
            arr[[i, 2]] = self.low[i];
            arr[[i, 3]] = self.close[i];
            arr[[i, 4]] = self.volume[i];
        }

        arr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_creation() {
        let klines = vec![
            KlineData {
                timestamp: 1000,
                open: 100.0,
                high: 110.0,
                low: 95.0,
                close: 105.0,
                volume: 1000.0,
                turnover: 100000.0,
            },
            KlineData {
                timestamp: 2000,
                open: 105.0,
                high: 115.0,
                low: 100.0,
                close: 110.0,
                volume: 1500.0,
                turnover: 150000.0,
            },
        ];

        let ohlcv = OHLCV::from_klines(&klines);
        assert_eq!(ohlcv.len(), 2);
        assert_eq!(ohlcv.close[0], 105.0);
        assert_eq!(ohlcv.close[1], 110.0);
    }
}
