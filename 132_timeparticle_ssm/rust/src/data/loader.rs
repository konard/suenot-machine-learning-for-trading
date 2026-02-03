//! Data loading utilities.

use crate::api::bybit::{BybitClient, Kline};
use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// OHLCV data structure.
#[derive(Debug, Clone)]
pub struct OhlcvData {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl OhlcvData {
    /// Create empty OHLCV data.
    pub fn new() -> Self {
        Self {
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Vec::new(),
        }
    }

    /// Create from klines.
    pub fn from_klines(klines: &[Kline]) -> Self {
        let mut data = Self::new();
        for kline in klines {
            data.open.push(kline.open);
            data.high.push(kline.high);
            data.low.push(kline.low);
            data.close.push(kline.close);
            data.volume.push(kline.volume);
        }
        data
    }

    /// Get length of data.
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    /// Convert to ndarray matrix.
    pub fn to_matrix(&self) -> Array2<f64> {
        let n = self.len();
        let mut matrix = Array2::zeros((n, 5));

        for i in 0..n {
            matrix[[i, 0]] = self.open[i];
            matrix[[i, 1]] = self.high[i];
            matrix[[i, 2]] = self.low[i];
            matrix[[i, 3]] = self.close[i];
            matrix[[i, 4]] = self.volume[i];
        }

        matrix
    }
}

impl Default for OhlcvData {
    fn default() -> Self {
        Self::new()
    }
}

/// Data loader for fetching and preprocessing market data.
pub struct DataLoader {
    client: BybitClient,
    buffer: VecDeque<Kline>,
    buffer_size: usize,
}

impl DataLoader {
    /// Create new data loader.
    pub fn new(buffer_size: usize) -> Self {
        Self {
            client: BybitClient::new(),
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
        }
    }

    /// Fetch historical data.
    pub fn fetch(&mut self, symbol: &str, interval: &str, limit: u32) -> Result<OhlcvData, String> {
        match self.client.fetch_klines(symbol, interval, limit) {
            Ok(klines) => {
                // Update buffer
                for kline in &klines {
                    if self.buffer.len() >= self.buffer_size {
                        self.buffer.pop_front();
                    }
                    self.buffer.push_back(kline.clone());
                }

                Ok(OhlcvData::from_klines(&klines))
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Get buffered data.
    pub fn get_buffer(&self) -> OhlcvData {
        let klines: Vec<Kline> = self.buffer.iter().cloned().collect();
        OhlcvData::from_klines(&klines)
    }

    /// Get latest price from buffer.
    pub fn latest_price(&self) -> Option<f64> {
        self.buffer.back().map(|k| k.close)
    }
}

/// Generate simulated OHLCV data for testing.
pub fn generate_simulated_data(n_points: usize, seed: u64) -> OhlcvData {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut close = vec![100.0];
    for _ in 1..n_points {
        let change: f64 = normal.sample(&mut rng) * 2.0;
        let new_close = (close.last().unwrap() + change).max(1.0);
        close.push(new_close);
    }

    let mut data = OhlcvData::new();
    for i in 0..n_points {
        let c = close[i];
        let spread: f64 = rng.gen::<f64>() * 3.0;

        data.close.push(c);
        data.high.push(c + spread.abs());
        data.low.push(c - spread.abs());
        data.open.push(data.low[i] + (data.high[i] - data.low[i]) * rng.gen::<f64>());
        data.volume.push(rng.gen::<f64>() * 1_000_000.0);
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_data() {
        let data = generate_simulated_data(100, 42);
        assert_eq!(data.len(), 100);
        assert!(!data.is_empty());

        let matrix = data.to_matrix();
        assert_eq!(matrix.shape(), &[100, 5]);
    }

    #[test]
    fn test_data_loader() {
        let loader = DataLoader::new(1000);
        assert!(loader.buffer.is_empty());
        assert!(loader.latest_price().is_none());
    }
}
