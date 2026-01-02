//! Utility functions

mod normalization;

pub use normalization::{normalize, normalize_minmax, standardize};

/// Calculate simple moving average
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return Vec::new();
    }

    data.windows(period)
        .map(|w| w.iter().sum::<f64>() / period as f64)
        .collect()
}

/// Calculate exponential moving average
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return Vec::new();
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = Vec::with_capacity(data.len());

    if data.len() >= period {
        let sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
        result.push(sma);

        for &value in &data[period..] {
            let prev = *result.last().unwrap();
            result.push((value - prev) * multiplier + prev);
        }
    }

    result
}

/// Calculate percentage change
pub fn pct_change(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return Vec::new();
    }

    data.windows(2)
        .map(|w| {
            if w[0] != 0.0 {
                (w[1] - w[0]) / w[0] * 100.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate log returns
pub fn log_returns(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return Vec::new();
    }

    data.windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate rolling window statistics
pub struct RollingWindow {
    data: Vec<f64>,
    window_size: usize,
}

impl RollingWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            data: Vec::with_capacity(window_size),
            window_size,
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.data.len() >= self.window_size {
            self.data.remove(0);
        }
        self.data.push(value);
    }

    pub fn is_full(&self) -> bool {
        self.data.len() >= self.window_size
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    pub fn std(&self) -> f64 {
        if self.data.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (self.data.len() - 1) as f64;
        variance.sqrt()
    }

    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::MAX, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::MIN, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 0.001);
        assert!((result[1] - 3.0).abs() < 0.001);
        assert!((result[2] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let result = ema(&data, 3);

        assert!(!result.is_empty());
    }

    #[test]
    fn test_rolling_window() {
        let mut window = RollingWindow::new(3);

        window.push(1.0);
        window.push(2.0);
        window.push(3.0);

        assert!(window.is_full());
        assert!((window.mean() - 2.0).abs() < 0.001);

        window.push(4.0);
        assert!((window.mean() - 3.0).abs() < 0.001);
    }
}
