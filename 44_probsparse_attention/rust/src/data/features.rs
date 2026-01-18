//! Вычисление технических признаков

use crate::api::Kline;

/// Набор технических признаков
#[derive(Debug, Clone, Default)]
pub struct Features {
    /// Log returns
    pub returns: Vec<f64>,
    /// Волатильность (скользящее std returns)
    pub volatility: Vec<f64>,
    /// Отношение объёма к скользящему среднему
    pub volume_ratio: Vec<f64>,
    /// Диапазон high-low нормализованный
    pub high_low_range: Vec<f64>,
    /// Диапазон close-open нормализованный
    pub close_open_range: Vec<f64>,
    /// Отношение быстрой MA к медленной MA
    pub ma_ratio: Vec<f64>,
    /// RSI
    pub rsi: Vec<f64>,
    /// Позиция относительно Bollinger Bands
    pub bb_position: Vec<f64>,
}

impl Features {
    /// Вычисляет все признаки из OHLCV данных
    ///
    /// # Arguments
    ///
    /// * `klines` - Исторические свечи
    /// * `lookback` - Окно для скользящих вычислений
    pub fn compute(klines: &[Kline], lookback: usize) -> Self {
        let n = klines.len();

        if n < lookback + 1 {
            return Self::default();
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Returns
        let mut returns = vec![0.0];
        for i in 1..n {
            returns.push((closes[i] / closes[i - 1]).ln());
        }

        // Volatility (rolling std of returns)
        let volatility = rolling_std(&returns, lookback);

        // Volume ratio
        let volume_ma = rolling_mean(&volumes, lookback);
        let volume_ratio: Vec<f64> = volumes.iter()
            .zip(volume_ma.iter())
            .map(|(v, ma)| if *ma > 0.0 { v / ma } else { 1.0 })
            .collect();

        // High-low range
        let high_low_range: Vec<f64> = klines.iter()
            .map(|k| (k.high - k.low) / k.close)
            .collect();

        // Close-open range
        let close_open_range: Vec<f64> = klines.iter()
            .map(|k| (k.close - k.open) / k.open)
            .collect();

        // MA ratio
        let ma_fast = rolling_mean(&closes, lookback / 2);
        let ma_slow = rolling_mean(&closes, lookback);
        let ma_ratio: Vec<f64> = ma_fast.iter()
            .zip(ma_slow.iter())
            .map(|(f, s)| if *s > 0.0 { f / s } else { 1.0 })
            .collect();

        // RSI
        let rsi = compute_rsi(&closes, lookback);

        // Bollinger Bands position
        let bb_position = compute_bb_position(&closes, lookback);

        Self {
            returns,
            volatility,
            volume_ratio,
            high_low_range,
            close_open_range,
            ma_ratio,
            rsi,
            bb_position,
        }
    }

    /// Возвращает матрицу признаков [n_samples, n_features]
    pub fn to_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.returns.len();
        let mut matrix = Vec::with_capacity(n);

        for i in 0..n {
            matrix.push(vec![
                self.returns.get(i).copied().unwrap_or(0.0),
                self.volatility.get(i).copied().unwrap_or(0.0),
                self.volume_ratio.get(i).copied().unwrap_or(1.0),
                self.high_low_range.get(i).copied().unwrap_or(0.0),
                self.close_open_range.get(i).copied().unwrap_or(0.0),
                self.ma_ratio.get(i).copied().unwrap_or(1.0),
            ]);
        }

        matrix
    }

    /// Нормализует признаки с использованием скользящей z-score
    pub fn normalize(&self, window: usize) -> Self {
        Self {
            returns: rolling_zscore(&self.returns, window),
            volatility: rolling_zscore(&self.volatility, window),
            volume_ratio: rolling_zscore(&self.volume_ratio, window),
            high_low_range: rolling_zscore(&self.high_low_range, window),
            close_open_range: rolling_zscore(&self.close_open_range, window),
            ma_ratio: rolling_zscore(&self.ma_ratio, window),
            rsi: rolling_zscore(&self.rsi, window),
            bb_position: self.bb_position.clone(), // Already normalized
        }
    }
}

/// Скользящее среднее
fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        if i < window - 1 {
            // Используем доступные данные
            let sum: f64 = data[..=i].iter().sum();
            result.push(sum / (i + 1) as f64);
        } else {
            let sum: f64 = data[i + 1 - window..=i].iter().sum();
            result.push(sum / window as f64);
        }
    }

    result
}

/// Скользящее стандартное отклонение
fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        if i < window - 1 {
            let slice = &data[..=i];
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
            result.push(var.sqrt());
        } else {
            let slice = &data[i + 1 - window..=i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            result.push(var.sqrt());
        }
    }

    result
}

/// Скользящая z-score нормализация
fn rolling_zscore(data: &[f64], window: usize) -> Vec<f64> {
    let mean = rolling_mean(data, window);
    let std = rolling_std(data, window);

    data.iter()
        .zip(mean.iter())
        .zip(std.iter())
        .map(|((x, m), s)| {
            if *s > 1e-8 { (x - m) / s } else { 0.0 }
        })
        .collect()
}

/// Вычисляет RSI
fn compute_rsi(closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    let mut rsi = vec![50.0; n];  // Default to neutral

    if n < period + 1 {
        return rsi;
    }

    // Calculate price changes
    let mut gains = Vec::with_capacity(n);
    let mut losses = Vec::with_capacity(n);

    gains.push(0.0);
    losses.push(0.0);

    for i in 1..n {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    // Calculate rolling average gain and loss
    let avg_gain = rolling_mean(&gains, period);
    let avg_loss = rolling_mean(&losses, period);

    // Calculate RSI
    for i in period..n {
        let rs = if avg_loss[i] > 0.0 {
            avg_gain[i] / avg_loss[i]
        } else {
            100.0  // No losses means RSI = 100
        };
        rsi[i] = 100.0 - (100.0 / (1.0 + rs));
    }

    rsi
}

/// Вычисляет позицию относительно Bollinger Bands
fn compute_bb_position(closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    let bb_mid = rolling_mean(closes, period);
    let bb_std = rolling_std(closes, period);

    closes.iter()
        .zip(bb_mid.iter())
        .zip(bb_std.iter())
        .map(|((c, mid), std)| {
            if *std > 1e-8 {
                (c - mid) / (2.0 * std)
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n).map(|i| Kline {
            timestamp: i as u64 * 3600000,
            open: 100.0 + (i as f64 * 0.1).sin(),
            high: 101.0 + (i as f64 * 0.1).sin(),
            low: 99.0 + (i as f64 * 0.1).sin(),
            close: 100.5 + (i as f64 * 0.1).sin(),
            volume: 1000.0 + (i as f64 * 100.0),
            turnover: 100000.0,
        }).collect()
    }

    #[test]
    fn test_features_compute() {
        let klines = create_test_klines(100);
        let features = Features::compute(&klines, 20);

        assert_eq!(features.returns.len(), 100);
        assert_eq!(features.volatility.len(), 100);
        assert_eq!(features.volume_ratio.len(), 100);
    }

    #[test]
    fn test_features_to_matrix() {
        let klines = create_test_klines(50);
        let features = Features::compute(&klines, 10);
        let matrix = features.to_matrix();

        assert_eq!(matrix.len(), 50);
        assert_eq!(matrix[0].len(), 6);
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&data, 3);

        assert_eq!(result.len(), 5);
        // First value is just the first element
        assert!((result[0] - 1.0).abs() < 1e-10);
        // Second is average of first two
        assert!((result[1] - 1.5).abs() < 1e-10);
        // Third is average of first three
        assert!((result[2] - 2.0).abs() < 1e-10);
        // Fourth is average of 2,3,4
        assert!((result[3] - 3.0).abs() < 1e-10);
        // Fifth is average of 3,4,5
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_bounds() {
        let klines = create_test_klines(100);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let rsi = compute_rsi(&closes, 14);

        for val in rsi {
            assert!(val >= 0.0 && val <= 100.0, "RSI should be between 0 and 100");
        }
    }

    #[test]
    fn test_bb_position() {
        let klines = create_test_klines(100);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let bb_pos = compute_bb_position(&closes, 20);

        // Most values should be between -2 and 2 (within 2 std devs)
        let in_range = bb_pos.iter()
            .skip(20)  // Skip initial period
            .filter(|&&x| x >= -2.0 && x <= 2.0)
            .count();

        assert!(in_range > 60, "Most BB positions should be within 2 std devs");
    }
}
