//! Извлечение признаков из OHLCV данных

use crate::data::Candle;
use ndarray::Array1;

/// Извлекатель технических признаков из свечных данных
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Использовать цену закрытия
    pub use_close: bool,
    /// Использовать объём
    pub use_volume: bool,
    /// Использовать high-low разброс
    pub use_range: bool,
    /// Использовать процентное изменение
    pub use_returns: bool,
    /// Использовать скользящие средние
    pub use_ma: bool,
    /// Период для скользящей средней
    pub ma_period: usize,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self {
            use_close: true,
            use_volume: true,
            use_range: true,
            use_returns: true,
            use_ma: false,
            ma_period: 20,
        }
    }
}

impl FeatureExtractor {
    /// Создаёт новый экстрактор признаков
    pub fn new() -> Self {
        Self::default()
    }

    /// Включает все доступные признаки
    pub fn all_features() -> Self {
        Self {
            use_close: true,
            use_volume: true,
            use_range: true,
            use_returns: true,
            use_ma: true,
            ma_period: 20,
        }
    }

    /// Только базовые OHLCV признаки
    pub fn basic() -> Self {
        Self {
            use_close: true,
            use_volume: true,
            use_range: false,
            use_returns: false,
            use_ma: false,
            ma_period: 20,
        }
    }

    /// Возвращает количество признаков
    pub fn feature_count(&self) -> usize {
        let mut count = 0;
        if self.use_close {
            count += 4;
        } // OHLC
        if self.use_volume {
            count += 1;
        }
        if self.use_range {
            count += 1;
        }
        if self.use_returns {
            count += 1;
        }
        if self.use_ma {
            count += 1;
        }
        count
    }

    /// Возвращает названия признаков
    pub fn feature_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.use_close {
            names.extend_from_slice(&["open", "high", "low", "close"]);
        }
        if self.use_volume {
            names.push("volume");
        }
        if self.use_range {
            names.push("range");
        }
        if self.use_returns {
            names.push("returns");
        }
        if self.use_ma {
            names.push("ma");
        }
        names
    }

    /// Извлекает признаки из одной свечи
    pub fn extract_single(&self, candle: &Candle, prev_close: Option<f64>) -> Vec<f64> {
        let mut features = Vec::new();

        if self.use_close {
            features.push(candle.open);
            features.push(candle.high);
            features.push(candle.low);
            features.push(candle.close);
        }

        if self.use_volume {
            features.push(candle.volume);
        }

        if self.use_range {
            features.push(candle.high - candle.low);
        }

        if self.use_returns {
            let ret = match prev_close {
                Some(prev) if prev > 0.0 => (candle.close - prev) / prev,
                _ => 0.0,
            };
            features.push(ret);
        }

        features
    }

    /// Извлекает признаки из списка свечей
    ///
    /// Возвращает 2D массив: [n_samples, n_features]
    pub fn extract(&self, candles: &[Candle]) -> Vec<Vec<f64>> {
        if candles.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(candles.len());

        // Для скользящей средней нужна история
        let ma_buffer: Vec<f64> = if self.use_ma {
            candles.iter().map(|c| c.close).collect()
        } else {
            Vec::new()
        };

        for (i, candle) in candles.iter().enumerate() {
            let prev_close = if i > 0 {
                Some(candles[i - 1].close)
            } else {
                None
            };

            let mut features = self.extract_single(candle, prev_close);

            // Добавляем скользящую среднюю
            if self.use_ma {
                let ma = self.calculate_ma(&ma_buffer, i, self.ma_period);
                features.push(ma);
            }

            result.push(features);
        }

        result
    }

    /// Извлекает только цены закрытия
    pub fn extract_close_prices(candles: &[Candle]) -> Array1<f64> {
        candles.iter().map(|c| c.close).collect()
    }

    /// Извлекает доходности (returns)
    pub fn extract_returns(candles: &[Candle]) -> Array1<f64> {
        let mut returns = Array1::zeros(candles.len());
        for i in 1..candles.len() {
            if candles[i - 1].close > 0.0 {
                returns[i] = (candles[i].close - candles[i - 1].close) / candles[i - 1].close;
            }
        }
        returns
    }

    /// Вычисляет простую скользящую среднюю
    fn calculate_ma(&self, prices: &[f64], current_idx: usize, period: usize) -> f64 {
        if current_idx < period - 1 {
            // Недостаточно данных - возвращаем текущую цену
            prices[current_idx]
        } else {
            let start = current_idx + 1 - period;
            let sum: f64 = prices[start..=current_idx].iter().sum();
            sum / period as f64
        }
    }
}

/// Рассчитывает RSI (Relative Strength Index)
pub fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![50.0; prices.len()];
    }

    let mut rsi = vec![50.0; prices.len()];
    let mut gains = Vec::with_capacity(prices.len());
    let mut losses = Vec::with_capacity(prices.len());

    // Вычисляем изменения
    gains.push(0.0);
    losses.push(0.0);

    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    // Вычисляем RSI
    for i in period..prices.len() {
        let avg_gain: f64 = gains[i - period + 1..=i].iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = losses[i - period + 1..=i].iter().sum::<f64>() / period as f64;

        if avg_loss == 0.0 {
            rsi[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    rsi
}

/// Рассчитывает MACD
pub fn calculate_macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = calculate_ema(prices, fast_period);
    let slow_ema = calculate_ema(prices, slow_period);

    let macd_line: Vec<f64> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = calculate_ema(&macd_line, signal_period);

    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    (macd_line, signal_line, histogram)
}

/// Рассчитывает EMA (Exponential Moving Average)
pub fn calculate_ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() {
        return Vec::new();
    }

    let mut ema = vec![0.0; prices.len()];
    let multiplier = 2.0 / (period as f64 + 1.0);

    // Первое значение EMA = первая цена
    ema[0] = prices[0];

    for i in 1..prices.len() {
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }

    ema
}

/// Рассчитывает Bollinger Bands
pub fn calculate_bollinger_bands(
    prices: &[f64],
    period: usize,
    std_dev: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let mut middle = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut lower = vec![0.0; n];

    for i in 0..n {
        if i < period - 1 {
            middle[i] = prices[i];
            upper[i] = prices[i];
            lower[i] = prices[i];
        } else {
            let start = i + 1 - period;
            let slice = &prices[start..=i];

            let mean: f64 = slice.iter().sum::<f64>() / period as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            middle[i] = mean;
            upper[i] = mean + std_dev * std;
            lower[i] = mean - std_dev * std;
        }
    }

    (upper, middle, lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles() -> Vec<Candle> {
        vec![
            Candle::new(0, 100.0, 105.0, 95.0, 102.0, 1000.0, 100000.0),
            Candle::new(1, 102.0, 108.0, 100.0, 107.0, 1200.0, 120000.0),
            Candle::new(2, 107.0, 110.0, 105.0, 108.0, 800.0, 80000.0),
        ]
    }

    #[test]
    fn test_feature_extractor() {
        let candles = create_test_candles();
        let extractor = FeatureExtractor::default();
        let features = extractor.extract(&candles);

        assert_eq!(features.len(), 3);
        assert_eq!(features[0].len(), extractor.feature_count());
    }

    #[test]
    fn test_extract_returns() {
        let candles = create_test_candles();
        let returns = FeatureExtractor::extract_returns(&candles);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.0).abs() < 1e-10); // Первый return = 0
        assert!((returns[1] - 0.049019607843137254).abs() < 1e-10); // (107-102)/102
    }

    #[test]
    fn test_rsi() {
        let prices = vec![
            44.0, 44.5, 43.0, 44.0, 44.5, 44.0, 43.5, 44.0, 45.0, 45.5, 46.0, 46.5, 46.0, 45.5,
            45.0,
        ];
        let rsi = calculate_rsi(&prices, 14);
        assert_eq!(rsi.len(), prices.len());
    }

    #[test]
    fn test_ema() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = calculate_ema(&prices, 3);
        assert_eq!(ema.len(), 5);
        assert!((ema[0] - 1.0).abs() < 1e-10);
    }
}
