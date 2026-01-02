//! Подготовка признаков для WaveNet

use crate::types::Candle;
use super::indicators::*;

/// Набор признаков для WaveNet
#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub features: Vec<Vec<f64>>,
    pub feature_names: Vec<String>,
}

impl FeatureSet {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            feature_names: Vec::new(),
        }
    }

    pub fn add_feature(&mut self, name: &str, values: Vec<f64>) {
        self.features.push(values);
        self.feature_names.push(name.to_string());
    }

    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    pub fn len(&self) -> usize {
        self.features.first().map(|f| f.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Получить данные для WaveNet (channels x timesteps)
    pub fn as_wavenet_input(&self) -> &[Vec<f64>] {
        &self.features
    }

    /// Нормализация признаков (z-score)
    pub fn normalize(&mut self) {
        for feature in &mut self.features {
            let valid_values: Vec<f64> = feature.iter().filter(|x| x.is_finite()).copied().collect();
            if valid_values.is_empty() {
                continue;
            }

            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            let variance = valid_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / valid_values.len() as f64;
            let std = variance.sqrt();

            if std > 1e-10 {
                for v in feature.iter_mut() {
                    if v.is_finite() {
                        *v = (*v - mean) / std;
                    }
                }
            }
        }
    }

    /// Заменить NaN на 0
    pub fn fill_nan(&mut self, value: f64) {
        for feature in &mut self.features {
            for v in feature.iter_mut() {
                if !v.is_finite() {
                    *v = value;
                }
            }
        }
    }
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Построитель признаков из свечей
pub struct FeatureBuilder {
    candles: Vec<Candle>,
}

impl FeatureBuilder {
    pub fn new(candles: Vec<Candle>) -> Self {
        Self { candles }
    }

    /// Построить базовые OHLCV признаки
    pub fn build_ohlcv(&self) -> FeatureSet {
        let mut fs = FeatureSet::new();

        // Нормализованные OHLCV (относительно close)
        let closes: Vec<f64> = self.candles.iter().map(|c| c.close).collect();

        // Open/Close ratio
        let open_close: Vec<f64> = self.candles
            .iter()
            .map(|c| (c.open - c.close) / c.close)
            .collect();
        fs.add_feature("open_close_ratio", open_close);

        // High/Close ratio
        let high_close: Vec<f64> = self.candles
            .iter()
            .map(|c| (c.high - c.close) / c.close)
            .collect();
        fs.add_feature("high_close_ratio", high_close);

        // Low/Close ratio
        let low_close: Vec<f64> = self.candles
            .iter()
            .map(|c| (c.low - c.close) / c.close)
            .collect();
        fs.add_feature("low_close_ratio", low_close);

        // Log returns
        let log_returns = log_returns(&closes);
        fs.add_feature("log_returns", log_returns);

        // Volume (normalized)
        let volumes: Vec<f64> = self.candles.iter().map(|c| c.volume).collect();
        let vol_mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let norm_volumes: Vec<f64> = volumes.iter().map(|v| v / vol_mean - 1.0).collect();
        fs.add_feature("volume_norm", norm_volumes);

        fs
    }

    /// Построить технические индикаторы
    pub fn build_indicators(&self) -> FeatureSet {
        let mut fs = FeatureSet::new();
        let closes: Vec<f64> = self.candles.iter().map(|c| c.close).collect();

        // RSI
        let rsi_vals = rsi(&closes, 14);
        let rsi_norm: Vec<f64> = rsi_vals.iter().map(|v| (v - 50.0) / 50.0).collect();
        fs.add_feature("rsi_14", rsi_norm);

        // MACD
        let macd_result = macd(&closes, 12, 26, 9);
        // Нормализуем MACD относительно цены
        let macd_norm: Vec<f64> = macd_result.macd_line
            .iter()
            .zip(closes.iter())
            .map(|(m, c)| m / c)
            .collect();
        fs.add_feature("macd", macd_norm);

        // MACD histogram
        let hist_norm: Vec<f64> = macd_result.histogram
            .iter()
            .zip(closes.iter())
            .map(|(h, c)| h / c)
            .collect();
        fs.add_feature("macd_hist", hist_norm);

        // Bollinger Bands position
        let bb = bollinger_bands(&closes, 20, 2.0);
        let bb_position: Vec<f64> = closes
            .iter()
            .zip(bb.upper.iter().zip(bb.lower.iter()))
            .map(|(c, (u, l))| {
                if (u - l).abs() > 1e-10 {
                    2.0 * (c - l) / (u - l) - 1.0
                } else {
                    0.0
                }
            })
            .collect();
        fs.add_feature("bb_position", bb_position);

        // Stochastic
        let stoch = stochastic(&self.candles, 14, 3);
        let stoch_norm: Vec<f64> = stoch.k.iter().map(|v| (v - 50.0) / 50.0).collect();
        fs.add_feature("stoch_k", stoch_norm);

        // ATR (volatility)
        let atr_vals = atr(&self.candles, 14);
        let atr_norm: Vec<f64> = atr_vals
            .iter()
            .zip(closes.iter())
            .map(|(a, c)| a / c)
            .collect();
        fs.add_feature("atr", atr_norm);

        // Momentum
        let mom = momentum(&closes, 10);
        let mom_norm: Vec<f64> = mom
            .iter()
            .zip(closes.iter())
            .map(|(m, c)| m / c)
            .collect();
        fs.add_feature("momentum_10", mom_norm);

        // ROC
        let roc_vals = roc(&closes, 10);
        let roc_norm: Vec<f64> = roc_vals.iter().map(|v| v / 100.0).collect();
        fs.add_feature("roc_10", roc_norm);

        fs
    }

    /// Построить полный набор признаков
    pub fn build_all(&self) -> FeatureSet {
        let mut ohlcv = self.build_ohlcv();
        let indicators = self.build_indicators();

        for (name, feature) in indicators.feature_names.iter().zip(indicators.features.iter()) {
            ohlcv.add_feature(name, feature.clone());
        }

        ohlcv
    }

    /// Построить целевую переменную (будущая доходность)
    pub fn build_target(&self, horizon: usize) -> Vec<f64> {
        let closes: Vec<f64> = self.candles.iter().map(|c| c.close).collect();
        let n = closes.len();

        let mut target = Vec::with_capacity(n);

        for i in 0..n {
            if i + horizon < n {
                target.push((closes[i + horizon] - closes[i]) / closes[i]);
            } else {
                target.push(f64::NAN);
            }
        }

        target
    }
}

/// Вычислить log returns
fn log_returns(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![0.0]; // Первое значение = 0
    for i in 1..data.len() {
        if data[i - 1] > 0.0 && data[i] > 0.0 {
            result.push((data[i] / data[i - 1]).ln());
        } else {
            result.push(f64::NAN);
        }
    }

    result
}

/// Создать окна для обучения
pub fn create_windows(
    features: &[Vec<f64>],
    target: &[f64],
    window_size: usize,
) -> (Vec<Vec<Vec<f64>>>, Vec<f64>) {
    let n = features[0].len();
    if n < window_size {
        return (Vec::new(), Vec::new());
    }

    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in window_size..n {
        if target[i - 1].is_finite() {
            let window: Vec<Vec<f64>> = features
                .iter()
                .map(|f| f[i - window_size..i].to_vec())
                .collect();

            x.push(window);
            y.push(target[i - 1]);
        }
    }

    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                timestamp: Utc::now(),
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 98.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
            })
            .collect()
    }

    #[test]
    fn test_feature_builder() {
        let candles = create_test_candles(100);
        let builder = FeatureBuilder::new(candles);
        let features = builder.build_all();

        assert!(features.num_features() > 0);
        assert_eq!(features.len(), 100);
    }

    #[test]
    fn test_feature_normalization() {
        let candles = create_test_candles(100);
        let builder = FeatureBuilder::new(candles);
        let mut features = builder.build_all();

        features.fill_nan(0.0);
        features.normalize();

        // После нормализации большинство значений должны быть в [-3, 3]
        for feature in &features.features {
            let valid: Vec<_> = feature.iter().filter(|x| x.is_finite()).collect();
            if !valid.is_empty() {
                let mean: f64 = valid.iter().copied().sum::<f64>() / valid.len() as f64;
                assert!(mean.abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_create_windows() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        ];
        let target = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        let (x, y) = create_windows(&features, &target, 3);

        assert!(!x.is_empty());
        assert_eq!(x.len(), y.len());
        assert_eq!(x[0].len(), 2); // 2 features
        assert_eq!(x[0][0].len(), 3); // window size 3
    }
}
