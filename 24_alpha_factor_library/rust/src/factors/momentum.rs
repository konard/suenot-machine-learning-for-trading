//! Индикаторы моментума
//!
//! - RSI (Relative Strength Index) - Индекс относительной силы
//! - ROC (Rate of Change) - Скорость изменения
//! - Momentum - Моментум
//! - Stochastic - Стохастик
//! - Williams %R
//! - CCI (Commodity Channel Index)

use super::utils;

/// RSI (Relative Strength Index) - Индекс относительной силы
///
/// RSI = 100 - (100 / (1 + RS))
/// RS = Average Gain / Average Loss
///
/// # Аргументы
/// - `data` - Массив цен
/// - `period` - Период RSI (обычно 14)
///
/// # Возвращает
/// Значения RSI в диапазоне [0, 100]
pub fn rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period + 1 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period];
    let mut gains = vec![0.0; data.len() - 1];
    let mut losses = vec![0.0; data.len() - 1];

    // Рассчитываем изменения
    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains[i - 1] = change;
        } else {
            losses[i - 1] = -change;
        }
    }

    // Первое среднее
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    // Первое RSI
    let rs = if avg_loss == 0.0 {
        100.0
    } else {
        avg_gain / avg_loss
    };
    result.push(100.0 - (100.0 / (1.0 + rs)));

    // Остальные значения (экспоненциальное сглаживание)
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;

        let rs = if avg_loss == 0.0 {
            100.0
        } else {
            avg_gain / avg_loss
        };
        result.push(100.0 - (100.0 / (1.0 + rs)));
    }

    result
}

/// ROC (Rate of Change) - Скорость изменения
///
/// ROC = ((Price - Price_n) / Price_n) * 100
///
/// # Аргументы
/// - `data` - Массив цен
/// - `period` - Период сравнения
pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() <= period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period];

    for i in period..data.len() {
        let prev = data[i - period];
        if prev != 0.0 {
            result.push(((data[i] - prev) / prev) * 100.0);
        } else {
            result.push(f64::NAN);
        }
    }

    result
}

/// Momentum - простой моментум
///
/// Momentum = Price - Price_n
pub fn momentum(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() <= period {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; period];

    for i in period..data.len() {
        result.push(data[i] - data[i - period]);
    }

    result
}

/// Результат стохастика
#[derive(Debug, Clone)]
pub struct StochasticResult {
    /// %K линия
    pub k: Vec<f64>,
    /// %D линия (сигнальная)
    pub d: Vec<f64>,
}

/// Stochastic Oscillator - Стохастический осциллятор
///
/// %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
/// %D = SMA(%K, d_period)
///
/// # Аргументы
/// - `highs` - Массив максимумов
/// - `lows` - Массив минимумов
/// - `closes` - Массив цен закрытия
/// - `k_period` - Период для %K (обычно 14)
/// - `d_period` - Период для %D (обычно 3)
pub fn stochastic(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    k_period: usize,
    d_period: usize,
) -> StochasticResult {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n < k_period {
        return StochasticResult {
            k: vec![f64::NAN; n],
            d: vec![f64::NAN; n],
        };
    }

    let mut k = vec![f64::NAN; k_period - 1];

    for i in (k_period - 1)..n {
        let window_high = highs[(i + 1 - k_period)..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let window_low = lows[(i + 1 - k_period)..=i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let range = window_high - window_low;
        if range > 0.0 {
            k.push(((closes[i] - window_low) / range) * 100.0);
        } else {
            k.push(50.0); // Если нет диапазона, возвращаем середину
        }
    }

    // %D = SMA(%K)
    let d = super::trend::sma(&k, d_period);

    StochasticResult { k, d }
}

/// Williams %R
///
/// %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100
///
/// Диапазон: [-100, 0]
pub fn williams_r(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n < period {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let window_high = highs[(i + 1 - period)..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let window_low = lows[(i + 1 - period)..=i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let range = window_high - window_low;
        if range > 0.0 {
            result.push(((window_high - closes[i]) / range) * -100.0);
        } else {
            result.push(-50.0);
        }
    }

    result
}

/// CCI (Commodity Channel Index)
///
/// CCI = (Typical Price - SMA(TP)) / (0.015 * Mean Deviation)
///
/// Typical Price = (High + Low + Close) / 3
pub fn cci(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n < period {
        return vec![f64::NAN; n];
    }

    // Typical Price
    let tp: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .zip(closes.iter())
        .map(|((h, l), c)| (h + l + c) / 3.0)
        .collect();

    let tp_sma = super::trend::sma(&tp, period);

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let sma_val = tp_sma[i];

        // Mean Deviation
        let mean_dev: f64 = tp[(i + 1 - period)..=i]
            .iter()
            .map(|x| (x - sma_val).abs())
            .sum::<f64>()
            / period as f64;

        if mean_dev > 0.0 {
            result.push((tp[i] - sma_val) / (0.015 * mean_dev));
        } else {
            result.push(0.0);
        }
    }

    result
}

/// ADX (Average Directional Index)
///
/// Измеряет силу тренда
pub fn adx(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() || n < period + 1 {
        return vec![f64::NAN; n];
    }

    // +DM и -DM
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    let mut tr = vec![0.0; n];

    for i in 1..n {
        let high_diff = highs[i] - highs[i - 1];
        let low_diff = lows[i - 1] - lows[i];

        if high_diff > low_diff && high_diff > 0.0 {
            plus_dm[i] = high_diff;
        }
        if low_diff > high_diff && low_diff > 0.0 {
            minus_dm[i] = low_diff;
        }

        // True Range
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
    }

    // Сглаженные значения
    let smooth_plus_dm = super::trend::ema(&plus_dm, period);
    let smooth_minus_dm = super::trend::ema(&minus_dm, period);
    let smooth_tr = super::trend::ema(&tr, period);

    // +DI и -DI
    let plus_di: Vec<f64> = smooth_plus_dm
        .iter()
        .zip(smooth_tr.iter())
        .map(|(dm, tr)| if *tr > 0.0 { (dm / tr) * 100.0 } else { 0.0 })
        .collect();

    let minus_di: Vec<f64> = smooth_minus_dm
        .iter()
        .zip(smooth_tr.iter())
        .map(|(dm, tr)| if *tr > 0.0 { (dm / tr) * 100.0 } else { 0.0 })
        .collect();

    // DX
    let dx: Vec<f64> = plus_di
        .iter()
        .zip(minus_di.iter())
        .map(|(p, m)| {
            let sum = p + m;
            if sum > 0.0 {
                ((p - m).abs() / sum) * 100.0
            } else {
                0.0
            }
        })
        .collect();

    // ADX = EMA(DX)
    super::trend::ema(&dx, period)
}

/// Awesome Oscillator
///
/// AO = SMA(Median Price, 5) - SMA(Median Price, 34)
pub fn awesome_oscillator(highs: &[f64], lows: &[f64]) -> Vec<f64> {
    let median: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .map(|(h, l)| (h + l) / 2.0)
        .collect();

    let sma_fast = super::trend::sma(&median, 5);
    let sma_slow = super::trend::sma(&median, 34);

    sma_fast
        .iter()
        .zip(sma_slow.iter())
        .map(|(f, s)| f - s)
        .collect()
}

/// Ultimate Oscillator
///
/// Комбинирует три периода
pub fn ultimate_oscillator(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period1: usize,
    period2: usize,
    period3: usize,
) -> Vec<f64> {
    let n = closes.len();
    if n < 2 {
        return vec![f64::NAN; n];
    }

    let mut bp = vec![0.0; n]; // Buying Pressure
    let mut tr = vec![0.0; n]; // True Range

    for i in 1..n {
        let prev_close = closes[i - 1];
        let true_low = lows[i].min(prev_close);
        let true_high = highs[i].max(prev_close);

        bp[i] = closes[i] - true_low;
        tr[i] = true_high - true_low;
    }

    let bp_sum1 = utils::rolling(&bp, period1, utils::sum);
    let bp_sum2 = utils::rolling(&bp, period2, utils::sum);
    let bp_sum3 = utils::rolling(&bp, period3, utils::sum);

    let tr_sum1 = utils::rolling(&tr, period1, utils::sum);
    let tr_sum2 = utils::rolling(&tr, period2, utils::sum);
    let tr_sum3 = utils::rolling(&tr, period3, utils::sum);

    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let avg1 = if tr_sum1[i] > 0.0 {
            bp_sum1[i] / tr_sum1[i]
        } else {
            0.0
        };
        let avg2 = if tr_sum2[i] > 0.0 {
            bp_sum2[i] / tr_sum2[i]
        } else {
            0.0
        };
        let avg3 = if tr_sum3[i] > 0.0 {
            bp_sum3[i] / tr_sum3[i]
        } else {
            0.0
        };

        // Веса: 4, 2, 1
        let uo = ((4.0 * avg1) + (2.0 * avg2) + avg3) / 7.0 * 100.0;
        result.push(uo);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_range() {
        let data: Vec<f64> = vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.0, 43.5, 43.25, 43.5, 44.0, 44.25, 44.5,
            44.25, 44.0, 43.5, 43.25, 43.0, 43.0, 43.25,
        ];
        let result = rsi(&data, 14);

        // RSI должен быть в диапазоне [0, 100]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "RSI out of range: {}", val);
            }
        }
    }

    #[test]
    fn test_roc() {
        let data = vec![100.0, 110.0, 121.0, 133.1];
        let result = roc(&data, 1);

        assert!(result[0].is_nan());
        assert!((result[1] - 10.0).abs() < 0.001); // 10% рост
        assert!((result[2] - 10.0).abs() < 0.001); // 10% рост
    }

    #[test]
    fn test_momentum() {
        let data = vec![100.0, 105.0, 110.0, 115.0, 120.0];
        let result = momentum(&data, 2);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 10.0); // 110 - 100
        assert_eq!(result[3], 10.0); // 115 - 105
        assert_eq!(result[4], 10.0); // 120 - 110
    }

    #[test]
    fn test_stochastic_range() {
        let highs = vec![10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 12.0, 11.0, 10.5, 11.0];
        let lows = vec![9.0, 10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 10.0, 9.5, 10.0];
        let closes = vec![9.5, 10.5, 11.5, 11.0, 12.0, 12.5, 11.5, 10.5, 10.0, 10.5];

        let result = stochastic(&highs, &lows, &closes, 5, 3);

        for val in result.k.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0);
            }
        }
    }

    #[test]
    fn test_williams_r_range() {
        let highs = vec![10.0, 11.0, 12.0, 11.5, 12.5];
        let lows = vec![9.0, 10.0, 11.0, 10.5, 11.5];
        let closes = vec![9.5, 10.5, 11.5, 11.0, 12.0];

        let result = williams_r(&highs, &lows, &closes, 3);

        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= -100.0 && *val <= 0.0);
            }
        }
    }
}
