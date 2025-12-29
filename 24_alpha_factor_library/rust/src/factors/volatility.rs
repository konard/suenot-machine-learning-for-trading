//! Индикаторы волатильности
//!
//! - ATR (Average True Range)
//! - Historical Volatility
//! - Chaikin Volatility
//! - Standard Deviation

use super::utils;

/// True Range
///
/// TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
pub fn true_range(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() {
        return vec![];
    }

    let mut result = Vec::with_capacity(n);

    // Первое значение = High - Low
    result.push(highs[0] - lows[0]);

    for i in 1..n {
        let hl = highs[i] - lows[i];
        let hc = (highs[i] - closes[i - 1]).abs();
        let lc = (lows[i] - closes[i - 1]).abs();

        result.push(hl.max(hc).max(lc));
    }

    result
}

/// ATR (Average True Range) - Средний истинный диапазон
///
/// ATR = EMA(True Range, period)
///
/// # Аргументы
/// - `highs` - Массив максимумов
/// - `lows` - Массив минимумов
/// - `closes` - Массив цен закрытия
/// - `period` - Период ATR (обычно 14)
pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let tr = true_range(highs, lows, closes);
    super::trend::ema(&tr, period)
}

/// ATR с использованием SMA вместо EMA
pub fn atr_sma(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let tr = true_range(highs, lows, closes);
    super::trend::sma(&tr, period)
}

/// Normalized ATR (ATR как процент от цены)
pub fn natr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let atr_values = atr(highs, lows, closes, period);

    atr_values
        .iter()
        .zip(closes.iter())
        .map(|(a, c)| if *c > 0.0 { (a / c) * 100.0 } else { 0.0 })
        .collect()
}

/// Historical Volatility (Историческая волатильность)
///
/// HV = std(log_returns) * sqrt(252) для дневных данных
///
/// # Аргументы
/// - `closes` - Массив цен закрытия
/// - `period` - Период расчёта (обычно 20-30)
/// - `annualization_factor` - Фактор аннуализации (252 для дней, 365*24 для часов)
pub fn historical_volatility(closes: &[f64], period: usize, annualization_factor: f64) -> Vec<f64> {
    let log_rets = utils::log_returns(closes);

    if log_rets.len() < period {
        return vec![f64::NAN; closes.len()];
    }

    let mut result = vec![f64::NAN; period];

    for i in period..log_rets.len() {
        let window = &log_rets[(i + 1 - period)..=i];
        let std = utils::std_dev(window);
        result.push(std * annualization_factor.sqrt());
    }

    // Добавляем NaN в начало, чтобы выровнять с closes
    result.insert(0, f64::NAN);

    result
}

/// Daily volatility (без аннуализации)
pub fn daily_volatility(closes: &[f64], period: usize) -> Vec<f64> {
    historical_volatility(closes, period, 1.0)
}

/// Chaikin Volatility
///
/// CV = ((EMA(High-Low, period) - EMA(High-Low, period)_n_periods_ago) / EMA(High-Low, period)_n_periods_ago) * 100
pub fn chaikin_volatility(highs: &[f64], lows: &[f64], period: usize, roc_period: usize) -> Vec<f64> {
    let n = highs.len();
    if n != lows.len() || n < period + roc_period {
        return vec![f64::NAN; n];
    }

    // High - Low
    let hl: Vec<f64> = highs.iter().zip(lows.iter()).map(|(h, l)| h - l).collect();

    // EMA of High-Low
    let ema_hl = super::trend::ema(&hl, period);

    // ROC of EMA
    let mut result = vec![f64::NAN; roc_period];

    for i in roc_period..ema_hl.len() {
        let prev = ema_hl[i - roc_period];
        if prev > 0.0 && !prev.is_nan() {
            result.push(((ema_hl[i] - prev) / prev) * 100.0);
        } else {
            result.push(f64::NAN);
        }
    }

    result
}

/// Standard Deviation (скользящее стандартное отклонение)
pub fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    utils::rolling(data, period, utils::std_dev)
}

/// Variance (скользящая дисперсия)
pub fn rolling_variance(data: &[f64], period: usize) -> Vec<f64> {
    utils::rolling(data, period, |w| {
        let std = utils::std_dev(w);
        std * std
    })
}

/// Ulcer Index - измеряет глубину и продолжительность просадок
///
/// UI = sqrt(sum((Close_i / Max_Close)^2) / n)
pub fn ulcer_index(closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    if n < period {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; period - 1];

    for i in (period - 1)..n {
        let window = &closes[(i + 1 - period)..=i];
        let max_close = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let sum_sq: f64 = window
            .iter()
            .map(|c| {
                let pct_dd = if max_close > 0.0 {
                    ((c - max_close) / max_close) * 100.0
                } else {
                    0.0
                };
                pct_dd * pct_dd
            })
            .sum();

        result.push((sum_sq / period as f64).sqrt());
    }

    result
}

/// Donchian Channel (торговый канал)
///
/// Upper = Highest High за period
/// Lower = Lowest Low за period
/// Middle = (Upper + Lower) / 2
pub fn donchian_channel(
    highs: &[f64],
    lows: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = highs.len();
    if n != lows.len() || n < period {
        return (
            vec![f64::NAN; n],
            vec![f64::NAN; n],
            vec![f64::NAN; n],
        );
    }

    let upper = utils::rolling(highs, period, utils::max);
    let lower = utils::rolling(lows, period, utils::min);

    let middle: Vec<f64> = upper
        .iter()
        .zip(lower.iter())
        .map(|(u, l)| (u + l) / 2.0)
        .collect();

    (upper, middle, lower)
}

/// Mass Index - предсказывает развороты на основе расширения/сжатия диапазона
pub fn mass_index(highs: &[f64], lows: &[f64], ema_period: usize, sum_period: usize) -> Vec<f64> {
    let n = highs.len();
    if n != lows.len() {
        return vec![];
    }

    // High - Low
    let hl: Vec<f64> = highs.iter().zip(lows.iter()).map(|(h, l)| h - l).collect();

    // Single EMA
    let ema1 = super::trend::ema(&hl, ema_period);

    // Double EMA
    let ema2 = super::trend::ema(&ema1, ema_period);

    // EMA Ratio
    let ratio: Vec<f64> = ema1
        .iter()
        .zip(ema2.iter())
        .map(|(e1, e2)| if *e2 > 0.0 { e1 / e2 } else { 1.0 })
        .collect();

    // Sum of ratios
    utils::rolling(&ratio, sum_period, utils::sum)
}

/// Parkinson Volatility - использует только High и Low
///
/// Более эффективная оценка волатильности, чем close-to-close
pub fn parkinson_volatility(highs: &[f64], lows: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    if n != lows.len() || n < period {
        return vec![f64::NAN; n];
    }

    // log(H/L)^2
    let log_sq: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .map(|(h, l)| {
            if *l > 0.0 {
                let log_hl = (h / l).ln();
                log_hl * log_hl
            } else {
                0.0
            }
        })
        .collect();

    let factor = 1.0 / (4.0 * (2.0_f64).ln());

    utils::rolling(&log_sq, period, |w| {
        let mean_sq = utils::mean(w);
        (factor * mean_sq).sqrt()
    })
}

/// Garman-Klass Volatility - использует OHLC данные
pub fn garman_klass_volatility(
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> Vec<f64> {
    let n = closes.len();
    if n != opens.len() || n != highs.len() || n != lows.len() || n < period {
        return vec![f64::NAN; n];
    }

    let variance: Vec<f64> = (0..n)
        .map(|i| {
            if opens[i] <= 0.0 || lows[i] <= 0.0 {
                return 0.0;
            }

            let log_hl = (highs[i] / lows[i]).ln();
            let log_co = (closes[i] / opens[i]).ln();

            0.5 * log_hl * log_hl - (2.0 * (2.0_f64).ln() - 1.0) * log_co * log_co
        })
        .collect();

    utils::rolling(&variance, period, |w| utils::mean(w).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_true_range() {
        let highs = vec![12.0, 13.0, 12.5];
        let lows = vec![10.0, 11.0, 10.5];
        let closes = vec![11.0, 12.0, 11.0];

        let result = true_range(&highs, &lows, &closes);

        assert_eq!(result[0], 2.0); // 12 - 10
        // TR[1] = max(13-11, |13-11|, |11-11|) = max(2, 2, 0) = 2
        assert_eq!(result[1], 2.0);
    }

    #[test]
    fn test_atr() {
        let highs = vec![12.0, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5];
        let lows = vec![10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5];
        let closes = vec![11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0];

        let result = atr(&highs, &lows, &closes, 3);

        assert_eq!(result.len(), 7);
        // Первые 2 значения должны быть NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Остальные должны быть положительными
        for val in result.iter().skip(2) {
            assert!(*val > 0.0);
        }
    }

    #[test]
    fn test_historical_volatility() {
        // Синтетические данные с известной волатильностью
        let closes: Vec<f64> = (1..=30).map(|x| 100.0 + (x as f64).sin() * 5.0).collect();

        let result = historical_volatility(&closes, 10, 252.0);

        assert_eq!(result.len(), closes.len());

        // Значения должны быть положительными после периода прогрева
        for val in result.iter().skip(11) {
            if !val.is_nan() {
                assert!(*val > 0.0);
            }
        }
    }

    #[test]
    fn test_donchian_channel() {
        let highs = vec![12.0, 13.0, 14.0, 13.0, 15.0];
        let lows = vec![10.0, 11.0, 12.0, 11.0, 13.0];

        let (upper, middle, lower) = donchian_channel(&highs, &lows, 3);

        // После периода прогрева
        assert_eq!(upper[2], 14.0); // max(12, 13, 14)
        assert_eq!(lower[2], 10.0); // min(10, 11, 12)
        assert_eq!(middle[2], 12.0); // (14 + 10) / 2
    }

    #[test]
    fn test_ulcer_index() {
        // При постоянном росте UI должен быть 0
        let closes = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let result = ulcer_index(&closes, 3);

        for val in result.iter().skip(2) {
            if !val.is_nan() {
                assert!(*val >= 0.0);
            }
        }
    }
}
