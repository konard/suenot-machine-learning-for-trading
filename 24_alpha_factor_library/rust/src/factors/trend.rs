//! Трендовые индикаторы
//!
//! - SMA (Simple Moving Average) - Простая скользящая средняя
//! - EMA (Exponential Moving Average) - Экспоненциальная скользящая средняя
//! - MACD (Moving Average Convergence Divergence) - Схождение/расхождение скользящих средних
//! - Bollinger Bands - Полосы Боллинджера

use super::utils;

/// Простая скользящая средняя (SMA)
///
/// SMA = (P1 + P2 + ... + Pn) / n
///
/// # Аргументы
/// - `data` - Массив цен
/// - `period` - Период усреднения
///
/// # Пример
/// ```
/// use alpha_factors::factors::sma;
/// let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
/// let sma_3 = sma(&prices, 3);
/// assert_eq!(sma_3[2], 11.0); // (10 + 11 + 12) / 3
/// ```
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    utils::rolling(data, period, utils::mean)
}

/// Экспоненциальная скользящая средняя (EMA)
///
/// EMA = Price * k + EMA_prev * (1 - k)
/// где k = 2 / (period + 1)
///
/// # Аргументы
/// - `data` - Массив цен
/// - `period` - Период EMA
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![];
    }

    if data.len() < period {
        return vec![f64::NAN; data.len()];
    }

    let k = 2.0 / (period as f64 + 1.0);
    let mut result = vec![f64::NAN; period - 1];

    // Первое значение EMA = SMA
    let initial_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
    result.push(initial_sma);

    // Последующие значения
    for i in period..data.len() {
        let prev_ema = result[i - 1];
        let current_ema = data[i] * k + prev_ema * (1.0 - k);
        result.push(current_ema);
    }

    result
}

/// Двойная экспоненциальная скользящая средняя (DEMA)
///
/// DEMA = 2 * EMA - EMA(EMA)
pub fn dema(data: &[f64], period: usize) -> Vec<f64> {
    let ema1 = ema(data, period);
    let ema2 = ema(&ema1, period);

    ema1.iter()
        .zip(ema2.iter())
        .map(|(e1, e2)| 2.0 * e1 - e2)
        .collect()
}

/// Тройная экспоненциальная скользящая средняя (TEMA)
///
/// TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
pub fn tema(data: &[f64], period: usize) -> Vec<f64> {
    let ema1 = ema(data, period);
    let ema2 = ema(&ema1, period);
    let ema3 = ema(&ema2, period);

    ema1.iter()
        .zip(ema2.iter())
        .zip(ema3.iter())
        .map(|((e1, e2), e3)| 3.0 * e1 - 3.0 * e2 + e3)
        .collect()
}

/// Взвешенная скользящая средняя (WMA)
///
/// WMA = (P1*n + P2*(n-1) + ... + Pn*1) / (n + (n-1) + ... + 1)
pub fn wma(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![];
    }

    let weight_sum: f64 = (1..=period).sum::<usize>() as f64;

    utils::rolling(data, period, |window| {
        let weighted_sum: f64 = window
            .iter()
            .enumerate()
            .map(|(i, &price)| price * (i + 1) as f64)
            .sum();
        weighted_sum / weight_sum
    })
}

/// Результат MACD
#[derive(Debug, Clone)]
pub struct MACDResult {
    /// Линия MACD
    pub macd_line: Vec<f64>,
    /// Сигнальная линия
    pub signal_line: Vec<f64>,
    /// Гистограмма
    pub histogram: Vec<f64>,
}

/// MACD (Moving Average Convergence Divergence)
///
/// - MACD Line = EMA(fast) - EMA(slow)
/// - Signal Line = EMA(MACD Line, signal_period)
/// - Histogram = MACD Line - Signal Line
///
/// # Аргументы
/// - `data` - Массив цен
/// - `fast_period` - Период быстрой EMA (обычно 12)
/// - `slow_period` - Период медленной EMA (обычно 26)
/// - `signal_period` - Период сигнальной линии (обычно 9)
pub fn macd(data: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> MACDResult {
    let ema_fast = ema(data, fast_period);
    let ema_slow = ema(data, slow_period);

    // MACD Line
    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(fast, slow)| fast - slow)
        .collect();

    // Signal Line
    let signal_line = ema(&macd_line, signal_period);

    // Histogram
    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    MACDResult {
        macd_line,
        signal_line,
        histogram,
    }
}

/// Результат Bollinger Bands
#[derive(Debug, Clone)]
pub struct BollingerResult {
    /// Верхняя полоса
    pub upper: Vec<f64>,
    /// Средняя линия (SMA)
    pub middle: Vec<f64>,
    /// Нижняя полоса
    pub lower: Vec<f64>,
    /// Ширина полос (bandwidth)
    pub bandwidth: Vec<f64>,
    /// %B индикатор
    pub percent_b: Vec<f64>,
}

/// Полосы Боллинджера (Bollinger Bands)
///
/// - Middle Band = SMA(period)
/// - Upper Band = Middle + (std_dev * num_std)
/// - Lower Band = Middle - (std_dev * num_std)
///
/// # Аргументы
/// - `data` - Массив цен
/// - `period` - Период SMA (обычно 20)
/// - `num_std` - Количество стандартных отклонений (обычно 2)
pub fn bollinger_bands(data: &[f64], period: usize, num_std: f64) -> BollingerResult {
    let middle = sma(data, period);
    let std = utils::rolling(data, period, utils::std_dev_pop);

    let mut upper = Vec::with_capacity(data.len());
    let mut lower = Vec::with_capacity(data.len());
    let mut bandwidth = Vec::with_capacity(data.len());
    let mut percent_b = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        let m = middle[i];
        let s = std[i] * num_std;

        let u = m + s;
        let l = m - s;

        upper.push(u);
        lower.push(l);

        // Bandwidth = (Upper - Lower) / Middle
        if m != 0.0 && !m.is_nan() {
            bandwidth.push((u - l) / m);
        } else {
            bandwidth.push(f64::NAN);
        }

        // %B = (Price - Lower) / (Upper - Lower)
        let band_width = u - l;
        if band_width != 0.0 && !band_width.is_nan() {
            percent_b.push((data[i] - l) / band_width);
        } else {
            percent_b.push(f64::NAN);
        }
    }

    BollingerResult {
        upper,
        middle,
        lower,
        bandwidth,
        percent_b,
    }
}

/// Канал Кельтнера (Keltner Channel)
///
/// - Middle = EMA(period)
/// - Upper = Middle + (ATR * multiplier)
/// - Lower = Middle - (ATR * multiplier)
pub fn keltner_channel(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
    multiplier: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let middle = ema(closes, period);
    let atr = super::volatility::atr(highs, lows, closes, period);

    let upper: Vec<f64> = middle
        .iter()
        .zip(atr.iter())
        .map(|(m, a)| m + a * multiplier)
        .collect();

    let lower: Vec<f64> = middle
        .iter()
        .zip(atr.iter())
        .map(|(m, a)| m - a * multiplier)
        .collect();

    (upper, middle, lower)
}

/// Parabolic SAR (Stop and Reverse)
///
/// Упрощённая реализация
pub fn parabolic_sar(highs: &[f64], lows: &[f64], af_start: f64, af_step: f64, af_max: f64) -> Vec<f64> {
    if highs.len() != lows.len() || highs.len() < 2 {
        return vec![];
    }

    let n = highs.len();
    let mut sar = vec![0.0; n];
    let mut is_long = true;
    let mut af = af_start;
    let mut ep = lows[0]; // Extreme point

    sar[0] = highs[0]; // Начинаем с максимума

    for i in 1..n {
        // Обновляем SAR
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1]);

        if is_long {
            // В длинной позиции
            if lows[i] < sar[i] {
                // Разворот на короткую
                is_long = false;
                sar[i] = ep;
                ep = lows[i];
                af = af_start;
            } else {
                // Обновляем EP и AF
                if highs[i] > ep {
                    ep = highs[i];
                    af = (af + af_step).min(af_max);
                }
            }
        } else {
            // В короткой позиции
            if highs[i] > sar[i] {
                // Разворот на длинную
                is_long = true;
                sar[i] = ep;
                ep = highs[i];
                af = af_start;
            } else {
                // Обновляем EP и AF
                if lows[i] < ep {
                    ep = lows[i];
                    af = (af + af_step).min(af_max);
                }
            }
        }
    }

    sar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0); // (1+2+3)/3
        assert_eq!(result[3], 3.0); // (2+3+4)/3
        assert_eq!(result[4], 4.0); // (3+4+5)/3
    }

    #[test]
    fn test_ema() {
        let data = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let result = ema(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Первое значение = SMA
        assert_eq!(result[2], 11.0); // (10+11+12)/3
    }

    #[test]
    fn test_macd() {
        let data: Vec<f64> = (1..=30).map(|x| x as f64).collect();
        let result = macd(&data, 12, 26, 9);

        assert_eq!(result.macd_line.len(), data.len());
        assert_eq!(result.signal_line.len(), data.len());
        assert_eq!(result.histogram.len(), data.len());
    }

    #[test]
    fn test_bollinger_bands() {
        let data = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0];
        let result = bollinger_bands(&data, 5, 2.0);

        assert_eq!(result.upper.len(), data.len());
        assert_eq!(result.middle.len(), data.len());
        assert_eq!(result.lower.len(), data.len());

        // Middle должна быть между upper и lower
        for i in 4..data.len() {
            assert!(result.upper[i] >= result.middle[i]);
            assert!(result.middle[i] >= result.lower[i]);
        }
    }
}
