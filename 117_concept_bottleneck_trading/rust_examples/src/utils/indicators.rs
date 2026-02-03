//! Technical indicators for trading analysis.

/// Calculate Simple Moving Average (SMA).
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    data.windows(period)
        .map(|w| w.iter().sum::<f64>() / period as f64)
        .collect()
}

/// Calculate Exponential Moving Average (EMA).
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return vec![];
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut result = Vec::with_capacity(data.len());

    // Initialize with SMA for first value
    if data.len() >= period {
        let initial_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
        result.push(initial_sma);

        for &price in &data[period..] {
            let prev_ema = *result.last().unwrap();
            let new_ema = (price - prev_ema) * multiplier + prev_ema;
            result.push(new_ema);
        }
    }

    result
}

/// Calculate Relative Strength Index (RSI).
pub fn rsi(closes: &[f64], period: usize) -> Vec<f64> {
    if closes.len() < period + 1 {
        return vec![];
    }

    let changes: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();
    let mut result = Vec::with_capacity(changes.len() - period + 1);

    for i in period..=changes.len() {
        let window = &changes[i - period..i];

        let gains: f64 = window.iter().filter(|&&x| x > 0.0).sum();
        let losses: f64 = window.iter().filter(|&&x| x < 0.0).map(|x| -x).sum();

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        let rsi_value = if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        };

        result.push(rsi_value);
    }

    result
}

/// Calculate MACD (Moving Average Convergence Divergence).
pub fn macd(
    closes: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = ema(closes, fast_period);
    let slow_ema = ema(closes, slow_period);

    if fast_ema.len() < slow_ema.len() || slow_ema.is_empty() {
        return (vec![], vec![], vec![]);
    }

    // Align EMAs
    let offset = fast_ema.len() - slow_ema.len();
    let macd_line: Vec<f64> = fast_ema[offset..]
        .iter()
        .zip(&slow_ema)
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = ema(&macd_line, signal_period);

    // Align MACD and signal
    let hist_offset = macd_line.len() - signal_line.len();
    let histogram: Vec<f64> = macd_line[hist_offset..]
        .iter()
        .zip(&signal_line)
        .map(|(m, s)| m - s)
        .collect();

    (macd_line, signal_line, histogram)
}

/// Calculate Bollinger Bands.
pub fn bollinger_bands(
    closes: &[f64],
    period: usize,
    num_std: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if closes.len() < period {
        return (vec![], vec![], vec![]);
    }

    let middle: Vec<f64> = sma(closes, period);
    let mut upper = Vec::with_capacity(middle.len());
    let mut lower = Vec::with_capacity(middle.len());

    for (i, &mid) in middle.iter().enumerate() {
        let window = &closes[i..i + period];
        let mean = mid;
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        upper.push(mid + num_std * std);
        lower.push(mid - num_std * std);
    }

    (upper, middle, lower)
}

/// Calculate Average True Range (ATR).
pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() < 2 || highs.len() != lows.len() || highs.len() != closes.len() {
        return vec![];
    }

    // Calculate True Range
    let mut tr = Vec::with_capacity(highs.len() - 1);
    for i in 1..highs.len() {
        let hl = highs[i] - lows[i];
        let hc = (highs[i] - closes[i - 1]).abs();
        let lc = (lows[i] - closes[i - 1]).abs();
        tr.push(hl.max(hc).max(lc));
    }

    // Calculate ATR as SMA of True Range
    sma(&tr, period)
}

/// Calculate Rate of Change (ROC).
pub fn roc(closes: &[f64], period: usize) -> Vec<f64> {
    if closes.len() <= period {
        return vec![];
    }

    closes
        .windows(period + 1)
        .map(|w| (w[period] - w[0]) / w[0] * 100.0)
        .collect()
}

/// Calculate standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Calculate rolling standard deviation.
pub fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    data.windows(period).map(|w| std_dev(w)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let closes = vec![44.0, 44.25, 44.5, 43.75, 44.5, 44.25, 44.75, 45.0, 45.5, 46.0];
        let result = rsi(&closes, 5);
        assert!(!result.is_empty());
        assert!(result.iter().all(|&x| x >= 0.0 && x <= 100.0));
    }

    #[test]
    fn test_bollinger_bands() {
        let closes: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let (upper, middle, lower) = bollinger_bands(&closes, 10, 2.0);

        assert!(!upper.is_empty());
        assert_eq!(upper.len(), middle.len());
        assert_eq!(middle.len(), lower.len());

        for ((&u, &m), &l) in upper.iter().zip(&middle).zip(&lower) {
            assert!(u > m);
            assert!(m > l);
        }
    }

    #[test]
    fn test_roc() {
        let closes = vec![100.0, 102.0, 105.0, 103.0, 108.0];
        let result = roc(&closes, 2);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 1e-10); // (105 - 100) / 100 * 100
    }
}
