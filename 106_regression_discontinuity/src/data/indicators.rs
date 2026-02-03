//! Technical indicators for RDD analysis.
//!
//! Implements common indicators used as running variables in RDD trading:
//! - RSI (Relative Strength Index)
//! - Moving averages (SMA, EMA)
//! - Price distance from thresholds

/// Compute Relative Strength Index (RSI).
///
/// RSI measures the speed and magnitude of price changes, ranging from 0 to 100.
/// Traditional thresholds: <30 = oversold, >70 = overbought.
///
/// # Arguments
/// * `prices` - Close prices
/// * `period` - Lookback period (typically 14)
pub fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    if n < period + 1 {
        return vec![50.0; n]; // Neutral RSI
    }

    let mut rsi = vec![50.0; n];

    // Calculate price changes
    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];

    for i in 1..n {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains[i] = change;
        } else {
            losses[i] = -change;
        }
    }

    // Initial average gain/loss (SMA)
    let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

    // First RSI value
    if avg_loss > 0.0 {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - (100.0 / (1.0 + rs));
    } else if avg_gain > 0.0 {
        rsi[period] = 100.0;
    } else {
        rsi[period] = 50.0;
    }

    // Subsequent RSI values using Wilder's smoothing
    for i in (period + 1)..n {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        } else if avg_gain > 0.0 {
            rsi[i] = 100.0;
        } else {
            rsi[i] = 50.0;
        }
    }

    rsi
}

/// Compute Simple Moving Average (SMA).
///
/// # Arguments
/// * `prices` - Price data
/// * `period` - Lookback period
pub fn compute_sma(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut sma = vec![f64::NAN; n];

    if n < period {
        return sma;
    }

    // Initial sum
    let mut sum: f64 = prices[..period].iter().sum();
    sma[period - 1] = sum / period as f64;

    // Rolling window
    for i in period..n {
        sum = sum - prices[i - period] + prices[i];
        sma[i] = sum / period as f64;
    }

    sma
}

/// Compute Exponential Moving Average (EMA).
///
/// # Arguments
/// * `prices` - Price data
/// * `period` - Lookback period (determines smoothing factor)
pub fn compute_ema(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut ema = vec![f64::NAN; n];

    if n < period {
        return ema;
    }

    // Smoothing factor
    let alpha = 2.0 / (period + 1) as f64;

    // Initialize with SMA
    let initial_sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
    ema[period - 1] = initial_sma;

    // EMA calculation
    for i in period..n {
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1];
    }

    ema
}

/// Compute price distance from a threshold (in percentage terms).
///
/// Useful for round number effects (e.g., distance from $100).
///
/// # Arguments
/// * `prices` - Price data
/// * `threshold` - The threshold price level
pub fn price_distance_from_threshold(prices: &[f64], threshold: f64) -> Vec<f64> {
    prices
        .iter()
        .map(|&p| (p - threshold) / threshold * 100.0)
        .collect()
}

/// Compute distance to nearest round number.
///
/// Useful for psychological price level analysis.
///
/// # Arguments
/// * `prices` - Price data
/// * `round_level` - Round number granularity (e.g., 100 for $100, $200, etc.)
pub fn distance_to_round_number(prices: &[f64], round_level: f64) -> Vec<f64> {
    prices
        .iter()
        .map(|&p| {
            let nearest_round = (p / round_level).round() * round_level;
            (p - nearest_round) / round_level * 100.0
        })
        .collect()
}

/// Identify RSI threshold crossings.
///
/// Returns indices where RSI crosses the threshold.
///
/// # Arguments
/// * `rsi` - RSI values
/// * `threshold` - Threshold to check (e.g., 30 for oversold)
/// * `direction` - "up" for crossing above, "down" for crossing below
pub fn find_threshold_crossings(rsi: &[f64], threshold: f64, direction: &str) -> Vec<usize> {
    let mut crossings = Vec::new();

    for i in 1..rsi.len() {
        match direction {
            "up" => {
                if rsi[i - 1] < threshold && rsi[i] >= threshold {
                    crossings.push(i);
                }
            }
            "down" => {
                if rsi[i - 1] > threshold && rsi[i] <= threshold {
                    crossings.push(i);
                }
            }
            _ => {}
        }
    }

    crossings
}

/// Compute MACD (Moving Average Convergence Divergence).
///
/// Returns (macd_line, signal_line, histogram).
///
/// # Arguments
/// * `prices` - Close prices
/// * `fast_period` - Fast EMA period (typically 12)
/// * `slow_period` - Slow EMA period (typically 26)
/// * `signal_period` - Signal line period (typically 9)
pub fn compute_macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let fast_ema = compute_ema(prices, fast_period);
    let slow_ema = compute_ema(prices, slow_period);

    // MACD line = Fast EMA - Slow EMA
    let mut macd_line = vec![f64::NAN; n];
    for i in 0..n {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }

    // Signal line = EMA of MACD line
    let valid_macd: Vec<f64> = macd_line.iter().filter(|&&x| !x.is_nan()).cloned().collect();
    let signal_ema = compute_ema(&valid_macd, signal_period);

    let mut signal_line = vec![f64::NAN; n];
    let first_valid = macd_line.iter().position(|&x| !x.is_nan()).unwrap_or(n);
    for (i, &val) in signal_ema.iter().enumerate() {
        if first_valid + i < n {
            signal_line[first_valid + i] = val;
        }
    }

    // Histogram = MACD - Signal
    let mut histogram = vec![f64::NAN; n];
    for i in 0..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    (macd_line, signal_line, histogram)
}

/// Compute Bollinger Bands.
///
/// Returns (middle_band, upper_band, lower_band, percent_b).
/// percent_b shows where price is relative to bands (0 = lower, 1 = upper).
///
/// # Arguments
/// * `prices` - Close prices
/// * `period` - SMA period (typically 20)
/// * `std_dev` - Standard deviation multiplier (typically 2)
pub fn compute_bollinger_bands(
    prices: &[f64],
    period: usize,
    std_dev: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let middle = compute_sma(prices, period);

    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    let mut percent_b = vec![f64::NAN; n];

    for i in (period - 1)..n {
        // Calculate standard deviation
        let slice = &prices[(i + 1 - period)..=i];
        let mean = middle[i];
        let variance: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        upper[i] = mean + std_dev * std;
        lower[i] = mean - std_dev * std;

        let band_width = upper[i] - lower[i];
        if band_width > 0.0 {
            percent_b[i] = (prices[i] - lower[i]) / band_width;
        }
    }

    (middle, upper, lower, percent_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let rsi = compute_rsi(&prices, 14);

        assert_eq!(rsi.len(), prices.len());
        // RSI should be between 0 and 100
        for &r in &rsi[14..] {
            assert!(r >= 0.0 && r <= 100.0, "RSI out of range: {}", r);
        }
    }

    #[test]
    fn test_sma() {
        let prices = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let sma = compute_sma(&prices, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 20.0).abs() < 1e-10); // (10+20+30)/3
        assert!((sma[3] - 30.0).abs() < 1e-10); // (20+30+40)/3
        assert!((sma[4] - 40.0).abs() < 1e-10); // (30+40+50)/3
    }

    #[test]
    fn test_ema() {
        let prices: Vec<f64> = (1..=10).map(|x| x as f64 * 10.0).collect();
        let ema = compute_ema(&prices, 5);

        assert_eq!(ema.len(), prices.len());
        assert!(ema[0].is_nan());
        assert!(!ema[4].is_nan()); // First valid EMA
    }

    #[test]
    fn test_threshold_crossings() {
        let rsi = vec![25.0, 28.0, 31.0, 35.0, 29.0, 30.0, 32.0];
        let crossings_up = find_threshold_crossings(&rsi, 30.0, "up");
        let crossings_down = find_threshold_crossings(&rsi, 30.0, "down");

        assert!(crossings_up.contains(&2)); // 28 -> 31 crosses above 30
        assert!(crossings_down.contains(&4)); // 35 -> 29 crosses below 30
    }
}
