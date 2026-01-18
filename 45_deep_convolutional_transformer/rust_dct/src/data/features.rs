//! Feature engineering for DCT model

use super::loader::OHLCV;
use ndarray::{Array1, Array2, Axis};

/// Compute technical indicators and features
pub fn compute_features(ohlcv: &OHLCV) -> Array2<f64> {
    let n = ohlcv.len();
    if n < 30 {
        return Array2::zeros((0, 0));
    }

    let mut features: Vec<Array1<f64>> = Vec::new();

    // Log returns
    let log_returns = compute_log_returns(&ohlcv.close);
    features.push(log_returns.clone());

    // Price ratios
    let hl_ratio = compute_hl_ratio(&ohlcv.high, &ohlcv.low, &ohlcv.close);
    features.push(hl_ratio);

    let oc_ratio = compute_oc_ratio(&ohlcv.open, &ohlcv.close);
    features.push(oc_ratio);

    // Moving average ratios
    for window in [5, 10, 20] {
        let ma = simple_moving_average(&ohlcv.close, window);
        let ma_ratio = &ohlcv.close / &ma;
        features.push(ma_ratio);
    }

    // Volatility
    let vol_5 = rolling_std(&log_returns, 5);
    let vol_20 = rolling_std(&log_returns, 20);
    features.push(vol_5);
    features.push(vol_20);

    // RSI
    let rsi = compute_rsi(&ohlcv.close, 14);
    let rsi_normalized = &rsi / 100.0;
    features.push(rsi_normalized);

    // MACD
    let ema_12 = exponential_moving_average(&ohlcv.close, 12);
    let ema_26 = exponential_moving_average(&ohlcv.close, 26);
    let macd = &ema_12 - &ema_26;
    let macd_signal = exponential_moving_average(&macd, 9);
    features.push(macd.clone());
    features.push(macd_signal);

    // Volume ratio
    let vol_ma = simple_moving_average(&ohlcv.volume, 20);
    let vol_ratio = &ohlcv.volume / &vol_ma.mapv(|x| if x == 0.0 { 1.0 } else { x });
    features.push(vol_ratio);

    // Bollinger Band position
    let bb_position = compute_bollinger_position(&ohlcv.close, 20);
    features.push(bb_position);

    // Stack features into 2D array
    let n_features = features.len();
    let mut result = Array2::zeros((n, n_features));

    for (i, feature) in features.iter().enumerate() {
        result.column_mut(i).assign(feature);
    }

    result
}

/// Compute log returns
fn compute_log_returns(prices: &Array1<f64>) -> Array1<f64> {
    let n = prices.len();
    let mut returns = Array1::zeros(n);

    for i in 1..n {
        if prices[i - 1] > 0.0 {
            returns[i] = (prices[i] / prices[i - 1]).ln();
        }
    }

    returns
}

/// Compute high-low ratio
fn compute_hl_ratio(high: &Array1<f64>, low: &Array1<f64>, close: &Array1<f64>) -> Array1<f64> {
    let hl_diff = high - low;
    &hl_diff / close.mapv(|x| if x == 0.0 { 1.0 } else { x })
}

/// Compute open-close ratio
fn compute_oc_ratio(open: &Array1<f64>, close: &Array1<f64>) -> Array1<f64> {
    (close - open) / open.mapv(|x| if x == 0.0 { 1.0 } else { x })
}

/// Simple moving average
fn simple_moving_average(data: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = data.len();
    let mut ma = Array1::zeros(n);

    for i in 0..n {
        if i >= window - 1 {
            let start = i + 1 - window;
            let sum: f64 = data.slice(ndarray::s![start..=i]).sum();
            ma[i] = sum / window as f64;
        } else {
            ma[i] = data[i];
        }
    }

    ma
}

/// Exponential moving average
fn exponential_moving_average(data: &Array1<f64>, span: usize) -> Array1<f64> {
    let n = data.len();
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut ema = Array1::zeros(n);

    ema[0] = data[0];
    for i in 1..n {
        ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1];
    }

    ema
}

/// Rolling standard deviation
fn rolling_std(data: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = data.len();
    let mut std = Array1::zeros(n);

    for i in 0..n {
        if i >= window - 1 {
            let start = i + 1 - window;
            let slice = data.slice(ndarray::s![start..=i]);
            let mean = slice.mean().unwrap_or(0.0);
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            std[i] = variance.sqrt();
        }
    }

    std
}

/// Compute RSI (Relative Strength Index)
fn compute_rsi(prices: &Array1<f64>, period: usize) -> Array1<f64> {
    let n = prices.len();
    let mut rsi = Array1::from_elem(n, 50.0);

    if n < period + 1 {
        return rsi;
    }

    // Calculate price changes
    let mut gains = Array1::zeros(n);
    let mut losses = Array1::zeros(n);

    for i in 1..n {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains[i] = change;
        } else {
            losses[i] = -change;
        }
    }

    // Calculate initial average gain/loss
    let mut avg_gain: f64 = gains.slice(ndarray::s![1..=period]).sum() / period as f64;
    let mut avg_loss: f64 = losses.slice(ndarray::s![1..=period]).sum() / period as f64;

    // Calculate RSI for each point
    for i in period..n {
        if i > period {
            avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
        }

        if avg_loss == 0.0 {
            rsi[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    rsi
}

/// Compute Bollinger Band position (0-1 scale)
fn compute_bollinger_position(prices: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = prices.len();
    let mut position = Array1::from_elem(n, 0.5);

    let ma = simple_moving_average(prices, window);
    let std = rolling_std(prices, window);

    for i in window - 1..n {
        let upper = ma[i] + 2.0 * std[i];
        let lower = ma[i] - 2.0 * std[i];
        let range = upper - lower;

        if range > 0.0 {
            position[i] = (prices[i] - lower) / range;
        }
    }

    position
}

/// Create movement labels
pub fn create_movement_labels(prices: &Array1<f64>, threshold: f64, horizon: usize) -> Array1<i32> {
    let n = prices.len();
    let mut labels = Array1::from_elem(n, 2i32); // Default to Stable

    for i in 0..n.saturating_sub(horizon) {
        let ret = (prices[i + horizon] - prices[i]) / prices[i];

        if ret > threshold {
            labels[i] = 0; // Up
        } else if ret < -threshold {
            labels[i] = 1; // Down
        }
        // else remains 2 (Stable)
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_moving_average() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let ma = simple_moving_average(&data, 3);

        assert!((ma[2] - 2.0).abs() < 1e-10);
        assert!((ma[3] - 3.0).abs() < 1e-10);
        assert!((ma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_bounds() {
        let prices = Array1::from_vec(vec![
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
            110.0, 108.0, 106.0, 108.0, 110.0, 112.0,
        ]);
        let rsi = compute_rsi(&prices, 14);

        for val in rsi.iter() {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }
}
