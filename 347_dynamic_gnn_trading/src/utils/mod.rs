//! Utility module for metrics and helpers

mod metrics;

pub use metrics::{Metrics, PerformanceTracker, TradeRecord};

use std::time::{SystemTime, UNIX_EPOCH};

/// Get current timestamp in milliseconds
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Get current timestamp in seconds
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Calculate exponential moving average
pub fn ema(values: &[f64], period: usize) -> Vec<f64> {
    if values.is_empty() || period == 0 {
        return vec![];
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = Vec::with_capacity(values.len());

    let mut ema_val = values[0];
    result.push(ema_val);

    for &val in &values[1..] {
        ema_val = alpha * val + (1.0 - alpha) * ema_val;
        result.push(ema_val);
    }

    result
}

/// Calculate simple moving average
pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
    if values.len() < period || period == 0 {
        return vec![];
    }

    let mut result = Vec::with_capacity(values.len() - period + 1);
    let mut sum: f64 = values[..period].iter().sum();
    result.push(sum / period as f64);

    for i in period..values.len() {
        sum = sum - values[i - period] + values[i];
        result.push(sum / period as f64);
    }

    result
}

/// Calculate standard deviation
pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;

    variance.sqrt()
}

/// Calculate returns from prices
pub fn returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| {
            if w[0] != 0.0 {
                (w[1] - w[0]) / w[0]
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate log returns from prices
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate cumulative returns
pub fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(returns.len());
    let mut cumulative = 1.0;

    for &r in returns {
        cumulative *= 1.0 + r;
        result.push(cumulative - 1.0);
    }

    result
}

/// Calculate drawdown series
pub fn drawdowns(equity_curve: &[f64]) -> Vec<f64> {
    if equity_curve.is_empty() {
        return vec![];
    }

    let mut peak = equity_curve[0];
    let mut drawdowns = Vec::with_capacity(equity_curve.len());

    for &val in equity_curve {
        if val > peak {
            peak = val;
        }
        let dd = if peak > 0.0 {
            (peak - val) / peak
        } else {
            0.0
        };
        drawdowns.push(dd);
    }

    drawdowns
}

/// Find maximum drawdown
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    let dds = drawdowns(equity_curve);
    dds.into_iter().fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema_vals = ema(&values, 3);
        assert_eq!(ema_vals.len(), 5);
        assert!(ema_vals[4] > ema_vals[0]);
    }

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma_vals = sma(&values, 3);
        assert_eq!(sma_vals.len(), 3);
        assert_eq!(sma_vals[0], 2.0); // (1+2+3)/3
    }

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let rets = returns(&prices);
        assert_eq!(rets.len(), 2);
        assert!((rets[0] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 90.0, 95.0, 80.0, 100.0];
        let mdd = max_drawdown(&equity);
        // Max DD from 110 to 80 = 27.27%
        assert!((mdd - 0.2727).abs() < 0.01);
    }
}
