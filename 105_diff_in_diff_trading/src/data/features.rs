//! Feature engineering for financial data.

use super::Candle;

/// Feature set computed from candle data.
#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub returns: Vec<f64>,
    pub log_returns: Vec<f64>,
    pub volatility: Vec<f64>,
    pub volume_change: Vec<f64>,
    pub high_low_ratio: Vec<f64>,
}

/// Compute simple returns from candle data.
pub fn compute_returns(candles: &[Candle]) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![];
    }

    candles
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect()
}

/// Compute log returns from candle data.
pub fn compute_log_returns(candles: &[Candle]) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![];
    }

    candles
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Compute rolling volatility (standard deviation of returns).
pub fn compute_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return vec![f64::NAN; returns.len()];
    }

    let mut result = vec![f64::NAN; window - 1];

    for i in (window - 1)..returns.len() {
        let window_data: Vec<f64> = returns[(i + 1 - window)..=i].to_vec();
        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
        let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / window_data.len() as f64;
        result.push(variance.sqrt());
    }

    result
}

/// Compute volume change from candle data.
pub fn compute_volume_change(candles: &[Candle]) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![];
    }

    candles
        .windows(2)
        .map(|w| {
            if w[0].volume > 0.0 {
                (w[1].volume - w[0].volume) / w[0].volume
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute all technical features from candle data.
pub fn compute_technical_features(candles: &[Candle], volatility_window: usize) -> FeatureSet {
    let returns = compute_returns(candles);
    let log_returns = compute_log_returns(candles);
    let volatility = compute_volatility(&returns, volatility_window);
    let volume_change = compute_volume_change(candles);

    let high_low_ratio: Vec<f64> = candles
        .iter()
        .map(|c| {
            if c.low > 0.0 {
                c.high / c.low
            } else {
                1.0
            }
        })
        .collect();

    FeatureSet {
        returns,
        log_returns,
        volatility,
        volume_change,
        high_low_ratio,
    }
}

/// Compute cumulative returns from a base index.
pub fn compute_cumulative_returns(candles: &[Candle], base_idx: usize) -> Vec<f64> {
    if candles.is_empty() || base_idx >= candles.len() {
        return vec![];
    }

    let base_price = candles[base_idx].close;

    candles
        .iter()
        .map(|c| (c.close - base_price) / base_price)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_test_candles() -> Vec<Candle> {
        let now = Utc::now();
        vec![
            Candle {
                timestamp: now - chrono::Duration::days(4),
                open: 100.0,
                high: 102.0,
                low: 99.0,
                close: 101.0,
                volume: 1000.0,
            },
            Candle {
                timestamp: now - chrono::Duration::days(3),
                open: 101.0,
                high: 105.0,
                low: 100.0,
                close: 104.0,
                volume: 1200.0,
            },
            Candle {
                timestamp: now - chrono::Duration::days(2),
                open: 104.0,
                high: 106.0,
                low: 102.0,
                close: 103.0,
                volume: 900.0,
            },
            Candle {
                timestamp: now - chrono::Duration::days(1),
                open: 103.0,
                high: 108.0,
                low: 101.0,
                close: 107.0,
                volume: 1500.0,
            },
            Candle {
                timestamp: now,
                open: 107.0,
                high: 109.0,
                low: 105.0,
                close: 106.0,
                volume: 1100.0,
            },
        ]
    }

    #[test]
    fn test_compute_returns() {
        let candles = make_test_candles();
        let returns = compute_returns(&candles);

        assert_eq!(returns.len(), 4);
        // First return: (104 - 101) / 101 ≈ 0.0297
        assert!((returns[0] - 0.0297).abs() < 0.001);
    }

    #[test]
    fn test_compute_volatility() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let vol = compute_volatility(&returns, 3);

        assert_eq!(vol.len(), 5);
        assert!(vol[0].is_nan());
        assert!(vol[1].is_nan());
        assert!(vol[2].is_finite());
    }

    #[test]
    fn test_compute_cumulative_returns() {
        let candles = make_test_candles();
        let cum_returns = compute_cumulative_returns(&candles, 0);

        assert_eq!(cum_returns.len(), 5);
        assert!((cum_returns[0] - 0.0).abs() < 0.001); // Base period
        assert!((cum_returns[4] - 0.0495).abs() < 0.001); // (106 - 101) / 101
    }
}
