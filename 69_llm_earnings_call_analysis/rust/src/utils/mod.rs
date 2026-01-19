//! Utility functions and types
//!
//! Common utilities for the earnings call analyzer.

use chrono::{DateTime, Utc};

/// Format timestamp to human-readable string
pub fn format_timestamp(timestamp: i64) -> String {
    let dt = DateTime::<Utc>::from_timestamp(timestamp / 1000, 0);
    match dt {
        Some(dt) => dt.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        None => format!("Invalid timestamp: {}", timestamp),
    }
}

/// Calculate percentage change
pub fn pct_change(old_value: f64, new_value: f64) -> f64 {
    if old_value == 0.0 {
        return 0.0;
    }
    (new_value - old_value) / old_value * 100.0
}

/// Calculate moving average
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window || window == 0 {
        return vec![];
    }

    data.windows(window)
        .map(|w| w.iter().sum::<f64>() / window as f64)
        .collect()
}

/// Calculate standard deviation
pub fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

    variance.sqrt()
}

/// Calculate z-score
pub fn z_score(value: f64, mean: f64, std: f64) -> f64 {
    if std == 0.0 {
        return 0.0;
    }
    (value - mean) / std
}

/// Normalize values to 0-1 range
pub fn normalize(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let range = max - min;
    if range == 0.0 {
        return vec![0.5; data.len()];
    }

    data.iter().map(|&x| (x - min) / range).collect()
}

/// Calculate exponential moving average
pub fn ema(data: &[f64], window: usize) -> Vec<f64> {
    if data.is_empty() || window == 0 {
        return vec![];
    }

    let multiplier = 2.0 / (window as f64 + 1.0);
    let mut result = Vec::with_capacity(data.len());

    // Start with SMA for the first point
    if data.len() >= window {
        let first_sma: f64 = data[..window].iter().sum::<f64>() / window as f64;
        result.push(first_sma);

        for &value in &data[window..] {
            let prev_ema = *result.last().unwrap();
            let new_ema = (value - prev_ema) * multiplier + prev_ema;
            result.push(new_ema);
        }
    }

    result
}

/// Clean and normalize text for analysis
pub fn clean_text(text: &str) -> String {
    text.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>()
        .join(" ")
        .to_lowercase()
}

/// Extract numbers from text
pub fn extract_numbers(text: &str) -> Vec<f64> {
    let re = regex::Regex::new(r"-?\d+\.?\d*").unwrap();
    re.find_iter(text)
        .filter_map(|m| m.as_str().parse().ok())
        .collect()
}

/// Calculate returns from price series
pub fn calculate_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Calculate cumulative returns
pub fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
    if returns.is_empty() {
        return vec![];
    }

    let mut cumulative = Vec::with_capacity(returns.len());
    let mut cum_product = 1.0;

    for &ret in returns {
        cum_product *= 1.0 + ret;
        cumulative.push(cum_product - 1.0);
    }

    cumulative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pct_change() {
        assert!((pct_change(100.0, 110.0) - 10.0).abs() < 0.001);
        assert!((pct_change(100.0, 90.0) - (-10.0)).abs() < 0.001);
        assert_eq!(pct_change(0.0, 100.0), 0.0);
    }

    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&data, 3);

        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 0.001);
        assert!((ma[1] - 3.0).abs() < 0.001);
        assert!((ma[2] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = std_dev(&data);

        // Expected: ~2.0
        assert!((std - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize(&data);

        assert_eq!(normalized.len(), 5);
        assert!((normalized[0] - 0.0).abs() < 0.001);
        assert!((normalized[4] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_clean_text() {
        let text = "  Hello   \n  World  \n";
        let cleaned = clean_text(text);

        assert_eq!(cleaned, "hello world");
    }

    #[test]
    fn test_extract_numbers() {
        let text = "Revenue grew 25.5% to $100 million";
        let numbers = extract_numbers(text);

        assert!(numbers.contains(&25.5));
        assert!(numbers.contains(&100.0));
    }

    #[test]
    fn test_calculate_returns() {
        let prices = vec![100.0, 110.0, 99.0];
        let returns = calculate_returns(&prices);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 0.001);
        assert!((returns[1] - (-0.1)).abs() < 0.001);
    }
}
