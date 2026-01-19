//! Performance Metrics
//!
//! Calculate various performance and risk metrics for simulation results.

use serde::{Deserialize, Serialize};

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Total return percentage
    pub total_return_pct: f64,
    /// Annualized return (CAGR)
    pub cagr: Option<f64>,
    /// Annualized volatility
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: Option<f64>,
    /// Sortino ratio
    pub sortino_ratio: Option<f64>,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: Option<f64>,
    /// Win rate
    pub win_rate: f64,
    /// Value at Risk (95%)
    pub var_95: f64,
    /// Tracking error vs fundamental
    pub tracking_error: Option<f64>,
    /// Final deviation from fundamental
    pub final_deviation_pct: Option<f64>,
    /// Correlation with fundamental
    pub fundamental_correlation: Option<f64>,
}

/// Calculate performance metrics from price history
pub fn calculate_performance_metrics(
    price_history: &[f64],
    fundamental_history: Option<&[f64]>,
    risk_free_rate: f64,
    periods_per_year: f64,
) -> PerformanceMetrics {
    let n = price_history.len();

    if n < 2 {
        return PerformanceMetrics {
            total_return: 0.0,
            total_return_pct: 0.0,
            cagr: None,
            volatility: 0.0,
            sharpe_ratio: None,
            sortino_ratio: None,
            max_drawdown: 0.0,
            calmar_ratio: None,
            win_rate: 0.0,
            var_95: 0.0,
            tracking_error: None,
            final_deviation_pct: None,
            fundamental_correlation: None,
        };
    }

    // Calculate returns
    let returns: Vec<f64> = price_history
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    // Total return
    let total_return = (price_history[n - 1] - price_history[0]) / price_history[0];

    // CAGR
    let years = (n - 1) as f64 / periods_per_year;
    let cagr = if years > 0.0 {
        Some((price_history[n - 1] / price_history[0]).powf(1.0 / years) - 1.0)
    } else {
        None
    };

    // Volatility
    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let volatility = variance.sqrt() * periods_per_year.sqrt();

    // Sharpe ratio
    let sharpe_ratio = if volatility > 0.0 {
        cagr.map(|c| (c - risk_free_rate) / volatility)
    } else {
        None
    };

    // Sortino ratio (downside deviation)
    let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    let sortino_ratio = if !negative_returns.is_empty() && cagr.is_some() {
        let downside_variance: f64 = negative_returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / negative_returns.len() as f64;
        let downside_vol = downside_variance.sqrt() * periods_per_year.sqrt();
        if downside_vol > 0.0 {
            Some((cagr.unwrap() - risk_free_rate) / downside_vol)
        } else {
            None
        }
    } else {
        None
    };

    // Maximum drawdown
    let mut cumulative = vec![1.0];
    for r in &returns {
        cumulative.push(cumulative.last().unwrap() * (1.0 + r));
    }

    let mut max_dd = 0.0;
    let mut peak = cumulative[0];
    for &val in &cumulative {
        if val > peak {
            peak = val;
        }
        let dd = (peak - val) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    // Calmar ratio
    let calmar_ratio = if max_dd > 0.0 {
        cagr.map(|c| c / max_dd)
    } else {
        None
    };

    // Win rate
    let positive_count = returns.iter().filter(|&&r| r > 0.0).count();
    let win_rate = positive_count as f64 / returns.len() as f64;

    // VaR 95%
    let mut sorted_returns = returns.clone();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let var_idx = (sorted_returns.len() as f64 * 0.05) as usize;
    let var_95 = sorted_returns.get(var_idx).copied().unwrap_or(0.0);

    // Fundamental-related metrics
    let (tracking_error, final_deviation_pct, fundamental_correlation) =
        if let Some(fundamentals) = fundamental_history {
            if fundamentals.len() == price_history.len() {
                // Tracking error
                let deviations: Vec<f64> = price_history.iter()
                    .zip(fundamentals.iter())
                    .map(|(p, f)| p - f)
                    .collect();
                let mean_dev: f64 = deviations.iter().sum::<f64>() / deviations.len() as f64;
                let var_dev: f64 = deviations.iter()
                    .map(|d| (d - mean_dev).powi(2))
                    .sum::<f64>() / deviations.len() as f64;
                let te = var_dev.sqrt();

                // Final deviation
                let final_dev = (price_history[n - 1] - fundamentals[n - 1]) / fundamentals[n - 1] * 100.0;

                // Correlation
                let mean_price: f64 = price_history.iter().sum::<f64>() / n as f64;
                let mean_fund: f64 = fundamentals.iter().sum::<f64>() / n as f64;

                let cov: f64 = price_history.iter()
                    .zip(fundamentals.iter())
                    .map(|(p, f)| (p - mean_price) * (f - mean_fund))
                    .sum::<f64>() / n as f64;

                let std_price = (price_history.iter()
                    .map(|p| (p - mean_price).powi(2))
                    .sum::<f64>() / n as f64).sqrt();

                let std_fund = (fundamentals.iter()
                    .map(|f| (f - mean_fund).powi(2))
                    .sum::<f64>() / n as f64).sqrt();

                let correlation = if std_price > 0.0 && std_fund > 0.0 {
                    Some(cov / (std_price * std_fund))
                } else {
                    None
                };

                (Some(te), Some(final_dev), correlation)
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

    PerformanceMetrics {
        total_return,
        total_return_pct: total_return * 100.0,
        cagr,
        volatility,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown: max_dd,
        calmar_ratio,
        win_rate,
        var_95,
        tracking_error,
        final_deviation_pct,
        fundamental_correlation,
    }
}

/// Bubble detection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubblePeriod {
    /// Start index
    pub start: usize,
    /// Peak index
    pub peak: usize,
    /// End index
    pub end: usize,
    /// Peak deviation from fundamental
    pub peak_deviation: f64,
    /// Duration in steps
    pub duration: usize,
}

/// Bubble analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubbleInfo {
    /// Whether bubble was detected
    pub bubble_detected: bool,
    /// Number of bubble periods
    pub num_bubbles: usize,
    /// Bubble periods
    pub bubble_periods: Vec<BubblePeriod>,
    /// Maximum deviation from fundamental
    pub max_deviation: f64,
    /// Percentage of time in bubble
    pub time_in_bubble_pct: f64,
}

/// Detect bubble formation in price series
pub fn detect_bubble(
    price_history: &[f64],
    fundamental_history: &[f64],
    bubble_threshold: f64,
) -> BubbleInfo {
    if price_history.len() != fundamental_history.len() || price_history.is_empty() {
        return BubbleInfo {
            bubble_detected: false,
            num_bubbles: 0,
            bubble_periods: Vec::new(),
            max_deviation: 0.0,
            time_in_bubble_pct: 0.0,
        };
    }

    // Calculate deviations
    let deviations: Vec<f64> = price_history.iter()
        .zip(fundamental_history.iter())
        .map(|(p, f)| (p - f) / f)
        .collect();

    // Find bubble periods
    let mut bubble_periods = Vec::new();
    let mut in_bubble = false;
    let mut start_idx = 0;
    let mut bubble_steps = 0;

    for (i, &dev) in deviations.iter().enumerate() {
        if dev > bubble_threshold {
            bubble_steps += 1;
            if !in_bubble {
                in_bubble = true;
                start_idx = i;
            }
        } else if in_bubble {
            in_bubble = false;
            // Find peak in this bubble period
            let peak_idx = (start_idx..i)
                .max_by(|&a, &b| {
                    price_history[a].partial_cmp(&price_history[b]).unwrap()
                })
                .unwrap_or(start_idx);

            bubble_periods.push(BubblePeriod {
                start: start_idx,
                peak: peak_idx,
                end: i,
                peak_deviation: deviations[peak_idx],
                duration: i - start_idx,
            });
        }
    }

    // Handle bubble at end
    if in_bubble {
        let peak_idx = (start_idx..deviations.len())
            .max_by(|&a, &b| {
                price_history[a].partial_cmp(&price_history[b]).unwrap()
            })
            .unwrap_or(start_idx);

        bubble_periods.push(BubblePeriod {
            start: start_idx,
            peak: peak_idx,
            end: deviations.len() - 1,
            peak_deviation: deviations[peak_idx],
            duration: deviations.len() - 1 - start_idx,
        });
    }

    let max_deviation = deviations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let time_in_bubble_pct = bubble_steps as f64 / deviations.len() as f64 * 100.0;

    BubbleInfo {
        bubble_detected: !bubble_periods.is_empty(),
        num_bubbles: bubble_periods.len(),
        bubble_periods,
        max_deviation,
        time_in_bubble_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_metrics() {
        let prices = vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0];
        let metrics = calculate_performance_metrics(&prices, None, 0.02, 252.0);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.volatility > 0.0);
        assert!(metrics.win_rate > 0.0);
    }

    #[test]
    fn test_bubble_detection() {
        let prices = vec![100.0, 110.0, 130.0, 160.0, 180.0, 150.0, 110.0, 100.0];
        let fundamentals = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];

        let bubble_info = detect_bubble(&prices, &fundamentals, 0.30);

        assert!(bubble_info.bubble_detected);
        assert!(bubble_info.num_bubbles >= 1);
        assert!(bubble_info.max_deviation > 0.5);
    }

    #[test]
    fn test_no_bubble() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let fundamentals = vec![100.0, 101.0, 102.0, 103.0, 104.0];

        let bubble_info = detect_bubble(&prices, &fundamentals, 0.30);

        assert!(!bubble_info.bubble_detected);
        assert_eq!(bubble_info.num_bubbles, 0);
    }
}
