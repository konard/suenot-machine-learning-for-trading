//! Feature engineering for SSM-GNN model inputs.

use super::bybit::Kline;

/// Compute technical features from kline data.
///
/// Returns a feature matrix (seq_len, 5) with columns:
/// [returns, volatility, rsi, sma_ratio, momentum]
pub fn compute_technical_features(klines: &[Kline], window: usize) -> Vec<Vec<f64>> {
    let n = klines.len();
    let mut features = vec![vec![0.0; 5]; n];

    // Returns
    for t in 1..n {
        let prev = klines[t - 1].close;
        if prev.abs() > 1e-10 {
            features[t][0] = (klines[t].close - prev) / prev;
        }
    }

    // Rolling volatility
    for t in window..n {
        let returns: Vec<f64> = (t - window..t).map(|i| features[i][0]).collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        features[t][1] = var.sqrt();
    }

    // RSI
    for t in window..n {
        let mut gains = 0.0;
        let mut losses = 0.0;
        let mut n_gains = 0;
        let mut n_losses = 0;

        for i in (t - window)..t {
            let ret = features[i][0];
            if ret > 0.0 {
                gains += ret;
                n_gains += 1;
            } else if ret < 0.0 {
                losses -= ret;
                n_losses += 1;
            }
        }

        let avg_gain = if n_gains > 0 { gains / n_gains as f64 } else { 0.0 };
        let avg_loss = if n_losses > 0 { losses / n_losses as f64 } else { 1e-10 };
        let rs = avg_gain / (avg_loss + 1e-10);
        features[t][2] = 100.0 - 100.0 / (1.0 + rs);
    }

    // SMA ratio
    for t in window..n {
        let sma: f64 = (t - window..t).map(|i| klines[i].close).sum::<f64>() / window as f64;
        if sma.abs() > 1e-10 {
            features[t][3] = klines[t].close / sma - 1.0;
        }
    }

    // Momentum (rate of change)
    for t in window..n {
        let prev = klines[t - window].close;
        if prev.abs() > 1e-10 {
            features[t][4] = (klines[t].close - prev) / prev;
        }
    }

    features
}

/// Build a correlation graph from multi-asset price data.
///
/// # Arguments
/// * `close_prices` - Vector of close price vectors, one per asset
/// * `threshold` - Minimum absolute correlation for edge creation
///
/// # Returns
/// (edge_sources, edge_targets, edge_weights)
pub fn build_correlation_graph(
    close_prices: &[Vec<f64>],
    threshold: f64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n_assets = close_prices.len();

    // Compute returns
    let returns: Vec<Vec<f64>> = close_prices
        .iter()
        .map(|prices| {
            (1..prices.len())
                .map(|t| {
                    if prices[t - 1].abs() > 1e-10 {
                        (prices[t] - prices[t - 1]) / prices[t - 1]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    // Compute correlation matrix
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    let mut weights = Vec::new();

    for i in 0..n_assets {
        for j in 0..n_assets {
            if i == j {
                continue;
            }

            let corr = pearson_correlation(&returns[i], &returns[j]);
            if corr.abs() >= threshold {
                sources.push(i);
                targets.push(j);
                weights.push(corr.abs());
            }
        }
    }

    // Fallback to fully connected if no edges
    if sources.is_empty() {
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i != j {
                    sources.push(i);
                    targets.push(j);
                    weights.push(1.0 / n_assets as f64);
                }
            }
        }
    }

    (sources, targets, weights)
}

/// Compute Pearson correlation between two vectors.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    let mean_x = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y = y[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_simulated_klines;

    #[test]
    fn test_compute_features() {
        let klines = generate_simulated_klines("BTCUSDT", 50);
        let features = compute_technical_features(&klines, 14);
        assert_eq!(features.len(), 50);
        assert_eq!(features[0].len(), 5);
    }

    #[test]
    fn test_correlation_graph() {
        let prices = vec![
            vec![100.0, 101.0, 102.0, 101.5, 103.0],
            vec![50.0, 50.5, 51.0, 50.8, 51.5],
            vec![200.0, 199.0, 198.0, 199.5, 197.0],
        ];

        let (src, dst, wt) = build_correlation_graph(&prices, 0.3);
        assert!(!src.is_empty());
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len(), wt.len());
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);
    }
}
