//! Example: PCA Analysis with Bybit Data
//!
//! This example demonstrates how to fetch real cryptocurrency data
//! from Bybit and perform PCA analysis on it.

use std::time::Duration;
use std::thread;

fn main() {
    println!("===========================================");
    println!("  PCA Analysis with Bybit Crypto Data");
    println!("===========================================");
    println!();

    // Configuration
    let n_symbols = 10;
    let n_candles = 100;
    let category = "linear"; // USDT perpetuals

    println!("Configuration:");
    println!("  Symbols: {} (top by 24h turnover)", n_symbols);
    println!("  Candles: {} daily", n_candles);
    println!("  Category: {} (USDT perpetuals)", category);
    println!();

    // Note: This example shows the API structure
    // In production, you would use the actual BybitClient
    println!("Note: This example demonstrates the workflow.");
    println!("For live data, run: cargo run -- fetch -n {} -l {}", n_symbols, n_candles);
    println!();

    // Simulate what the analysis would look like with mock data
    demonstrate_with_mock_data(n_symbols, n_candles);
}

fn demonstrate_with_mock_data(n_symbols: usize, n_candles: usize) {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let symbols = vec![
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT",
    ];
    let symbols: Vec<&str> = symbols.into_iter().take(n_symbols).collect();

    println!("Generating mock data for demonstration...");
    println!("Symbols: {:?}", symbols);
    println!();

    // Generate mock returns with realistic crypto characteristics
    let mut rng = rand::thread_rng();
    let market_dist = Normal::new(0.001, 0.04).unwrap(); // High volatility
    let idio_dist = Normal::new(0.0, 0.03).unwrap();

    // Market factor (BTC-like)
    let market_factor: Vec<f64> = (0..n_candles)
        .map(|_| market_dist.sample(&mut rng))
        .collect();

    // Generate returns
    let mut returns: Vec<Vec<f64>> = Vec::new();
    let betas = [1.0, 1.2, 0.8, 0.9, 1.1, 1.3, 1.5, 0.7, 1.0, 0.85];

    for (i, _symbol) in symbols.iter().enumerate() {
        let beta = betas[i % betas.len()];
        let asset_returns: Vec<f64> = market_factor
            .iter()
            .map(|&m| beta * m + idio_dist.sample(&mut rng))
            .collect();
        returns.push(asset_returns);
    }

    // Calculate return statistics
    println!("Return Statistics:");
    println!("{:-<70}", "");
    println!("{:<12} {:>10} {:>10} {:>10} {:>10}", "Symbol", "Mean%", "Std%", "Min%", "Max%");
    println!("{:-<70}", "");

    for (i, symbol) in symbols.iter().enumerate() {
        let ret = &returns[i];
        let mean = ret.iter().sum::<f64>() / ret.len() as f64;
        let var = ret.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (ret.len() - 1) as f64;
        let std = var.sqrt();
        let min = ret.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = ret.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!(
            "{:<12} {:>9.2}% {:>9.2}% {:>9.2}% {:>9.2}%",
            symbol,
            mean * 100.0,
            std * 100.0,
            min * 100.0,
            max * 100.0
        );
    }
    println!();

    // Calculate correlation matrix
    println!("Correlation Matrix:");
    println!("{:-<80}", "");
    print!("{:<12}", "");
    for symbol in symbols.iter().take(8) {
        print!(" {:>8}", &symbol[..symbol.len().min(8)]);
    }
    println!();

    for (i, symbol) in symbols.iter().enumerate().take(8) {
        print!("{:<12}", &symbol[..symbol.len().min(12)]);
        for j in 0..8.min(n_symbols) {
            let corr = correlation(&returns[i], &returns[j]);
            print!(" {:>8.2}", corr);
        }
        println!();
    }
    println!();

    // Perform PCA
    println!("PCA Analysis:");
    println!("{:-<50}", "");

    // Simplified PCA using power iteration
    let cov = covariance(&returns);
    let (eigenvalues, eigenvectors) = power_iteration_pca(&cov, 5);

    let total_var: f64 = eigenvalues.iter().sum();
    let mut cumulative = 0.0;

    println!("{:>5} {:>12} {:>10} {:>12}", "PC", "Eigenvalue", "Var%", "Cumulative%");
    println!("{:-<50}", "");

    for (i, &ev) in eigenvalues.iter().enumerate() {
        let var_pct = ev / total_var * 100.0;
        cumulative += var_pct;
        println!("{:>5} {:>12.6} {:>9.1}% {:>11.1}%", i + 1, ev, var_pct, cumulative);
    }
    println!();

    // Interpretation
    println!("Interpretation:");
    println!("{:-<60}", "");
    println!("PC1 ({:.1}%): Market factor - all cryptos move together", eigenvalues[0] / total_var * 100.0);
    if eigenvalues.len() > 1 {
        println!("PC2 ({:.1}%): Alt/BTC rotation factor", eigenvalues[1] / total_var * 100.0);
    }
    if eigenvalues.len() > 2 {
        println!("PC3 ({:.1}%): Sector-specific factor", eigenvalues[2] / total_var * 100.0);
    }
    println!();

    // Top loadings for PC1
    println!("PC1 Loadings (Market Factor):");
    if !eigenvectors.is_empty() {
        let mut loadings: Vec<(&str, f64)> = symbols
            .iter()
            .map(|&s| s)
            .zip(eigenvectors[0].iter().copied())
            .collect();
        loadings.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        for (symbol, loading) in loadings {
            println!("  {:>12}: {:>8.4}", symbol, loading);
        }
    }

    println!();
    println!("===========================================");
    println!("  Analysis Complete");
    println!("===========================================");
    println!();
    println!("To use real Bybit data, run:");
    println!("  cargo run -- fetch -n 10 -l 100");
    println!("  cargo run -- analyze -i data/prices.csv");
    println!("  cargo run -- portfolio -i data/prices.csv -n 4");
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n as usize {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

fn covariance(returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_assets = returns.len();
    let n_periods = returns[0].len() as f64;

    // Calculate means
    let means: Vec<f64> = returns
        .iter()
        .map(|r| r.iter().sum::<f64>() / n_periods)
        .collect();

    // Calculate covariance
    let mut cov = vec![vec![0.0; n_assets]; n_assets];

    for i in 0..n_assets {
        for j in 0..n_assets {
            let mut sum = 0.0;
            for t in 0..n_periods as usize {
                sum += (returns[i][t] - means[i]) * (returns[j][t] - means[j]);
            }
            cov[i][j] = sum / (n_periods - 1.0);
        }
    }

    cov
}

fn power_iteration_pca(cov: &[Vec<f64>], n_components: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = cov.len();
    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Vec::new();
    let mut work = cov.to_vec();

    for _ in 0..n_components.min(n) {
        // Initialize vector
        let mut v = vec![1.0 / (n as f64).sqrt(); n];

        // Power iteration
        for _ in 0..50 {
            // Matrix-vector multiply
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += work[i][j] * v[j];
                }
            }

            // Normalize
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                new_v.iter_mut().for_each(|x| *x /= norm);
            }

            v = new_v;
        }

        // Calculate eigenvalue
        let mut av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += work[i][j] * v[j];
            }
        }
        let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(&a, &b)| a * b).sum();

        eigenvalues.push(eigenvalue.max(0.0));
        eigenvectors.push(v.clone());

        // Deflate
        for i in 0..n {
            for j in 0..n {
                work[i][j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}
