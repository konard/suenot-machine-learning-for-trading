//! Example: Eigenportfolios
//!
//! This example demonstrates how to construct eigenportfolios
//! from PCA of asset returns using synthetic crypto-like data.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
    println!("===========================================");
    println!("  Eigenportfolios from PCA");
    println!("===========================================");
    println!();

    // Simulate crypto returns
    let symbols = vec![
        "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "AVAX", "LINK",
    ];
    let n_periods = 365;
    let n_assets = symbols.len();

    println!("Simulating {} days of returns for {} crypto assets...", n_periods, n_assets);
    println!();

    let returns = generate_crypto_returns(n_periods, n_assets);

    // Calculate covariance matrix
    let cov = covariance_matrix(&returns);

    println!("Covariance Matrix (top-left 5x5):");
    for i in 0..5.min(n_assets) {
        print!("  ");
        for j in 0..5.min(n_assets) {
            print!("{:>8.5} ", cov[[i, j]]);
        }
        println!();
    }
    println!();

    // Perform eigendecomposition
    let (eigenvalues, eigenvectors) = simple_eigen(&cov);

    // Display explained variance
    let total_var: f64 = eigenvalues.iter().sum();
    println!("Explained Variance by Component:");
    println!("{:-<50}", "");
    let mut cumulative = 0.0;
    for (i, &ev) in eigenvalues.iter().take(5).enumerate() {
        let ratio = ev / total_var;
        cumulative += ratio;
        println!(
            "  PC{}: {:>6.2}% (cumulative: {:>6.2}%)",
            i + 1,
            ratio * 100.0,
            cumulative * 100.0
        );
    }
    println!();

    // Create eigenportfolios
    println!("Eigenportfolio Weights:");
    println!("{:-<70}", "");

    for pc_idx in 0..4.min(n_assets) {
        let weights = normalize_weights(&eigenvectors.column(pc_idx).to_vec());

        println!("\nPortfolio {} (explains {:.1}% variance):", pc_idx + 1, eigenvalues[pc_idx] / total_var * 100.0);

        // Show top holdings
        let mut holdings: Vec<(&str, f64)> = symbols.iter().map(|s| *s).zip(weights.iter().copied()).collect();
        holdings.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        for (symbol, weight) in holdings.iter().take(5) {
            let direction = if *weight > 0.0 { "LONG" } else { "SHORT" };
            println!("    {:>6}: {:>7.2}% ({})", symbol, weight * 100.0, direction);
        }
    }
    println!();

    // Calculate portfolio returns
    println!("Portfolio Performance:");
    println!("{:-<60}", "");
    println!("{:<20} {:>10} {:>10} {:>10}", "Portfolio", "Return%", "Vol%", "Sharpe");
    println!("{:-<60}", "");

    // Equal weight (market)
    let equal_weights: Vec<f64> = vec![1.0 / n_assets as f64; n_assets];
    let market_ret = portfolio_returns(&returns, &equal_weights);
    let (market_mean, market_vol) = mean_and_vol(&market_ret);
    let market_sharpe = market_mean / market_vol * (365.0_f64).sqrt();
    println!(
        "{:<20} {:>9.2}% {:>9.2}% {:>10.2}",
        "Market (Equal Wt)",
        market_mean * 365.0 * 100.0,
        market_vol * (365.0_f64).sqrt() * 100.0,
        market_sharpe
    );

    // Eigenportfolios
    for pc_idx in 0..4.min(n_assets) {
        let weights = normalize_weights(&eigenvectors.column(pc_idx).to_vec());
        let port_ret = portfolio_returns(&returns, &weights);
        let (mean_ret, vol) = mean_and_vol(&port_ret);
        let sharpe = mean_ret / vol * (365.0_f64).sqrt();

        println!(
            "{:<20} {:>9.2}% {:>9.2}% {:>10.2}",
            format!("Eigenportfolio {}", pc_idx + 1),
            mean_ret * 365.0 * 100.0,
            vol * (365.0_f64).sqrt() * 100.0,
            sharpe
        );
    }

    println!();
    println!("Note: Portfolio 1 typically behaves like 'the market'");
    println!("      Other portfolios capture different risk factors");
}

fn generate_crypto_returns(n_periods: usize, n_assets: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let market_normal = Normal::new(0.001, 0.03).unwrap();
    let idio_normal = Normal::new(0.0, 0.02).unwrap();

    let mut returns = Array2::zeros((n_periods, n_assets));

    // Generate market factor
    let market_factor: Vec<f64> = (0..n_periods)
        .map(|_| market_normal.sample(&mut rng))
        .collect();

    // Each asset has different beta to market
    let betas: Vec<f64> = (0..n_assets)
        .map(|i| 0.5 + (i as f64 / n_assets as f64) * 1.0)
        .collect();

    for t in 0..n_periods {
        for i in 0..n_assets {
            let market_component = betas[i] * market_factor[t];
            let idio_component = idio_normal.sample(&mut rng);
            returns[[t, i]] = market_component + idio_component;
        }
    }

    returns
}

fn covariance_matrix(returns: &Array2<f64>) -> Array2<f64> {
    let n = returns.nrows() as f64;
    let d = returns.ncols();

    // Calculate mean
    let mut mean = vec![0.0; d];
    for i in 0..returns.nrows() {
        for j in 0..d {
            mean[j] += returns[[i, j]];
        }
    }
    mean.iter_mut().for_each(|x| *x /= n);

    // Calculate covariance
    let mut cov = Array2::zeros((d, d));
    for i in 0..returns.nrows() {
        for j in 0..d {
            for k in 0..d {
                cov[[j, k]] += (returns[[i, j]] - mean[j]) * (returns[[i, k]] - mean[k]);
            }
        }
    }

    cov /= n - 1.0;
    cov
}

fn simple_eigen(cov: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = cov.nrows();
    let mut eigenvalues = vec![0.0; n];
    let mut eigenvectors = Array2::eye(n);
    let mut work = cov.clone();

    // Power iteration for each eigenvalue
    for i in 0..n {
        let mut v = Array1::zeros(n);
        v[i % n] = 1.0;

        // Power iteration
        for _ in 0..100 {
            let mut new_v = Array1::zeros(n);
            for j in 0..n {
                for k in 0..n {
                    new_v[j] += work[[j, k]] * v[k];
                }
            }

            // Normalize
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                new_v /= norm;
            }

            v = new_v;
        }

        // Eigenvalue = v^T * A * v
        let mut av = Array1::zeros(n);
        for j in 0..n {
            for k in 0..n {
                av[j] += work[[j, k]] * v[k];
            }
        }
        let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(&a, &b)| a * b).sum();

        eigenvalues[i] = eigenvalue.max(0.0);
        for j in 0..n {
            eigenvectors[[j, i]] = v[j];
        }

        // Deflate
        for j in 0..n {
            for k in 0..n {
                work[[j, k]] -= eigenvalue * v[j] * v[k];
            }
        }
    }

    // Sort by eigenvalue (descending)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_eigenvectors = Array2::zeros((n, n));
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        for j in 0..n {
            sorted_eigenvectors[[j, new_idx]] = eigenvectors[[j, old_idx]];
        }
    }

    (sorted_eigenvalues, sorted_eigenvectors)
}

fn normalize_weights(weights: &[f64]) -> Vec<f64> {
    let sum: f64 = weights.iter().sum();
    if sum.abs() > 1e-10 {
        weights.iter().map(|&w| w / sum).collect()
    } else {
        weights.to_vec()
    }
}

fn portfolio_returns(returns: &Array2<f64>, weights: &[f64]) -> Vec<f64> {
    let n = returns.nrows();
    let mut port_ret = vec![0.0; n];

    for t in 0..n {
        for (i, &w) in weights.iter().enumerate() {
            port_ret[t] += w * returns[[t, i]];
        }
    }

    port_ret
}

fn mean_and_vol(returns: &[f64]) -> (f64, f64) {
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);

    (mean, var.sqrt())
}
