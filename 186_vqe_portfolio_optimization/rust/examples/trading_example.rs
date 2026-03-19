//! # VQE Portfolio Optimization - Trading Example
//!
//! Fetches real market data from Bybit for BTC, ETH, and SOL,
//! computes covariance matrix from returns, runs VQE to find
//! the optimal portfolio, and compares with classical solutions.

use ndarray::{Array1, Array2};
use vqe_portfolio_optimization::*;

/// Fallback: generate synthetic data if the API is unavailable.
fn synthetic_data() -> (Array2<f64>, Array1<f64>) {
    println!("Using synthetic market data as fallback...\n");

    // Realistic crypto covariance matrix (annualized, daily scale ~ /252)
    let cov = Array2::from_shape_vec(
        (3, 3),
        vec![
            0.0012, 0.0009, 0.0010, // BTC
            0.0009, 0.0018, 0.0012, // ETH
            0.0010, 0.0012, 0.0025, // SOL
        ],
    )
    .unwrap();

    // Expected daily returns
    let means = Array1::from(vec![0.0003, 0.0005, 0.0007]);

    (cov, means)
}

fn main() {
    println!("==========================================================");
    println!("  VQE Portfolio Optimization - Crypto Trading Example");
    println!("==========================================================\n");

    let asset_names = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    // ---------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // ---------------------------------------------------------------
    println!("Step 1: Fetching market data from Bybit...\n");

    let mut all_returns: Vec<Vec<f64>> = Vec::new();
    let mut fetch_ok = true;

    for symbol in &symbols {
        match fetch_bybit_klines(symbol, "D", 100) {
            Ok(prices) => {
                println!(
                    "  {} : {} candles fetched, latest close = {:.2}",
                    symbol,
                    prices.len(),
                    prices.last().unwrap_or(&0.0)
                );
                let rets = compute_log_returns(&prices);
                all_returns.push(rets);
            }
            Err(e) => {
                println!("  {} : Failed to fetch ({}).", symbol, e);
                fetch_ok = false;
                break;
            }
        }
    }

    let (cov_matrix, expected_returns) = if fetch_ok && !all_returns.is_empty() {
        // Trim all return series to the same length
        let min_len = all_returns.iter().map(|r| r.len()).min().unwrap_or(0);
        if min_len < 10 {
            println!("\n  Not enough data points. Falling back to synthetic data.");
            synthetic_data()
        } else {
            let trimmed: Vec<Vec<f64>> = all_returns
                .iter()
                .map(|r| r[r.len() - min_len..].to_vec())
                .collect();
            println!("\n  Using {} return observations.", min_len);
            compute_covariance_and_means(&trimmed)
        }
    } else {
        synthetic_data()
    };

    // ---------------------------------------------------------------
    // Step 2: Display covariance matrix
    // ---------------------------------------------------------------
    let names_ref: Vec<&str> = asset_names.iter().map(|s| s.as_ref()).collect();
    print_covariance_matrix(&names_ref, &cov_matrix);

    println!("\nExpected Daily Returns:");
    for (i, name) in asset_names.iter().enumerate() {
        println!("  {:<12} {:>10.6}", name, expected_returns[i]);
    }

    // ---------------------------------------------------------------
    // Step 3: Build QUBO matrix
    // ---------------------------------------------------------------
    println!("\nStep 3: Building QUBO matrix...\n");

    let n_assets = asset_names.len();
    let k = 2; // Select 2 out of 3 assets
    let lambda_risk = 1.0;
    let lambda_return = 100.0; // Scale up to make returns significant
    let lambda_budget = 2.0;

    let qubo = build_qubo_matrix(
        &cov_matrix,
        &expected_returns,
        k,
        lambda_risk,
        lambda_return,
        lambda_budget,
    );

    println!("QUBO Matrix ({}x{}):", n_assets, n_assets);
    for i in 0..n_assets {
        for j in 0..n_assets {
            print!("{:>10.4}", qubo[[i, j]]);
        }
        println!();
    }

    // ---------------------------------------------------------------
    // Step 4: Brute-force classical solution
    // ---------------------------------------------------------------
    println!("\nStep 4: Classical brute-force solution...\n");

    let (bf_bits, bf_cost) = brute_force_solve(&qubo, n_assets);
    println!("  Brute-force optimal bits: {:?}", bf_bits);
    println!("  Brute-force QUBO cost:    {:.6}", bf_cost);

    let bf_n_selected = bf_bits.iter().filter(|&&b| b).count().max(1);
    let bf_weights: Vec<f64> = bf_bits
        .iter()
        .map(|&b| if b { 1.0 / bf_n_selected as f64 } else { 0.0 })
        .collect();
    print_portfolio(&names_ref, &bf_weights);

    let (bf_ret, bf_var) =
        classical_equal_weight_portfolio(&cov_matrix, &expected_returns, &bf_bits);
    println!(
        "  Expected return: {:.6}, Variance: {:.6}, Sharpe (daily): {:.4}",
        bf_ret,
        bf_var,
        if bf_var > 0.0 {
            bf_ret / bf_var.sqrt()
        } else {
            0.0
        }
    );

    // ---------------------------------------------------------------
    // Step 5: Classical best-subset search
    // ---------------------------------------------------------------
    println!("\nStep 5: Classical best-subset (select {} of {})...\n", k, n_assets);

    let (cl_bits, cl_ret, cl_var) = classical_best_subset(
        &cov_matrix,
        &expected_returns,
        n_assets,
        k,
        lambda_risk,
        lambda_return,
    );
    let cl_weights: Vec<f64> = cl_bits
        .iter()
        .map(|&b| if b { 1.0 / k as f64 } else { 0.0 })
        .collect();
    println!("  Best classical subset: {:?}", cl_bits);
    print_portfolio(&names_ref, &cl_weights);
    println!(
        "  Expected return: {:.6}, Variance: {:.6}, Sharpe (daily): {:.4}",
        cl_ret,
        cl_var,
        if cl_var > 0.0 {
            cl_ret / cl_var.sqrt()
        } else {
            0.0
        }
    );

    // ---------------------------------------------------------------
    // Step 6: VQE optimization
    // ---------------------------------------------------------------
    println!("\nStep 6: Running VQE optimization...\n");

    let config = VqeConfig {
        n_qubits: n_assets,
        n_layers: 3,
        n_restarts: 10,
        n_iterations: 200,
        perturbation_size: 0.3,
        n_perturbations: 10,
    };

    println!(
        "  Config: {} qubits, {} layers, {} restarts, {} iterations",
        config.n_qubits, config.n_layers, config.n_restarts, config.n_iterations
    );

    let result = run_vqe(&qubo, &config);

    println!("\n  VQE optimal energy:    {:.6}", result.optimal_energy);
    println!("  VQE optimal portfolio: {:?}", result.optimal_portfolio);
    print_portfolio(&names_ref, &result.portfolio_weights);

    let (vqe_ret, vqe_var) = classical_equal_weight_portfolio(
        &cov_matrix,
        &expected_returns,
        &result.optimal_portfolio,
    );
    println!(
        "  Expected return: {:.6}, Variance: {:.6}, Sharpe (daily): {:.4}",
        vqe_ret,
        vqe_var,
        if vqe_var > 0.0 {
            vqe_ret / vqe_var.sqrt()
        } else {
            0.0
        }
    );

    // ---------------------------------------------------------------
    // Step 7: Comparison summary
    // ---------------------------------------------------------------
    println!("\n==========================================================");
    println!("  Comparison Summary");
    println!("==========================================================\n");
    println!(
        "  {:>20} {:>12} {:>12} {:>12}",
        "Method", "Return", "Variance", "QUBO Cost"
    );
    println!("  {:-<58}", "");
    println!(
        "  {:>20} {:>12.6} {:>12.6} {:>12.6}",
        "Brute-force", bf_ret, bf_var, bf_cost
    );
    println!(
        "  {:>20} {:>12.6} {:>12.6} {:>12.6}",
        "Classical subset",
        cl_ret,
        cl_var,
        qubo_cost(&qubo, &cl_bits)
    );
    println!(
        "  {:>20} {:>12.6} {:>12.6} {:>12.6}",
        "VQE", vqe_ret, vqe_var, result.optimal_energy
    );

    println!("\n  VQE vs Brute-force energy gap: {:.6}", result.optimal_energy - bf_cost);

    if (result.optimal_energy - bf_cost).abs() < 1.0 {
        println!("  -> VQE found a near-optimal solution!");
    } else {
        println!("  -> VQE solution has room for improvement. Try more restarts/layers.");
    }

    println!("\n==========================================================");
    println!("  Done.");
    println!("==========================================================");
}
