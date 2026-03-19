//! Trading example: HHL algorithm simulation vs classical solver
//!
//! Fetches BTCUSDT and ETHUSDT hourly data from Bybit, constructs a
//! covariance matrix, and solves for minimum-variance portfolio weights
//! using both the HHL simulation and classical Gaussian elimination.

use anyhow::Result;
use hhl_algorithm_finance::{
    build_covariance_matrix, compare_solvers, fetch_bybit_klines, log_returns,
};

fn main() -> Result<()> {
    println!("=== HHL Algorithm Finance - Trading Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Fetch market data from Bybit
    // -----------------------------------------------------------------------
    let symbols = ["BTCUSDT", "ETHUSDT"];
    let interval = "60"; // 1-hour candles
    let limit = 200;

    println!("Fetching data from Bybit...");
    let mut all_returns: Vec<Vec<f64>> = Vec::new();

    for symbol in &symbols {
        match fetch_bybit_klines(symbol, interval, limit) {
            Ok(klines) => {
                let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
                let rets = log_returns(&closes);
                println!(
                    "  {} : {} candles -> {} returns (mean={:.6}, std={:.6})",
                    symbol,
                    klines.len(),
                    rets.len(),
                    mean_f64(&rets),
                    std_f64(&rets),
                );
                all_returns.push(rets);
            }
            Err(e) => {
                println!("  {} : failed to fetch ({}), using synthetic data", symbol, e);
                all_returns.push(generate_synthetic_returns(symbol));
            }
        }
    }

    // Trim to equal length
    let min_len = all_returns.iter().map(|r| r.len()).min().unwrap_or(0);
    if min_len < 2 {
        println!("\nNot enough data points. Exiting.");
        return Ok(());
    }
    for r in &mut all_returns {
        r.truncate(min_len);
    }
    println!("\nUsing {} return observations per asset.\n", min_len);

    // -----------------------------------------------------------------------
    // 2. Build covariance matrix
    // -----------------------------------------------------------------------
    let cov = build_covariance_matrix(&all_returns);
    println!("Covariance matrix:");
    println!("  [{:.8}  {:.8}]", cov[[0, 0]], cov[[0, 1]]);
    println!("  [{:.8}  {:.8}]\n", cov[[1, 0]], cov[[1, 1]]);

    // Condition number approximation for 2x2
    let trace = cov[[0, 0]] + cov[[1, 1]];
    let det = cov[[0, 0]] * cov[[1, 1]] - cov[[0, 1]] * cov[[1, 0]];
    let disc = ((trace * trace - 4.0 * det).max(0.0)).sqrt();
    let lambda_max = (trace + disc) / 2.0;
    let lambda_min = (trace - disc) / 2.0;
    if lambda_min.abs() > 1e-14 {
        println!(
            "Eigenvalues: {:.8}, {:.8}  (condition number: {:.2})\n",
            lambda_max,
            lambda_min,
            lambda_max / lambda_min
        );
    }

    // -----------------------------------------------------------------------
    // 3. Solve portfolio weights: HHL simulation vs Classical
    // -----------------------------------------------------------------------
    match compare_solvers(&cov) {
        Ok((w_hhl, w_classical, hhl_us, classical_us)) => {
            println!("--- Minimum-Variance Portfolio Weights ---\n");
            println!(
                "  {:>10}  {:>12}  {:>12}",
                "Asset", "HHL Sim", "Classical"
            );
            println!("  {:>10}  {:>12}  {:>12}", "-----", "-------", "---------");
            for (i, symbol) in symbols.iter().enumerate() {
                println!(
                    "  {:>10}  {:>12.6}  {:>12.6}",
                    symbol, w_hhl[i], w_classical[i]
                );
            }

            let diff: f64 = w_hhl
                .iter()
                .zip(w_classical.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            println!("\n  Sum of absolute weight differences: {:.2e}", diff);
            println!(
                "  HHL simulation time:  {} us",
                hhl_us
            );
            println!(
                "  Classical solve time: {} us",
                classical_us
            );
            println!("\n  Note: For 2x2 systems the classical solver is faster.");
            println!("  HHL's advantage emerges for large-scale systems (N >> 1000).");
        }
        Err(e) => {
            println!("Solver comparison failed: {}", e);
        }
    }

    // -----------------------------------------------------------------------
    // 4. Demonstrate HHL on a synthetic well-conditioned system
    // -----------------------------------------------------------------------
    println!("\n\n=== Synthetic 2x2 System Demo ===\n");
    let a = [[4.0, 1.0], [1.0, 3.0]];
    let b = [1.0, 2.0];

    println!("System: A = [[4, 1], [1, 3]],  b = [1, 2]\n");

    match hhl_algorithm_finance::hhl_simulate_2x2(&a, &b) {
        Ok(x_hhl) => {
            let x_classical =
                hhl_algorithm_finance::gaussian_elimination(
                    &vec![vec![4.0, 1.0], vec![1.0, 3.0]],
                    &vec![1.0, 2.0],
                )
                .unwrap();

            println!("  HHL solution:       x = [{:.6}, {:.6}]", x_hhl[0], x_hhl[1]);
            println!(
                "  Classical solution: x = [{:.6}, {:.6}]",
                x_classical[0], x_classical[1]
            );

            // Verify
            let ax0 = a[0][0] * x_hhl[0] + a[0][1] * x_hhl[1];
            let ax1 = a[1][0] * x_hhl[0] + a[1][1] * x_hhl[1];
            println!("\n  Verification: A * x_hhl = [{:.6}, {:.6}]", ax0, ax1);
            println!("  Expected:     b         = [{:.6}, {:.6}]", b[0], b[1]);
        }
        Err(e) => println!("  HHL failed: {}", e),
    }

    println!("\nDone.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mean_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_f64(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean_f64(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

fn generate_synthetic_returns(symbol: &str) -> Vec<f64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    symbol.hash(&mut hasher);
    let seed = hasher.finish();

    // Simple LCG PRNG for reproducibility without extra deps in the example
    let mut state = seed;
    let n = 100;
    let mut returns = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (state >> 33) as f64 / (1u64 << 31) as f64; // [0, 1)
        let r = (u - 0.5) * 0.02; // centered around 0, +/- 1%
        returns.push(r);
    }
    returns
}
