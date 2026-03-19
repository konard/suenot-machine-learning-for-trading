//! # VQE Portfolio Optimization - Trading Example
//!
//! Fetches BTC and ETH prices from Bybit, formulates portfolio as QUBO,
//! runs VQE-inspired classical optimization, and compares with benchmarks.

use vqe_portfolio_optimization::*;

fn main() -> anyhow::Result<()> {
    println!("=============================================================");
    println!("  VQE Portfolio Optimization - Bybit Crypto Portfolio");
    println!("=============================================================\n");

    // Asset configuration
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"];
    let asset_names = vec!["BTC", "ETH", "SOL", "BNB"];
    let n_assets = symbols.len();

    // Try to fetch real data from Bybit, fall back to synthetic data
    println!("Fetching price data from Bybit...\n");
    let all_returns = match fetch_all_returns(&symbols) {
        Ok(returns) => {
            println!("Successfully fetched real market data.\n");
            returns
        }
        Err(e) => {
            println!("Could not fetch Bybit data: {}. Using synthetic data.\n", e);
            generate_synthetic_returns(n_assets, 30)
        }
    };

    // Compute statistics
    let (mean_returns, cov_matrix) = compute_statistics(&all_returns);

    println!("--- Asset Statistics ---");
    for (i, name) in asset_names.iter().enumerate() {
        println!(
            "  {}: mean daily return = {:.6}, volatility = {:.6}",
            name,
            mean_returns[i],
            cov_matrix[[i, i]].sqrt()
        );
    }
    println!();

    // Configuration
    let bits_per_asset = 2;
    let risk_aversion = 0.5;
    let penalty = 2.0;
    let n_layers = 2;
    let max_iterations = 300;

    // =========================================================================
    // Strategy 1: Equal Weight
    // =========================================================================
    let equal_weights = vec![1.0 / n_assets as f64; n_assets];
    let eq_ret = portfolio_return(&equal_weights, &mean_returns);
    let eq_risk = portfolio_risk(&equal_weights, &cov_matrix);
    let eq_sharpe = sharpe_ratio(&equal_weights, &mean_returns, &cov_matrix);

    println!("=== Strategy 1: Equal Weight ===");
    print_weights(&asset_names, &equal_weights);
    println!("  Expected daily return: {:.6}", eq_ret);
    println!("  Daily risk (std dev):  {:.6}", eq_risk);
    println!("  Sharpe ratio:          {:.4}", eq_sharpe);
    println!();

    // =========================================================================
    // Strategy 2: Classical Mean-Variance
    // =========================================================================
    let mv_weights = classical_mean_variance(&mean_returns, &cov_matrix, risk_aversion);
    let mv_ret = portfolio_return(&mv_weights, &mean_returns);
    let mv_risk = portfolio_risk(&mv_weights, &cov_matrix);
    let mv_sharpe = sharpe_ratio(&mv_weights, &mean_returns, &cov_matrix);

    println!("=== Strategy 2: Classical Mean-Variance ===");
    print_weights(&asset_names, &mv_weights);
    println!("  Expected daily return: {:.6}", mv_ret);
    println!("  Daily risk (std dev):  {:.6}", mv_risk);
    println!("  Sharpe ratio:          {:.4}", mv_sharpe);
    println!();

    // =========================================================================
    // Strategy 3: VQE Quantum-Inspired (Discrete)
    // =========================================================================
    println!("=== Strategy 3: VQE Quantum-Inspired ===");
    println!("  Running VQE optimization ({} qubits, {} layers, {} iterations)...",
        n_assets * bits_per_asset, n_layers, max_iterations);

    let result = run_portfolio_optimization(
        &mean_returns,
        &cov_matrix,
        n_assets,
        bits_per_asset,
        risk_aversion,
        penalty,
        n_layers,
        max_iterations,
    );

    let vqe_weights = &result.weights;
    let vqe_ret = portfolio_return(vqe_weights, &mean_returns);
    let vqe_risk = portfolio_risk(vqe_weights, &cov_matrix);
    let vqe_sharpe = sharpe_ratio(vqe_weights, &mean_returns, &cov_matrix);

    println!("  Converged after {} iterations", result.iterations);
    println!("  Optimal energy: {:.6}", result.optimal_energy);
    println!("  Optimal bitstring: {:0width$b}", result.optimal_bitstring, width = n_assets * bits_per_asset);
    print_weights(&asset_names, vqe_weights);
    println!("  Expected daily return: {:.6}", vqe_ret);
    println!("  Daily risk (std dev):  {:.6}", vqe_risk);
    println!("  Sharpe ratio:          {:.4}", vqe_sharpe);
    println!();

    // =========================================================================
    // Comparison
    // =========================================================================
    println!("=== Comparison Summary ===");
    println!("{:<25} {:>10} {:>10} {:>10}", "Strategy", "Return", "Risk", "Sharpe");
    println!("{:-<25} {:-^10} {:-^10} {:-^10}", "", "", "", "");
    println!("{:<25} {:>10.6} {:>10.6} {:>10.4}", "Equal Weight", eq_ret, eq_risk, eq_sharpe);
    println!("{:<25} {:>10.6} {:>10.6} {:>10.4}", "Classical MV", mv_ret, mv_risk, mv_sharpe);
    println!("{:<25} {:>10.6} {:>10.6} {:>10.4}", "VQE Quantum-Inspired", vqe_ret, vqe_risk, vqe_sharpe);
    println!();

    // Highlight improvements
    if vqe_sharpe > eq_sharpe {
        println!(
            "VQE improved Sharpe ratio by {:.2}% over equal weight.",
            (vqe_sharpe / eq_sharpe - 1.0) * 100.0
        );
    }

    println!("\nNote: VQE uses discrete {}-bit weights (multiples of 1/{}).",
        bits_per_asset, (1 << bits_per_asset) - 1);
    println!("This naturally regularizes the portfolio, avoiding extreme allocations.");

    Ok(())
}

/// Fetch returns for all symbols from Bybit.
fn fetch_all_returns(symbols: &[&str]) -> anyhow::Result<Vec<Vec<f64>>> {
    let mut all_returns = Vec::new();
    for symbol in symbols {
        let prices = fetch_bybit_klines(symbol, "D", 60)?;
        if prices.len() < 10 {
            anyhow::bail!("Insufficient price data for {}", symbol);
        }
        let returns = compute_log_returns(&prices);
        all_returns.push(returns);
    }
    Ok(all_returns)
}

/// Generate synthetic return data for testing.
fn generate_synthetic_returns(n_assets: usize, n_days: usize) -> Vec<Vec<f64>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Realistic daily return parameters for crypto assets
    let means = [0.001, 0.0015, 0.002, 0.0008]; // BTC, ETH, SOL, BNB
    let vols = [0.03, 0.04, 0.06, 0.035];

    let mut all_returns = Vec::new();
    for i in 0..n_assets {
        let returns: Vec<f64> = (0..n_days)
            .map(|_| {
                let z: f64 = rng.gen_range(-3.0..3.0);
                means[i % means.len()] + vols[i % vols.len()] * z
            })
            .collect();
        all_returns.push(returns);
    }
    all_returns
}

/// Print portfolio weights nicely.
fn print_weights(names: &[&str], weights: &[f64]) {
    println!("  Weights:");
    for (name, w) in names.iter().zip(weights.iter()) {
        println!("    {}: {:.4} ({:.1}%)", name, w, w * 100.0);
    }
}
