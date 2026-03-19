//! # Grover Search Trading Example
//!
//! Fetches multiple crypto assets from Bybit, defines portfolio selection
//! criteria, runs Grover search to find qualifying portfolios, and compares
//! with brute-force classical search (timing).
//!
//! Run with: `cargo run --example trading_example`

use grover_search_trading::*;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("=============================================================");
    println!("  Grover Search Trading - Portfolio Optimization Example");
    println!("=============================================================\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch asset data from Bybit
    // -----------------------------------------------------------------------
    let symbols = &[
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOTUSDT",
        "AVAXUSDT",
        "MATICUSDT",
    ];

    println!("Fetching market data from Bybit for {} assets...\n", symbols.len());

    let assets = match fetch_bybit_tickers(symbols).await {
        Ok(a) if !a.is_empty() => {
            println!("Successfully fetched {} assets from Bybit:\n", a.len());
            for asset in &a {
                println!(
                    "  {:>10}  price: {:>12.4}  24h change: {:>+7.2}%  vol: {:>14.2}",
                    asset.symbol,
                    asset.last_price,
                    asset.price_change_pct * 100.0,
                    asset.volume_24h
                );
            }
            println!();
            a
        }
        Ok(_) | Err(_) => {
            println!("Could not fetch live data from Bybit. Using synthetic data.\n");
            generate_synthetic_assets(symbols)
        }
    };

    let num_assets = assets.len();
    let num_qubits = num_assets;
    let total_portfolios = 1usize << num_qubits;

    println!("Portfolio search space: 2^{} = {} possible subsets\n", num_qubits, total_portfolios);

    // -----------------------------------------------------------------------
    // Step 2: Prepare returns and covariance
    // -----------------------------------------------------------------------
    let returns: Vec<f64> = assets.iter().map(|a| a.price_change_pct).collect();
    let covariance = build_covariance_matrix(&assets);
    let asset_symbols: Vec<String> = assets.iter().map(|a| a.symbol.clone()).collect();

    println!("Asset returns (24h %):");
    for (i, r) in returns.iter().enumerate() {
        println!("  {:>10}: {:>+7.4}%", asset_symbols[i], r * 100.0);
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 3: Define trading criteria
    // -----------------------------------------------------------------------
    let criteria = TradingCriteria {
        min_return: 0.005,   // Minimum 0.5% return
        max_risk: 0.10,      // Maximum 10% volatility
    };

    println!("Trading Criteria:");
    println!("  Minimum portfolio return: {:.2}%", criteria.min_return * 100.0);
    println!("  Maximum portfolio risk:   {:.2}%", criteria.max_risk * 100.0);
    println!();

    // -----------------------------------------------------------------------
    // Step 4: Classical brute-force search
    // -----------------------------------------------------------------------
    println!("--- Classical Brute-Force Search ---\n");

    let classical_start = Instant::now();
    let marked_states = build_trading_oracle(&returns, &covariance, &criteria);
    let classical_duration = classical_start.elapsed();

    println!(
        "Found {} qualifying portfolios out of {} (evaluated all {}).",
        marked_states.len(),
        total_portfolios,
        total_portfolios
    );
    println!("Classical search time: {:?}\n", classical_duration);

    if marked_states.is_empty() {
        println!("No portfolios meet the criteria. Try relaxing constraints.");
        println!("Adjusting criteria to min_return=0.001, max_risk=0.15...\n");

        let relaxed_criteria = TradingCriteria {
            min_return: 0.001,
            max_risk: 0.15,
        };
        let marked_states = build_trading_oracle(&returns, &covariance, &relaxed_criteria);

        if marked_states.is_empty() {
            println!("Still no qualifying portfolios. Market conditions may be unfavorable.");
            println!("Running Grover search with synthetic marked states for demonstration.\n");
            run_demo_grover(num_qubits, &asset_symbols);
            return;
        }

        run_grover_search(num_qubits, &marked_states, &asset_symbols, &returns, &covariance, &relaxed_criteria, total_portfolios);
    } else {
        // Show some qualifying portfolios
        println!("Sample qualifying portfolios:");
        for (i, &mask) in marked_states.iter().take(5).enumerate() {
            let portfolio = decode_portfolio(mask, &asset_symbols);
            let selected: Vec<usize> = (0..num_assets).filter(|&j| mask & (1 << j) != 0).collect();
            let count = selected.len() as f64;
            let w = 1.0 / count;
            let ret: f64 = selected.iter().map(|&j| returns[j] * w).sum();
            let mut var = 0.0;
            for &a in &selected {
                for &b in &selected {
                    var += w * w * covariance[a][b];
                }
            }
            println!(
                "  {}. [{}] return={:>+7.4}% risk={:>7.4}%",
                i + 1,
                portfolio.join(", "),
                ret * 100.0,
                var.sqrt() * 100.0
            );
        }
        println!();

        run_grover_search(num_qubits, &marked_states, &asset_symbols, &returns, &covariance, &criteria, total_portfolios);
    }
}

fn run_grover_search(
    num_qubits: usize,
    marked_states: &[usize],
    asset_symbols: &[String],
    returns: &[f64],
    covariance: &[Vec<f64>],
    _criteria: &TradingCriteria,
    total_portfolios: usize,
) {
    // -----------------------------------------------------------------------
    // Step 5: Grover quantum search
    // -----------------------------------------------------------------------
    println!("--- Grover Quantum Search (Simulated) ---\n");

    let num_marked = marked_states.len();
    let iters = optimal_iterations(total_portfolios, num_marked);
    println!(
        "Search space: {} states, {} marked solutions",
        total_portfolios, num_marked
    );
    println!(
        "Optimal Grover iterations: {} (vs {} classical evaluations)",
        iters, total_portfolios
    );
    println!(
        "Speedup factor: {:.1}x\n",
        total_portfolios as f64 / iters as f64
    );

    let grover_start = Instant::now();
    let mut gs = GroverSearch::new(num_qubits, marked_states.to_vec());
    let _probs = gs.run();
    let grover_duration = grover_start.elapsed();

    println!("Grover search time: {:?}", grover_duration);

    // Show top results
    let top_results = gs.measure_top_k(5);
    println!("\nTop 5 results by measurement probability:\n");

    let num_assets = returns.len();
    for (rank, (idx, prob)) in top_results.iter().enumerate() {
        let portfolio = decode_portfolio(*idx, asset_symbols);
        let is_solution = marked_states.contains(idx);
        let selected: Vec<usize> = (0..num_assets).filter(|&j| idx & (1 << j) != 0).collect();

        if !selected.is_empty() {
            let count = selected.len() as f64;
            let w = 1.0 / count;
            let ret: f64 = selected.iter().map(|&j| returns[j] * w).sum();
            let mut var = 0.0;
            for &a in &selected {
                for &b in &selected {
                    var += w * w * covariance[a][b];
                }
            }
            println!(
                "  {}. state={:0width$b} prob={:.4} {} [{}] ret={:>+.4}% risk={:.4}%",
                rank + 1,
                idx,
                prob,
                if is_solution { "VALID" } else { "-----" },
                portfolio.join(", "),
                ret * 100.0,
                var.sqrt() * 100.0,
                width = num_assets
            );
        } else {
            println!(
                "  {}. state={:0width$b} prob={:.4} (empty portfolio)",
                rank + 1,
                idx,
                prob,
                width = num_assets
            );
        }
    }

    // Verify Grover found a valid solution
    let (top_idx, top_prob) = gs.measure_top();
    if marked_states.contains(&top_idx) {
        println!("\nGrover search successfully identified a qualifying portfolio!");
        println!("Top measurement probability: {:.4}", top_prob);
    } else {
        println!("\nNote: Top state is not a qualifying portfolio (may need iteration tuning).");
    }

    // -----------------------------------------------------------------------
    // Step 6: Comparison summary
    // -----------------------------------------------------------------------
    println!("\n--- Performance Comparison ---\n");
    println!("  Classical evaluations:    {}", total_portfolios);
    println!("  Grover iterations:        {}", iters);
    println!(
        "  Theoretical speedup:      {:.1}x",
        total_portfolios as f64 / iters as f64
    );
    println!("  Classical wall time:      {:?}", grover_duration);
    println!("  Grover sim wall time:     {:?}", grover_duration);
    println!();
    println!("Note: On a real quantum computer, each Grover iteration is a single");
    println!("quantum circuit execution. The classical simulation overhead would");
    println!("disappear, and the quadratic speedup would be fully realized.");
    println!();
    println!("For {} assets, a real quantum computer would evaluate ~{} portfolios",
        num_assets, iters);
    println!("instead of all {} — a {:.0}x reduction.",
        total_portfolios,
        total_portfolios as f64 / iters.max(1) as f64);
}

fn run_demo_grover(num_qubits: usize, asset_symbols: &[String]) {
    println!("--- Demo: Grover Search with Synthetic Targets ---\n");

    // Mark a few arbitrary states for demonstration
    let marked = vec![3, 7, 12];
    let total = 1usize << num_qubits;
    let iters = optimal_iterations(total, marked.len());

    println!("Marking states {:?} as solutions for demonstration.", marked);
    println!("Optimal iterations: {}\n", iters);

    let mut gs = GroverSearch::new(num_qubits, marked.clone());
    gs.run();

    let top = gs.measure_top_k(5);
    println!("Top 5 measurement results:");
    for (rank, (idx, prob)) in top.iter().enumerate() {
        let portfolio = decode_portfolio(*idx, asset_symbols);
        let tag = if marked.contains(idx) { "MARKED" } else { "------" };
        println!(
            "  {}. state={:0width$b} prob={:.4} {} [{}]",
            rank + 1,
            idx,
            prob,
            tag,
            portfolio.join(", "),
            width = num_qubits
        );
    }
}

/// Generate synthetic asset data when Bybit API is unavailable.
fn generate_synthetic_assets(symbols: &[&str]) -> Vec<AssetData> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let base_prices = [45000.0, 2800.0, 120.0, 0.55, 0.35, 6.5, 28.0, 0.85];

    symbols
        .iter()
        .enumerate()
        .map(|(i, &sym)| {
            let base = if i < base_prices.len() {
                base_prices[i]
            } else {
                10.0
            };
            let change_pct: f64 = rng.gen_range(-0.05..0.08);
            let volatility: f64 = rng.gen_range(0.02..0.06);

            AssetData {
                symbol: sym.to_string(),
                last_price: base * (1.0 + change_pct),
                price_change_pct: change_pct,
                high_24h: base * (1.0 + volatility),
                low_24h: base * (1.0 - volatility),
                volume_24h: rng.gen_range(1_000_000.0..100_000_000.0),
            }
        })
        .collect()
}
