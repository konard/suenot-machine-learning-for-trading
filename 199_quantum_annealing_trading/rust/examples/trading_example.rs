//! Trading Example: Portfolio Optimization with Quantum Annealing
//!
//! This example fetches multiple crypto assets from Bybit, formulates
//! portfolio selection as a QUBO problem, and solves it with both
//! simulated annealing and simulated quantum annealing, comparing
//! solution quality and convergence.

use anyhow::Result;
use quantum_annealing_trading::*;

const SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
];

const TARGET_ASSETS: usize = 4; // Select 4 out of 10 assets
const KLINE_LIMIT: u32 = 60; // 60 daily candles
const RISK_AVERSION: f64 = 2.0;
const PENALTY: f64 = 10.0;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Quantum Annealing Portfolio Optimization ===\n");
    println!("Fetching data for {} crypto assets from Bybit...\n", SYMBOLS.len());

    // Fetch kline data for all symbols
    let mut all_returns: Vec<Vec<f64>> = Vec::new();
    let mut valid_symbols: Vec<&str> = Vec::new();

    for symbol in SYMBOLS {
        match fetch_klines(symbol, "D", KLINE_LIMIT).await {
            Ok(klines) => {
                if klines.len() < 10 {
                    println!("  [SKIP] {} - insufficient data ({} candles)", symbol, klines.len());
                    continue;
                }
                let returns = compute_log_returns(&klines);
                println!(
                    "  [OK] {} - {} returns, last close: {:.2}",
                    symbol,
                    returns.len(),
                    klines.last().map(|k| k.close).unwrap_or(0.0)
                );
                all_returns.push(returns);
                valid_symbols.push(symbol);
            }
            Err(e) => {
                println!("  [ERR] {} - {}", symbol, e);
            }
        }
    }

    let n = valid_symbols.len();
    if n < TARGET_ASSETS {
        anyhow::bail!(
            "Not enough assets with valid data ({} < {})",
            n,
            TARGET_ASSETS
        );
    }

    println!("\n--- Building QUBO Problem ---\n");

    // Build covariance matrix and expected returns
    let covariance = build_covariance_matrix(&all_returns);
    let expected_returns = compute_mean_returns(&all_returns);

    println!("Covariance matrix ({0}x{0}):", n);
    for i in 0..n {
        let row: Vec<String> = (0..n)
            .map(|j| format!("{:>9.6}", covariance[[i, j]]))
            .collect();
        println!("  {}: [{}]", valid_symbols[i], row.join(", "));
    }

    println!("\nExpected daily returns:");
    for i in 0..n {
        println!("  {}: {:.6}", valid_symbols[i], expected_returns[i]);
    }

    // Build portfolio QUBO
    let builder = PortfolioQuboBuilder::new(
        covariance.clone(),
        expected_returns.clone(),
        RISK_AVERSION,
        TARGET_ASSETS,
        PENALTY,
    );
    let problem = builder.build_qubo();

    println!("\nQUBO matrix constructed: {} variables", problem.num_variables);

    // Analyze energy landscape
    println!("\n--- Energy Landscape Analysis ---\n");
    let stats = EnergyLandscapeAnalyzer::energy_distribution(&problem, 10000);
    println!("Random sampling (10000 solutions):");
    println!("  Min energy:     {:.4}", stats.min);
    println!("  Max energy:     {:.4}", stats.max);
    println!("  Mean energy:    {:.4}", stats.mean);
    println!("  Std dev:        {:.4}", stats.std_dev);
    println!("  Median:         {:.4}", stats.median);
    println!("  10th percentile:{:.4}", stats.percentile_10);

    let local_min_frac = EnergyLandscapeAnalyzer::local_minima_fraction(&problem, 5000);
    println!("  Local minima fraction: {:.2}%", local_min_frac * 100.0);

    // Solve with Simulated Annealing
    println!("\n--- Simulated Annealing (Classical) ---\n");
    let sa_solver = SimulatedAnnealingSolver::new(
        5000,
        10.0,
        AnnealingSchedule::Exponential { alpha: 3.0 },
    );

    let mut sa_best_energy = f64::MAX;
    let mut sa_best_solution = vec![0u8; n];
    let mut sa_best_history = Vec::new();
    let num_runs = 20;

    for run in 0..num_runs {
        let result = sa_solver.solve(&problem);
        if result.best_energy < sa_best_energy {
            sa_best_energy = result.best_energy;
            sa_best_solution = result.best_solution;
            sa_best_history = result.energy_history;
        }
        if run == 0 || (run + 1) % 5 == 0 {
            println!("  Run {}/{}: best energy so far = {:.6}", run + 1, num_runs, sa_best_energy);
        }
    }

    let sa_selected: Vec<&str> = sa_best_solution
        .iter()
        .enumerate()
        .filter(|(_, &x)| x == 1)
        .map(|(i, _)| valid_symbols[i])
        .collect();
    let sa_count: u8 = sa_best_solution.iter().sum();

    println!("\nSA Best solution:");
    println!("  Energy:    {:.6}", sa_best_energy);
    println!("  Selected:  {} assets: {:?}", sa_count, sa_selected);
    println!(
        "  Convergence: {:.6} -> {:.6} ({} sweeps)",
        sa_best_history.first().unwrap_or(&0.0),
        sa_best_history.last().unwrap_or(&0.0),
        sa_best_history.len()
    );

    // Solve with Simulated Quantum Annealing
    println!("\n--- Simulated Quantum Annealing ---\n");
    let sqa_solver = SimulatedQuantumAnnealingSolver::new(
        2000,
        8,    // Trotter slices
        5.0,  // Initial transverse field
        1.0,  // Temperature
        AnnealingSchedule::Exponential { alpha: 2.5 },
    );

    let mut sqa_best_energy = f64::MAX;
    let mut sqa_best_solution = vec![0u8; n];
    let mut sqa_best_history = Vec::new();

    for run in 0..num_runs {
        let result = sqa_solver.solve(&problem);
        if result.best_energy < sqa_best_energy {
            sqa_best_energy = result.best_energy;
            sqa_best_solution = result.best_solution;
            sqa_best_history = result.energy_history;
        }
        if run == 0 || (run + 1) % 5 == 0 {
            println!("  Run {}/{}: best energy so far = {:.6}", run + 1, num_runs, sqa_best_energy);
        }
    }

    let sqa_selected: Vec<&str> = sqa_best_solution
        .iter()
        .enumerate()
        .filter(|(_, &x)| x == 1)
        .map(|(i, _)| valid_symbols[i])
        .collect();
    let sqa_count: u8 = sqa_best_solution.iter().sum();

    println!("\nSQA Best solution:");
    println!("  Energy:    {:.6}", sqa_best_energy);
    println!("  Selected:  {} assets: {:?}", sqa_count, sqa_selected);
    println!(
        "  Convergence: {:.6} -> {:.6} ({} sweeps)",
        sqa_best_history.first().unwrap_or(&0.0),
        sqa_best_history.last().unwrap_or(&0.0),
        sqa_best_history.len()
    );

    // Comparison
    println!("\n--- Comparison ---\n");
    println!(
        "  SA  energy: {:.6} | Selected: {:?}",
        sa_best_energy, sa_selected
    );
    println!(
        "  SQA energy: {:.6} | Selected: {:?}",
        sqa_best_energy, sqa_selected
    );

    let improvement = if sa_best_energy.abs() > 1e-10 {
        (sa_best_energy - sqa_best_energy) / sa_best_energy.abs() * 100.0
    } else {
        0.0
    };

    if sqa_best_energy < sa_best_energy {
        println!(
            "\n  SQA found a better solution by {:.2}%",
            improvement.abs()
        );
    } else if sa_best_energy < sqa_best_energy {
        println!(
            "\n  SA found a better solution by {:.2}%",
            improvement.abs()
        );
    } else {
        println!("\n  Both methods found equivalent solutions");
    }

    // Energy barrier between SA and SQA solutions
    let barrier = EnergyLandscapeAnalyzer::energy_barrier(&problem, &sa_best_solution, &sqa_best_solution);
    println!("  Energy barrier between solutions: {:.6}", barrier);

    println!("\n=== Done ===");
    Ok(())
}
