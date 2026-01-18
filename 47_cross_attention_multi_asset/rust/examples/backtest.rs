//! Backtest Example
//!
//! Demonstrates how to run backtests with the Cross-Attention model.
//!
//! Usage:
//!     cargo run --release --example backtest -- --initial-capital 100000

use cross_attention_multi_asset::strategy::{
    Backtest, BacktestConfig, SignalGenerator, SignalConfig,
};

#[derive(Debug)]
struct BacktestArgs {
    initial_capital: f64,
    transaction_cost: f64,
    rebalance_freq: usize,
    n_steps: usize,
    n_assets: usize,
}

impl Default for BacktestArgs {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            rebalance_freq: 24,
            n_steps: 1000,
            n_assets: 5,
        }
    }
}

fn parse_args() -> BacktestArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut backtest_args = BacktestArgs::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--initial-capital" => {
                if i + 1 < args.len() {
                    backtest_args.initial_capital = args[i + 1].parse().unwrap_or(100_000.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--transaction-cost" => {
                if i + 1 < args.len() {
                    backtest_args.transaction_cost = args[i + 1].parse().unwrap_or(0.001);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--rebalance-freq" => {
                if i + 1 < args.len() {
                    backtest_args.rebalance_freq = args[i + 1].parse().unwrap_or(24);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    backtest_args
}

/// Simple random number for generating mock market data
mod rand {
    static mut SEED: u64 = 42;

    pub fn random() -> f64 {
        unsafe {
            SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
            (SEED >> 33) as f64 / (1u64 << 31) as f64
        }
    }

    pub fn randn() -> f64 {
        // Box-Muller transform for normal distribution
        let u1 = random();
        let u2 = random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Generate realistic mock market data
fn generate_market_data(n_steps: usize, n_assets: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Parameters for different assets
    let mean_returns = vec![0.0001, 0.00015, 0.0002, 0.00005, 0.00012];
    let volatilities = vec![0.02, 0.025, 0.03, 0.015, 0.022];
    let correlations = vec![
        vec![1.0, 0.7, 0.5, 0.3, 0.4],
        vec![0.7, 1.0, 0.6, 0.4, 0.5],
        vec![0.5, 0.6, 1.0, 0.3, 0.6],
        vec![0.3, 0.4, 0.3, 1.0, 0.2],
        vec![0.4, 0.5, 0.6, 0.2, 1.0],
    ];

    // Generate correlated returns
    let mut returns: Vec<Vec<f64>> = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        // Generate independent random normals
        let z: Vec<f64> = (0..n_assets).map(|_| rand::randn()).collect();

        // Apply correlation structure (simplified Cholesky)
        let mut correlated_z = vec![0.0; n_assets];
        for i in 0..n_assets {
            for j in 0..=i {
                correlated_z[i] += correlations[i][j] * z[j];
            }
            // Normalize
            let factor: f64 = (0..=i).map(|j| correlations[i][j].powi(2)).sum::<f64>().sqrt();
            if factor > 0.0 {
                correlated_z[i] /= factor;
            }
        }

        // Apply mean and volatility
        let step_returns: Vec<f64> = (0..n_assets)
            .map(|i| mean_returns[i.min(4)] + volatilities[i.min(4)] * correlated_z[i])
            .collect();

        returns.push(step_returns);
    }

    // Generate model-predicted weights (slightly biased towards high-return assets)
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(n_steps);

    for t in 0..n_steps {
        // Base weights
        let mut w: Vec<f64> = (0..n_assets)
            .map(|i| {
                let base = 1.0 / n_assets as f64;
                let momentum = if t >= 20 {
                    returns[t - 20..t]
                        .iter()
                        .map(|r| r[i])
                        .sum::<f64>()
                        * 10.0
                } else {
                    0.0
                };
                (base + momentum).max(0.0)
            })
            .collect();

        // Normalize
        let total: f64 = w.iter().sum();
        if total > 0.0 {
            for wi in &mut w {
                *wi /= total;
            }
        }

        weights.push(w);
    }

    (weights, returns)
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("Cross-Attention Multi-Asset Trading - Backtest Example");
    println!("{}", "=".repeat(60));

    let args = parse_args();
    println!("\nBacktest configuration:");
    println!("  Initial capital: ${:.2}", args.initial_capital);
    println!("  Transaction cost: {:.2}%", args.transaction_cost * 100.0);
    println!("  Rebalance frequency: {} steps", args.rebalance_freq);
    println!("  Number of steps: {}", args.n_steps);
    println!("  Number of assets: {}", args.n_assets);

    // Generate mock market data
    println!("\n{}", "-".repeat(40));
    println!("Generating market data...");
    println!("{}", "-".repeat(40));

    let (weights, returns) = generate_market_data(args.n_steps, args.n_assets);

    let symbols: Vec<String> = vec![
        "BTC".to_string(),
        "ETH".to_string(),
        "SOL".to_string(),
        "AVAX".to_string(),
        "DOT".to_string(),
    ];

    let timestamps: Vec<i64> = (0..args.n_steps)
        .map(|i| 1704067200000 + i as i64 * 3600000) // Starting from 2024-01-01
        .collect();

    // Print return statistics
    println!("\nReturn Statistics:");
    for (i, sym) in symbols.iter().enumerate() {
        let asset_returns: Vec<f64> = returns.iter().map(|r| r[i]).collect();
        let mean = asset_returns.iter().sum::<f64>() / args.n_steps as f64;
        let variance = asset_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / args.n_steps as f64;
        let std = variance.sqrt();

        println!(
            "  {}: mean={:.4}%, std={:.4}%",
            sym,
            mean * 100.0,
            std * 100.0
        );
    }

    // Run model backtest
    println!("\n{}", "-".repeat(40));
    println!("Running Model Backtest...");
    println!("{}", "-".repeat(40));

    let config = BacktestConfig {
        initial_capital: args.initial_capital,
        transaction_cost: args.transaction_cost,
        rebalance_freq: args.rebalance_freq,
        max_position: 0.5,
        allow_short: false,
        slippage: 0.0005,
    };

    let backtest = Backtest::new(config.clone());
    let model_result = backtest.run(&weights, &returns, &symbols, &timestamps);

    // Run baseline backtest
    println!("\n{}", "-".repeat(40));
    println!("Running Baseline Backtest (Equal Weight)...");
    println!("{}", "-".repeat(40));

    let baseline_result = backtest.compare_baseline(&returns, &symbols, &timestamps);

    // Print results
    println!("\n{}", "-".repeat(40));
    println!("Performance Comparison");
    println!("{}", "-".repeat(40));

    println!("\n{:<25} {:>15} {:>15}", "", "Model", "Baseline");
    println!("{}", "-".repeat(55));

    println!(
        "{:<25} {:>14.2}% {:>14.2}%",
        "Total Return",
        model_result.metrics.total_return * 100.0,
        baseline_result.metrics.total_return * 100.0
    );

    println!(
        "{:<25} {:>14.2}% {:>14.2}%",
        "Annualized Return",
        model_result.metrics.annualized_return * 100.0,
        baseline_result.metrics.annualized_return * 100.0
    );

    println!(
        "{:<25} {:>15.3} {:>15.3}",
        "Sharpe Ratio",
        model_result.metrics.sharpe_ratio,
        baseline_result.metrics.sharpe_ratio
    );

    println!(
        "{:<25} {:>15.3} {:>15.3}",
        "Sortino Ratio",
        model_result.metrics.sortino_ratio,
        baseline_result.metrics.sortino_ratio
    );

    println!(
        "{:<25} {:>15.3} {:>15.3}",
        "Calmar Ratio",
        model_result.metrics.calmar_ratio,
        baseline_result.metrics.calmar_ratio
    );

    println!(
        "{:<25} {:>14.2}% {:>14.2}%",
        "Max Drawdown",
        model_result.metrics.max_drawdown * 100.0,
        baseline_result.metrics.max_drawdown * 100.0
    );

    println!(
        "{:<25} {:>14.2}% {:>14.2}%",
        "Volatility",
        model_result.metrics.volatility * 100.0,
        baseline_result.metrics.volatility * 100.0
    );

    println!(
        "{:<25} {:>14.2}% {:>14.2}%",
        "Win Rate",
        model_result.metrics.win_rate * 100.0,
        baseline_result.metrics.win_rate * 100.0
    );

    println!(
        "{:<25} ${:>14.2} ${:>14.2}",
        "Transaction Costs",
        model_result.metrics.total_transaction_costs,
        baseline_result.metrics.total_transaction_costs
    );

    println!(
        "{:<25} {:>15} {:>15}",
        "Number of Trades",
        model_result.metrics.n_trades,
        baseline_result.metrics.n_trades
    );

    // Print final values
    let final_model_value = model_result.steps.last().unwrap().portfolio_value;
    let final_baseline_value = baseline_result.steps.last().unwrap().portfolio_value;

    println!("\n{}", "-".repeat(40));
    println!("Final Portfolio Values");
    println!("{}", "-".repeat(40));
    println!("Model:    ${:.2}", final_model_value);
    println!("Baseline: ${:.2}", final_baseline_value);
    println!("Outperformance: ${:.2} ({:.2}%)",
             final_model_value - final_baseline_value,
             (final_model_value / final_baseline_value - 1.0) * 100.0);

    // Signal analysis
    println!("\n{}", "-".repeat(40));
    println!("Signal Analysis");
    println!("{}", "-".repeat(40));

    let signal_config = SignalConfig::default();
    let signal_generator = SignalGenerator::new(signal_config);

    // Analyze last few predicted weights
    println!("\nLast 5 weight predictions:");
    println!("{:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
             "Step", &symbols[0], &symbols[1], &symbols[2], &symbols[3], &symbols[4]);

    for i in (args.n_steps - 5)..args.n_steps {
        let w = &weights[i];
        println!(
            "{:>8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            i, w[0], w[1], w[2], w[3], w[4]
        );
    }

    // Generate signals from last weights
    let last_weights = &weights[args.n_steps - 1];
    let signals = signal_generator.generate_from_weights(last_weights, &symbols);

    println!("\nGenerated Signals:");
    for signal in &signals {
        println!(
            "  {} - {:?} (strength: {:.3}, target: {:.3})",
            signal.symbol, signal.action, signal.strength, signal.target_weight
        );
    }

    // Equity curve summary
    println!("\n{}", "-".repeat(40));
    println!("Equity Curve Summary");
    println!("{}", "-".repeat(40));

    let sample_points = [0, 249, 499, 749, args.n_steps - 1];
    println!("{:>8} {:>15} {:>15}", "Step", "Model Value", "Baseline Value");

    for &idx in &sample_points {
        if idx < model_result.steps.len() {
            println!(
                "{:>8} ${:>14.2} ${:>14.2}",
                idx,
                model_result.steps[idx].portfolio_value,
                baseline_result.steps[idx].portfolio_value
            );
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Backtest example complete!");
    println!("{}", "=".repeat(60));
}
