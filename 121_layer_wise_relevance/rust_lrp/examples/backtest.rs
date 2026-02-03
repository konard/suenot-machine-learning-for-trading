//! Example: Run backtest with LRP insights

use clap::Parser;
use rust_lrp::{
    model::{LRPNetwork, LRPConfig},
    data::{prepare_data, normalize_dataset, generate_sample_data},
    strategy::{backtest, BacktestConfig},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Initial capital
    #[arg(short, long, default_value = "100000")]
    capital: f64,

    /// Transaction cost (fraction)
    #[arg(short, long, default_value = "0.001")]
    transaction_cost: f64,

    /// Confidence threshold
    #[arg(long, default_value = "0.55")]
    confidence_threshold: f64,

    /// Use LRP-based position sizing
    #[arg(long, default_value = "true")]
    use_lrp_sizing: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("LRP Backtest");
    println!("============\n");

    println!("Configuration:");
    println!("  Initial Capital: ${:.2}", args.capital);
    println!("  Transaction Cost: {:.2}%", args.transaction_cost * 100.0);
    println!("  Confidence Threshold: {:.2}%", args.confidence_threshold * 100.0);
    println!("  LRP Position Sizing: {}", args.use_lrp_sizing);

    // Create model
    let config = LRPConfig::default()
        .with_epsilon(0.01)
        .with_gamma(0.25);

    // Generate sample data
    println!("\nPreparing data...");
    let features = generate_sample_data(1000);
    let mut dataset = prepare_data(&features, 60, 1);

    println!("  Total samples: {}", dataset.len());

    // Normalize
    normalize_dataset(&mut dataset);

    // Split into train/test
    let (train, _val, test) = dataset.split(0.7, 0.15);

    println!("  Train: {}, Test: {}", train.len(), test.len());

    // Create model
    let mut model = LRPNetwork::new(
        dataset.input_dim(),
        vec![128, 64, 32],
        2,
        config,
    );

    // Run backtest
    println!("\nRunning backtest...");

    let backtest_config = BacktestConfig {
        initial_capital: args.capital,
        transaction_cost: args.transaction_cost,
        confidence_threshold: args.confidence_threshold,
        max_position: 1.0,
        use_lrp_sizing: args.use_lrp_sizing,
    };

    let results = backtest(&mut model, &test, backtest_config);

    // Print results
    println!("\n=====================================");
    println!("         BACKTEST RESULTS");
    println!("=====================================\n");

    println!("Performance Metrics:");
    println!("  Total Return:     {:>10.2}%", results.total_return * 100.0);
    println!("  Final Capital:    ${:>10.2}", results.final_capital);
    println!("  Sharpe Ratio:     {:>10.2}", results.sharpe_ratio);
    println!("  Sortino Ratio:    {:>10.2}", results.sortino_ratio);
    println!("  Max Drawdown:     {:>10.2}%", results.max_drawdown * 100.0);

    println!("\nTrading Statistics:");
    println!("  Win Rate:         {:>10.2}%", results.win_rate * 100.0);
    println!("  Profit Factor:    {:>10.2}", results.profit_factor);
    println!("  Number of Trades: {:>10}", results.n_trades);

    println!("\nLRP Insights:");
    println!("  Avg Confidence:   {:>10.2}%", results.avg_confidence * 100.0);
    println!("  Avg Coherence:    {:>10.4}", results.avg_coherence);

    println!("\nTop Feature Importance:");
    for (name, imp) in results.feature_importance.iter().take(5) {
        println!("  {:<20} {:>6.1}%", name, imp * 100.0);
    }

    println!("\n=====================================");

    Ok(())
}
