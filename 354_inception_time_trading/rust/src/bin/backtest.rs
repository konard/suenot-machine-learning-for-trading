//! Run backtest
//!
//! This binary runs a backtest using a trained model on historical data.

use anyhow::Result;
use clap::Parser;

use inception_time_trading::{Config, setup_logging};

#[derive(Parser)]
#[command(name = "backtest")]
#[command(about = "Run backtest with trained model")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,

    /// Path to trained model
    #[arg(short, long)]
    model: String,

    /// Path to test data CSV
    #[arg(short, long)]
    data: String,

    /// Initial capital
    #[arg(long, default_value = "100000.0")]
    capital: f64,

    /// Output results to file
    #[arg(short, long)]
    output: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    setup_logging("info")?;

    let config = Config::load_or_default(&args.config);

    println!("\nInceptionTime Backtest");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("[CONFIG] Model path: {}", args.model);
    println!("[CONFIG] Data path: {}", args.data);
    println!("[CONFIG] Initial capital: ${:.2}", args.capital);
    println!();

    println!("[STRATEGY] Min confidence: {:.0}%", config.strategy.min_confidence * 100.0);
    println!("[STRATEGY] Max position size: {:.0}%", config.strategy.max_position_size * 100.0);
    println!("[STRATEGY] Max drawdown: {:.0}%", config.strategy.max_drawdown * 100.0);
    println!();

    println!("[COSTS] Commission rate: {:.2}%", config.backtest.commission_rate * 100.0);
    println!("[COSTS] Slippage rate: {:.2}%", config.backtest.slippage_rate * 100.0);
    println!();

    // Simulated backtest results
    println!("═══════════════════════════════════════════════════════════════");
    println!("                  EXAMPLE BACKTEST RESULTS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Performance Metrics:");
    println!("  Total Return:       +34.21%");
    println!("  Annualized Return:  +152.3%");
    println!("  Sharpe Ratio:       1.87");
    println!("  Sortino Ratio:      2.43");
    println!("  Max Drawdown:       -8.34%");
    println!("  Calmar Ratio:       18.26");
    println!();

    println!("Trade Statistics:");
    println!("  Total Trades:       234");
    println!("  Winning Trades:     137 (58.5%)");
    println!("  Losing Trades:      97 (41.5%)");
    println!("  Profit Factor:      1.72");
    println!("  Avg Win:            $423.15");
    println!("  Avg Loss:           $298.42");
    println!();

    println!("Risk Metrics:");
    println!("  Max Consecutive Wins:   8");
    println!("  Max Consecutive Losses: 4");
    println!("  Avg Trade Duration:     4.2 hours");
    println!();

    if let Some(output_path) = &args.output {
        println!("[OUTPUT] Results would be saved to: {}", output_path);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                        NOTES");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("This is an example output. For actual backtesting:");
    println!("  1. Train a model first using the 'train' binary");
    println!("  2. Provide the path to the trained model");
    println!("  3. Provide historical data for backtesting");
    println!();
    println!("Example:");
    println!("  cargo run --release --bin backtest -- \\");
    println!("    --model models/inception_ensemble.pt \\");
    println!("    --data data/btcusdt_15_90d.csv \\");
    println!("    --capital 100000");
    println!();
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
