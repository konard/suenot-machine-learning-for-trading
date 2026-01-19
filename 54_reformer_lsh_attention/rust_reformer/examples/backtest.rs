//! Example: Backtest a trading strategy using Reformer
//!
//! This example demonstrates how to:
//! 1. Fetch historical data
//! 2. Create a Reformer model
//! 3. Run a backtest simulation
//! 4. Analyze the results

use clap::Parser;
use reformer::{
    BybitClient, ReformerConfig, ReformerModel, AttentionType,
    BacktestConfig, run_backtest,
};

/// Backtest a Reformer-based trading strategy
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Initial capital
    #[arg(long, default_value = "10000")]
    initial_capital: f64,

    /// Position size (fraction of capital)
    #[arg(long, default_value = "0.1")]
    position_size: f64,

    /// Stop loss percentage
    #[arg(long, default_value = "0.02")]
    stop_loss: f64,

    /// Take profit percentage
    #[arg(long, default_value = "0.04")]
    take_profit: f64,

    /// Number of historical candles
    #[arg(long, default_value = "2000")]
    history: usize,

    /// Show trade details
    #[arg(long)]
    show_trades: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Reformer Backtest Example ===");
    println!("Symbol: {}", args.symbol);
    println!("Initial Capital: ${:.2}", args.initial_capital);
    println!("Position Size: {:.0}%", args.position_size * 100.0);
    println!("Stop Loss: {:.1}%", args.stop_loss * 100.0);
    println!("Take Profit: {:.1}%", args.take_profit * 100.0);
    println!();

    // Fetch historical data
    println!("Fetching historical data...");
    let client = BybitClient::new();
    let klines = client.get_extended_klines(&args.symbol, "60", args.history).await?;
    println!("Fetched {} klines", klines.len());

    // Calculate date range
    if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
        let first_date = chrono::DateTime::from_timestamp_millis(first.timestamp as i64)
            .map(|d| d.format("%Y-%m-%d").to_string())
            .unwrap_or_default();
        let last_date = chrono::DateTime::from_timestamp_millis(last.timestamp as i64)
            .map(|d| d.format("%Y-%m-%d").to_string())
            .unwrap_or_default();
        println!("Date range: {} to {}", first_date, last_date);

        let price_change = (last.close - first.close) / first.close * 100.0;
        println!("Buy & Hold Return: {:.2}%", price_change);
    }

    // Create model
    println!("\nCreating Reformer model...");
    let model_config = ReformerConfig {
        seq_len: 168,
        n_features: 10,
        d_model: 64,
        n_heads: 4,
        d_ff: 256,
        n_layers: 4,
        n_hash_rounds: 4,
        n_buckets: 16,
        prediction_horizon: 24,
        attention_type: AttentionType::LSH,
        ..Default::default()
    };

    let model = ReformerModel::new(model_config);

    // Configure backtest
    let backtest_config = BacktestConfig {
        initial_capital: args.initial_capital,
        position_size: args.position_size,
        stop_loss: args.stop_loss,
        take_profit: args.take_profit,
        commission: 0.001,
        slippage: 0.0005,
        min_confidence: 0.0,
    };

    // Run backtest
    println!("\nRunning backtest...");
    let results = run_backtest(&model, &klines, backtest_config);

    // Display results
    println!("\n{}", results);

    // Additional analysis
    println!("\n=== Risk Analysis ===");
    let var_95 = calculate_var(&results.equity_curve, 0.95);
    let var_99 = calculate_var(&results.equity_curve, 0.99);
    println!("Value at Risk (95%): {:.2}%", var_95 * 100.0);
    println!("Value at Risk (99%): {:.2}%", var_99 * 100.0);

    // Calmar ratio
    let calmar = if results.max_drawdown > 0.0 {
        results.annualized_return / results.max_drawdown
    } else {
        0.0
    };
    println!("Calmar Ratio: {:.3}", calmar);

    // Trade analysis
    if !results.trades.is_empty() {
        println!("\n=== Trade Analysis ===");

        let long_trades: Vec<_> = results.trades.iter().filter(|t| t.size > 0.0).collect();
        let short_trades: Vec<_> = results.trades.iter().filter(|t| t.size < 0.0).collect();

        println!("Long Trades: {}", long_trades.len());
        println!("Short Trades: {}", short_trades.len());

        let avg_duration: f64 = results
            .trades
            .iter()
            .map(|t| (t.exit_time - t.entry_time) as f64 / 3600000.0)
            .sum::<f64>()
            / results.trades.len() as f64;
        println!("Average Trade Duration: {:.1} hours", avg_duration);

        // Exit reason breakdown
        let mut exit_reasons = std::collections::HashMap::new();
        for trade in &results.trades {
            *exit_reasons.entry(trade.exit_reason.clone()).or_insert(0) += 1;
        }

        println!("\nExit Reasons:");
        for (reason, count) in &exit_reasons {
            let pct = *count as f64 / results.trades.len() as f64 * 100.0;
            println!("  {}: {} ({:.1}%)", reason, count, pct);
        }
    }

    // Show individual trades
    if args.show_trades && !results.trades.is_empty() {
        println!("\n=== Trade Log ===");
        println!("{:<12} {:<12} {:>10} {:>10} {:>10} {:<15}",
            "Entry", "Exit", "Entry $", "Exit $", "P&L %", "Reason");
        println!("{}", "-".repeat(70));

        for trade in results.trades.iter().take(20) {
            let entry_date = chrono::DateTime::from_timestamp_millis(trade.entry_time as i64)
                .map(|d| d.format("%m/%d %H:%M").to_string())
                .unwrap_or_default();
            let exit_date = chrono::DateTime::from_timestamp_millis(trade.exit_time as i64)
                .map(|d| d.format("%m/%d %H:%M").to_string())
                .unwrap_or_default();

            println!(
                "{:<12} {:<12} {:>10.2} {:>10.2} {:>+10.2}% {:<15}",
                entry_date,
                exit_date,
                trade.entry_price,
                trade.exit_price,
                trade.return_pct * 100.0,
                trade.exit_reason
            );
        }

        if results.trades.len() > 20 {
            println!("... and {} more trades", results.trades.len() - 20);
        }
    }

    // Equity curve statistics
    println!("\n=== Equity Curve ===");
    let max_equity = results.equity_curve.iter().cloned().fold(0.0_f64, f64::max);
    let min_equity = results.equity_curve.iter().cloned().fold(f64::MAX, f64::min);

    println!("Peak Equity: ${:.2}", max_equity);
    println!("Trough Equity: ${:.2}", min_equity);
    println!("Final Equity: ${:.2}", results.final_capital);

    // Plot simple ASCII equity curve
    println!("\nEquity Curve (simplified):");
    print_ascii_chart(&results.equity_curve, 50, 10);

    println!("\n=== Disclaimer ===");
    println!("This backtest uses simplified signal generation.");
    println!("Real performance may differ significantly.");
    println!("Past performance does not guarantee future results.");

    Ok(())
}

/// Calculate Value at Risk
fn calculate_var(equity_curve: &[f64], confidence: f64) -> f64 {
    if equity_curve.len() < 2 {
        return 0.0;
    }

    let mut returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((1.0 - confidence) * returns.len() as f64) as usize;
    returns.get(idx).copied().unwrap_or(0.0).abs()
}

/// Print a simple ASCII chart
fn print_ascii_chart(data: &[f64], width: usize, height: usize) {
    if data.is_empty() {
        return;
    }

    let min_val = data.iter().cloned().fold(f64::MAX, f64::min);
    let max_val = data.iter().cloned().fold(f64::MIN, f64::max);
    let range = max_val - min_val;

    if range < 1e-10 {
        return;
    }

    // Sample data to fit width
    let step = data.len().saturating_sub(1) / width.saturating_sub(1);
    let step = step.max(1);

    let sampled: Vec<f64> = data.iter().step_by(step).cloned().collect();

    // Create chart
    for row in (0..height).rev() {
        let threshold = min_val + range * row as f64 / (height - 1) as f64;

        print!("{:>10.0} |", threshold);

        for &val in &sampled {
            if val >= threshold {
                print!("█");
            } else {
                print!(" ");
            }
        }
        println!();
    }

    // X axis
    print!("{:>10} +", "");
    for _ in 0..sampled.len() {
        print!("-");
    }
    println!();
}
