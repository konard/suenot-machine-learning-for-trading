//! Backtest associative memory trading strategy
//!
//! Usage:
//!   cargo run --bin backtest -- --input data/btc_hourly.csv --output results.csv

use anyhow::Result;
use associative_memory_trading::{
    data::OHLCVSeries,
    strategy::{BacktestConfig, Backtester, SignalConfig, PositionSizingMode},
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "backtest")]
#[command(about = "Backtest associative memory trading strategy")]
struct Args {
    /// Input CSV file with OHLCV data
    #[arg(short, long)]
    input: String,

    /// Output CSV file for results
    #[arg(short, long, default_value = "backtest_results.csv")]
    output: String,

    /// Pattern lookback period
    #[arg(long, default_value = "20")]
    lookback: usize,

    /// Forward period for labels
    #[arg(long, default_value = "5")]
    forward: usize,

    /// Warmup period
    #[arg(long, default_value = "100")]
    warmup: usize,

    /// Memory size
    #[arg(long, default_value = "500")]
    memory_size: usize,

    /// Minimum confidence threshold
    #[arg(long, default_value = "0.3")]
    min_confidence: f64,

    /// Train/test split ratio
    #[arg(long, default_value = "0.7")]
    train_ratio: f64,

    /// Transaction cost (fraction)
    #[arg(long, default_value = "0.001")]
    cost: f64,

    /// Position sizing mode (fixed, linear, quadratic, kelly)
    #[arg(long, default_value = "linear")]
    sizing: String,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    log::info!("Loading data from {}", args.input);
    let data = OHLCVSeries::from_csv(&args.input, "SYMBOL", "interval")?;
    log::info!("Loaded {} candles", data.len());

    // Parse sizing mode
    let sizing_mode = match args.sizing.as_str() {
        "fixed" => PositionSizingMode::Fixed,
        "quadratic" => PositionSizingMode::Quadratic,
        "kelly" => PositionSizingMode::Kelly,
        _ => PositionSizingMode::Linear,
    };

    // Configure backtest
    let config = BacktestConfig {
        pattern_lookback: args.lookback,
        forward_period: args.forward,
        warmup_period: args.warmup,
        signal_config: SignalConfig {
            min_confidence: args.min_confidence,
            sizing_mode,
            ..Default::default()
        },
        transaction_cost: args.cost,
        slippage: 0.0005,
        memory_size: args.memory_size,
        train_ratio: args.train_ratio,
    };

    println!("\n=== Backtest Configuration ===");
    println!("Pattern Lookback:  {}", config.pattern_lookback);
    println!("Forward Period:    {}", config.forward_period);
    println!("Warmup Period:     {}", config.warmup_period);
    println!("Memory Size:       {}", config.memory_size);
    println!("Min Confidence:    {:.1}%", config.signal_config.min_confidence * 100.0);
    println!("Train Ratio:       {:.1}%", config.train_ratio * 100.0);
    println!("Transaction Cost:  {:.2}%", config.transaction_cost * 100.0);
    println!("Position Sizing:   {:?}", config.signal_config.sizing_mode);

    // Run backtest
    log::info!("Running backtest...");
    let backtester = Backtester::new(config);
    let result = backtester.run(&data)?;

    // Print results
    result.print_summary();

    // Save to CSV
    result.to_csv(&args.output)?;
    log::info!("Results saved to {}", args.output);

    // Print trade statistics
    if !result.trades.is_empty() {
        println!("=== Trade Statistics ===\n");

        let mut pnls: Vec<f64> = result.trades.iter().map(|t| t.pnl_pct).collect();
        pnls.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let best = pnls.last().unwrap_or(&0.0);
        let worst = pnls.first().unwrap_or(&0.0);
        let median = pnls[pnls.len() / 2];

        println!("Best Trade:   {:.2}%", best * 100.0);
        println!("Worst Trade:  {:.2}%", worst * 100.0);
        println!("Median Trade: {:.2}%", median * 100.0);

        // Holding time
        let holding_times: Vec<f64> = result.trades
            .iter()
            .map(|t| (t.exit_time - t.entry_time).num_hours() as f64)
            .collect();

        let avg_holding = holding_times.iter().sum::<f64>() / holding_times.len() as f64;
        println!("Avg Holding:  {:.1} hours", avg_holding);

        // Long vs Short
        let long_trades: Vec<&_> = result.trades.iter().filter(|t| t.direction > 0.0).collect();
        let short_trades: Vec<&_> = result.trades.iter().filter(|t| t.direction < 0.0).collect();

        let long_pnl: f64 = long_trades.iter().map(|t| t.pnl).sum();
        let short_pnl: f64 = short_trades.iter().map(|t| t.pnl).sum();

        println!("\nLong Trades:  {} (PnL: {:.2}%)", long_trades.len(), long_pnl * 100.0);
        println!("Short Trades: {} (PnL: {:.2}%)", short_trades.len(), short_pnl * 100.0);
    }

    // Monthly breakdown if enough data
    if result.steps.len() > 720 { // More than 30 days of hourly data
        println!("\n=== Monthly Performance ===");

        let mut monthly_returns: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

        for step in &result.steps {
            let month = step.timestamp.format("%Y-%m").to_string();
            *monthly_returns.entry(month).or_insert(0.0) += step.pnl;
        }

        let mut months: Vec<_> = monthly_returns.into_iter().collect();
        months.sort_by(|a, b| a.0.cmp(&b.0));

        for (month, ret) in months {
            let direction = if ret >= 0.0 { "+" } else { "" };
            println!("  {}: {}{:.2}%", month, direction, ret * 100.0);
        }
    }

    Ok(())
}
