//! Example: SMA Crossover Strategy with Event-Driven Backtest
//!
//! This example demonstrates a complete event-driven backtest
//! using the SMA Crossover strategy, similar to backtrader in Python.
//!
//! Usage:
//!   cargo run --example sma_crossover -- --symbol BTCUSDT --fast 10 --slow 50

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use rust_backtester::{
    api::BybitClient,
    backtest::{BacktestConfig, BacktestEngine, BrokerConfig},
    models::Timeframe,
    strategies::SmaCrossover,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "sma_crossover")]
#[command(about = "Run SMA Crossover backtest on cryptocurrency data")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Timeframe
    #[arg(short, long, default_value = "1h")]
    timeframe: String,

    /// Number of days to backtest
    #[arg(short, long, default_value_t = 60)]
    days: i64,

    /// Fast SMA period
    #[arg(long, default_value_t = 10)]
    fast: usize,

    /// Slow SMA period
    #[arg(long, default_value_t = 50)]
    slow: usize,

    /// Initial capital
    #[arg(short, long, default_value_t = 10000.0)]
    capital: f64,

    /// Trading fee (percentage)
    #[arg(long, default_value_t = 0.1)]
    fee: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║          SMA CROSSOVER EVENT-DRIVEN BACKTEST                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Parse timeframe
    let timeframe = Timeframe::from_str(&args.timeframe)
        .ok_or_else(|| anyhow::anyhow!("Invalid timeframe: {}", args.timeframe))?;

    println!("Configuration:");
    println!("  Symbol:     {}", args.symbol);
    println!("  Timeframe:  {}", timeframe);
    println!("  Period:     {} days", args.days);
    println!("  Fast SMA:   {}", args.fast);
    println!("  Slow SMA:   {}", args.slow);
    println!("  Capital:    ${:.2}", args.capital);
    println!("  Fee:        {}%\n", args.fee);

    // Fetch data
    info!("Fetching historical data from Bybit...");
    let client = BybitClient::new();
    let end = Utc::now();
    let start = end - Duration::days(args.days);

    let candles = client
        .get_historical_klines(&args.symbol, timeframe, start, end)
        .await?;

    if candles.is_empty() {
        println!("No data available. Exiting.");
        return Ok(());
    }

    println!("Fetched {} candles", candles.len());
    println!(
        "Period: {} to {}",
        candles.first().unwrap().timestamp.format("%Y-%m-%d"),
        candles.last().unwrap().timestamp.format("%Y-%m-%d")
    );

    // Create strategy
    let mut strategy = SmaCrossover::new(args.fast, args.slow);

    // Create backtest engine with custom configuration
    let config = BacktestConfig {
        broker: BrokerConfig {
            initial_cash: args.capital,
            fee_rate: args.fee / 100.0,
            slippage: 0.0005, // 0.05%
            ..Default::default()
        },
        position_size: 0.95, // Use 95% of capital
        warmup_period: args.slow + 10,
    };

    let mut engine = BacktestEngine::with_config(config);

    // Run backtest
    info!("Running backtest...");
    let result = engine.run(&mut strategy, &candles);

    // Print results
    result.print_report();

    // Additional analysis
    println!("═══════════════════════════════════════════════════════════════");
    println!("                      ADDITIONAL ANALYSIS                       ");
    println!("═══════════════════════════════════════════════════════════════");

    // Calculate buy & hold for comparison
    let first_price = candles.first().unwrap().close;
    let last_price = candles.last().unwrap().close;
    let bnh_return = (last_price / first_price - 1.0) * 100.0;

    println!(
        "Buy & Hold Return:    {:>10.2}%",
        bnh_return
    );
    println!(
        "Strategy vs B&H:      {:>10.2}%",
        result.total_return_pct - bnh_return
    );

    // Risk-adjusted metrics
    if result.sharpe_ratio > 1.0 {
        println!("\nRisk Assessment: ACCEPTABLE");
        println!("  - Sharpe Ratio > 1.0 indicates good risk-adjusted returns");
    } else if result.sharpe_ratio > 0.0 {
        println!("\nRisk Assessment: MODERATE");
        println!("  - Sharpe Ratio between 0 and 1 indicates acceptable returns");
    } else {
        println!("\nRisk Assessment: POOR");
        println!("  - Negative Sharpe Ratio indicates poor risk-adjusted returns");
    }

    if result.max_drawdown_pct > 20.0 {
        println!("  - WARNING: Maximum drawdown exceeds 20%");
    }

    println!("\n═══════════════════════════════════════════════════════════════");

    // Save results
    let output_path = std::path::PathBuf::from(format!(
        "backtest_{}_{}_{}_{}.json",
        args.symbol.to_lowercase(),
        args.fast,
        args.slow,
        chrono::Utc::now().format("%Y%m%d_%H%M%S")
    ));

    result.save_json(&output_path)?;
    println!("\nResults saved to: {}", output_path.display());

    Ok(())
}
