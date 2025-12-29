//! Example: ML Signal Strategy Backtest
//!
//! This example demonstrates how to use pre-computed ML signals
//! for backtesting, similar to the ML4T workflow in Python.
//!
//! In practice, you would:
//! 1. Train an ML model in Python (sklearn, PyTorch, etc.)
//! 2. Generate predictions/signals
//! 3. Export signals to CSV/JSON
//! 4. Load and backtest in Rust
//!
//! Usage:
//!   cargo run --example ml_signals -- --symbol BTCUSDT

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use rust_backtester::{
    api::BybitClient,
    backtest::{BacktestConfig, BacktestEngine, BrokerConfig},
    models::Timeframe,
    strategies::{generate_mock_signals, MlSignalStrategy, RsiStrategy, SmaCrossover, Strategy},
    utils::PerformanceMetrics,
};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "ml_signals")]
#[command(about = "Backtest using ML-generated signals")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Timeframe
    #[arg(short, long, default_value = "1h")]
    timeframe: String,

    /// Number of days to backtest
    #[arg(short, long, default_value_t = 90)]
    days: i64,

    /// Initial capital
    #[arg(short, long, default_value_t = 10000.0)]
    capital: f64,

    /// Path to signals file (optional, will generate mock signals if not provided)
    #[arg(long)]
    signals_file: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              ML SIGNALS BACKTEST EXAMPLE                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Parse timeframe
    let timeframe = Timeframe::from_str(&args.timeframe)
        .ok_or_else(|| anyhow::anyhow!("Invalid timeframe: {}", args.timeframe))?;

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
        "Period: {} to {}\n",
        candles.first().unwrap().timestamp.format("%Y-%m-%d"),
        candles.last().unwrap().timestamp.format("%Y-%m-%d")
    );

    // Create broker configuration
    let broker_config = BrokerConfig {
        initial_cash: args.capital,
        fee_rate: 0.001, // 0.1%
        slippage: 0.0005,
        ..Default::default()
    };

    let config = BacktestConfig {
        broker: broker_config.clone(),
        position_size: 0.95,
        warmup_period: 50,
    };

    // Generate or load ML signals
    let signals = match &args.signals_file {
        Some(path) => {
            println!("Loading signals from: {}", path);
            let path = std::path::Path::new(path);
            if path.extension().map(|e| e == "csv").unwrap_or(false) {
                MlSignalStrategy::from_csv_file(path, 0.3, -0.3)?
            } else {
                MlSignalStrategy::from_json_file(path, 0.3, -0.3)?
            }
        }
        None => {
            println!("Generating mock ML signals (momentum-based)...");
            let mock_signals = generate_mock_signals(&candles, 20);
            println!("Generated {} signals\n", mock_signals.len());
            MlSignalStrategy::from_vec(mock_signals, 0.3, -0.3)
                .with_name("Mock ML Momentum")
        }
    };

    // Run backtest with ML signals
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    RUNNING BACKTESTS                           ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut results = Vec::new();

    // 1. ML Signal Strategy
    {
        let mut strategy = signals.clone();
        let mut engine = BacktestEngine::with_config(config.clone());
        let result = engine.run(&mut strategy, &candles);
        println!("ML Signal Strategy:");
        println!("  Return: {:>+.2}%  Sharpe: {:.2}  MaxDD: {:.2}%",
            result.total_return_pct, result.sharpe_ratio, result.max_drawdown_pct);
        results.push(("ML Signals", result));
    }

    // 2. SMA Crossover (baseline)
    {
        let mut strategy = SmaCrossover::new(10, 50);
        let mut engine = BacktestEngine::with_config(config.clone());
        let result = engine.run(&mut strategy, &candles);
        println!("SMA Crossover (10/50):");
        println!("  Return: {:>+.2}%  Sharpe: {:.2}  MaxDD: {:.2}%",
            result.total_return_pct, result.sharpe_ratio, result.max_drawdown_pct);
        results.push(("SMA 10/50", result));
    }

    // 3. RSI Strategy
    {
        let mut strategy = RsiStrategy::default_params();
        let mut engine = BacktestEngine::with_config(config.clone());
        let result = engine.run(&mut strategy, &candles);
        println!("RSI Strategy (14, 30/70):");
        println!("  Return: {:>+.2}%  Sharpe: {:.2}  MaxDD: {:.2}%",
            result.total_return_pct, result.sharpe_ratio, result.max_drawdown_pct);
        results.push(("RSI", result));
    }

    // Buy and Hold comparison
    let first_price = candles.first().unwrap().close;
    let last_price = candles.last().unwrap().close;
    let bnh_return = (last_price / first_price - 1.0) * 100.0;

    println!("\nBuy & Hold:");
    println!("  Return: {:>+.2}%", bnh_return);

    // Summary table
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    STRATEGY COMPARISON                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Strategy      │ Return   │ Sharpe │ Sortino │ MaxDD │ Trades ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    for (name, result) in &results {
        println!(
            "║ {:<13} │ {:>+7.2}% │ {:>6.2} │ {:>7.2} │ {:>4.1}% │ {:>6} ║",
            name,
            result.total_return_pct,
            result.sharpe_ratio,
            result.sortino_ratio,
            result.max_drawdown_pct,
            result.total_trades
        );
    }

    println!(
        "║ {:<13} │ {:>+7.2}% │    N/A │     N/A │   N/A │    N/A ║",
        "Buy & Hold", bnh_return
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Find best strategy
    let best = results
        .iter()
        .max_by(|a, b| a.1.sharpe_ratio.partial_cmp(&b.1.sharpe_ratio).unwrap())
        .unwrap();

    println!("\nBest Strategy (by Sharpe Ratio): {}", best.0);
    best.1.print_report();

    // ML Workflow Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    ML4T WORKFLOW SUMMARY                       ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("This example demonstrates the ML4T workflow:");
    println!();
    println!("1. DATA COLLECTION");
    println!("   - Fetched {} candles from Bybit API", candles.len());
    println!("   - Symbol: {}, Timeframe: {}", args.symbol, timeframe);
    println!();
    println!("2. SIGNAL GENERATION");
    println!("   - In production: Train ML model (sklearn, PyTorch, etc.)");
    println!("   - Generate predictions on historical data");
    println!("   - Export signals to CSV/JSON");
    println!();
    println!("3. BACKTESTING");
    println!("   - Load signals and market data");
    println!("   - Run event-driven simulation");
    println!("   - Account for fees and slippage");
    println!();
    println!("4. EVALUATION");
    println!("   - Compare against benchmarks (Buy & Hold, SMA, RSI)");
    println!("   - Check risk metrics (Sharpe, MaxDD)");
    println!("   - Validate on out-of-sample data");
    println!();
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
