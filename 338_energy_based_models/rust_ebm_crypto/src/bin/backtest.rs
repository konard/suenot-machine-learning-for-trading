//! Backtest EBM trading strategy
//!
//! Usage:
//!   cargo run --bin backtest -- --symbol BTCUSDT --limit 5000

use clap::Parser;
use log::info;
use rust_ebm_crypto::data::{BybitClient, OhlcvData};
use rust_ebm_crypto::strategy::{BacktestConfig, BacktestEngine};

#[derive(Parser, Debug)]
#[command(author, version, about = "Backtest EBM trading strategy")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles for backtest
    #[arg(short, long, default_value = "5000")]
    limit: usize,

    /// Initial capital
    #[arg(long, default_value = "10000.0")]
    capital: f64,

    /// Commission rate
    #[arg(long, default_value = "0.001")]
    commission: f64,

    /// Warmup period
    #[arg(long, default_value = "100")]
    warmup: usize,

    /// Input CSV file (optional, otherwise fetch from Bybit)
    #[arg(long)]
    input: Option<String>,

    /// Show individual trades
    #[arg(long)]
    show_trades: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    // Load or fetch data
    let data = if let Some(input_path) = &args.input {
        info!("Loading data from {}", input_path);
        OhlcvData::from_csv(input_path, &args.symbol, &args.interval)?
    } else {
        info!(
            "Fetching {} candles for {} ({})",
            args.limit, args.symbol, args.interval
        );
        let client = BybitClient::public();
        client.get_historical_klines(&args.symbol, &args.interval, args.limit, None)?
    };

    info!("Data loaded: {} candles", data.len());

    if data.len() < args.warmup + 100 {
        eprintln!(
            "Not enough data for backtest. Need at least {} candles, got {}",
            args.warmup + 100,
            data.len()
        );
        std::process::exit(1);
    }

    // Print data summary
    println!("\n=== Data Summary ===");
    println!("Symbol:   {}", data.symbol);
    println!("Interval: {}", data.interval);
    println!("Candles:  {}", data.len());
    if !data.data.is_empty() {
        let first = &data.data[0];
        let last = data.data.last().unwrap();
        println!(
            "Period:   {} to {}",
            first.datetime().format("%Y-%m-%d %H:%M"),
            last.datetime().format("%Y-%m-%d %H:%M")
        );

        let returns = data.returns();
        let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
        println!("Buy & Hold Return: {:.2}%", total_return * 100.0);
    }

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: args.capital,
        commission_rate: args.commission,
        slippage_rate: 0.0005,
        allow_short: true,
        warmup_period: args.warmup,
    };

    // Run backtest
    println!("\n=== Running Backtest ===");
    let mut engine = BacktestEngine::new(config);
    let results = engine.run(&data.data);

    // Print results
    results.print_summary();

    // Show individual trades if requested
    if args.show_trades && !results.trades.is_empty() {
        println!("\n=== Trade History ===");
        println!(
            "{:<20} {:<20} {:>8} {:>12} {:>12} {:>10} {}",
            "Entry Time", "Exit Time", "Side", "Entry", "Exit", "Return", "Reason"
        );
        println!("{}", "-".repeat(100));

        for trade in &results.trades {
            let entry_time = chrono::DateTime::from_timestamp(trade.entry_time / 1000, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "N/A".to_string());
            let exit_time = chrono::DateTime::from_timestamp(trade.exit_time / 1000, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "{:<20} {:<20} {:>8} {:>12.4} {:>12.4} {:>10.2}% {}",
                entry_time,
                exit_time,
                trade.side.as_str(),
                trade.entry_price,
                trade.exit_price,
                trade.return_pct * 100.0,
                trade.exit_reason
            );
        }
    }

    // Summary comparison
    println!("\n=== Strategy vs Buy & Hold ===");
    let buy_hold_return: f64 = data.returns().iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
    let strategy_return = results.total_return;
    let alpha = strategy_return - buy_hold_return;

    println!("Buy & Hold Return:  {:>10.2}%", buy_hold_return * 100.0);
    println!("Strategy Return:    {:>10.2}%", strategy_return * 100.0);
    println!("Alpha:              {:>10.2}%", alpha * 100.0);
    println!(
        "Outperformance:     {:>10}",
        if alpha > 0.0 { "YES" } else { "NO" }
    );

    Ok(())
}
