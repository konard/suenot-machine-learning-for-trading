//! Backtest ESN trading strategy
//!
//! Usage: cargo run --bin backtest -- --model model.bin --data data.csv

use anyhow::Result;
use clap::Parser;
use esn_trading::{
    EchoStateNetwork,
    api::Kline,
    trading::{FeatureEngineering, SignalGenerator, TradingSignal, Backtest, BacktestConfig},
};
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Parser, Debug)]
#[command(author, version, about = "Backtest ESN trading strategy")]
struct Args {
    /// Model file
    #[arg(short, long)]
    model: String,

    /// Data file (CSV)
    #[arg(short, long)]
    data: String,

    /// Initial capital
    #[arg(long, default_value = "10000")]
    capital: f64,

    /// Commission rate
    #[arg(long, default_value = "0.0004")]
    commission: f64,

    /// Position size (fraction of capital)
    #[arg(long, default_value = "0.1")]
    position_size: f64,

    /// Stop loss percentage
    #[arg(long, default_value = "0.02")]
    stop_loss: f64,

    /// Take profit percentage
    #[arg(long, default_value = "0.04")]
    take_profit: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("Loading model from {}...", args.model);
    let mut esn = EchoStateNetwork::load(&args.model)?;
    println!("Model loaded!");

    println!("Loading data from {}...", args.data);
    let klines = load_klines(&args.data)?;
    println!("Loaded {} klines", klines.len());

    // Feature engineering (must match training)
    println!("\nEngineering features...");
    let fe = FeatureEngineering::new()
        .add_returns(10)
        .add_volatility(20)
        .add_rsi(14)
        .add_momentum(5)
        .add_bollinger(20, 2.0);

    let features = fe.transform(&klines);
    let lookback = fe.required_lookback();

    // Generate signals
    println!("Generating trading signals...");
    let mut signal_generator = SignalGenerator::new()
        .with_smoothing(3)
        .with_confidence_threshold(0.1);

    esn.reset_state();
    let mut signals = Vec::new();

    // Pad with Hold for lookback period
    for _ in 0..lookback {
        signals.push(TradingSignal::Hold);
    }

    for feature in &features {
        let prediction = esn.step(feature);
        let (signal, _confidence) = signal_generator.generate(&prediction);
        signals.push(signal);
    }

    // Align with klines
    let signals = &signals[..klines.len()];

    // Count signals
    let buy_count = signals.iter().filter(|s| matches!(s, TradingSignal::Buy | TradingSignal::StrongBuy)).count();
    let sell_count = signals.iter().filter(|s| matches!(s, TradingSignal::Sell | TradingSignal::StrongSell)).count();
    let hold_count = signals.iter().filter(|s| matches!(s, TradingSignal::Hold)).count();

    println!("\nSignal distribution:");
    println!("  Buy:  {} ({:.1}%)", buy_count, buy_count as f64 / signals.len() as f64 * 100.0);
    println!("  Sell: {} ({:.1}%)", sell_count, sell_count as f64 / signals.len() as f64 * 100.0);
    println!("  Hold: {} ({:.1}%)", hold_count, hold_count as f64 / signals.len() as f64 * 100.0);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: args.capital,
        commission: args.commission,
        slippage: 0.0001,
        position_size_pct: args.position_size,
        leverage: 1.0,
        stop_loss: Some(args.stop_loss),
        take_profit: Some(args.take_profit),
    };

    println!("\nBacktest Configuration:");
    println!("  Initial Capital: ${:.2}", config.initial_capital);
    println!("  Commission: {:.4}%", config.commission * 100.0);
    println!("  Position Size: {:.1}%", config.position_size_pct * 100.0);
    println!("  Stop Loss: {:.1}%", args.stop_loss * 100.0);
    println!("  Take Profit: {:.1}%", args.take_profit * 100.0);

    // Run backtest
    println!("\nRunning backtest...");
    let backtest = Backtest::new(config);
    let result = backtest.run(&klines, signals);

    println!();
    result.print_summary();

    // Save equity curve
    let equity_path = "equity_curve.csv";
    let mut file = File::create(equity_path)?;
    use std::io::Write;
    writeln!(file, "index,equity")?;
    for (i, equity) in result.equity_curve.iter().enumerate() {
        writeln!(file, "{},{}", i, equity)?;
    }
    println!("\nEquity curve saved to {}", equity_path);

    // Save trades
    if !result.trades.is_empty() {
        let trades_path = "trades.csv";
        let mut file = File::create(trades_path)?;
        writeln!(file, "entry_time,exit_time,side,entry_price,exit_price,size,pnl,return_pct,commission")?;
        for trade in &result.trades {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{}",
                trade.entry_time,
                trade.exit_time,
                trade.side,
                trade.entry_price,
                trade.exit_price,
                trade.size,
                trade.pnl,
                trade.return_pct,
                trade.commission
            )?;
        }
        println!("Trade history saved to {}", trades_path);
    }

    Ok(())
}

fn load_klines(path: &str) -> Result<Vec<Kline>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut klines = Vec::new();
    let mut first_line = true;

    for line in reader.lines() {
        let line = line?;
        if first_line {
            first_line = false;
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 7 {
            klines.push(Kline {
                start_time: parts[0].parse().unwrap_or(0),
                open: parts[1].parse().unwrap_or(0.0),
                high: parts[2].parse().unwrap_or(0.0),
                low: parts[3].parse().unwrap_or(0.0),
                close: parts[4].parse().unwrap_or(0.0),
                volume: parts[5].parse().unwrap_or(0.0),
                turnover: parts[6].parse().unwrap_or(0.0),
            });
        }
    }

    klines.sort_by_key(|k| k.start_time);
    Ok(klines)
}
