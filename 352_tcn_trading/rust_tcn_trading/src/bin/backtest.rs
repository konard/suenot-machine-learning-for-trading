//! Run backtest using TCN signals
//!
//! Usage:
//!     cargo run --bin backtest -- --symbol BTCUSDT

use anyhow::Result;
use clap::Parser;
use rust_tcn_trading::api::{BybitClient, TimeFrame};
use rust_tcn_trading::features::{Normalizer, TechnicalIndicators};
use rust_tcn_trading::tcn::{TCN, TCNConfig};
use rust_tcn_trading::trading::{
    BacktestConfig, BacktestEngine, RiskConfig, RiskManager, SignalGenerator,
};

/// Run backtest on cryptocurrency data
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Time interval
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of candles for backtesting
    #[arg(short, long, default_value_t = 1000)]
    limit: u32,

    /// Initial capital
    #[arg(long, default_value_t = 100000.0)]
    capital: f64,

    /// Commission rate (as decimal)
    #[arg(long, default_value_t = 0.001)]
    commission: f64,

    /// Confidence threshold for trading
    #[arg(long, default_value_t = 0.6)]
    threshold: f64,

    /// Use rules-based strategy instead of TCN
    #[arg(long, default_value_t = false)]
    rules_based: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== TCN Backtest for {} ===\n", args.symbol);

    // Parse timeframe
    let timeframe = TimeFrame::from_str(&args.interval)
        .ok_or_else(|| anyhow::anyhow!("Invalid interval: {}", args.interval))?;

    // Fetch data
    println!("Fetching market data...");
    let client = BybitClient::new();
    let data = client
        .get_klines(&args.symbol, timeframe, Some(args.limit), None, None)
        .await?;

    println!("Fetched {} candles", data.len());

    if data.len() < 200 {
        return Err(anyhow::anyhow!(
            "Not enough data. Need at least 200 candles, got {}",
            data.len()
        ));
    }

    // Calculate features
    println!("Calculating technical indicators...");
    let features = TechnicalIndicators::calculate_all(&data.candles);
    println!("Calculated {} features", features.num_features);

    // Normalize features
    let mut normalizer = Normalizer::zscore();
    let normalized = normalizer.fit_transform(&features.data);

    // Create TCN model
    println!("Creating TCN model...");
    let config = TCNConfig {
        input_size: features.num_features,
        output_size: 3,
        num_channels: vec![32, 32, 32],
        kernel_size: 3,
        dropout: 0.2,
    };

    let tcn = TCN::new(config);
    println!("  Receptive field: {} bars", tcn.receptive_field());

    // Create signal generator
    let signal_gen = SignalGenerator::new(tcn, args.threshold, args.threshold);

    // Generate signals
    println!("\nGenerating trading signals...");
    let seq_len = 50; // Window size for each prediction
    let mut signals = Vec::new();

    for i in 0..data.len() {
        if i < seq_len {
            // Not enough history, generate neutral signal
            signals.push(rust_tcn_trading::trading::TradingSignal::neutral());
        } else {
            // Create feature window
            let window = normalized.slice(ndarray::s![.., (i - seq_len)..i]).to_owned();
            let feature_matrix = rust_tcn_trading::features::FeatureMatrix {
                feature_names: features.feature_names.clone(),
                data: window,
                num_features: features.num_features,
                seq_len,
            };

            let signal = signal_gen.generate_signal(&feature_matrix);
            signals.push(signal);
        }
    }

    // Count signals
    let long_signals = signals.iter().filter(|s| s.signal_type == rust_tcn_trading::trading::SignalType::Long).count();
    let short_signals = signals.iter().filter(|s| s.signal_type == rust_tcn_trading::trading::SignalType::Short).count();
    let neutral_signals = signals.iter().filter(|s| s.signal_type == rust_tcn_trading::trading::SignalType::Neutral).count();

    println!("\nSignal distribution:");
    println!("  Long:    {} ({:.1}%)", long_signals, long_signals as f64 / signals.len() as f64 * 100.0);
    println!("  Short:   {} ({:.1}%)", short_signals, short_signals as f64 / signals.len() as f64 * 100.0);
    println!("  Neutral: {} ({:.1}%)", neutral_signals, neutral_signals as f64 / signals.len() as f64 * 100.0);

    // Configure backtest
    let backtest_config = BacktestConfig {
        initial_capital: args.capital,
        commission: args.commission,
        slippage: 0.0005,
        allow_short: true,
        use_margin: false,
        margin_requirement: 0.2,
    };

    let risk_config = RiskConfig::default();
    let risk_manager = RiskManager::new(risk_config);

    let backtest_engine = BacktestEngine::new(backtest_config, risk_manager);

    // Run backtest
    println!("\n=== Running Backtest ===\n");
    println!("Configuration:");
    println!("  Initial capital: ${:.2}", args.capital);
    println!("  Commission:      {:.2}%", args.commission * 100.0);
    println!("  Threshold:       {:.0}%", args.threshold * 100.0);

    let result = backtest_engine.run(&signals, &data.candles, &args.symbol);

    // Print results
    println!("\n{}", result.summary());

    // Additional metrics
    println!("Additional Metrics:");
    println!("  Total trades:      {}", result.total_trades);
    if result.total_trades > 0 {
        let winning_trades = result.trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = result.trades.iter().filter(|t| t.pnl < 0.0).count();

        println!("  Winning trades:    {}", winning_trades);
        println!("  Losing trades:     {}", losing_trades);

        if !result.trades.is_empty() {
            let avg_win: f64 = result.trades.iter()
                .filter(|t| t.pnl > 0.0)
                .map(|t| t.pnl)
                .sum::<f64>() / winning_trades.max(1) as f64;

            let avg_loss: f64 = result.trades.iter()
                .filter(|t| t.pnl < 0.0)
                .map(|t| t.pnl.abs())
                .sum::<f64>() / losing_trades.max(1) as f64;

            println!("  Average win:       ${:.2}", avg_win);
            println!("  Average loss:      ${:.2}", avg_loss);
            println!("  Win/Loss ratio:    {:.2}", avg_win / avg_loss.max(1.0));
        }
    }

    // Buy and hold comparison
    let first_price = data.candles.first().map(|c| c.close).unwrap_or(0.0);
    let last_price = data.candles.last().map(|c| c.close).unwrap_or(0.0);
    let buy_hold_return = (last_price / first_price - 1.0) * 100.0;

    println!("\nBenchmark Comparison:");
    println!("  Buy & Hold return: {:.2}%", buy_hold_return);
    println!("  Strategy return:   {:.2}%", result.total_return * 100.0);
    println!("  Alpha:             {:.2}%", result.total_return * 100.0 - buy_hold_return);

    // Print first few trades
    if !result.trades.is_empty() {
        println!("\nRecent Trades (last 5):");
        for trade in result.trades.iter().rev().take(5) {
            let trade_type = match trade.trade_type {
                rust_tcn_trading::trading::SignalType::Long => "LONG ",
                rust_tcn_trading::trading::SignalType::Short => "SHORT",
                rust_tcn_trading::trading::SignalType::Neutral => "HOLD ",
            };

            let exit_price = trade.exit_price.unwrap_or(0.0);
            let pnl_pct = trade.return_pct * 100.0;

            println!(
                "  {} @ ${:.2} -> ${:.2} | P&L: ${:.2} ({:+.2}%)",
                trade_type, trade.entry_price, exit_price, trade.pnl, pnl_pct
            );
        }
    }

    // Save equity curve to CSV if there are results
    if !result.equity_curve.is_empty() {
        let output_file = format!("backtest_{}_equity.csv", args.symbol.to_lowercase());
        let mut file = std::fs::File::create(&output_file)?;
        use std::io::Write;

        writeln!(file, "timestamp,equity,position,unrealized_pnl")?;
        for point in &result.equity_curve {
            writeln!(
                file,
                "{},{:.2},{:.6},{:.2}",
                point.timestamp.format("%Y-%m-%d %H:%M:%S"),
                point.equity,
                point.position,
                point.unrealized_pnl
            )?;
        }
        println!("\nEquity curve saved to: {}", output_file);
    }

    println!("\n=== Backtest Complete ===");

    Ok(())
}
