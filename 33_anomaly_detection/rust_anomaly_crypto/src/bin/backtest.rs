//! Backtest anomaly-based trading strategy
//!
//! Usage: cargo run --bin backtest -- --symbol BTCUSDT --days 30

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;

use rust_anomaly_crypto::{
    anomaly::{AnomalyDetector, EnsembleDetector},
    data::{BybitClient, BybitConfig},
    strategy::{PositionManager, Signal, SignalConfig, SignalGenerator},
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Backtest anomaly-based trading strategy")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Days of historical data
    #[arg(short, long, default_value_t = 7)]
    days: i64,

    /// Anomaly threshold for reduce signal
    #[arg(long, default_value_t = 0.7)]
    reduce_threshold: f64,

    /// Anomaly threshold for exit signal
    #[arg(long, default_value_t = 1.5)]
    exit_threshold: f64,

    /// Enable contrarian trading
    #[arg(long)]
    contrarian: bool,

    /// Initial position size
    #[arg(long, default_value_t = 1.0)]
    position_size: f64,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

struct BacktestMetrics {
    total_return: f64,
    max_drawdown: f64,
    sharpe_ratio: f64,
    win_rate: f64,
    num_trades: usize,
    anomalies_detected: usize,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║            Anomaly Detection Strategy Backtest                ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Symbol: {:10} | Period: {} days | Interval: {:4}          ║",
             args.symbol, args.days, args.interval);
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Fetch historical data
    println!("Fetching {} days of {} data...", args.days, args.symbol);

    let client = BybitClient::new(BybitConfig::new());
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(args.days);

    let data = client.get_historical_klines(&args.symbol, &args.interval, start_time, end_time)?;

    println!("Loaded {} candles", data.len());
    println!(
        "Period: {} to {}",
        data.data.first().map(|c| c.timestamp.to_string()).unwrap_or_default(),
        data.data.last().map(|c| c.timestamp.to_string()).unwrap_or_default()
    );
    println!();

    // Calculate returns
    let returns = data.returns();
    let closes = data.closes();

    if returns.is_empty() {
        println!("Not enough data for backtest");
        return Ok(());
    }

    // Detect anomalies
    println!("Detecting anomalies...");
    let mut detector = EnsembleDetector::new().with_threshold(0.5);
    detector.fit(&returns);
    let anomaly_result = detector.detect(&returns);

    println!(
        "Found {} anomalies ({:.2}% of data)",
        anomaly_result.anomaly_count(),
        anomaly_result.anomaly_rate() * 100.0
    );
    println!();

    // Run backtest
    println!("Running backtest...");

    let signal_config = SignalConfig {
        reduce_threshold: args.reduce_threshold,
        exit_threshold: args.exit_threshold,
        enable_contrarian: args.contrarian,
        ..Default::default()
    };

    let mut signal_generator = SignalGenerator::new(signal_config);
    let mut position_manager = PositionManager::new(args.position_size);

    // Track equity curve
    let mut equity_curve: Vec<f64> = vec![1.0];
    let mut peak_equity = 1.0;
    let mut max_drawdown = 0.0;
    let mut trade_returns: Vec<f64> = Vec::new();
    let mut last_entry_equity = 1.0;

    // Start with a position (for comparison)
    let initial_signal = rust_anomaly_crypto::strategy::TradingSignal::entry_long(0.0, 1.0);
    position_manager.process_signal(&initial_signal, closes[0]);

    for i in 1..returns.len() {
        let anomaly_score = anomaly_result.normalized_scores.get(i).cloned().unwrap_or(0.0);
        let current_return = returns[i - 1];
        let current_price = closes[i];

        // Generate signal
        let signal = signal_generator.generate(anomaly_score, current_return);

        // Process signal
        let (action, _) = position_manager.process_signal(&signal, current_price);

        // Update equity
        let position_return = position_manager.position.signed_size() * current_return;
        let current_equity = equity_curve.last().unwrap_or(&1.0) * (1.0 + position_return);
        equity_curve.push(current_equity);

        // Track drawdown
        if current_equity > peak_equity {
            peak_equity = current_equity;
        }
        let drawdown = (peak_equity - current_equity) / peak_equity;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }

        // Track trade returns
        if signal.signal == Signal::ExitAll {
            let trade_return = (current_equity - last_entry_equity) / last_entry_equity;
            trade_returns.push(trade_return);
        } else if signal.signal.is_entry() {
            last_entry_equity = current_equity;
        }

        // Verbose output
        if args.verbose && signal.signal != Signal::Hold {
            println!(
                "[{}] {} | Price: ${:.2} | Score: {:.2} | Equity: {:.4}",
                data.data[i + 1].timestamp.format("%Y-%m-%d %H:%M"),
                action,
                current_price,
                anomaly_score,
                current_equity
            );
        }
    }

    // Close final position
    if let Some(&final_price) = closes.last() {
        position_manager.close_all(final_price);
    }

    // Calculate metrics
    let final_equity = equity_curve.last().unwrap_or(&1.0);
    let total_return = (final_equity - 1.0) * 100.0;

    // Sharpe ratio (simplified)
    let equity_returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();
    let mean_return: f64 = equity_returns.iter().sum::<f64>() / equity_returns.len() as f64;
    let variance: f64 = equity_returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / equity_returns.len() as f64;
    let volatility = variance.sqrt();
    let sharpe = if volatility > 0.0 {
        mean_return / volatility * (252.0_f64).sqrt() // Annualized
    } else {
        0.0
    };

    // Win rate
    let wins = trade_returns.iter().filter(|&&r| r > 0.0).count();
    let win_rate = if !trade_returns.is_empty() {
        wins as f64 / trade_returns.len() as f64
    } else {
        0.0
    };

    // Buy and hold comparison
    let buy_hold_return = (closes.last().unwrap_or(&1.0) / closes.first().unwrap_or(&1.0) - 1.0) * 100.0;

    // Print results
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    Backtest Results                           ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Strategy Performance:                                         ║");
    println!("║   Total Return:      {:>+8.2}%                                 ║", total_return);
    println!("║   Max Drawdown:      {:>8.2}%                                  ║", max_drawdown * 100.0);
    println!("║   Sharpe Ratio:      {:>8.2}                                   ║", sharpe);
    println!("║   Number of Trades:  {:>8}                                    ║", position_manager.trade_count());
    println!("║   Win Rate:          {:>8.2}%                                  ║", win_rate * 100.0);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Comparison:                                                   ║");
    println!("║   Buy & Hold Return: {:>+8.2}%                                 ║", buy_hold_return);
    println!("║   Alpha:             {:>+8.2}%                                 ║", total_return - buy_hold_return);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Anomaly Statistics:                                           ║");
    println!("║   Anomalies Detected: {:>7}                                   ║", anomaly_result.anomaly_count());
    println!("║   Anomaly Rate:      {:>8.2}%                                  ║", anomaly_result.anomaly_rate() * 100.0);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Interpretation
    println!();
    if total_return > buy_hold_return {
        println!("📈 Strategy outperformed buy & hold by {:.2}%", total_return - buy_hold_return);
    } else {
        println!("📉 Strategy underperformed buy & hold by {:.2}%", buy_hold_return - total_return);
    }

    if max_drawdown < 0.10 {
        println!("✅ Low max drawdown ({:.2}%)", max_drawdown * 100.0);
    } else if max_drawdown < 0.20 {
        println!("⚠️  Moderate max drawdown ({:.2}%)", max_drawdown * 100.0);
    } else {
        println!("🚨 High max drawdown ({:.2}%)", max_drawdown * 100.0);
    }

    if sharpe > 1.0 {
        println!("✅ Good risk-adjusted returns (Sharpe: {:.2})", sharpe);
    } else if sharpe > 0.0 {
        println!("⚠️  Moderate risk-adjusted returns (Sharpe: {:.2})", sharpe);
    } else {
        println!("🚨 Poor risk-adjusted returns (Sharpe: {:.2})", sharpe);
    }

    Ok(())
}
