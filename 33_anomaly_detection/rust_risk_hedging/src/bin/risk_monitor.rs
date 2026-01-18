//! Real-time risk monitoring
//!
//! Usage: cargo run --bin risk_monitor -- --symbol BTCUSDT

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use rust_risk_hedging::{
    anomaly::{AnomalyLevel, EnsembleDetector},
    data::{symbols, BybitClient},
    features::CryptoRiskFeatures,
    risk::{AlertConfig, AlertGenerator, HedgingStrategy, SignalHistory},
};
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about = "Real-time cryptocurrency risk monitor")]
struct Args {
    /// Trading symbols (comma-separated)
    #[arg(short, long, default_value = "BTCUSDT,ETHUSDT")]
    symbols: String,

    /// Update interval in seconds
    #[arg(short, long, default_value = "60")]
    interval: u64,

    /// Portfolio value
    #[arg(short, long, default_value = "100000")]
    portfolio: f64,

    /// Number of updates (0 = infinite)
    #[arg(short, long, default_value = "10")]
    updates: usize,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let symbols: Vec<&str> = args.symbols.split(',').collect();

    println!("{}", "=== Real-Time Risk Monitor ===".bold());
    println!("Monitoring: {}", args.symbols.cyan());
    println!("Update interval: {}s", args.interval);
    println!("Portfolio value: ${:.2}", args.portfolio);
    println!();

    let client = BybitClient::public();
    let detector = EnsembleDetector::default();
    let strategy = HedgingStrategy::default();
    let mut alert_gen = AlertGenerator::new(AlertConfig::default());
    let mut signal_history = SignalHistory::default();

    let mut update_count = 0;
    let max_updates = if args.updates == 0 { usize::MAX } else { args.updates };

    loop {
        if update_count >= max_updates {
            break;
        }

        clear_screen();
        println!("{}", "=== Real-Time Risk Monitor ===".bold());
        println!("Update: {} | {}", update_count + 1, chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        println!("{}", "=".repeat(60));
        println!();

        let mut portfolio_risk = 0.0;
        let mut symbol_count = 0.0;

        for symbol in &symbols {
            match analyze_symbol(&client, &detector, symbol) {
                Ok((score, level, features)) => {
                    print_symbol_status(symbol, score, level, &features);

                    portfolio_risk += score;
                    symbol_count += 1.0;

                    // Check for alerts
                    if alert_gen.should_alert(level) {
                        let alert = alert_gen.generate_alert(level, score);
                        println!("\n{}", alert.format_colored());
                    }
                }
                Err(e) => {
                    println!("{}: {}", symbol.red(), e);
                }
            }
            println!();
        }

        // Portfolio-level summary
        if symbol_count > 0.0 {
            let avg_risk = portfolio_risk / symbol_count;
            let avg_level = AnomalyLevel::from_score(avg_risk);

            println!("{}", "=== Portfolio Summary ===".bold());
            print_portfolio_summary(avg_risk, avg_level, &strategy, args.portfolio);

            // Update signal history
            signal_history.add(rust_risk_hedging::risk::RiskSignal::from_level(avg_level), avg_risk);

            // Show trend
            let trend = signal_history.trend();
            println!("\nRisk Trend: {:?} - {}", trend, trend.description());
        }

        update_count += 1;

        if update_count < max_updates {
            println!("\n{}", format!("Next update in {}s... (Ctrl+C to exit)", args.interval).dimmed());
            std::thread::sleep(Duration::from_secs(args.interval));
        }
    }

    println!("\n{}", "Monitoring complete.".green());
    Ok(())
}

fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
}

fn analyze_symbol(
    client: &BybitClient,
    detector: &EnsembleDetector,
    symbol: &str,
) -> Result<(f64, AnomalyLevel, CryptoRiskFeatures)> {
    let data = client.get_klines(symbol, "60", 200, None, None)?;

    if data.is_empty() {
        return Err(anyhow::anyhow!("No data"));
    }

    let features = CryptoRiskFeatures::from_ohlcv(&data);
    let results = detector.detect_from_ohlcv(&data);

    let latest = results.last().ok_or_else(|| anyhow::anyhow!("No results"))?;

    Ok((latest.score, latest.level, features))
}

fn print_symbol_status(
    symbol: &str,
    score: f64,
    level: AnomalyLevel,
    features: &CryptoRiskFeatures,
) {
    let level_str = match level {
        AnomalyLevel::Normal => "NORMAL".green(),
        AnomalyLevel::Elevated => "ELEVATED".yellow(),
        AnomalyLevel::High => "HIGH".truecolor(255, 165, 0),
        AnomalyLevel::Extreme => "EXTREME".red().bold(),
    };

    let indicator = match level {
        AnomalyLevel::Normal => "●".green(),
        AnomalyLevel::Elevated => "●".yellow(),
        AnomalyLevel::High => "●".truecolor(255, 165, 0),
        AnomalyLevel::Extreme => "◉".red(),
    };

    println!("{} {} [{}]", indicator, symbol.bold(), level_str);

    // Risk bar
    let bar_len = 30;
    let filled = (score * bar_len as f64) as usize;
    let bar: String = (0..bar_len)
        .map(|i| if i < filled { '█' } else { '░' })
        .collect();

    let colored_bar = if score < 0.5 {
        bar.green()
    } else if score < 0.7 {
        bar.yellow()
    } else if score < 0.9 {
        bar.truecolor(255, 165, 0)
    } else {
        bar.red()
    };

    println!("  Risk: [{}] {:.1}%", colored_bar, score * 100.0);

    // Key metrics
    println!(
        "  24h Change: {:+.2}% | Volume Ratio: {:.2}x | Range: {:.2}%",
        features.change_24h,
        features.volume_ratio,
        features.range_ratio
    );

    // Composite score
    let crypto_score = features.crypto_risk_score();
    println!("  Crypto Risk Score: {:.2}", crypto_score);
}

fn print_portfolio_summary(
    avg_risk: f64,
    level: AnomalyLevel,
    strategy: &HedgingStrategy,
    portfolio_value: f64,
) {
    println!("Average Risk Score: {:.2}", avg_risk);

    let level_str = match level {
        AnomalyLevel::Normal => "NORMAL".green(),
        AnomalyLevel::Elevated => "ELEVATED".yellow(),
        AnomalyLevel::High => "HIGH".truecolor(255, 165, 0),
        AnomalyLevel::Extreme => "EXTREME".red().bold(),
    };

    println!("Portfolio Risk Level: {}", level_str);

    // Hedging recommendation
    let allocation = strategy.decide(avg_risk, portfolio_value);

    if allocation.total_hedge_pct > 0.0 {
        println!("\n{}", "Recommended Hedge:".bold());
        println!("  Total: {:.1}% of portfolio", allocation.total_hedge_pct * 100.0);

        let amounts = allocation.dollar_amounts(portfolio_value);
        for (instrument, amount) in &amounts {
            println!("  {:?}: ${:.2}", instrument, amount);
        }
    } else {
        println!("\n{}", "No hedging required at current risk levels.".green());
    }
}
