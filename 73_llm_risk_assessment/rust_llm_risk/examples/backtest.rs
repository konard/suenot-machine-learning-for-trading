//! Example: Backtest risk-based trading strategy.
//!
//! This example demonstrates how to:
//! 1. Fetch historical price data
//! 2. Generate synthetic risk scores (or use real ones)
//! 3. Run a backtest
//! 4. Analyze results
//!
//! Run with: cargo run --example backtest

use anyhow::Result;
use clap::Parser;
use llm_risk_assessment::data::{BybitClient, OHLCV};
use llm_risk_assessment::risk::{RiskScore, RiskDirection, Confidence};
use llm_risk_assessment::strategy::{BacktestConfig, Backtester, RiskBasedTrader, TraderConfig};
use rand::Rng;
use tracing::info;
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(name = "backtest")]
#[command(about = "Backtest risk-based trading strategy")]
struct Args {
    /// Symbol to backtest
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of days of historical data
    #[arg(short, long, default_value = "30")]
    days: u32,

    /// Interval (1, 5, 15, 30, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Initial capital
    #[arg(short, long, default_value = "10000")]
    capital: f64,

    /// Use synthetic risk scores (no LLM required)
    #[arg(long, default_value = "true")]
    synthetic: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Starting backtest example");
    println!("\n=== Risk-Based Trading Backtest ===\n");
    println!("Symbol:   {}", args.symbol);
    println!("Days:     {}", args.days);
    println!("Interval: {}", args.interval);
    println!("Capital:  ${:.2}", args.capital);

    // Fetch historical data
    println!("\nFetching historical data...");
    let client = BybitClient::new();
    let price_data = client
        .fetch_historical_klines(&args.symbol, &args.interval, args.days)
        .await?;

    println!("Fetched {} candles", price_data.len());

    if price_data.is_empty() {
        println!("No data available for backtest");
        return Ok(());
    }

    // Generate risk scores
    println!("\nGenerating risk scores...");
    let risk_scores = if args.synthetic {
        generate_synthetic_risk_scores(&price_data)
    } else {
        println!("Note: For real LLM-based risk scores, use the assess_risk example");
        generate_synthetic_risk_scores(&price_data)
    };

    println!("Generated {} risk scores", risk_scores.len());

    // Configure the trader
    let trader_config = TraderConfig {
        risk_threshold_long: 4.0,
        risk_threshold_short: 7.0,
        max_position_size: 1.0,
        stop_loss_pct: 0.05,
        take_profit_pct: 0.10,
        allow_shorts: true,
    };

    let mut trader = RiskBasedTrader::with_config(trader_config);

    // Configure the backtester
    let backtest_config = BacktestConfig {
        initial_capital: args.capital,
        trading_fee: 0.001,  // 0.1%
        slippage: 0.0005,    // 0.05%
        compound: true,
    };

    let backtester = Backtester::with_config(backtest_config);

    // Run backtest
    println!("\nRunning backtest...");
    let result = backtester.run(&price_data, &risk_scores, &mut trader);

    // Print results
    result.print_summary();

    // Detailed trade analysis
    if !result.trades.is_empty() {
        println!("\n=== Trade Details ===");
        println!(
            "{:<5} {:<10} {:<12} {:<12} {:<10} {:<10}",
            "#", "Direction", "Entry", "Exit", "PnL", "Return%"
        );
        println!("{}", "-".repeat(65));

        for (i, trade) in result.trades.iter().enumerate().take(10) {
            let direction = if trade.direction == 1 { "LONG" } else { "SHORT" };
            println!(
                "{:<5} {:<10} ${:<11.2} ${:<11.2} ${:<9.2} {:<10.2}%",
                i + 1,
                direction,
                trade.entry_price,
                trade.exit_price,
                trade.net_pnl,
                trade.return_pct
            );
        }

        if result.trades.len() > 10 {
            println!("... and {} more trades", result.trades.len() - 10);
        }

        // Risk score analysis
        println!("\n=== Entry Risk Score Analysis ===");
        let long_entries: Vec<f64> = result
            .trades
            .iter()
            .filter(|t| t.direction == 1)
            .map(|t| t.entry_risk_score)
            .collect();

        let short_entries: Vec<f64> = result
            .trades
            .iter()
            .filter(|t| t.direction == -1)
            .map(|t| t.entry_risk_score)
            .collect();

        if !long_entries.is_empty() {
            let avg_long_risk: f64 = long_entries.iter().sum::<f64>() / long_entries.len() as f64;
            println!("Average risk score at long entry:  {:.2}", avg_long_risk);
        }

        if !short_entries.is_empty() {
            let avg_short_risk: f64 = short_entries.iter().sum::<f64>() / short_entries.len() as f64;
            println!("Average risk score at short entry: {:.2}", avg_short_risk);
        }
    }

    // Compare with buy-and-hold
    println!("\n=== Strategy vs Buy-and-Hold ===");
    if let (Some(first), Some(last)) = (price_data.first(), price_data.last()) {
        let bh_return = (last.close - first.close) / first.close * 100.0;
        let bh_final = args.capital * (1.0 + bh_return / 100.0);

        println!("Strategy Return:    {:.2}%", result.total_return_pct);
        println!("Buy-and-Hold Return: {:.2}%", bh_return);
        println!();
        println!("Strategy Final:     ${:.2}", result.final_value);
        println!("Buy-and-Hold Final: ${:.2}", bh_final);

        let outperformance = result.total_return_pct - bh_return;
        if outperformance > 0.0 {
            println!("\nStrategy outperformed by {:.2}%", outperformance);
        } else {
            println!("\nStrategy underperformed by {:.2}%", -outperformance);
        }
    }

    println!("\nBacktest complete!");

    Ok(())
}

/// Generate synthetic risk scores based on price momentum.
fn generate_synthetic_risk_scores(price_data: &[OHLCV]) -> Vec<RiskScore> {
    let mut rng = rand::thread_rng();

    // Calculate momentum
    let window = 20;
    let mut risk_scores = Vec::new();

    for (i, candle) in price_data.iter().enumerate() {
        let momentum = if i >= window {
            let past_price = price_data[i - window].close;
            (candle.close - past_price) / past_price
        } else {
            0.0
        };

        // Convert momentum to risk score
        // Negative momentum = higher risk
        let base_risk = 5.0 - momentum * 50.0;
        let base_risk = base_risk.clamp(1.0, 10.0);

        // Add some randomness
        let noise = rng.gen_range(-1.0..1.0);
        let market_risk = (base_risk + noise).clamp(1.0, 10.0);

        // Other risk dimensions with some correlation to market risk
        let credit_risk = (base_risk + rng.gen_range(-1.5..1.5)).clamp(1.0, 10.0);
        let liquidity_risk = (base_risk * 0.8 + rng.gen_range(-1.0..1.0)).clamp(1.0, 10.0);
        let operational_risk = (5.0 + rng.gen_range(-2.0..2.0)).clamp(1.0, 10.0);
        let regulatory_risk = (5.0 + rng.gen_range(-1.5..1.5)).clamp(1.0, 10.0);
        let sentiment_risk = (base_risk + rng.gen_range(-2.0..2.0)).clamp(1.0, 10.0);

        let mut score = RiskScore::new(
            market_risk,
            credit_risk,
            liquidity_risk,
            operational_risk,
            regulatory_risk,
            sentiment_risk,
        );

        // Set direction based on momentum
        score = if momentum > 0.02 {
            score.with_direction(RiskDirection::Decreasing)
        } else if momentum < -0.02 {
            score.with_direction(RiskDirection::Increasing)
        } else {
            score.with_direction(RiskDirection::Stable)
        };

        // Set confidence
        let confidence = if momentum.abs() > 0.05 {
            Confidence::High
        } else if momentum.abs() > 0.02 {
            Confidence::Medium
        } else {
            Confidence::Low
        };
        score = score.with_confidence(confidence);

        risk_scores.push(score);
    }

    risk_scores
}
