//! Backtest hedging strategy on historical data
//!
//! Usage: cargo run --bin hedging_backtest -- --symbol BTCUSDT

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use colored::Colorize;
use rust_risk_hedging::{
    anomaly::EnsembleDetector,
    data::BybitClient,
    risk::{HedgingStrategy, Portfolio, PortfolioTracker, Position, PositionType},
};
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(author, version, about = "Backtest hedging strategy")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of candles for backtest
    #[arg(short, long, default_value = "500")]
    limit: usize,

    /// Initial portfolio value
    #[arg(short, long, default_value = "100000")]
    portfolio: f64,

    /// Position size (percentage of portfolio)
    #[arg(long, default_value = "0.5")]
    position_size: f64,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("{}", "=== Hedging Strategy Backtest ===".bold());
    println!();

    // Fetch historical data
    println!("Fetching historical data for {}...", args.symbol.cyan());
    let client = BybitClient::public();
    let data = client.get_klines(&args.symbol, "60", args.limit, None, None)?;

    if data.len() < 100 {
        println!("{}", "Insufficient data for backtest!".red());
        return Ok(());
    }

    println!("Loaded {} candles", data.len());

    // Initialize detector and strategy
    let detector = EnsembleDetector::default();
    let strategy = HedgingStrategy::default();

    // Run backtest with hedging
    println!("\n{}", "Running backtest WITH hedging...".cyan());
    let (result_with_hedge, trades_with) = run_backtest(
        &data,
        &detector,
        &strategy,
        args.portfolio,
        args.position_size,
        true,
    );

    // Run backtest without hedging for comparison
    println!("\n{}", "Running backtest WITHOUT hedging...".cyan());
    let (result_without_hedge, trades_without) = run_backtest(
        &data,
        &detector,
        &strategy,
        args.portfolio,
        args.position_size,
        false,
    );

    // Compare results
    println!("\n{}", "=== Backtest Results Comparison ===".bold());
    println!();

    println!("{:30} {:>15} {:>15}", "", "With Hedge", "Without Hedge");
    println!("{}", "-".repeat(60));

    println!(
        "{:30} {:>15.2} {:>15.2}",
        "Final Portfolio Value",
        result_with_hedge.final_value,
        result_without_hedge.final_value
    );

    println!(
        "{:30} {:>14.2}% {:>14.2}%",
        "Total Return",
        result_with_hedge.total_return,
        result_without_hedge.total_return
    );

    println!(
        "{:30} {:>14.2}% {:>14.2}%",
        "Maximum Drawdown",
        result_with_hedge.max_drawdown,
        result_without_hedge.max_drawdown
    );

    println!(
        "{:30} {:>14.2}% {:>14.2}%",
        "Average Drawdown",
        result_with_hedge.avg_drawdown,
        result_without_hedge.avg_drawdown
    );

    println!(
        "{:30} {:>14.2}% {:>14.2}%",
        "Volatility (Annualized)",
        result_with_hedge.volatility,
        result_without_hedge.volatility
    );

    println!(
        "{:30} {:>15.2} {:>15.2}",
        "Sharpe Ratio",
        result_with_hedge.sharpe_ratio,
        result_without_hedge.sharpe_ratio
    );

    // Hedging specific metrics
    println!("\n{}", "=== Hedging Metrics ===".bold());
    println!("Hedge Activations: {}", trades_with.hedge_activations);
    println!("Average Hedge Size: {:.1}%", trades_with.avg_hedge_size * 100.0);
    println!("Total Hedge Cost: ${:.2}", trades_with.total_hedge_cost);
    println!("Hedge Cost (%): {:.2}%", trades_with.total_hedge_cost / args.portfolio * 100.0);

    // Improvement analysis
    println!("\n{}", "=== Hedging Impact ===".bold());

    let dd_reduction = result_without_hedge.max_drawdown - result_with_hedge.max_drawdown;
    let return_diff = result_with_hedge.total_return - result_without_hedge.total_return;

    if dd_reduction > 0.0 {
        println!(
            "Drawdown Reduction: {} {:.2}%",
            "↓".green(),
            dd_reduction
        );
    } else {
        println!(
            "Drawdown Change: {} {:.2}%",
            "↑".red(),
            -dd_reduction
        );
    }

    if return_diff > 0.0 {
        println!(
            "Return Improvement: {} {:.2}%",
            "↑".green(),
            return_diff
        );
    } else {
        println!(
            "Return Cost: {} {:.2}%",
            "↓".yellow(),
            -return_diff
        );
    }

    // Efficiency ratio
    let hedge_cost_pct = trades_with.total_hedge_cost / args.portfolio * 100.0;
    if hedge_cost_pct > 0.0 && dd_reduction > 0.0 {
        let efficiency = dd_reduction / hedge_cost_pct;
        println!("Hedge Efficiency: {:.2}x (DD reduction per % cost)", efficiency);
    }

    // Crisis performance
    println!("\n{}", "=== Crisis Period Analysis ===".bold());
    analyze_crisis_periods(&data, &result_with_hedge, &result_without_hedge);

    Ok(())
}

#[derive(Debug, Clone)]
struct BacktestResult {
    final_value: f64,
    total_return: f64,
    max_drawdown: f64,
    avg_drawdown: f64,
    volatility: f64,
    sharpe_ratio: f64,
    equity_curve: Vec<f64>,
}

#[derive(Debug, Clone)]
struct TradeMetrics {
    hedge_activations: usize,
    avg_hedge_size: f64,
    total_hedge_cost: f64,
}

fn run_backtest(
    data: &rust_risk_hedging::data::OHLCVSeries,
    detector: &EnsembleDetector,
    strategy: &HedgingStrategy,
    initial_capital: f64,
    position_size: f64,
    use_hedging: bool,
) -> (BacktestResult, TradeMetrics) {
    let mut portfolio = Portfolio::new(initial_capital);
    let mut tracker = PortfolioTracker::new(portfolio.clone());

    let mut equity_curve = Vec::new();
    let mut drawdowns = Vec::new();
    let mut returns = Vec::new();

    let mut hedge_activations = 0;
    let mut total_hedge_size = 0.0;
    let mut total_hedge_cost = 0.0;

    // Open initial position
    let initial_price = data.data[0].close;
    let position_value = initial_capital * position_size;
    let quantity = position_value / initial_price;

    portfolio.add_position(Position::new(
        data.symbol.clone(),
        quantity,
        initial_price,
        PositionType::Long,
    ));

    let lookback = 50; // Minimum candles for detection

    for i in lookback..data.len() {
        let current_candle = &data.data[i];
        let current_price = current_candle.close;

        // Update portfolio with current price
        let mut prices = HashMap::new();
        prices.insert(data.symbol.clone(), current_price);
        tracker.update(&prices);

        // Get subset of data for detection
        let subset_data = rust_risk_hedging::data::OHLCVSeries::with_data(
            data.symbol.clone(),
            data.interval.clone(),
            data.data[0..=i].to_vec(),
        );

        // Detect anomalies
        let results = detector.detect_from_ohlcv(&subset_data);

        if let Some(latest) = results.last() {
            if use_hedging {
                // Apply hedging
                let allocation = strategy.decide(latest.score, tracker.portfolio().total_value());

                if allocation.total_hedge_pct > 0.0 {
                    hedge_activations += 1;
                    total_hedge_size += allocation.total_hedge_pct;

                    // Simulate hedge cost
                    let hedge_cost =
                        allocation.estimated_annual_cost * tracker.portfolio().total_value()
                            / 252.0; // Daily cost
                    total_hedge_cost += hedge_cost;
                }

                tracker.portfolio_mut().apply_hedge(&allocation);
            }
        }

        // Record metrics
        let current_value = tracker.portfolio().total_value() - total_hedge_cost;
        equity_curve.push(current_value);

        let dd = tracker.current_drawdown();
        drawdowns.push(dd);

        if equity_curve.len() > 1 {
            let prev = equity_curve[equity_curve.len() - 2];
            if prev > 0.0 {
                returns.push((current_value - prev) / prev);
            }
        }
    }

    // Calculate final metrics
    let final_value = *equity_curve.last().unwrap_or(&initial_capital);
    let total_return = (final_value - initial_capital) / initial_capital * 100.0;
    let max_drawdown = drawdowns.iter().cloned().fold(0.0_f64, f64::max);
    let avg_drawdown = drawdowns.iter().sum::<f64>() / drawdowns.len().max(1) as f64;

    let volatility = if returns.len() > 1 {
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt() * (252.0_f64).sqrt() * 100.0 // Annualized
    } else {
        0.0
    };

    let sharpe_ratio = if volatility > 0.0 {
        (total_return - 5.0) / volatility // Assuming 5% risk-free rate
    } else {
        0.0
    };

    let avg_hedge_size = if hedge_activations > 0 {
        total_hedge_size / hedge_activations as f64
    } else {
        0.0
    };

    (
        BacktestResult {
            final_value,
            total_return,
            max_drawdown,
            avg_drawdown,
            volatility,
            sharpe_ratio,
            equity_curve,
        },
        TradeMetrics {
            hedge_activations,
            avg_hedge_size,
            total_hedge_cost,
        },
    )
}

fn analyze_crisis_periods(
    data: &rust_risk_hedging::data::OHLCVSeries,
    with_hedge: &BacktestResult,
    without_hedge: &BacktestResult,
) {
    // Find periods with large drawdowns
    let returns = data.returns();
    let mut crisis_periods = Vec::new();

    let mut in_crisis = false;
    let mut crisis_start = 0;

    for (i, &ret) in returns.iter().enumerate() {
        if ret < -3.0 && !in_crisis {
            // 3% drop starts crisis
            in_crisis = true;
            crisis_start = i;
        } else if ret > 1.0 && in_crisis {
            // Recovery ends crisis
            crisis_periods.push((crisis_start, i));
            in_crisis = false;
        }
    }

    if crisis_periods.is_empty() {
        println!("No significant crisis periods detected.");
        return;
    }

    println!("Found {} crisis periods:", crisis_periods.len());

    for (idx, (start, end)) in crisis_periods.iter().enumerate().take(5) {
        if *start >= with_hedge.equity_curve.len() || *end >= with_hedge.equity_curve.len() {
            continue;
        }

        let offset = 50; // Lookback offset
        let adj_start = (*start).saturating_sub(offset);
        let adj_end = (*end).saturating_sub(offset).min(with_hedge.equity_curve.len() - 1);

        if adj_start >= with_hedge.equity_curve.len() || adj_end >= with_hedge.equity_curve.len() {
            continue;
        }

        let with_dd = if adj_start < adj_end && with_hedge.equity_curve[adj_start] > 0.0 {
            (with_hedge.equity_curve[adj_start] - with_hedge.equity_curve[adj_end])
                / with_hedge.equity_curve[adj_start]
                * 100.0
        } else {
            0.0
        };

        let without_dd = if adj_start < adj_end && without_hedge.equity_curve[adj_start] > 0.0 {
            (without_hedge.equity_curve[adj_start] - without_hedge.equity_curve[adj_end])
                / without_hedge.equity_curve[adj_start]
                * 100.0
        } else {
            0.0
        };

        let protection = without_dd - with_dd;

        println!(
            "  Crisis {}: DD with hedge: {:.1}%, without: {:.1}%, protection: {:.1}%",
            idx + 1,
            with_dd,
            without_dd,
            protection
        );
    }
}
