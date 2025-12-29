//! Backtest Comparison Example
//!
//! Compares online learning vs batch learning vs monthly retrain approaches.
//!
//! Run with: cargo run --example backtest_comparison

use online_learning::api::BybitClient;
use online_learning::backtest::BacktestEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Online vs Batch Learning Comparison ===\n");

    // Fetch data
    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    println!("Fetching {} data from Bybit...", symbol);
    let candles = client.get_klines(symbol, "1h", 1000).await?;
    println!("Fetched {} candles ({:.1} days of hourly data)\n",
        candles.len(),
        candles.len() as f64 / 24.0);

    // Create backtest engine
    let engine = BacktestEngine::new(candles.clone())
        .with_periods(vec![12, 24, 48, 96])
        .with_transaction_cost(0.001)
        .with_signal_threshold(0.0005);

    println!("Backtest Configuration:");
    println!("  Momentum periods: [12, 24, 48, 96] hours");
    println!("  Transaction cost: 0.1%");
    println!("  Signal threshold: 0.05%");
    println!();

    // === Run Backtests ===
    println!("Running backtests...\n");

    // 1. Online Learning
    print!("  1. Online Learning... ");
    let online_result = engine.run_online(0.01)?;
    println!("Done!");

    // 2. Static Model (train once)
    print!("  2. Static Model (train on first 25%)... ");
    let static_result = engine.run_static(candles.len() / 4)?;
    println!("Done!");

    // 3. Monthly Retrain
    print!("  3. Monthly Retrain (every 720 hours)... ");
    let monthly_result = engine.run_monthly_retrain(100, 720)?;
    println!("Done!");

    // === Display Results ===
    println!("\n=== Comparison Results ===\n");

    // Header
    println!("{:20} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Approach", "Return", "Sharpe", "Max DD", "Trades", "Win Rate");
    println!("{}", "-".repeat(80));

    // Online Learning
    println!("{:20} {:>11.2}% {:>12.2} {:>11.2}% {:>12} {:>11.1}%",
        "Online Learning",
        online_result.total_return * 100.0,
        online_result.sharpe_ratio,
        online_result.max_drawdown * 100.0,
        online_result.n_trades,
        online_result.win_rate * 100.0);

    // Static Model
    println!("{:20} {:>11.2}% {:>12.2} {:>11.2}% {:>12} {:>11.1}%",
        "Static Model",
        static_result.total_return * 100.0,
        static_result.sharpe_ratio,
        static_result.max_drawdown * 100.0,
        static_result.n_trades,
        static_result.win_rate * 100.0);

    // Monthly Retrain
    println!("{:20} {:>11.2}% {:>12.2} {:>11.2}% {:>12} {:>11.1}%",
        "Monthly Retrain",
        monthly_result.total_return * 100.0,
        monthly_result.sharpe_ratio,
        monthly_result.max_drawdown * 100.0,
        monthly_result.n_trades,
        monthly_result.win_rate * 100.0);

    // === Analysis ===
    println!("\n=== Analysis ===\n");

    // Best approach
    let approaches = vec![
        ("Online Learning", online_result.sharpe_ratio, online_result.total_return),
        ("Static Model", static_result.sharpe_ratio, static_result.total_return),
        ("Monthly Retrain", monthly_result.sharpe_ratio, monthly_result.total_return),
    ];

    let best_sharpe = approaches
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    let best_return = approaches
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    println!("Best Sharpe Ratio: {} ({:.2})", best_sharpe.0, best_sharpe.1);
    println!("Best Total Return: {} ({:.2}%)", best_return.0, best_return.2 * 100.0);

    // Compare online vs static
    let online_vs_static = online_result.sharpe_ratio - static_result.sharpe_ratio;
    if online_vs_static > 0.0 {
        println!("\nOnline learning outperformed static model by {:.2} Sharpe points", online_vs_static);
        println!("This suggests the market experienced concept drift during the test period.");
    } else {
        println!("\nStatic model performed better, suggesting relatively stable market conditions.");
    }

    // === Equity Curve Summary ===
    println!("\n=== Equity Curve Summary ===\n");

    // Show equity at key points
    let n = online_result.cumulative_returns.len();
    if n > 4 {
        let quarters = [0, n/4, n/2, 3*n/4, n-1];

        println!("{:>8} {:>15} {:>15} {:>15}",
            "Period", "Online", "Static", "Monthly");
        println!("{}", "-".repeat(56));

        for &i in &quarters {
            let online_eq = 1.0 + online_result.cumulative_returns.get(i).unwrap_or(&0.0);
            let static_eq = 1.0 + static_result.cumulative_returns.get(i).unwrap_or(&0.0);
            let monthly_eq = 1.0 + monthly_result.cumulative_returns.get(i).unwrap_or(&0.0);

            let period_label = match i {
                0 => "Start",
                _ if i == n/4 => "25%",
                _ if i == n/2 => "50%",
                _ if i == 3*n/4 => "75%",
                _ => "End",
            };

            println!("{:>8} {:>15.4} {:>15.4} {:>15.4}",
                period_label, online_eq, static_eq, monthly_eq);
        }
    }

    // === Recommendations ===
    println!("\n=== Recommendations ===\n");

    if online_result.sharpe_ratio > static_result.sharpe_ratio {
        println!("✓ Online learning is recommended for this market:");
        println!("  - Market shows signs of non-stationarity");
        println!("  - Continuous adaptation provides edge");
        println!("  - Consider monitoring drift frequency for parameter tuning");
    } else {
        println!("✓ Static or periodic retrain may be sufficient:");
        println!("  - Market appears relatively stable");
        println!("  - Less computational overhead with batch training");
        println!("  - Consider monthly or quarterly model updates");
    }

    if online_result.max_drawdown > 0.15 {
        println!("\n⚠ Warning: High maximum drawdown detected");
        println!("  - Consider adding position sizing rules");
        println!("  - May need tighter stop-loss thresholds");
    }

    if online_result.n_trades > 500 {
        println!("\n⚠ High trading frequency detected");
        println!("  - Transaction costs may be significant");
        println!("  - Consider increasing signal threshold");
    }

    println!("\nNote: Past performance does not guarantee future results.");
    println!("Always validate strategies on out-of-sample data before live trading.");

    Ok(())
}
