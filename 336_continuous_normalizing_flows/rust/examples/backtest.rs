//! Backtesting Example
//!
//! This example demonstrates:
//! - Running a backtest with CNF trader
//! - Analyzing performance metrics
//! - Comparing with buy-and-hold strategy

use cnf_trading::{
    api::BybitClient,
    backtest::Backtester,
    cnf::ContinuousNormalizingFlow,
    trading::CNFTrader,
    utils::{compute_features_batch, normalize_features, generate_synthetic_candles},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== CNF Trading Backtest Example ===\n");

    // Try to fetch real data, fall back to synthetic
    let candles = match fetch_real_data().await {
        Ok(c) => {
            println!("Using real Bybit data\n");
            c
        }
        Err(e) => {
            println!("Could not fetch real data ({}), using synthetic data\n", e);
            generate_synthetic_candles(1000, 50000.0)
        }
    };

    println!("Total candles: {}", candles.len());
    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        println!("Price range: {:.2} to {:.2}", first.close, last.close);
        let buy_hold_return = last.close / first.close - 1.0;
        println!("Buy & Hold return: {:.2}%\n", buy_hold_return * 100.0);
    }

    // Split data into train and test
    let train_size = (candles.len() as f64 * 0.7) as usize;
    let train_candles = &candles[..train_size];
    let test_candles = &candles[train_size..];

    println!("Training period: {} candles", train_candles.len());
    println!("Testing period: {} candles\n", test_candles.len());

    // Compute and normalize training features
    let lookback = 20;
    let train_features = compute_features_batch(train_candles, lookback);
    let (_, means, stds) = normalize_features(&train_features);

    // Create model and trader
    println!("Creating CNF model and trader...");
    let cnf = ContinuousNormalizingFlow::new(9, 64, 3);

    let trader = CNFTrader::new(cnf)
        .with_normalization(means, stds)
        .with_likelihood_threshold(-15.0)
        .with_confidence_threshold(0.4)
        .with_lookback(lookback);

    // Run backtest on test data
    println!("Running backtest...\n");
    let backtester = Backtester::new(lookback)
        .with_transaction_cost(0.0005); // 0.05% transaction cost

    let mut trader_for_backtest = trader;
    let results = backtester.run(&mut trader_for_backtest, test_candles);

    // Print results
    println!("=== Backtest Results ===\n");
    results.metrics.print_summary();

    // Calculate buy and hold for comparison
    if let (Some(first), Some(last)) = (test_candles.first(), test_candles.last()) {
        let buy_hold_return = last.close / first.close - 1.0;
        println!("\n=== Comparison with Buy & Hold ===");
        println!("Buy & Hold Return:   {:.4}", buy_hold_return);
        println!("Strategy Return:     {:.4}", results.metrics.total_return);
        println!("Excess Return:       {:.4}", results.metrics.total_return - buy_hold_return);
    }

    // Show sample of entries
    println!("\n=== Sample Backtest Entries ===");
    println!("{:<20} {:>10} {:>8} {:>8} {:>10} {:>12}",
             "Time", "Price", "Signal", "Position", "PnL", "Cumulative");
    println!("{}", "-".repeat(80));

    for entry in results.entries.iter().take(20) {
        let signal_str = match entry.signal {
            cnf_trading::trading::SignalType::Long => "LONG",
            cnf_trading::trading::SignalType::Short => "SHORT",
            cnf_trading::trading::SignalType::Neutral => "FLAT",
        };

        println!("{:<20} {:>10.2} {:>8} {:>8.4} {:>10.6} {:>12.6}",
                 entry.timestamp.format("%Y-%m-%d %H:%M"),
                 entry.close,
                 signal_str,
                 entry.position,
                 entry.pnl,
                 entry.cumulative_pnl);
    }

    if results.entries.len() > 20 {
        println!("... ({} more entries)", results.entries.len() - 20);
    }

    // Analyze signal distribution
    println!("\n=== Signal Distribution ===");
    let long_count = results.entries.iter()
        .filter(|e| matches!(e.signal, cnf_trading::trading::SignalType::Long))
        .count();
    let short_count = results.entries.iter()
        .filter(|e| matches!(e.signal, cnf_trading::trading::SignalType::Short))
        .count();
    let neutral_count = results.entries.iter()
        .filter(|e| matches!(e.signal, cnf_trading::trading::SignalType::Neutral))
        .count();

    let total = results.entries.len() as f64;
    println!("Long:    {} ({:.1}%)", long_count, long_count as f64 / total * 100.0);
    println!("Short:   {} ({:.1}%)", short_count, short_count as f64 / total * 100.0);
    println!("Neutral: {} ({:.1}%)", neutral_count, neutral_count as f64 / total * 100.0);

    // Analyze P&L distribution
    println!("\n=== P&L Analysis ===");
    let pnls: Vec<f64> = results.entries.iter().map(|e| e.pnl).collect();
    let positive_pnls: Vec<f64> = pnls.iter().filter(|&&p| p > 0.0).cloned().collect();
    let negative_pnls: Vec<f64> = pnls.iter().filter(|&&p| p < 0.0).cloned().collect();

    if !positive_pnls.is_empty() {
        let avg_win = positive_pnls.iter().sum::<f64>() / positive_pnls.len() as f64;
        println!("Average Win:  {:.6}", avg_win);
    }

    if !negative_pnls.is_empty() {
        let avg_loss = negative_pnls.iter().sum::<f64>() / negative_pnls.len() as f64;
        println!("Average Loss: {:.6}", avg_loss);
    }

    if !positive_pnls.is_empty() && !negative_pnls.is_empty() {
        let avg_win = positive_pnls.iter().sum::<f64>() / positive_pnls.len() as f64;
        let avg_loss = negative_pnls.iter().sum::<f64>() / negative_pnls.len() as f64;
        println!("Win/Loss Ratio: {:.4}", avg_win / avg_loss.abs());
    }

    println!("\n=== Backtest Complete ===");

    Ok(())
}

async fn fetch_real_data() -> anyhow::Result<Vec<cnf_trading::utils::Candle>> {
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "60", 1000).await?;

    if candles.len() < 200 {
        anyhow::bail!("Not enough candles fetched");
    }

    Ok(candles)
}
