//! Example: Full pipeline from data to backtest
//!
//! This example demonstrates the complete workflow:
//! 1. Fetch data from Bybit
//! 2. Compute features
//! 3. Generate labels
//! 4. Extract rules
//! 5. Backtest strategy
//!
//! Run with: cargo run --example full_pipeline

use anyhow::Result;
use rule_extraction_trading::{
    data::{compute_features, BybitClient, StockData},
    extraction::TrepanExtractor,
    trading::{backtest, buy_and_hold_benchmark, RuleBasedStrategy},
};

fn main() -> Result<()> {
    println!("==============================================");
    println!("    Rule Extraction Trading - Full Pipeline   ");
    println!("==============================================\n");

    // Configuration
    let symbol = "BTCUSDT";
    let interval = "60"; // 1 hour
    let limit = 200;
    let initial_capital = 100_000.0;

    // Step 1: Fetch data from Bybit
    println!("Step 1: Fetching data from Bybit...");
    println!("  Symbol: {}", symbol);
    println!("  Interval: {} minutes", interval);
    println!("  Candles: {}", limit);

    let client = BybitClient::new();
    let klines = client.get_klines(symbol, interval, Some(limit))?;

    println!("  Fetched {} candles", klines.len());
    println!(
        "  Date range: {} to {}",
        klines.first().map(|k| k.timestamp.format("%Y-%m-%d %H:%M")).unwrap_or_default(),
        klines.last().map(|k| k.timestamp.format("%Y-%m-%d %H:%M")).unwrap_or_default()
    );

    // Step 2: Prepare data and compute features
    println!("\nStep 2: Computing features...");

    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let stock_data: Vec<StockData> = klines
        .iter()
        .map(|k| StockData {
            date: k.timestamp.to_string(),
            open: k.open,
            high: k.high,
            low: k.low,
            close: k.close,
            volume: k.volume,
        })
        .collect();

    let (features, feature_names) = compute_features(&stock_data);

    println!("  Features: {:?}", feature_names);
    println!("  Samples: {}", features.len());

    // Show sample features
    if let Some(last_feat) = features.last() {
        println!("\n  Latest feature values:");
        for (name, value) in feature_names.iter().zip(last_feat.iter()) {
            println!("    {}: {:.4}", name, value);
        }
    }

    // Step 3: Generate labels (next period direction)
    println!("\nStep 3: Generating training labels...");

    let labels: Vec<i32> = prices
        .windows(2)
        .map(|w| if w[1] > w[0] { 1 } else { -1 })
        .chain(std::iter::once(0))
        .collect();

    let buy_count = labels.iter().filter(|&&l| l == 1).count();
    let sell_count = labels.iter().filter(|&&l| l == -1).count();

    println!("  Label distribution:");
    println!("    BUY (up):   {} ({:.1}%)", buy_count, buy_count as f64 / labels.len() as f64 * 100.0);
    println!("    SELL (down):{} ({:.1}%)", sell_count, sell_count as f64 / labels.len() as f64 * 100.0);

    // Step 4: Extract rules using TREPAN-like algorithm
    println!("\nStep 4: Extracting trading rules...");

    let extractor = TrepanExtractor::new(feature_names.clone())
        .with_max_depth(5)
        .with_min_samples_leaf(15);

    let ruleset = extractor.extract_from_predictions(&features, &labels);

    println!("  Extracted {} rules:", ruleset.rules.len());
    println!();

    for (i, rule) in ruleset.rules.iter().enumerate() {
        println!("  Rule {}: {}", i + 1, rule.to_string(&feature_names));
    }

    // Evaluate rule quality
    let fidelity = ruleset.calculate_fidelity(&features, &labels);
    let coverage = ruleset.calculate_coverage(&features);
    let avg_complexity = ruleset.average_complexity();

    println!("\n  Rule Quality Metrics:");
    println!("    Fidelity: {:.1}%", fidelity * 100.0);
    println!("    Coverage: {:.1}%", coverage * 100.0);
    println!("    Avg complexity: {:.2} conditions/rule", avg_complexity);

    // Step 5: Backtest the strategy
    println!("\nStep 5: Running backtest...");
    println!("  Initial capital: ${:.2}", initial_capital);

    let strategy = RuleBasedStrategy::from_ruleset(ruleset);
    let result = backtest(&strategy, &features, &prices, initial_capital);

    // Compare with buy-and-hold
    let benchmark = buy_and_hold_benchmark(&prices, initial_capital);

    // Print results
    println!("\n==============================================");
    println!("                 RESULTS                      ");
    println!("==============================================");

    println!("\n--- Rule-Based Strategy ---");
    println!("  Total Return:     {:>10.2}%", result.total_return * 100.0);
    println!("  Sharpe Ratio:     {:>10.2}", result.sharpe_ratio);
    println!("  Max Drawdown:     {:>10.2}%", result.max_drawdown * 100.0);
    println!("  Number of Trades: {:>10}", result.n_trades);
    println!("  Win Rate:         {:>10.1}%", result.win_rate * 100.0);
    println!(
        "  Final Value:      ${:>9.2}",
        result.equity_curve.last().unwrap_or(&initial_capital)
    );

    println!("\n--- Buy-and-Hold Benchmark ---");
    println!("  Total Return:     {:>10.2}%", benchmark.total_return * 100.0);
    println!("  Sharpe Ratio:     {:>10.2}", benchmark.sharpe_ratio);
    println!("  Max Drawdown:     {:>10.2}%", benchmark.max_drawdown * 100.0);
    println!(
        "  Final Value:      ${:>9.2}",
        benchmark.equity_curve.last().unwrap_or(&initial_capital)
    );

    // Performance comparison
    println!("\n--- Comparison ---");
    let excess_return = result.total_return - benchmark.total_return;
    let sign = if excess_return >= 0.0 { "+" } else { "" };
    println!("  Excess Return:    {:>10}{}",
             sign, format!("{:.2}%", excess_return * 100.0));
    println!(
        "  Sharpe Diff:      {:>10.2}",
        result.sharpe_ratio - benchmark.sharpe_ratio
    );

    // Show recent trades with explanations
    if !result.trades.is_empty() {
        println!("\n==============================================");
        println!("              RECENT TRADES                   ");
        println!("==============================================");

        for (i, trade) in result.trades.iter().take(5).enumerate() {
            let trade_time = klines.get(trade.idx).map(|k| k.timestamp.format("%Y-%m-%d %H:%M").to_string()).unwrap_or_default();

            println!("\nTrade {} - {} at ${:.2}", i + 1, trade.trade_type, trade.price);
            println!("  Time: {}", trade_time);
            println!("  Size: {:.6}", trade.size);
            println!("  Reasons:");
            for exp in &trade.explanation {
                println!("    - {}", exp);
            }
        }
    }

    // Summary
    println!("\n==============================================");
    println!("                 SUMMARY                      ");
    println!("==============================================");
    println!();

    if result.total_return > benchmark.total_return {
        println!("The rule-based strategy OUTPERFORMED buy-and-hold");
        println!("by {:.2}% ({:.2}% vs {:.2}%)",
                 excess_return * 100.0,
                 result.total_return * 100.0,
                 benchmark.total_return * 100.0);
    } else {
        println!("The rule-based strategy UNDERPERFORMED buy-and-hold");
        println!("by {:.2}% ({:.2}% vs {:.2}%)",
                 excess_return.abs() * 100.0,
                 result.total_return * 100.0,
                 benchmark.total_return * 100.0);
    }

    if result.max_drawdown < benchmark.max_drawdown {
        println!("\nHowever, the strategy had LOWER risk:");
        println!("  Max drawdown: {:.2}% vs {:.2}%",
                 result.max_drawdown * 100.0,
                 benchmark.max_drawdown * 100.0);
    }

    println!("\n[Note: Past performance does not guarantee future results]");

    Ok(())
}
