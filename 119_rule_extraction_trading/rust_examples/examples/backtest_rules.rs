//! Example: Backtesting a rule-based trading strategy
//!
//! This example demonstrates how to backtest trading rules
//! on historical data and evaluate performance.
//!
//! Run with: cargo run --example backtest_rules

use anyhow::Result;
use rule_extraction_trading::{
    data::{compute_features, StockData},
    rules::{Condition, Operator, Rule},
    trading::{backtest, buy_and_hold_benchmark, RuleBasedStrategy},
};

fn main() -> Result<()> {
    println!("=== Rule-Based Strategy Backtest ===\n");

    // Generate synthetic price data
    let prices = generate_price_data(500);
    let stock_data = generate_stock_data(&prices);

    // Compute features
    let (features, feature_names) = compute_features(&stock_data);

    println!("Data prepared:");
    println!("  - {} price points", prices.len());
    println!("  - {} features: {:?}", feature_names.len(), feature_names);
    println!(
        "  - Price range: ${:.2} - ${:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Define trading rules
    let rules = create_trading_rules();

    println!("\n=== Trading Rules ===");
    for (i, rule) in rules.iter().enumerate() {
        println!("Rule {}: {}", i + 1, rule.to_string(&feature_names));
    }

    // Create strategy and run backtest
    let strategy = RuleBasedStrategy::new(rules, feature_names.clone());
    let initial_capital = 100_000.0;

    println!("\n=== Backtest Configuration ===");
    println!("  Initial capital: ${:.2}", initial_capital);
    println!("  Start price: ${:.2}", prices.first().unwrap_or(&0.0));
    println!("  End price: ${:.2}", prices.last().unwrap_or(&0.0));

    // Run backtest
    let result = backtest(&strategy, &features, &prices, initial_capital);

    // Print results
    println!("\n=== Rule-Based Strategy Results ===");
    println!("{:-<50}", "");
    println!("Total Return:       {:>12.2}%", result.total_return * 100.0);
    println!("Sharpe Ratio:       {:>12.2}", result.sharpe_ratio);
    println!("Max Drawdown:       {:>12.2}%", result.max_drawdown * 100.0);
    println!("Number of Trades:   {:>12}", result.n_trades);
    println!("Win Rate:           {:>12.1}%", result.win_rate * 100.0);
    println!("Avg Trade Return:   {:>12.2}%", result.avg_trade_return * 100.0);
    println!(
        "Final Portfolio:    ${:>11.2}",
        result.equity_curve.last().unwrap_or(&initial_capital)
    );

    // Compare with buy-and-hold
    let benchmark = buy_and_hold_benchmark(&prices, initial_capital);

    println!("\n=== Buy-and-Hold Benchmark ===");
    println!("{:-<50}", "");
    println!("Total Return:       {:>12.2}%", benchmark.total_return * 100.0);
    println!("Sharpe Ratio:       {:>12.2}", benchmark.sharpe_ratio);
    println!("Max Drawdown:       {:>12.2}%", benchmark.max_drawdown * 100.0);
    println!(
        "Final Portfolio:    ${:>11.2}",
        benchmark.equity_curve.last().unwrap_or(&initial_capital)
    );

    // Performance comparison
    println!("\n=== Performance Comparison ===");
    println!("{:-<50}", "");
    let outperformance = result.total_return - benchmark.total_return;
    let outperform_str = if outperformance > 0.0 {
        format!("+{:.2}%", outperformance * 100.0)
    } else {
        format!("{:.2}%", outperformance * 100.0)
    };
    println!("Excess Return:      {:>12}", outperform_str);
    println!(
        "Sharpe Difference:  {:>12.2}",
        result.sharpe_ratio - benchmark.sharpe_ratio
    );

    // Show trade history
    if !result.trades.is_empty() {
        println!("\n=== Trade History (first 10) ===");
        println!("{:-<80}", "");
        println!(
            "{:>6} {:>8} {:>12} {:>12}",
            "Index", "Type", "Price", "Size"
        );
        println!("{:-<80}", "");

        for trade in result.trades.iter().take(10) {
            println!(
                "{:>6} {:>8} {:>12.2} {:>12.4}",
                trade.idx, trade.trade_type, trade.price, trade.size
            );
        }

        if result.trades.len() > 10 {
            println!("... ({} more trades)", result.trades.len() - 10);
        }
    }

    // Show trades with explanations
    println!("\n=== Trade Explanations (first 3) ===");
    for trade in result.trades.iter().take(3) {
        println!("\n[{}] {} at ${:.2}", trade.idx, trade.trade_type, trade.price);
        for exp in &trade.explanation {
            println!("  - {}", exp);
        }
    }

    // Equity curve statistics
    if !result.equity_curve.is_empty() {
        let min_equity = result
            .equity_curve
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_equity = result
            .equity_curve
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        println!("\n=== Equity Curve Statistics ===");
        println!("  Min equity: ${:.2}", min_equity);
        println!("  Max equity: ${:.2}", max_equity);
        println!(
            "  Equity range: ${:.2}",
            max_equity - min_equity
        );
    }

    Ok(())
}

/// Generate synthetic price data with trends and mean reversion
fn generate_price_data(n: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut prices = Vec::with_capacity(n);
    let mut price = 100.0;
    let mut trend = 0.0;

    for i in 0..n {
        // Change trend occasionally
        if i % 50 == 0 {
            trend = rng.gen_range(-0.002..0.002);
        }

        // Random walk with trend
        let noise: f64 = rng.gen_range(-0.02..0.02);
        let mean_reversion = (100.0 - price) * 0.001;

        price *= 1.0 + trend + noise + mean_reversion;
        price = price.max(50.0).min(200.0); // Keep in range

        prices.push(price);
    }

    prices
}

/// Generate stock data from prices
fn generate_stock_data(prices: &[f64]) -> Vec<StockData> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    prices
        .iter()
        .enumerate()
        .map(|(i, &close)| {
            let volatility: f64 = rng.gen_range(0.005..0.02);
            let open = close * (1.0 + rng.gen_range(-volatility..volatility));
            let high = close.max(open) * (1.0 + rng.gen_range(0.0..volatility));
            let low = close.min(open) * (1.0 - rng.gen_range(0.0..volatility));
            let volume = rng.gen_range(100000.0..500000.0);

            StockData {
                date: format!("2024-{:02}-{:02}", (i / 30) % 12 + 1, i % 28 + 1),
                open,
                high,
                low,
                close,
                volume,
            }
        })
        .collect()
}

/// Create a set of trading rules
fn create_trading_rules() -> Vec<Rule> {
    vec![
        // Rule 1: RSI oversold with positive MACD
        Rule::new(
            vec![
                Condition::new(0, Operator::LessOrEqual, 35.0),
                Condition::new(1, Operator::GreaterThan, 0.0),
            ],
            1, // BUY
            0.75,
        ),
        // Rule 2: RSI overbought with negative MACD
        Rule::new(
            vec![
                Condition::new(0, Operator::GreaterThan, 65.0),
                Condition::new(1, Operator::LessOrEqual, 0.0),
            ],
            -1, // SELL
            0.75,
        ),
        // Rule 3: Strong uptrend (SMA ratio > 1.02)
        Rule::new(
            vec![
                Condition::new(2, Operator::GreaterThan, 1.02),
                Condition::new(3, Operator::LessOrEqual, 0.025),
            ],
            1, // BUY
            0.65,
        ),
        // Rule 4: Strong downtrend with high volatility
        Rule::new(
            vec![
                Condition::new(2, Operator::LessOrEqual, 0.98),
                Condition::new(3, Operator::GreaterThan, 0.02),
            ],
            -1, // SELL
            0.70,
        ),
        // Rule 5: Volume spike with positive momentum
        Rule::new(
            vec![
                Condition::new(4, Operator::GreaterThan, 0.2),
                Condition::new(1, Operator::GreaterThan, 0.5),
            ],
            1, // BUY
            0.60,
        ),
    ]
}
