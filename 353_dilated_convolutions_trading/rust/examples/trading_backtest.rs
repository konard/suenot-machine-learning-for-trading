//! Example: Trading Backtest with Dilated Convolutions
//!
//! This example demonstrates:
//! 1. Creating a trading strategy with dilated convolutions
//! 2. Backtesting on historical data
//! 3. Analyzing performance metrics
//! 4. Comparing different model configurations

use dilated_conv_trading::api::Interval;
use dilated_conv_trading::conv::DilatedConvStack;
use dilated_conv_trading::strategy::{PositionSizer, TradingStrategy};
use dilated_conv_trading::utils::{
    equity_curve, max_drawdown, profit_factor, sharpe_ratio, sortino_ratio, win_rate,
};
use dilated_conv_trading::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Trading Backtest with Dilated Convolutions ===\n");

    // 1. Fetch historical data
    println!("1. Fetching historical data...\n");

    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    let klines = client
        .get_klines_with_interval(symbol, Interval::Hour1, 1000)
        .await?;

    println!("   Symbol: {}", symbol);
    println!("   Timeframe: 1 hour");
    println!("   Candles: {}", klines.len());

    let first_time = chrono::DateTime::from_timestamp_millis(klines.first().unwrap().timestamp);
    let last_time = chrono::DateTime::from_timestamp_millis(klines.last().unwrap().timestamp);
    println!(
        "   Period: {:?} to {:?}",
        first_time.map(|t| t.format("%Y-%m-%d").to_string()),
        last_time.map(|t| t.format("%Y-%m-%d").to_string())
    );

    // 2. Create and compare strategies
    println!("\n2. Testing different model configurations...\n");

    let configs = [
        ("Small (4 layers)", vec![1, 2, 4, 8]),
        ("Medium (6 layers)", vec![1, 2, 4, 8, 16, 32]),
        ("Large (8 layers)", vec![1, 2, 4, 8, 16, 32, 64, 128]),
    ];

    let mut results = Vec::new();

    for (name, dilation_rates) in &configs {
        println!("   Testing: {}", name);
        println!("     Dilations: {:?}", dilation_rates);

        let model = DilatedConvStack::new(5, 32, dilation_rates);
        println!("     Receptive field: {} bars ({} hours)",
            model.receptive_field(), model.receptive_field());

        let strategy = TradingStrategy::new(model);
        let backtest = strategy.backtest(&klines);

        println!("     Total Return: {:.2}%", backtest.total_return * 100.0);
        println!("     Sharpe Ratio: {:.2}", backtest.sharpe_ratio);
        println!();

        results.push((name, backtest));
    }

    // 3. Detailed analysis of best strategy
    println!("3. Detailed Analysis of Best Strategy...\n");

    // Find best by Sharpe ratio
    let (best_name, best_result) = results
        .iter()
        .max_by(|a, b| a.1.sharpe_ratio.partial_cmp(&b.1.sharpe_ratio).unwrap())
        .unwrap();

    println!("   Best configuration: {}\n", best_name);
    println!("{}", best_result);

    // 4. Run full backtest with the best configuration
    println!("4. Full Backtest with Equity Curve...\n");

    let best_model = DilatedConvStack::new(5, 32, &[1, 2, 4, 8, 16, 32]);
    let strategy = TradingStrategy::new(best_model);
    let signals = strategy.generate_signals(&klines);

    // Calculate returns
    let mut strategy_returns = Vec::new();
    let mut buy_and_hold_returns = Vec::new();

    for i in 1..klines.len() {
        let ret = (klines[i].close - klines[i - 1].close) / klines[i - 1].close;
        let position = signals[i - 1].to_multiplier();

        strategy_returns.push(ret * position);
        buy_and_hold_returns.push(ret);
    }

    // Calculate metrics
    let strategy_equity = equity_curve(&strategy_returns);
    let bh_equity = equity_curve(&buy_and_hold_returns);

    println!("   Strategy Metrics:");
    println!("     Final equity: {:.4}x", strategy_equity.last().unwrap_or(&1.0));
    println!("     Sharpe ratio: {:.2}", sharpe_ratio(&strategy_returns, 0.0, 365.0 * 24.0));
    println!("     Sortino ratio: {:.2}", sortino_ratio(&strategy_returns, 0.0, 365.0 * 24.0));
    println!("     Max drawdown: {:.2}%", max_drawdown(&strategy_equity) * 100.0);
    println!("     Win rate: {:.2}%", win_rate(&strategy_returns) * 100.0);
    println!("     Profit factor: {:.2}", profit_factor(&strategy_returns));

    println!("\n   Buy & Hold Metrics:");
    println!("     Final equity: {:.4}x", bh_equity.last().unwrap_or(&1.0));
    println!("     Sharpe ratio: {:.2}", sharpe_ratio(&buy_and_hold_returns, 0.0, 365.0 * 24.0));
    println!("     Max drawdown: {:.2}%", max_drawdown(&bh_equity) * 100.0);

    // 5. Signal analysis
    println!("\n5. Signal Distribution...\n");

    use dilated_conv_trading::strategy::Signal;
    let mut signal_counts = std::collections::HashMap::new();
    for signal in &signals {
        *signal_counts.entry(format!("{:?}", signal)).or_insert(0) += 1;
    }

    let total = signals.len();
    for (signal, count) in &signal_counts {
        let pct = *count as f64 / total as f64 * 100.0;
        println!("   {:12}: {:4} ({:.1}%)", signal, count, pct);
    }

    // 6. Position sizing example
    println!("\n6. Position Sizing Example...\n");

    let sizer = PositionSizer::new();
    let current_price = klines.last().unwrap().close;
    let capital = 10000.0;

    println!("   Capital: ${:.2}", capital);
    println!("   Current price: ${:.2}", current_price);

    // Simulate different confidence levels
    for confidence in [0.3, 0.5, 0.7, 0.9] {
        let pred = dilated_conv_trading::strategy::Prediction::new(1.0, 0.02, 0.01);
        let size = sizer.calculate_size(&pred, capital, current_price);
        println!(
            "   Confidence {:.0}%: position size = {:.4} {} (${:.2})",
            confidence * 100.0,
            size,
            symbol.replace("USDT", ""),
            size * current_price
        );
    }

    // 7. Equity curve visualization (ASCII)
    println!("\n7. Equity Curve (ASCII)...\n");

    let step = strategy_equity.len() / 50;
    let sampled: Vec<f64> = strategy_equity.iter().step_by(step.max(1)).cloned().collect();
    let min_eq = sampled.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_eq = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("   {:.2}x |", max_eq);
    for i in (0..10).rev() {
        let threshold = min_eq + (max_eq - min_eq) * (i as f64 / 10.0);
        let line: String = sampled
            .iter()
            .map(|&eq| if eq >= threshold { '█' } else { ' ' })
            .collect();
        if i == 5 {
            println!("         | {}", line);
        } else {
            println!("         | {}", line);
        }
    }
    println!("   {:.2}x |{}", min_eq, "─".repeat(sampled.len()));
    println!("          Start{}End", " ".repeat(sampled.len() - 8));

    println!("\n=== Backtest Complete ===");

    Ok(())
}
