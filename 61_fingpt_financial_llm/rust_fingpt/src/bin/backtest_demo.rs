//! FinGPT Backtesting Demo
//!
//! This example demonstrates how to backtest a FinGPT-based trading strategy
//! using historical price data and simulated news sentiment.

use fingpt_trading::{
    SentimentAnalyzer, Backtester, BacktestConfig, MarketDataLoader, DataSource,
};
use fingpt_trading::backtest::generate_mock_sentiment;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(70));
    println!("FinGPT Strategy Backtesting Demo");
    println!("{}", "=".repeat(70));

    // Initialize components
    let analyzer = SentimentAnalyzer::new_mock();
    let loader = MarketDataLoader::new_mock();

    // Generate mock historical data
    println!("\nGenerating mock historical data...");
    let price_data = loader.load("BTCUSDT", DataSource::Crypto, 90).await?;
    let sentiment_data = generate_mock_sentiment(90);

    println!("  Price data points: {}", price_data.len());
    println!("  Sentiment events: {}", sentiment_data.len());
    if let (Some(first), Some(last)) = (price_data.first(), price_data.last()) {
        println!(
            "  Date range: {} to {}",
            first.timestamp.format("%Y-%m-%d"),
            last.timestamp.format("%Y-%m-%d")
        );
    }

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size_pct: 0.1,
        sentiment_threshold: 0.3,
        stop_loss_pct: 0.05,
        take_profit_pct: 0.10,
        max_positions: 3,
        commission_pct: 0.001,
    };

    println!("\nBacktest Configuration:");
    println!("  Initial Capital: ${:.2}", config.initial_capital);
    println!("  Position Size: {:.0}%", config.position_size_pct * 100.0);
    println!("  Sentiment Threshold: {}", config.sentiment_threshold);
    println!("  Stop Loss: {:.0}%", config.stop_loss_pct * 100.0);
    println!("  Take Profit: {:.0}%", config.take_profit_pct * 100.0);
    println!("  Max Positions: {}", config.max_positions);
    println!("  Commission: {:.2}%", config.commission_pct * 100.0);

    // Run backtest
    println!("\n{}", "=".repeat(70));
    println!("Running Backtest...");
    println!("{}", "=".repeat(70));

    let backtester = Backtester::new(analyzer.clone(), config.clone());
    let result = backtester.run(&price_data, &sentiment_data)?;

    // Display results
    println!("\n{:=^70}", " BACKTEST RESULTS ");

    println!("\nPerformance Summary:");
    println!("  Total Return: {:.2}%", result.total_return * 100.0);
    println!("  Final Portfolio Value: ${:.2}", result.final_value);
    println!(
        "  Profit/Loss: ${:+.2}",
        result.final_value - config.initial_capital
    );

    println!("\nRisk Metrics:");
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Sortino Ratio: {:.2}", result.sortino_ratio);
    println!("  Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("  Volatility (Ann.): {:.2}%", result.volatility * 100.0);

    println!("\nTrading Statistics:");
    println!("  Total Trades: {}", result.total_trades);
    println!("  Winning Trades: {}", result.winning_trades);
    println!("  Losing Trades: {}", result.losing_trades);
    println!("  Win Rate: {:.1}%", result.win_rate * 100.0);

    // Display individual trades
    if !result.trades.is_empty() {
        println!("\n{:=^70}", " TRADE LOG ");
        println!(
            "\n{:<12} {:<8} {:<10} {:<12} {:<12} {:<10}",
            "Date", "Type", "Size", "Entry", "Exit", "P/L"
        );
        println!("{}", "-".repeat(70));

        for trade in result.trades.iter().take(10) {
            println!(
                "{:<12} {:<8} {:<10.4} ${:<10.2} ${:<10.2} ${:+.2}",
                trade.entry_time.format("%Y-%m-%d"),
                trade.side.to_uppercase(),
                trade.size,
                trade.entry_price,
                trade.exit_price,
                trade.pnl
            );
        }

        if result.trades.len() > 10 {
            println!("... and {} more trades", result.trades.len() - 10);
        }
    }

    // Compare strategies
    println!("\n{}", "=".repeat(70));
    println!("STRATEGY COMPARISON");
    println!("{}", "=".repeat(70));

    let strategies = vec![
        (
            "Conservative",
            BacktestConfig {
                initial_capital: 100_000.0,
                position_size_pct: 0.05,
                sentiment_threshold: 0.5,
                stop_loss_pct: 0.03,
                take_profit_pct: 0.06,
                max_positions: 2,
                commission_pct: 0.001,
            },
        ),
        (
            "Moderate",
            BacktestConfig {
                initial_capital: 100_000.0,
                position_size_pct: 0.10,
                sentiment_threshold: 0.3,
                stop_loss_pct: 0.05,
                take_profit_pct: 0.10,
                max_positions: 3,
                commission_pct: 0.001,
            },
        ),
        (
            "Aggressive",
            BacktestConfig {
                initial_capital: 100_000.0,
                position_size_pct: 0.20,
                sentiment_threshold: 0.2,
                stop_loss_pct: 0.08,
                take_profit_pct: 0.15,
                max_positions: 5,
                commission_pct: 0.001,
            },
        ),
    ];

    println!(
        "\n{:<15} {:<12} {:<12} {:<12} {:<12}",
        "Strategy", "Return", "Sharpe", "Max DD", "Win Rate"
    );
    println!("{}", "-".repeat(70));

    for (name, cfg) in strategies {
        let bt = Backtester::new(SentimentAnalyzer::new_mock(), cfg);
        let res = bt.run(&price_data, &sentiment_data)?;

        println!(
            "{:<15} {:+.1}%        {:.2}         {:.1}%        {:.0}%",
            name,
            res.total_return * 100.0,
            res.sharpe_ratio,
            res.max_drawdown * 100.0,
            res.win_rate * 100.0
        );
    }

    // Monthly breakdown
    println!("\n{}", "=".repeat(70));
    println!("MONTHLY PERFORMANCE");
    println!("{}", "=".repeat(70));

    if !result.equity_curve.is_empty() {
        use std::collections::BTreeMap;

        let mut monthly: BTreeMap<String, (f64, f64)> = BTreeMap::new();
        let mut prev_value = config.initial_capital;

        for point in &result.equity_curve {
            let month_key = point.timestamp.format("%Y-%m").to_string();
            monthly
                .entry(month_key)
                .and_modify(|e| e.1 = point.equity)
                .or_insert((prev_value, point.equity));
            prev_value = point.equity;
        }

        println!(
            "\n{:<12} {:<15} {:<15} {:<12}",
            "Month", "Start Value", "End Value", "Return"
        );
        println!("{}", "-".repeat(55));

        for (month, (start, end)) in &monthly {
            let ret = (end - start) / start;
            println!(
                "{:<12} ${:<13.2} ${:<13.2} {:+.2}%",
                month,
                start,
                end,
                ret * 100.0
            );
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("DISCLAIMER");
    println!("{}", "=".repeat(70));
    println!(
        r#"
This backtest uses simulated data and mock sentiment analysis.
Past performance does not guarantee future results.
This is for educational purposes only - not financial advice.
Always do your own research before trading.
"#
    );

    println!("{}", "=".repeat(70));
    println!("Demo completed successfully!");
    println!("{}", "=".repeat(70));

    Ok(())
}
