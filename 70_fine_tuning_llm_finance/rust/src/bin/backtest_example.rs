//! Backtesting Example
//!
//! Demonstrates how to use the LlmBacktester for testing trading strategies
//! with simulated historical data.

use llm_finance::{LlmBacktester, BacktestConfig, YahooFinanceLoader, SignalType};

fn main() {
    println!("=== Trading Strategy Backtesting ===\n");

    // Load historical data
    println!("--- Loading Historical Data ---");
    let loader = YahooFinanceLoader::new();
    let series = loader.get_daily("AAPL", "1y").expect("Failed to load data");

    let prices: Vec<f64> = series.data.iter().map(|d| d.close).collect();

    println!("Historical data summary:");
    println!("  - Symbol: {}", series.symbol);
    println!("  - Data points: {}", prices.len());
    println!("  - Start price: ${:.2}", prices.first().unwrap_or(&0.0));
    println!("  - End price: ${:.2}", prices.last().unwrap_or(&0.0));
    println!("  - Min price: ${:.2}", prices.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  - Max price: ${:.2}", prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Calculate some indicators
    println!("\n--- Technical Indicators ---");
    let sma_20 = series.sma(20);
    let rsi = series.rsi(14);

    if let Some(last_sma) = sma_20.last().and_then(|v| *v) {
        println!("  - Current SMA(20): ${:.2}", last_sma);
    }
    if let Some(last_rsi) = rsi.last().and_then(|v| *v) {
        println!("  - Current RSI(14): {:.2}", last_rsi);
    }

    // Generate simple trading signals based on RSI
    println!("\n--- Generating Trading Signals ---");
    let signals: Vec<SignalType> = rsi.iter().map(|r| {
        match r {
            Some(rsi_val) if *rsi_val < 30.0 => SignalType::Buy,  // Oversold
            Some(rsi_val) if *rsi_val > 70.0 => SignalType::Sell, // Overbought
            _ => SignalType::Hold,
        }
    }).collect();

    let buy_count = signals.iter().filter(|&&s| s == SignalType::Buy).count();
    let sell_count = signals.iter().filter(|&&s| s == SignalType::Sell).count();
    let hold_count = signals.iter().filter(|&&s| s == SignalType::Hold).count();

    println!("Signal distribution:");
    println!("  - Buy signals: {}", buy_count);
    println!("  - Sell signals: {}", sell_count);
    println!("  - Hold signals: {}", hold_count);

    // Configure backtester
    println!("\n--- Backtest Configuration ---");
    let config = BacktestConfig::default()
        .with_capital(100_000.0)
        .with_position_size(0.2)
        .with_stop_loss(0.05)
        .with_take_profit(0.15);

    println!("Configuration:");
    println!("  - Initial capital: ${:.2}", config.initial_capital);
    println!("  - Position size: {:.1}%", config.position_size * 100.0);
    println!("  - Transaction cost: {:.2}%", config.transaction_cost * 100.0);
    println!("  - Stop loss: {:.1}%", config.stop_loss * 100.0);
    println!("  - Take profit: {:.1}%", config.take_profit * 100.0);
    println!("  - Max positions: {}", config.max_positions);

    // Run backtest
    println!("\n--- Running Backtest ---");
    let mut backtester = LlmBacktester::new(config);
    let result = backtester.run(&prices, &signals);

    // Display results
    println!("\n=== Backtest Results ===");
    println!("Performance Metrics:");
    println!("  - Final capital: ${:.2}", result.final_capital);
    println!("  - Total return: {:.2}%", result.total_return * 100.0);
    println!("  - Annualized return: {:.2}%", result.annualized_return * 100.0);
    println!("  - Sharpe ratio: {:.3}", result.sharpe_ratio);
    println!("  - Sortino ratio: {:.3}", result.sortino_ratio);
    println!("  - Max drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("  - Volatility (ann.): {:.2}%", result.volatility * 100.0);

    println!("\nTrading Statistics:");
    println!("  - Total trades: {}", result.total_trades);
    println!("  - Winning trades: {}", result.winning_trades);
    println!("  - Losing trades: {}", result.losing_trades);
    println!("  - Win rate: {:.2}%", result.win_rate * 100.0);
    println!("  - Avg win: {:.2}%", result.avg_win * 100.0);
    println!("  - Avg loss: {:.2}%", result.avg_loss * 100.0);
    println!("  - Profit factor: {:.2}", result.profit_factor);

    println!("\nProfit/Loss:");
    println!("  - Net profit: ${:.2}", result.final_capital - result.initial_capital);

    // Risk-adjusted returns
    println!("\nRisk Analysis:");
    if result.max_drawdown > 0.0 {
        let calmar_ratio = result.annualized_return / result.max_drawdown;
        println!("  - Calmar ratio: {:.3}", calmar_ratio);
    }

    // Buy and hold comparison
    let buy_hold_return = (prices.last().unwrap_or(&1.0) / prices.first().unwrap_or(&1.0)) - 1.0;
    println!("\n--- Strategy vs Buy-and-Hold ---");
    println!("  - Strategy return: {:.2}%", result.total_return * 100.0);
    println!("  - Buy-and-hold return: {:.2}%", buy_hold_return * 100.0);
    let outperformance = result.total_return - buy_hold_return;
    if outperformance > 0.0 {
        println!("  - Outperformance: +{:.2}%", outperformance * 100.0);
    } else {
        println!("  - Underperformance: {:.2}%", outperformance * 100.0);
    }

    // Print full report
    result.print_report();

    println!("\nBacktesting example completed successfully!");
}
