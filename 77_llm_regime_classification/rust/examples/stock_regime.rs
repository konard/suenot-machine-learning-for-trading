//! Example: Stock Market Regime Classification
//!
//! This example demonstrates how to classify market regimes
//! for stock market data using S&P 500 (SPY) ETF.
//!
//! Run with: cargo run --example stock_regime

use regime_classification::{
    data::{DataLoader, YahooFinanceLoader},
    classifier::{MarketRegime, StatisticalClassifier, TransitionDetector},
    signals::SignalGenerator,
    backtest::Backtester,
    evaluate::RegimeEvaluator,
};

fn main() {
    println!("{}", "=".repeat(60));
    println!("Stock Market Regime Classification Example");
    println!("{}", "=".repeat(60));
    println!();

    // Step 1: Load data
    println!("Step 1: Loading SPY data...");
    let loader = YahooFinanceLoader::new();

    let spy_data = match loader.get_daily("SPY", "1y") {
        Ok(data) => {
            println!("  Loaded {} days of data", data.len());
            data
        }
        Err(_) => {
            println!("  Using mock data for demonstration");
            loader.generate_mock_data("SPY", 252)
        }
    };

    let prices = spy_data.close_prices();
    println!(
        "  Price range: ${:.2} - ${:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // Step 2: Initialize classifier
    println!("Step 2: Initializing regime classifier...");
    let classifier = StatisticalClassifier::new(20, 0.02, 0.001);
    println!("  Parameters: window=20, vol_threshold=2%, trend_threshold=0.1%");
    println!();

    // Step 3: Classify current regime
    println!("Step 3: Classifying current market regime...");
    let result = classifier.classify(&spy_data);

    println!("  Current Regime: {}", result.regime.as_str());
    println!("  Confidence: {:.1}%", result.confidence * 100.0);
    println!("  Explanation: {}", result.explanation);
    println!();

    // Step 4: Generate historical classifications
    println!("Step 4: Generating historical regime classifications...");

    let window = 20;
    let mut regime_results = Vec::new();

    for i in window..spy_data.len() {
        let window_data = spy_data.slice(i - window, i + 1);
        let regime_result = classifier.classify(&window_data);
        regime_results.push(regime_result);
    }

    // Count regimes
    let mut counts = std::collections::HashMap::new();
    for r in &regime_results {
        *counts.entry(r.regime).or_insert(0) += 1;
    }

    println!("  Regime Distribution:");
    let total = regime_results.len();
    for regime in MarketRegime::all() {
        let count = counts.get(&regime).unwrap_or(&0);
        let pct = *count as f64 / total as f64 * 100.0;
        println!("    {:15}: {:4} ({:5.1}%)", regime.as_str(), count, pct);
    }
    println!();

    // Step 5: Generate trading signals
    println!("Step 5: Generating trading signals...");
    let signal_gen = SignalGenerator::new(0.6);
    let signal = signal_gen.generate_signal(&result);

    println!("  Signal Type: {}", signal.signal_type.as_str());
    println!("  Position Size: {:.1}%", signal.position_size * 100.0);
    if let Some(sl) = signal.stop_loss {
        println!("  Stop Loss: {:.1}%", sl * 100.0);
    }
    if let Some(tp) = signal.take_profit {
        println!("  Take Profit: {:.1}%", tp * 100.0);
    }
    println!("  Reasoning: {}", signal.reasoning);
    println!();

    // Step 6: Run backtest
    println!("Step 6: Running backtest...");
    let backtester = Backtester::new(100000.0, 0.001, 0.0005);

    let trimmed_data = spy_data.slice(window, spy_data.len());
    let backtest_result = backtester.run(&trimmed_data, &regime_results, &signal_gen);

    println!("  Total Return: {:.2}%", backtest_result.total_return * 100.0);
    println!(
        "  Annualized Return: {:.2}%",
        backtest_result.annualized_return * 100.0
    );
    println!("  Sharpe Ratio: {:.2}", backtest_result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", backtest_result.max_drawdown * 100.0);
    println!("  Win Rate: {:.1}%", backtest_result.win_rate * 100.0);
    println!("  Number of Trades: {}", backtest_result.num_trades);
    println!();

    // Step 7: Generate report
    println!("Step 7: Generating evaluation report...");
    let evaluator = RegimeEvaluator::new();
    let returns = spy_data.returns();
    let report = evaluator.generate_report(&regime_results, &returns[window - 1..]);
    println!();
    println!("{}", report);

    println!();
    println!("{}", "=".repeat(60));
    println!("Example completed successfully!");
    println!("{}", "=".repeat(60));
}
