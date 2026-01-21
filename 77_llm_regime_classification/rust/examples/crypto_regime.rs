//! Example: Cryptocurrency Regime Classification (Bybit)
//!
//! This example demonstrates how to classify market regimes
//! for cryptocurrency data using Bybit exchange data.
//!
//! Run with: cargo run --example crypto_regime

use regime_classification::{
    data::{BybitDataLoader, DataLoader},
    classifier::{MarketRegime, StatisticalClassifier, TransitionDetector},
    signals::{SignalGenerator, PositionSizer},
    backtest::Backtester,
    evaluate::RegimeEvaluator,
};

fn main() {
    println!("{}", "=".repeat(60));
    println!("Cryptocurrency Regime Classification Example (Bybit)");
    println!("{}", "=".repeat(60));
    println!();

    // Step 1: Load Bybit data
    println!("Step 1: Loading BTC/USDT data from Bybit...");
    let loader = BybitDataLoader::new();

    let btc_data = match loader.get_klines("BTCUSDT", "60", 1000) {
        Ok(data) => {
            println!("  Loaded {} hourly bars", data.len());
            data
        }
        Err(_) => {
            println!("  Using mock data for demonstration");
            loader.generate_mock_crypto_data("BTCUSDT", 1000)
        }
    };

    let prices = btc_data.close_prices();
    println!(
        "  Price range: ${:.2} - ${:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Also load ETH
    println!("  Loading ETH/USDT data...");
    let eth_data = match loader.get_klines("ETHUSDT", "60", 1000) {
        Ok(data) => {
            println!("  Loaded {} hourly bars for ETH", data.len());
            data
        }
        Err(_) => {
            println!("  Using mock data for ETH");
            loader.generate_mock_crypto_data("ETHUSDT", 1000)
        }
    };
    println!();

    // Step 2: Initialize crypto-specific classifier
    println!("Step 2: Initializing crypto regime classifier...");
    println!("  (Note: Crypto uses higher volatility thresholds)");

    let classifier = StatisticalClassifier::for_crypto();
    println!("  Parameters: window=24, vol_threshold=5%, trend_threshold=0.3%");
    println!();

    // Step 3: Classify current regime
    println!("Step 3: Classifying current BTC regime...");
    let btc_result = classifier.classify(&btc_data);

    println!("  BTC Current Regime: {}", btc_result.regime.as_str());
    println!("  Confidence: {:.1}%", btc_result.confidence * 100.0);
    println!("  Explanation: {}", btc_result.explanation);
    println!();

    // Also classify ETH
    println!("  Classifying current ETH regime...");
    let eth_result = classifier.classify(&eth_data);
    println!("  ETH Current Regime: {}", eth_result.regime.as_str());
    println!("  Confidence: {:.1}%", eth_result.confidence * 100.0);
    println!();

    // Step 4: Generate historical classifications
    println!("Step 4: Generating historical regime classifications...");

    let window = 24;
    let mut regime_results = Vec::new();

    for i in window..btc_data.len() {
        let window_data = btc_data.slice(i - window, i + 1);
        let regime_result = classifier.classify(&window_data);
        regime_results.push(regime_result);
    }

    // Count regimes
    let mut counts = std::collections::HashMap::new();
    for r in &regime_results {
        *counts.entry(r.regime).or_insert(0) += 1;
    }

    println!("  BTC Regime Distribution (hourly):");
    let total = regime_results.len();
    for regime in MarketRegime::all() {
        let count = counts.get(&regime).unwrap_or(&0);
        let pct = *count as f64 / total as f64 * 100.0;
        println!("    {:15}: {:4} ({:5.1}%)", regime.as_str(), count, pct);
    }
    println!();

    // Step 5: Calculate position sizing
    println!("Step 5: Calculating position sizes...");

    let signal_gen = SignalGenerator::new(0.5);
    let signal = signal_gen.generate_signal(&btc_result);

    let position_sizer = PositionSizer::new(0.5, 0.02, true);

    // Calculate volatility
    let returns = btc_data.returns();
    let recent_returns: Vec<f64> = returns.iter().rev().take(24).copied().collect();
    let volatility = if recent_returns.len() > 1 {
        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (recent_returns.len() - 1) as f64;
        variance.sqrt() * (24.0_f64).sqrt() * (365.0_f64).sqrt()
    } else {
        0.5
    };

    let current_price = prices.last().unwrap_or(&45000.0);
    let position = position_sizer.calculate_position(&signal, 100000.0, *current_price, volatility);

    println!("  Signal: {}", signal.signal_type.as_str());
    println!("  Current BTC Price: ${:.2}", position.entry_price);
    println!("  Annualized Volatility: {:.1}%", volatility * 100.0);
    println!(
        "  Position Size: ${:.2} ({:.1}% of capital)",
        position.position_dollars,
        position.position_pct * 100.0
    );
    println!("  BTC Units: {:.6}", position.position_units);
    if let Some(sl) = position.stop_loss_price {
        println!("  Stop Loss Price: ${:.2}", sl);
    }
    println!();

    // Step 6: Run backtest
    println!("Step 6: Running backtest on BTC...");
    let backtester = Backtester::new(100000.0, 0.001, 0.001);

    let trimmed_data = btc_data.slice(window, btc_data.len());
    let backtest_result = backtester.run(&trimmed_data, &regime_results, &signal_gen);

    println!("  Total Return: {:.2}%", backtest_result.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", backtest_result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", backtest_result.max_drawdown * 100.0);
    println!("  Win Rate: {:.1}%", backtest_result.win_rate * 100.0);
    println!("  Number of Trades: {}", backtest_result.num_trades);
    println!();

    // Step 7: Compare BTC vs ETH
    println!("Step 7: Comparing BTC vs ETH regimes...");

    let mut eth_results = Vec::new();
    for i in window..eth_data.len().min(btc_data.len()) {
        let window_data = eth_data.slice(i - window, i + 1);
        let eth_result = classifier.classify(&window_data);
        eth_results.push(eth_result);
    }

    let min_len = regime_results.len().min(eth_results.len());
    let agreement = regime_results
        .iter()
        .take(min_len)
        .zip(eth_results.iter().take(min_len))
        .filter(|(btc, eth)| btc.regime == eth.regime)
        .count() as f64
        / min_len as f64;

    println!("  BTC-ETH Regime Agreement: {:.1}%", agreement * 100.0);
    println!();

    // Step 8: Generate report
    println!("Step 8: Generating evaluation report...");
    let evaluator = RegimeEvaluator::new();
    let btc_returns = btc_data.returns();
    let report = evaluator.generate_report(&regime_results, &btc_returns[window - 1..]);
    println!();
    println!("{}", report);

    println!();
    println!("{}", "=".repeat(60));
    println!("Crypto regime classification example completed!");
    println!("{}", "=".repeat(60));
}
