//! Pattern Trading Strategy Example
//!
//! This example demonstrates how to use the Matching Network with the
//! PatternTradingStrategy for generating trading signals.

use matching_networks_finance::{
    MatchingNetwork,
    MarketPattern,
    PatternTradingStrategy,
    Position,
    TradingSignal,
    data::{StockDataLoader, FeatureExtractor},
};
use ndarray::{Array1, Array2};

fn main() {
    println!("=== Pattern Trading Strategy Example ===\n");

    // Generate synthetic market data
    let loader = StockDataLoader::new();

    // Create various market scenarios for the support set
    println!("Generating market data...");
    let uptrend_data = loader.generate_pattern_data("uptrend", 100, 100.0);
    let downtrend_data = loader.generate_pattern_data("downtrend", 100, 100.0);
    let consolidation_data = loader.generate_pattern_data("sideways", 100, 100.0);
    let breakout_data = loader.generate_pattern_data("breakout", 100, 100.0);
    let reversal_data = loader.generate_pattern_data("reversal", 100, 100.0);

    // Extract features
    let extractor = FeatureExtractor::new();
    let window_size = 50;
    let step = 10;

    let mut support_features = Vec::new();
    let mut support_labels = Vec::new();

    // Extract features and assign labels
    let datasets = [
        (&uptrend_data, MarketPattern::TrendContinuation),
        (&downtrend_data, MarketPattern::TrendReversal),
        (&consolidation_data, MarketPattern::Consolidation),
        (&breakout_data, MarketPattern::Breakout),
        (&reversal_data, MarketPattern::FalseBreakout),
    ];

    for (data, pattern) in &datasets {
        let features = extractor.extract_windows(data, window_size, step);
        for row in features.rows() {
            support_features.push(row.to_vec());
            support_labels.push(pattern.to_index());
        }
    }

    if support_features.is_empty() {
        println!("Error: Could not extract features from data");
        return;
    }

    let feature_dim = support_features[0].len();
    let num_support = support_features.len();

    println!("Support set: {} samples, {} features each\n", num_support, feature_dim);

    // Convert to arrays
    let support_flat: Vec<f64> = support_features.into_iter().flatten().collect();
    let support_features = Array2::from_shape_vec(
        (num_support, feature_dim),
        support_flat,
    ).unwrap();
    let support_labels = Array1::from_vec(support_labels);

    // Create the matching network and strategy
    let network = MatchingNetwork::new(
        feature_dim,
        64,
        32,
        true, // use FCE
    );

    let mut strategy = PatternTradingStrategy::new(network, 0.6); // 60% confidence threshold
    strategy.set_support_set(support_features.clone(), support_labels.clone());

    println!("Strategy Configuration:");
    println!("  Minimum confidence: {:.0}%", strategy.min_confidence() * 100.0);
    println!();

    // Generate test data (new market conditions)
    println!("Simulating trading on new market data...\n");
    let test_data = loader.generate_pattern_data("breakout", 200, 100.0);

    // Simulate trading
    let mut equity = 10000.0;
    let position_size = 1000.0;

    println!("{:<10} {:<15} {:<20} {:<15} {:<10}",
             "Bar", "Signal", "Pattern", "Confidence", "Equity");
    println!("{}", "-".repeat(70));

    for i in window_size..test_data.len() {
        let window = &test_data[i - window_size..=i];

        if let Some((signal, pattern, confidence)) = strategy.generate_signal(window) {
            // Simple P&L simulation
            let current_price = test_data[i].close;
            let prev_price = test_data[i - 1].close;
            let price_change = (current_price - prev_price) / prev_price;

            // Update equity based on position
            match strategy.position() {
                Position::Long => {
                    equity *= 1.0 + price_change;
                }
                Position::Short => {
                    equity *= 1.0 - price_change;
                }
                Position::Flat => {}
            }

            // Only print when we get a non-Hold signal or periodically
            if signal != TradingSignal::Hold || i % 20 == 0 {
                let signal_str = match signal {
                    TradingSignal::Buy => "BUY",
                    TradingSignal::Sell => "SELL",
                    TradingSignal::Hold => "HOLD",
                    TradingSignal::Exit => "EXIT",
                };

                println!(
                    "{:<10} {:<15} {:<20} {:<15.1}% ${:<10.2}",
                    i - window_size,
                    signal_str,
                    format!("{}", pattern),
                    confidence * 100.0,
                    equity
                );
            }

            // Process signal
            let _ = strategy.on_bar(window);
        }
    }

    println!();

    // Performance summary
    let summary = strategy.summary();
    println!("=== Trading Performance Summary ===");
    println!("Total Trades: {}", summary.total_trades);
    println!("Winning Trades: {}", summary.winning_trades);
    println!("Losing Trades: {}", summary.losing_trades);
    println!("Win Rate: {:.1}%", summary.win_rate * 100.0);
    println!("Total Return: {:.2}%", summary.total_return * 100.0);
    println!("Average Return per Trade: {:.2}%", summary.avg_return * 100.0);
    println!("Max Drawdown: {:.2}%", summary.max_drawdown * 100.0);
    println!("Sharpe Ratio: {:.2}", summary.sharpe_ratio);
    println!();

    // Final equity
    let initial_equity = 10000.0;
    let return_pct = (equity - initial_equity) / initial_equity * 100.0;
    println!("Simulation Results:");
    println!("  Initial Equity: ${:.2}", initial_equity);
    println!("  Final Equity: ${:.2}", equity);
    println!("  Return: {:.2}%", return_pct);

    println!("\n=== Example Complete ===");
}
