//! Example: Multi-Asset Regime Classification
//!
//! This example demonstrates how to classify market regimes
//! across multiple assets (stocks and crypto) and detect
//! cross-market relationships.
//!
//! Run with: cargo run --example multi_asset_regime

use regime_classification::{
    data::{DataLoader, YahooFinanceLoader, BybitDataLoader, OHLCVData},
    classifier::{MarketRegime, StatisticalClassifier, TransitionDetector, RegimeResult},
    signals::SignalGenerator,
};
use std::collections::HashMap;

fn main() {
    println!("{}", "=".repeat(60));
    println!("Multi-Asset Regime Classification Example");
    println!("{}", "=".repeat(60));
    println!();

    // Step 1: Load multiple asset data
    println!("Step 1: Loading multi-asset data...");

    let stock_loader = YahooFinanceLoader::new();
    let crypto_loader = BybitDataLoader::new();

    let mut assets: HashMap<&str, (OHLCVData, &str)> = HashMap::new();

    // Load stock data
    let stock_symbols = ["SPY", "QQQ", "IWM"];
    for symbol in &stock_symbols {
        let data = match stock_loader.get_daily(symbol, "1y") {
            Ok(d) => {
                println!("  Loaded {}: {} days", symbol, d.len());
                d
            }
            Err(_) => {
                println!("  Using mock data for {}", symbol);
                stock_loader.generate_mock_data(symbol, 252)
            }
        };
        assets.insert(symbol, (data, "stock"));
    }

    // Load crypto data
    let crypto_symbols = ["BTCUSDT", "ETHUSDT"];
    for symbol in &crypto_symbols {
        let data = match crypto_loader.get_klines(symbol, "D", 365) {
            Ok(d) => {
                println!("  Loaded {}: {} days", symbol, d.len());
                d
            }
            Err(_) => {
                println!("  Using mock data for {}", symbol);
                crypto_loader.generate_mock_crypto_data(symbol, 365)
            }
        };
        assets.insert(symbol, (data, "crypto"));
    }
    println!();

    // Step 2: Initialize classifiers
    println!("Step 2: Initializing classifiers...");
    let stock_classifier = StatisticalClassifier::new(20, 0.02, 0.001);
    let crypto_classifier = StatisticalClassifier::for_crypto();
    println!("  Stock classifier: 20-day window, 2% vol threshold");
    println!("  Crypto classifier: 24-day window, 5% vol threshold");
    println!();

    // Step 3: Classify each asset
    println!("Step 3: Classifying current regimes for all assets...");

    let mut current_regimes: HashMap<&str, RegimeResult> = HashMap::new();

    for (symbol, (data, asset_type)) in &assets {
        let classifier = if *asset_type == "stock" {
            &stock_classifier
        } else {
            &crypto_classifier
        };

        let result = classifier.classify(data);
        println!(
            "  {:10}: {:15} (confidence: {:.1}%)",
            symbol,
            result.regime.as_str(),
            result.confidence * 100.0
        );
        current_regimes.insert(symbol, result);
    }
    println!();

    // Step 4: Analyze cross-asset relationships
    println!("Step 4: Analyzing cross-asset regime relationships...");

    // Group by regime
    let mut regime_groups: HashMap<MarketRegime, Vec<&str>> = HashMap::new();
    for (symbol, result) in &current_regimes {
        regime_groups
            .entry(result.regime)
            .or_insert_with(Vec::new)
            .push(symbol);
    }

    println!("  Assets grouped by current regime:");
    for (regime, symbols) in &regime_groups {
        println!("    {}: {}", regime.as_str(), symbols.join(", "));
    }
    println!();

    // Calculate agreement
    let regimes: Vec<MarketRegime> = current_regimes.values().map(|r| r.regime).collect();
    let mut regime_counts: HashMap<MarketRegime, usize> = HashMap::new();
    for r in &regimes {
        *regime_counts.entry(*r).or_insert(0) += 1;
    }
    let most_common = regime_counts.iter().max_by_key(|(_, c)| *c).unwrap();
    let agreement = *most_common.1 as f64 / regimes.len() as f64;

    println!("  Market regime agreement: {:.1}%", agreement * 100.0);
    println!("  Dominant regime: {}", most_common.0.as_str());
    println!();

    // Step 5: Generate historical analysis
    println!("Step 5: Generating historical regime analysis...");

    let window = 20;
    let history_length = 100;

    let mut regime_histories: HashMap<&str, Vec<RegimeResult>> = HashMap::new();

    for (symbol, (data, asset_type)) in &assets {
        let classifier = if *asset_type == "stock" {
            &stock_classifier
        } else {
            &crypto_classifier
        };

        let mut history = Vec::new();
        let start_idx = window.max(data.len().saturating_sub(history_length));

        for i in start_idx..data.len() {
            let window_data = data.slice(i.saturating_sub(window), i + 1);
            let result = classifier.classify(&window_data);
            history.push(result);
        }

        regime_histories.insert(symbol, history);
    }

    // Calculate pairwise agreement
    println!("  Pairwise regime agreement matrix:");
    let symbols: Vec<&str> = assets.keys().copied().collect();

    print!("    {:10}", "");
    for s in &symbols {
        print!("{:10}", s);
    }
    println!();

    for s1 in &symbols {
        print!("    {:10}", s1);
        for s2 in &symbols {
            if s1 == s2 {
                print!("{:10}", "--");
            } else {
                let h1 = regime_histories.get(s1).unwrap();
                let h2 = regime_histories.get(s2).unwrap();
                let min_len = h1.len().min(h2.len());

                let agree = h1
                    .iter()
                    .take(min_len)
                    .zip(h2.iter().take(min_len))
                    .filter(|(r1, r2)| r1.regime == r2.regime)
                    .count() as f64
                    / min_len as f64;

                print!("{:>9.0}%", agree * 100.0);
            }
        }
        println!();
    }
    println!();

    // Step 6: Detect transitions
    println!("Step 6: Detecting regime transitions...");

    for (symbol, history) in &regime_histories {
        let mut detector = TransitionDetector::new(3);
        let mut transitions = 0;

        for result in history {
            let (changed, _) = detector.update(result.clone());
            if changed {
                transitions += 1;
            }
        }

        println!("  {}: {} regime transitions detected", symbol, transitions);
    }
    println!();

    // Step 7: Generate consensus signals
    println!("Step 7: Generating portfolio signals based on regime consensus...");

    let signal_gen = SignalGenerator::new(0.6);

    // Calculate consensus
    let avg_confidence: f64 = current_regimes.values().map(|r| r.confidence).sum::<f64>()
        / current_regimes.len() as f64;

    let consensus_result = RegimeResult::new(
        *most_common.0,
        avg_confidence,
        avg_confidence,
        &format!("Consensus across {} assets", assets.len()),
    );

    let signal = signal_gen.generate_signal(&consensus_result);

    println!("  Consensus Regime: {}", most_common.0.as_str());
    println!("  Average Confidence: {:.1}%", avg_confidence * 100.0);
    println!("  Portfolio Signal: {}", signal.signal_type.as_str());
    println!("  Suggested Position: {:.1}%", signal.position_size * 100.0);
    println!();

    // Step 8: Risk analysis
    println!("Step 8: Analyzing risk by market regime...");

    for (symbol, (data, _)) in &assets {
        let history = regime_histories.get(symbol).unwrap();
        let returns = data.returns();

        if returns.len() < history.len() {
            continue;
        }

        let aligned_returns = &returns[returns.len() - history.len()..];

        println!("  {}:", symbol);
        for regime in MarketRegime::all() {
            let regime_returns: Vec<f64> = history
                .iter()
                .enumerate()
                .filter(|(_, r)| r.regime == regime)
                .filter_map(|(i, _)| aligned_returns.get(i).copied())
                .collect();

            if !regime_returns.is_empty() {
                let avg_ret = regime_returns.iter().sum::<f64>() / regime_returns.len() as f64 * 252.0;
                let variance = regime_returns
                    .iter()
                    .map(|r| (r - avg_ret / 252.0).powi(2))
                    .sum::<f64>()
                    / regime_returns.len() as f64;
                let vol = variance.sqrt() * (252.0_f64).sqrt();

                println!(
                    "    {:15}: Return={:+.1}%, Vol={:.1}%",
                    regime.as_str(),
                    avg_ret * 100.0,
                    vol * 100.0
                );
            }
        }
    }
    println!();

    // Step 9: Summary
    println!("{}", "=".repeat(60));
    println!("MULTI-ASSET REGIME SUMMARY");
    println!("{}", "=".repeat(60));
    println!();

    println!("Current Market State:");
    println!("  Dominant Regime: {}", most_common.0.as_str());
    println!("  Agreement Level: {:.1}%", agreement * 100.0);
    println!();

    println!("Recommended Actions:");
    match *most_common.0 {
        MarketRegime::Bull => {
            println!("  - Markets aligned in BULL regime");
            println!("  - Consider increasing equity exposure");
            println!("  - Trend-following strategies may work well");
        }
        MarketRegime::Bear => {
            println!("  - Markets aligned in BEAR regime");
            println!("  - Consider reducing risk exposure");
            println!("  - Defensive positioning recommended");
        }
        MarketRegime::HighVolatility => {
            println!("  - High volatility across markets");
            println!("  - Reduce position sizes");
            println!("  - Wait for clearer direction");
        }
        MarketRegime::Sideways => {
            println!("  - Markets in sideways consolidation");
            println!("  - Range-trading strategies may work");
            println!("  - Avoid strong directional bets");
        }
        MarketRegime::Crisis => {
            println!("  - CRISIS regime detected!");
            println!("  - Maximum defensive positioning");
            println!("  - Consider safe haven assets");
        }
    }
    println!();

    if agreement < 0.5 {
        println!("  NOTE: Low agreement between assets");
        println!("  Markets may be transitioning between regimes");
        println!("  Consider asset-specific strategies");
    }
    println!();

    println!("{}", "=".repeat(60));
    println!("Multi-asset regime analysis completed!");
    println!("{}", "=".repeat(60));
}
