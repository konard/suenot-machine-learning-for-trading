//! Trading Strategy with Distilled Decision Tree
//!
//! This example demonstrates a complete trading workflow:
//! 1. Load market data
//! 2. Compute technical features
//! 3. Simulate teacher model predictions
//! 4. Distill to interpretable tree
//! 5. Backtest and compare performance

use decision_tree_distillation::{
    model::{DecisionTree, DistillationConfig},
    data::{generate_synthetic_data, TechnicalFeatures},
    backtest::{Backtester, BacktestConfig},
    trading::{TradingSignal, RiskParams},
};
use ndarray::Array1;

#[tokio::main]
async fn main() {
    println!("Trading Strategy with Distilled Decision Tree");
    println!("{}", "=".repeat(60));

    // 1. Generate synthetic market data
    println!("\n1. Loading market data...");
    let market_data = generate_synthetic_data(1000, 50000.0);
    println!("   Loaded {} candles", market_data.len());
    println!(
        "   Price range: ${:.2} - ${:.2}",
        market_data.close.iter().cloned().fold(f64::INFINITY, f64::min),
        market_data.close.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // 2. Compute technical features
    println!("\n2. Computing technical features...");
    let tech_features = TechnicalFeatures::compute(&market_data, 1)
        .expect("Failed to compute features");
    println!("   Features: {:?}", tech_features.feature_names);
    println!("   Shape: {} samples x {} features",
        tech_features.features.nrows(),
        tech_features.features.ncols()
    );

    // 3. Simulate teacher model (in practice, this would be a trained complex model)
    println!("\n3. Simulating teacher model predictions...");
    let teacher_predictions = simulate_teacher_predictions(&tech_features);
    let buy_signals = teacher_predictions.iter().filter(|&&p| p > 0.6).count();
    let sell_signals = teacher_predictions.iter().filter(|&&p| p < 0.4).count();
    println!("   Buy signals: {}", buy_signals);
    println!("   Sell signals: {}", sell_signals);
    println!("   Hold signals: {}", teacher_predictions.len() - buy_signals - sell_signals);

    // 4. Distill to interpretable tree
    println!("\n4. Distilling to decision tree...");
    let config = DistillationConfig {
        max_depth: 5,
        min_samples_leaf: 30,
        min_samples_split: 60,
        feature_names: Some(tech_features.feature_names.clone()),
        class_names: Some(vec!["SELL".to_string(), "BUY".to_string()]),
    };

    let mut tree = DecisionTree::new(config);
    tree.fit(&tech_features.features, &teacher_predictions)
        .expect("Failed to fit tree");

    println!("   Tree depth: {}", tree.depth());
    println!("   Number of rules: {}", tree.n_leaves());

    // Calculate fidelity
    let fidelity = tree.fidelity_score(&tech_features.features, &teacher_predictions)
        .expect("Failed to calculate fidelity");
    println!("   Fidelity to teacher: {:.2}%", fidelity * 100.0);

    // 5. Extract and display trading rules
    println!("\n5. Extracted Trading Rules:");
    println!("{}", "-".repeat(60));
    let rules = tree.extract_rules().expect("Failed to extract rules");
    for (i, rule) in rules.iter().enumerate().take(10) {
        println!("   Rule {}: {}", i + 1, rule);
    }
    if rules.len() > 10 {
        println!("   ... and {} more rules", rules.len() - 10);
    }

    // 6. Make student predictions
    println!("\n6. Making student predictions...");
    let student_predictions = tree.predict(&tech_features.features)
        .expect("Failed to predict");

    // 7. Backtest both models
    println!("\n7. Backtesting models...");
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.1,
        transaction_cost: 0.001,
        slippage: 0.0005,
        risk_free_rate: 0.02,
    };

    let backtester = Backtester::new(backtest_config);

    // Get prices and timestamps for backtesting
    let start_idx = market_data.len() - tech_features.features.nrows();
    let prices: Vec<f64> = market_data.close[start_idx..].to_vec();
    let timestamps = market_data.timestamp[start_idx..].to_vec();

    let comparison = backtester.compare(
        &teacher_predictions,
        &student_predictions,
        &prices,
        &timestamps,
    );

    println!("{}", comparison.summary());

    // 8. Generate trading signals for current state
    println!("\n8. Current Market Analysis:");
    println!("{}", "-".repeat(60));
    let last_idx = tech_features.features.nrows() - 1;
    let last_features = tech_features.features.row(last_idx);
    let last_prediction = student_predictions[last_idx];
    let signal = TradingSignal::from_probability(last_prediction);

    println!("   Latest feature values:");
    for (name, &value) in tech_features.feature_names.iter().zip(last_features.iter()) {
        println!("     {}: {:.4}", name, value);
    }
    println!("\n   Model prediction: {:.2}%", last_prediction * 100.0);
    println!("   Trading signal: {}", signal);
    println!("   Position multiplier: {:.2}", signal.position_multiplier());

    // 9. Risk analysis
    println!("\n9. Risk Parameters:");
    println!("{}", "-".repeat(60));
    let risk_params = RiskParams::default();
    println!("   Max position size: {:.1}%", risk_params.max_position_size * 100.0);
    println!("   Max positions: {}", risk_params.max_positions);
    println!("   Max loss per trade: {:.1}%", risk_params.max_loss_per_trade * 100.0);
    println!("   Min confidence threshold: {:.1}%", risk_params.min_confidence * 100.0);

    let should_trade = risk_params.validate_trade(last_prediction.abs(), 0);
    println!("\n   Trade validation: {}", if should_trade { "APPROVED" } else { "REJECTED" });

    println!("\n{}", "=".repeat(60));
    println!("Strategy analysis complete!");
}

/// Simulate a complex teacher model's predictions based on technical features
fn simulate_teacher_predictions(features: &TechnicalFeatures) -> Array1<f64> {
    let n_samples = features.features.nrows();
    let mut predictions = Array1::<f64>::zeros(n_samples);

    for i in 0..n_samples {
        let row = features.features.row(i);

        // Feature indices based on TechnicalFeatures::compute order:
        // 0: rsi_14, 1: macd, 2: macd_signal, 3: macd_histogram
        // 4: bb_position_20, 5: atr_14, 6: adx_14, 7: volume_ratio
        // 8: roc_10, 9: volatility_20

        let rsi = row[0];
        let macd_histogram = row[3];
        let bb_position = row[4];
        let adx = row[6];
        let volume_ratio = row[7];

        // Simulate complex model logic
        let mut score = 0.5;

        // RSI signals
        if rsi < 30.0 {
            score += 0.2;  // Oversold - bullish
        } else if rsi > 70.0 {
            score -= 0.2;  // Overbought - bearish
        }

        // MACD signals
        if macd_histogram > 0.0 {
            score += 0.15 * (macd_histogram / 0.5).min(1.0);
        } else {
            score -= 0.15 * (macd_histogram.abs() / 0.5).min(1.0);
        }

        // Bollinger Band position
        if bb_position < 0.2 {
            score += 0.1;  // Near lower band - bullish
        } else if bb_position > 0.8 {
            score -= 0.1;  // Near upper band - bearish
        }

        // ADX for trend strength
        if adx > 25.0 {
            // Strong trend - amplify signal
            let amplifier = 1.0 + (adx - 25.0) / 50.0;
            score = 0.5 + (score - 0.5) * amplifier;
        }

        // Volume confirmation
        if volume_ratio > 1.5 {
            // High volume - more conviction
            let amplifier = 1.0 + (volume_ratio - 1.0) / 2.0;
            score = 0.5 + (score - 0.5) * amplifier;
        }

        predictions[i] = score.max(0.0).min(1.0);
    }

    predictions
}
