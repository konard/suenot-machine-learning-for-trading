//! Market Regime Prediction Example
//!
//! This example demonstrates using few-shot learning to classify
//! market regimes and generate trading signals.
//!
//! Run with: cargo run --example regime_prediction

use few_shot_market_prediction::strategy::{
    MarketRegime, RegimeClassifier, RegimeClassification,
    SignalGenerator, SignalConfig, SignalType,
    RiskManager, RiskConfig,
};
use std::time::Duration;

fn main() {
    println!("=== Market Regime Prediction with Few-Shot Learning ===\n");

    // Define all market regimes we want to classify
    let regimes = MarketRegime::all();
    println!("Classifying into {} market regimes:", regimes.len());
    for regime in &regimes {
        println!("  - {} (position bias: {:.1})", regime, regime.position_bias());
    }
    println!();

    // Create regime classifier
    let mut classifier = RegimeClassifier::new(0.6);
    println!("Regime classifier created with 60% confidence threshold\n");

    // Simulate some regime predictions
    // In real usage, these probabilities would come from the few-shot model
    let test_scenarios = [
        ("Strong momentum rally", vec![0.75, 0.15, 0.03, 0.02, 0.01, 0.02, 0.01, 0.01]),
        ("Mild uptrend", vec![0.15, 0.55, 0.20, 0.05, 0.02, 0.02, 0.00, 0.01]),
        ("Choppy sideways", vec![0.05, 0.10, 0.60, 0.10, 0.05, 0.08, 0.01, 0.01]),
        ("Bearish breakdown", vec![0.02, 0.03, 0.05, 0.15, 0.60, 0.10, 0.04, 0.01]),
        ("High volatility", vec![0.10, 0.10, 0.10, 0.10, 0.10, 0.40, 0.05, 0.05]),
        ("Market crash", vec![0.01, 0.02, 0.02, 0.10, 0.15, 0.15, 0.50, 0.05]),
        ("Recovery phase", vec![0.15, 0.15, 0.10, 0.05, 0.05, 0.10, 0.05, 0.35]),
    ];

    println!("--- Regime Classification Results ---\n");

    // Create signal generator
    let signal_config = SignalConfig {
        min_confidence: 0.5,
        max_position_size: 0.2,
        default_take_profit: 3.0,
        default_stop_loss: 1.5,
        use_regime_adjustment: true,
        cooldown: Duration::from_secs(0), // No cooldown for demo
    };
    let mut signal_generator = SignalGenerator::new(signal_config);

    // Create risk manager
    let risk_config = RiskConfig::default();
    let risk_manager = RiskManager::new(risk_config, 100_000.0);

    for (scenario_name, probs) in &test_scenarios {
        println!("Scenario: {}", scenario_name);

        // Classify the regime
        let classification = classifier.classify_with_uncertainty(probs);

        println!("  Classification: {}", classification.description());
        println!("  Should act: {}", classification.should_act());

        // Update classifier history for transition analysis
        classifier.update_history(classification.regime, classification.confidence);

        // Generate trading signal
        signal_generator.clear_cooldowns(); // Reset for demo
        let signal = signal_generator.generate("BTCUSDT", &classification);

        println!("  Signal: {:?}", signal.signal_type);
        if signal.signal_type.requires_action() {
            println!("    Position size: {:.1}%", signal.position_size * 100.0);
            if let Some(tp) = signal.take_profit {
                println!("    Take profit: {:.1}%", tp);
            }
            if let Some(sl) = signal.stop_loss {
                println!("    Stop loss: {:.1}%", sl);
            }
            println!("    Expected value: {:.2}", signal.expected_value());

            // Check risk
            let risk_check = risk_manager.check_signal(&signal, 0);
            println!("    Risk approved: {}", risk_check.approved);
            if !risk_check.rejections.is_empty() {
                println!("    Rejections: {:?}", risk_check.rejections);
            }
        }
        println!();
    }

    // Demonstrate regime transitions
    println!("--- Regime Transition Analysis ---\n");

    // Simulate a sequence of regimes
    let regime_sequence = [
        MarketRegime::Sideways,
        MarketRegime::WeakUptrend,
        MarketRegime::StrongUptrend,
        MarketRegime::Volatile,
        MarketRegime::WeakDowntrend,
        MarketRegime::Crash,
        MarketRegime::Recovery,
        MarketRegime::Sideways,
    ];

    for regime in &regime_sequence {
        classifier.update_history(*regime, 0.8);
    }

    println!("After observing {} regime changes:", regime_sequence.len());
    if let Some(dominant) = classifier.dominant_regime(5) {
        println!("  Dominant regime (last 5): {}", dominant);
    }
    println!();

    // Check transition probabilities
    let transitions = [
        (MarketRegime::StrongUptrend, MarketRegime::Volatile),
        (MarketRegime::Volatile, MarketRegime::Crash),
        (MarketRegime::Crash, MarketRegime::Recovery),
        (MarketRegime::Recovery, MarketRegime::Sideways),
    ];

    println!("Transition probabilities from history:");
    for (from, to) in &transitions {
        let prob = classifier.transition_probability(*from, *to);
        let likely = classifier.is_transition_likely(*from, *to);
        println!(
            "  {} -> {}: {:.1}% (likely: {})",
            from, to, prob * 100.0, likely
        );
    }

    // Show risk summary
    println!("\n--- Current Risk Summary ---");
    println!("{}", risk_manager.risk_summary());

    // Demonstrate different risk profiles
    println!("\n--- Risk Profile Comparison ---");
    let profiles = [
        ("Default", RiskConfig::default()),
        ("Conservative", RiskConfig::conservative()),
        ("Aggressive", RiskConfig::aggressive()),
    ];

    for (name, config) in &profiles {
        println!("  {}: max_position={:.0}%, max_exposure={:.0}%, daily_limit={:.0}%",
            name,
            config.max_position_size * 100.0,
            config.max_total_exposure * 100.0,
            config.daily_loss_limit * 100.0
        );
    }

    println!("\n=== Example Complete ===");
}
