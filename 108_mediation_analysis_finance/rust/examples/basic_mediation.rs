//! Basic mediation analysis example
//!
//! This example demonstrates how to use the mediation analysis library
//! for financial trading signals.

use mediation_analysis::{MediationAnalyzer, MediationType};
use rand::Rng;

fn main() {
    println!("=== Mediation Analysis Example ===\n");

    // Create analyzer with 60-period lookback
    let mut analyzer = MediationAnalyzer::new(60);

    // Simulate financial data with known mediation structure
    // X = Market returns (treatment)
    // M = Trading volume (mediator)
    // Y = Stock returns (outcome)

    let a_true = 0.5;      // X → M path
    let b_true = 0.4;      // M → Y path
    let c_prime_true = 0.3; // Direct effect

    println!("Simulating data with:");
    println!("  a path (X→M): {:.2}", a_true);
    println!("  b path (M→Y): {:.2}", b_true);
    println!("  Direct effect (c'): {:.2}", c_prime_true);
    println!("  True indirect effect (a×b): {:.2}", a_true * b_true);
    println!("  True total effect: {:.2}", c_prime_true + a_true * b_true);
    println!();

    let mut rng = rand::thread_rng();

    // Generate 100 observations
    for i in 0..100 {
        // Simulate market return (treatment)
        let x = rng.gen_range(-0.02..0.02);

        // Simulate trading volume (mediator) - affected by market return
        let m = a_true * x + rng.gen_range(-0.01..0.01);

        // Simulate stock return (outcome) - affected by both X and M
        let y = c_prime_true * x + b_true * m + rng.gen_range(-0.015..0.015);

        analyzer.update(x, m, y);
    }

    // Run analysis
    println!("Running mediation analysis...\n");

    if let Some(results) = analyzer.analyze() {
        println!("=== Results ===\n");

        println!("Path Coefficients:");
        println!("  a (X → M):   {:.4} (SE: {:.4})", results.a_path, results.a_path_se);
        println!("  b (M → Y|X): {:.4} (SE: {:.4})", results.b_path, results.b_path_se);
        println!();

        println!("Effect Decomposition:");
        println!("  Total Effect:    {:.4} (SE: {:.4})", results.total_effect, results.total_effect_se);
        println!("  Direct Effect:   {:.4} (SE: {:.4})", results.direct_effect, results.direct_effect_se);
        println!("  Indirect Effect: {:.4} (SE: {:.4})", results.indirect_effect, results.indirect_effect_se);
        println!();

        println!("Sobel Test:");
        println!("  Z-statistic: {:.4}", results.sobel_z);
        println!("  P-value:     {:.4}", results.sobel_pvalue);
        println!();

        println!("Proportion Mediated: {:.1}%", results.proportion_mediated * 100.0);
        println!();

        // Determine mediation type
        let med_type = results.mediation_type(0.05);
        let type_str = match med_type {
            MediationType::Full => "Full Mediation",
            MediationType::Partial => "Partial Mediation",
            MediationType::DirectOnly => "Direct Effect Only",
            MediationType::None => "No Mediation",
        };
        println!("Mediation Type: {}", type_str);

        // Generate trading signal
        println!("\n=== Trading Signal Generation ===\n");

        // Current observation
        let current_x = 0.01;  // Positive market return
        let current_m = 0.008; // High volume

        if let Some(signal) = analyzer.generate_signal(current_x, current_m, 0.3, 0.1) {
            let direction = match signal.signal {
                1 => "LONG",
                -1 => "SHORT",
                _ => "NEUTRAL",
            };

            println!("Current X (market return): {:.4}", current_x);
            println!("Current M (volume): {:.4}", current_m);
            println!();
            println!("Signal: {}", direction);
            println!("Confidence: {:.2}%", signal.confidence * 100.0);
            println!("Mediator Activated: {}", signal.mediator_activated);
            println!("Expected Effect: {:.4}", signal.expected_effect);
        }
    } else {
        println!("Analysis failed - not enough data");
    }
}
