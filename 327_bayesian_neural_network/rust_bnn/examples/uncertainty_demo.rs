//! Example: Demonstrate uncertainty quantification with BNN.
//!
//! This example shows how Bayesian Neural Networks provide
//! uncertainty estimates alongside predictions.

use bayesian_nn_trading::bnn::{BNNConfig, BayesianNN};
use bayesian_nn_trading::bnn::uncertainty::UncertaintyEstimate;
use colored::Colorize;
use ndarray::Array2;
use rand::Rng;

fn main() {
    println!("{}", "=".repeat(60).blue());
    println!(
        "{}",
        "Bayesian Neural Network - Uncertainty Demo".blue().bold()
    );
    println!("{}", "=".repeat(60).blue());

    // Create a simple BNN
    let config = BNNConfig::with_input_dim(10)
        .hidden_dims(vec![32, 16])
        .num_classes(3)
        .n_samples_inference(50);

    let mut model = BayesianNN::new(config);

    println!(
        "\n{} Created BNN with {} parameters",
        "[OK]".green(),
        model.num_parameters()
    );

    // Generate sample inputs
    let mut rng = rand::thread_rng();

    // Case 1: Input similar to training distribution (low uncertainty expected)
    println!("\n{}", "Case 1: Typical Input".yellow().bold());
    let typical_input = Array2::from_shape_fn((1, 10), |_| rng.gen_range(-1.0..1.0));

    let estimate1 = model.predict_with_uncertainty(&typical_input);
    print_uncertainty_analysis(&estimate1, "Typical");

    // Case 2: Input with extreme values (higher uncertainty expected)
    println!("\n{}", "Case 2: Extreme Input".yellow().bold());
    let extreme_input = Array2::from_shape_fn((1, 10), |_| rng.gen_range(-10.0..10.0));

    let estimate2 = model.predict_with_uncertainty(&extreme_input);
    print_uncertainty_analysis(&estimate2, "Extreme");

    // Case 3: Zero input (edge case)
    println!("\n{}", "Case 3: Zero Input".yellow().bold());
    let zero_input = Array2::zeros((1, 10));

    let estimate3 = model.predict_with_uncertainty(&zero_input);
    print_uncertainty_analysis(&estimate3, "Zero");

    // Demonstrate trading decision based on uncertainty
    println!("\n{}", "Trading Decision Examples:".blue().bold());

    demonstrate_trading_decision(&estimate1, "Typical Input");
    demonstrate_trading_decision(&estimate2, "Extreme Input");
    demonstrate_trading_decision(&estimate3, "Zero Input");

    // Show Monte Carlo sampling
    println!("\n{}", "Monte Carlo Sampling Visualization:".blue().bold());
    demonstrate_mc_sampling(&mut model, &typical_input);

    println!("\n{}", "Demo Complete!".green().bold());
}

fn print_uncertainty_analysis(estimate: &UncertaintyEstimate, label: &str) {
    println!("  {} Analysis:", label.cyan());
    println!("    Predicted Class: {}", estimate.predicted_class());
    println!(
        "    Probability:     {:.2}%",
        estimate.predicted_probability() * 100.0
    );
    println!("    Confidence:      {:.4}", estimate.mean_confidence());
    println!(
        "    Epistemic Unc:   {:.6}",
        estimate.mean_epistemic()
    );

    // Interpret uncertainty level
    let uncertainty = estimate.mean_epistemic();
    let level = if uncertainty < 0.01 {
        "LOW".green()
    } else if uncertainty < 0.05 {
        "MEDIUM".yellow()
    } else {
        "HIGH".red()
    };
    println!("    Uncertainty Level: {}", level);
}

fn demonstrate_trading_decision(estimate: &UncertaintyEstimate, input_type: &str) {
    let confidence = estimate.mean_confidence();
    let uncertainty = estimate.mean_epistemic();
    let prob = estimate.predicted_probability();

    let confidence_threshold = 0.3;
    let prob_threshold = 0.55;

    println!("\n  {} {}:", "Input:".cyan(), input_type);

    let decision = if confidence < confidence_threshold {
        format!("NO TRADE - Low confidence ({:.2})", confidence).red()
    } else if prob < prob_threshold {
        format!("NO TRADE - Low probability ({:.2})", prob).yellow()
    } else {
        // Calculate position size using Kelly-like formula
        let base_size = 0.25;
        let uncertainty_penalty = uncertainty.sqrt();
        let position_size = base_size * confidence * (1.0 - uncertainty_penalty);

        format!(
            "TRADE - Position size: {:.1}% of portfolio",
            position_size * 100.0
        )
        .green()
    };

    println!("    Decision: {}", decision);
}

fn demonstrate_mc_sampling(model: &mut BayesianNN, input: &Array2<f64>) {
    println!("  Running 10 Monte Carlo samples:");

    for i in 0..10 {
        let output = model.forward(input, true);

        // Apply softmax manually
        let row = output.row(0);
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
        let probs: Vec<f64> = row.iter().map(|&v| (v - max_val).exp() / exp_sum).collect();

        let predicted_class = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let class_names = ["UP", "NEUTRAL", "DOWN"];
        println!(
            "    Sample {}: {} ({:.1}%), {} ({:.1}%), {} ({:.1}%) -> {}",
            i + 1,
            class_names[0],
            probs[0] * 100.0,
            class_names[1],
            probs[1] * 100.0,
            class_names[2],
            probs[2] * 100.0,
            class_names[predicted_class].yellow()
        );
    }

    println!("\n  Note: Variance across samples = Epistemic Uncertainty");
}
