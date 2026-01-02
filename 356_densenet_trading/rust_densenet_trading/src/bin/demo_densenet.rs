//! DenseNet Architecture Demo
//!
//! Demonstrates the DenseNet architecture for trading.
//!
//! Usage:
//!   cargo run --bin demo_densenet

use ndarray::Array2;
use rust_densenet_trading::densenet::{DenseNet, DenseNetConfig};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           DenseNet Trading Architecture Demo                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Demo different DenseNet configurations
    let configs = vec![
        ("Tiny (HFT)", DenseNetConfig::tiny()),
        ("Small (Intraday)", DenseNetConfig::small()),
        ("Medium (Swing)", DenseNetConfig::medium()),
        ("Large (Research)", DenseNetConfig::large()),
    ];

    println!("DenseNet Variants:");
    println!("{:-<70}", "");
    println!(
        "{:<20} {:>15} {:>15} {:>15}",
        "Variant", "Blocks", "Growth Rate", "Parameters"
    );
    println!("{:-<70}", "");

    for (name, config) in &configs {
        let blocks: String = config
            .block_config
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join("-");
        let params = config.estimate_parameters();

        println!(
            "{:<20} {:>15} {:>15} {:>15}",
            name,
            blocks,
            config.growth_rate,
            format_params(params)
        );
    }

    println!();
    println!("{:=<70}", "");
    println!();

    // Create and demonstrate the small model
    println!("Creating Small DenseNet model for demonstration...");
    println!();

    let config = DenseNetConfig::small();
    let model = DenseNet::new(config.clone());

    println!("{}", model.summary());

    // Create sample input
    println!("Running forward pass with sample data...");
    println!();

    let input = Array2::from_shape_fn((config.input_features, config.sequence_length), |(i, j)| {
        // Create some synthetic price-like features
        let base = 100.0;
        let trend = j as f64 * 0.01;
        let noise = ((i * j) as f64 * 0.1).sin() * 0.5;
        base + trend + noise
    });

    println!(
        "Input shape: [{}, {}]",
        input.dim().0,
        input.dim().1
    );

    // Run prediction
    let prediction = model.forward(&input, false);

    println!();
    println!("Prediction Results:");
    println!("{:-<50}", "");
    println!("  Short probability:  {:.4}", prediction.probabilities[0]);
    println!("  Hold probability:   {:.4}", prediction.probabilities[1]);
    println!("  Long probability:   {:.4}", prediction.probabilities[2]);
    println!("  Predicted action:   {}", prediction.signal());
    println!("  Confidence:         {:.2}%", prediction.confidence * 100.0);
    println!();

    // Demonstrate feature extraction importance
    println!("DenseNet Key Concepts:");
    println!("{:-<70}", "");
    println!();

    println!("1. DENSE CONNECTIVITY:");
    println!("   Each layer receives feature maps from ALL previous layers.");
    println!("   This enables:");
    println!("   • Feature reuse: Early patterns remain accessible");
    println!("   • Better gradient flow: No vanishing gradients");
    println!("   • Parameter efficiency: Fewer parameters needed");
    println!();

    println!("2. GROWTH RATE (k={}):", config.growth_rate);
    println!("   Each layer adds {} new feature maps.", config.growth_rate);
    println!(
        "   After {} layers, we accumulate many rich features.",
        config.block_config.iter().sum::<usize>()
    );
    println!();

    println!("3. TRANSITION LAYERS:");
    println!(
        "   Compression factor θ = {} reduces channel count",
        config.compression
    );
    println!("   Average pooling reduces spatial dimensions");
    println!("   This keeps the model computationally efficient");
    println!();

    println!("4. BOTTLENECK LAYERS:");
    if config.use_bottleneck {
        println!("   1x1 convolution before 3x1 reduces computation");
        println!(
            "   Bottleneck factor: {} (reduces to {}k channels)",
            config.bottleneck_factor,
            config.bottleneck_factor * config.growth_rate
        );
    } else {
        println!("   Disabled in this configuration");
    }
    println!();

    // Trading application specifics
    println!("{:=<70}", "");
    println!();
    println!("TRADING APPLICATION:");
    println!();

    println!("Input Features ({} total):", config.input_features);
    println!("  • Price features: returns, range, body");
    println!("  • Volume features: volume ratio, OBV");
    println!("  • Momentum: RSI, MACD, ROC");
    println!("  • Volatility: ATR, Bollinger position");
    println!("  • Trend: SMA/EMA ratios at multiple periods");
    println!();

    println!("Sequence Length: {} candles", config.sequence_length);
    println!("  For 15-minute candles: {:.1} hours of history", config.sequence_length as f64 * 15.0 / 60.0);
    println!("  For 1-hour candles: {:.1} days of history", config.sequence_length as f64 / 24.0);
    println!();

    println!("Output:");
    println!("  • 3 classes: Short (-1), Hold (0), Long (+1)");
    println!("  • Confidence score for position sizing");
    println!("  • Can add volatility prediction for risk management");
    println!();

    println!("Done!");
}

fn format_params(params: usize) -> String {
    if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else {
        params.to_string()
    }
}
