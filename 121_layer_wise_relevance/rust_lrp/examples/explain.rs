//! Example: Generate LRP explanations

use clap::Parser;
use ndarray::Array1;
use rust_lrp::{
    model::{LRPNetwork, LRPConfig},
    data::{prepare_data, generate_sample_data, FEATURE_NAMES},
    strategy::analyze_signal,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of samples to explain
    #[arg(short, long, default_value = "5")]
    num_samples: usize,

    /// Top N features to show
    #[arg(short, long, default_value = "5")]
    top_features: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("LRP Explanation Generator");
    println!("=========================\n");

    // Create model and data
    let config = LRPConfig::default()
        .with_epsilon(0.01)
        .with_gamma(0.25);

    let features = generate_sample_data(500);
    let dataset = prepare_data(&features, 60, 1);

    let mut model = LRPNetwork::new(
        dataset.input_dim(),
        vec![128, 64, 32],
        2,
        config,
    );

    let feature_names: Vec<String> = FEATURE_NAMES.iter().map(|s| s.to_string()).collect();

    println!("Analyzing {} samples...\n", args.num_samples);

    for i in 0..args.num_samples.min(dataset.len()) {
        let sample = &dataset.samples[i];
        let input = sample.x.view()
            .into_shape(dataset.input_dim())
            .unwrap()
            .to_owned();

        let analysis = analyze_signal(&mut model, &input, &feature_names);

        println!("Sample {}:", i + 1);
        println!("  Signal: {:?}", analysis.signal);
        println!("  Confidence: {:.2}%", analysis.confidence * 100.0);
        println!("  Coherence: {:.4}", analysis.coherence);
        println!("  Top {} contributing features:", args.top_features);

        for (name, rel) in analysis.relevances.iter().take(args.top_features) {
            let direction = if *rel > 0.0 { "+" } else { "-" };
            println!("    {} {}: {:.1}%", direction, name, rel.abs());
        }

        // Verify conservation
        let (conserved, error) = model.verify_conservation(&input, 0.1);
        println!("  Conservation: {} (error: {:.4})", conserved, error);

        println!();
    }

    // Aggregate feature importance
    println!("Aggregate Feature Importance:");
    println!("------------------------------");

    let mut importance = vec![0.0; feature_names.len()];
    let mut count = 0;

    for sample in dataset.samples.iter().take(100) {
        let input = sample.x.view()
            .into_shape(dataset.input_dim())
            .unwrap()
            .to_owned();

        let relevance = model.explain(&input, None);

        let n_features = feature_names.len();
        let n_timesteps = relevance.len() / n_features;

        for t in 0..n_timesteps {
            for f in 0..n_features {
                let idx = t * n_features + f;
                if idx < relevance.len() {
                    importance[f] += relevance[idx].abs();
                }
            }
        }
        count += 1;
    }

    // Normalize
    let total: f64 = importance.iter().sum();
    let mut pairs: Vec<(String, f64)> = feature_names
        .iter()
        .zip(importance.iter())
        .map(|(name, &imp)| (name.clone(), imp / total * 100.0))
        .collect();

    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (name, imp) in pairs.iter().take(10) {
        println!("  {}: {:.1}%", name, imp);
    }

    Ok(())
}
