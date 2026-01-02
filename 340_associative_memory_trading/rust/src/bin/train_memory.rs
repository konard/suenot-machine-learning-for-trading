//! Train associative memory on historical data
//!
//! Usage:
//!   cargo run --bin train_memory -- --input data/btc_hourly.csv --output memory.json

use anyhow::Result;
use associative_memory_trading::{
    data::OHLCVSeries,
    features::{patterns_to_matrix, PatternBuilder, PatternConfig, FeatureSet},
    memory::{DenseAssociativeMemory, PatternMemoryManager},
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "train_memory")]
#[command(about = "Train associative memory on historical data")]
struct Args {
    /// Input CSV file with OHLCV data
    #[arg(short, long)]
    input: String,

    /// Output file for trained memory
    #[arg(short, long, default_value = "memory.json")]
    output: String,

    /// Pattern lookback period
    #[arg(long, default_value = "20")]
    lookback: usize,

    /// Forward period for labels
    #[arg(long, default_value = "5")]
    forward: usize,

    /// Maximum patterns to store
    #[arg(long, default_value = "500")]
    max_patterns: usize,

    /// Beta (temperature) parameter
    #[arg(long, default_value = "1.0")]
    beta: f64,

    /// Feature set (basic, full, technical)
    #[arg(long, default_value = "full")]
    features: String,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    log::info!("Loading data from {}", args.input);
    let data = OHLCVSeries::from_csv(&args.input, "SYMBOL", "interval")?;
    log::info!("Loaded {} candles", data.len());

    // Configure pattern builder
    let feature_set = match args.features.as_str() {
        "basic" => FeatureSet::Basic,
        "technical" => FeatureSet::Technical,
        _ => FeatureSet::Full,
    };

    let config = PatternConfig {
        lookback: args.lookback,
        forward: args.forward,
        normalize: true,
        feature_set,
    };

    let builder = PatternBuilder::with_config(config);
    let pattern_dim = builder.pattern_dim();

    log::info!("Building patterns with {} features", pattern_dim);

    // Build patterns
    let patterns = builder.build_patterns(&data);
    log::info!("Built {} patterns", patterns.len());

    if patterns.is_empty() {
        return Err(anyhow::anyhow!("No patterns could be built from data"));
    }

    // Convert to matrix format
    let (features, labels) = patterns_to_matrix(&patterns);

    // Create and train Dense Associative Memory
    let memory_size = args.max_patterns.min(patterns.len());
    let mut memory = DenseAssociativeMemory::new(memory_size, pattern_dim, args.beta);

    // Use subset of patterns if too many
    if patterns.len() > memory_size {
        log::info!("Sampling {} patterns from {}", memory_size, patterns.len());

        // Take evenly spaced patterns
        let step = patterns.len() / memory_size;
        let mut sampled_features = ndarray::Array2::zeros((memory_size, pattern_dim));
        let mut sampled_labels = ndarray::Array1::zeros(memory_size);

        for i in 0..memory_size {
            let src_idx = i * step;
            for j in 0..pattern_dim {
                sampled_features[[i, j]] = features[[src_idx, j]];
            }
            sampled_labels[i] = labels[src_idx];
        }

        memory.store(&sampled_features, &sampled_labels);
    } else {
        memory.store(&features, &labels);
    }

    // Create pattern memory manager and save
    let mut manager = PatternMemoryManager::new(memory_size, pattern_dim);

    for pattern in patterns.iter().take(memory_size) {
        if let Some(label) = pattern.label {
            manager.add_pattern(
                pattern.features.as_slice().unwrap(),
                label,
                pattern.timestamp,
            );
        }
    }

    manager.save(&args.output)?;
    log::info!("Saved memory to {}", args.output);

    // Print statistics
    let stats = memory.stats();
    println!("\n=== Memory Statistics ===");
    println!("Capacity:      {}", stats.capacity);
    println!("Stored:        {}", stats.current_size);
    println!("Pattern Dim:   {}", stats.pattern_dim);
    println!("Beta:          {:.2}", stats.beta);
    println!("Utilization:   {:.1}%", stats.utilization * 100.0);

    // Test retrieval on last pattern
    if let Some(last_pattern) = patterns.last() {
        let (pred, conf) = memory.predict(&last_pattern.features);
        println!("\n=== Test Retrieval ===");
        println!("Query: last pattern");
        println!("Prediction: {:.4}", pred);
        println!("Confidence: {:.2}%", conf * 100.0);
        println!("Actual label: {:.4}", last_pattern.label.unwrap_or(0.0));

        // Get top similar patterns
        let top_k = memory.retrieve_top_k(&last_pattern.features, 5);
        println!("\nTop 5 similar patterns:");
        for (idx, sim, val) in top_k {
            println!("  Pattern {}: similarity={:.3}, outcome={:.4}", idx, sim, val);
        }
    }

    // Label distribution
    let positive = labels.iter().filter(|&&l| l > 0.0).count();
    let negative = labels.iter().filter(|&&l| l < 0.0).count();
    println!("\n=== Label Distribution ===");
    println!("Positive: {} ({:.1}%)", positive, positive as f64 / patterns.len() as f64 * 100.0);
    println!("Negative: {} ({:.1}%)", negative, negative as f64 / patterns.len() as f64 * 100.0);

    let mean_label: f64 = labels.iter().sum::<f64>() / labels.len() as f64;
    let std_label: f64 = (labels.iter().map(|l| (l - mean_label).powi(2)).sum::<f64>()
        / labels.len() as f64).sqrt();
    println!("Mean:     {:.4}", mean_label);
    println!("Std:      {:.4}", std_label);

    Ok(())
}
