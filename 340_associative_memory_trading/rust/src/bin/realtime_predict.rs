//! Real-time prediction using associative memory
//!
//! Usage:
//!   cargo run --bin realtime_predict -- --symbol BTCUSDT --memory memory.json

use anyhow::Result;
use associative_memory_trading::{
    data::{BybitClient, intervals},
    features::{PatternBuilder, PatternConfig, FeatureSet},
    memory::PatternMemoryManager,
    strategy::{SignalConfig, SignalGenerator},
};
use chrono::Utc;
use clap::Parser;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "realtime_predict")]
#[command(about = "Real-time prediction using associative memory")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Memory file to load
    #[arg(short, long)]
    memory: Option<String>,

    /// Pattern lookback period
    #[arg(long, default_value = "20")]
    lookback: usize,

    /// Minimum confidence threshold
    #[arg(long, default_value = "0.3")]
    min_confidence: f64,

    /// Update interval in seconds
    #[arg(long, default_value = "60")]
    interval: u64,

    /// Run once and exit
    #[arg(long, default_value = "false")]
    once: bool,

    /// Candle interval
    #[arg(long, default_value = "60")]
    candle_interval: String,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("=== Associative Memory Real-time Predictor ===\n");
    println!("Symbol:     {}", args.symbol);
    println!("Lookback:   {} candles", args.lookback);
    println!("Interval:   {} minutes", args.candle_interval);
    println!("Confidence: {:.0}% minimum", args.min_confidence * 100.0);

    // Initialize components
    let client = BybitClient::public();

    let config = PatternConfig {
        lookback: args.lookback,
        forward: 5,
        normalize: true,
        feature_set: FeatureSet::Full,
    };
    let pattern_builder = PatternBuilder::with_config(config);
    let pattern_dim = pattern_builder.pattern_dim();

    let signal_config = SignalConfig {
        min_confidence: args.min_confidence,
        ..Default::default()
    };
    let signal_generator = SignalGenerator::new(signal_config.clone());

    // Load or create memory
    let mut manager = PatternMemoryManager::new(500, pattern_dim);

    if let Some(memory_path) = &args.memory {
        match manager.load(memory_path) {
            Ok(_) => {
                log::info!("Loaded memory from {}", memory_path);
                let stats = manager.stats();
                println!("Memory:     {} patterns loaded\n", stats.n_patterns);
            }
            Err(e) => {
                log::warn!("Could not load memory: {}. Starting fresh.", e);
                println!("Memory:     Starting with empty memory\n");
            }
        }
    } else {
        println!("Memory:     No memory file specified, building from recent data\n");

        // Fetch historical data to build initial memory
        log::info!("Fetching historical data to build memory...");
        let end_time = Utc::now();
        let start_time = end_time - chrono::Duration::days(30);

        let historical = client.get_historical_klines(
            &args.symbol,
            &args.candle_interval,
            start_time,
            end_time,
        )?;

        let patterns = pattern_builder.build_patterns(&historical);
        log::info!("Built {} patterns from historical data", patterns.len());

        for pattern in &patterns {
            if let Some(label) = pattern.label {
                manager.add_pattern(
                    pattern.features.as_slice().unwrap(),
                    label,
                    pattern.timestamp,
                );
            }
        }

        let stats = manager.stats();
        println!("Memory:     Built {} patterns from 30 days of data\n", stats.n_patterns);
    }

    // Main prediction loop
    loop {
        match predict_once(&client, &args, &pattern_builder, &mut manager, &signal_generator) {
            Ok(_) => {}
            Err(e) => {
                log::error!("Prediction error: {}", e);
            }
        }

        if args.once {
            break;
        }

        println!("\nNext update in {} seconds...", args.interval);
        std::thread::sleep(Duration::from_secs(args.interval));
        println!("\n{}", "=".repeat(50));
    }

    Ok(())
}

fn predict_once(
    client: &BybitClient,
    args: &Args,
    pattern_builder: &PatternBuilder,
    manager: &mut PatternMemoryManager,
    signal_generator: &SignalGenerator,
) -> Result<()> {
    let now = Utc::now();
    println!("\n[{}]", now.format("%Y-%m-%d %H:%M:%S UTC"));

    // Fetch current data
    let data = client.get_klines(
        &args.symbol,
        &args.candle_interval,
        args.lookback + 10,
        None,
        None,
    )?;

    if data.len() < args.lookback {
        return Err(anyhow::anyhow!("Insufficient data"));
    }

    // Get current price
    let current_price = data.data.last().unwrap().close;
    let prev_price = data.data[data.len() - 2].close;
    let price_change = (current_price - prev_price) / prev_price * 100.0;

    println!("\nCurrent Price: ${:.2} ({:+.2}%)", current_price, price_change);

    // Build current pattern
    let pattern = match pattern_builder.build_current_pattern(&data) {
        Some(p) => p,
        None => return Err(anyhow::anyhow!("Could not build pattern")),
    };

    // Get prediction
    let signal = signal_generator.generate_from_manager(manager, pattern.features.as_slice().unwrap());

    // Display prediction
    println!("\n--- Prediction ---");
    println!("Direction:   {}", match signal.direction {
        d if d > 0.0 => "LONG ↑",
        d if d < 0.0 => "SHORT ↓",
        _ => "NEUTRAL ━"
    });
    println!("Confidence:  {:.1}%", signal.confidence * 100.0);
    println!("Actionable:  {}", if signal.is_actionable { "YES" } else { "NO" });

    if signal.is_actionable {
        println!("Position:    {:.0}%", signal.position_size * 100.0);
    }

    // Get similar patterns
    let (patterns, outcomes, similarities) = manager.query(
        pattern.features.as_slice().unwrap(),
        5,
    );

    if !patterns.is_empty() {
        println!("\n--- Similar Historical Patterns ---");
        for (i, (_, outcome, sim)) in patterns.iter().zip(outcomes.iter()).zip(similarities.iter()).enumerate() {
            let (_, outcome) = (patterns[i].clone(), outcome);
            let direction = if *outcome > 0.0 { "↑" } else if *outcome < 0.0 { "↓" } else { "━" };
            println!("  #{}: similarity={:.1}%, outcome={:+.2}% {}",
                i + 1, sim * 100.0, outcome * 100.0, direction);
        }

        // Consensus
        let positive = outcomes.iter().filter(|&&o| o > 0.0).count();
        let consensus = if outcomes.len() > 0 {
            positive as f64 / outcomes.len() as f64 * 100.0
        } else {
            50.0
        };
        println!("\nConsensus: {:.0}% bullish", consensus);
    }

    // Memory stats
    let stats = manager.stats();
    println!("\n--- Memory Stats ---");
    println!("Patterns:    {}/{}", stats.n_patterns, stats.capacity);
    println!("Retrievals:  {}", stats.total_retrievals);

    Ok(())
}
