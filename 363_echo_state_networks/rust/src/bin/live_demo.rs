//! Live prediction demo (paper trading)
//!
//! Usage: cargo run --bin live_demo -- --symbol BTCUSDT

use anyhow::Result;
use clap::Parser;
use esn_trading::{
    EchoStateNetwork, ESNConfig,
    api::BybitClient,
    trading::{FeatureEngineering, SignalGenerator},
};
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about = "Live ESN prediction demo")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Update interval in seconds
    #[arg(short, long, default_value = "60")]
    interval: u64,

    /// Model file (optional, will train on startup if not provided)
    #[arg(short, long)]
    model: Option<String>,

    /// Number of historical candles for warmup
    #[arg(long, default_value = "500")]
    warmup: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("=== ESN Live Demo ===");
    println!("Symbol: {}", args.symbol);
    println!("Update interval: {}s", args.interval);

    let client = BybitClient::new();

    // Load or train model
    let mut esn = if let Some(model_path) = &args.model {
        println!("\nLoading model from {}...", model_path);
        EchoStateNetwork::load(model_path)?
    } else {
        println!("\nTraining new model on historical data...");

        // Fetch historical data
        let klines = client.get_klines(&args.symbol, "60", args.warmup, None).await?;
        println!("Fetched {} historical klines", klines.len());

        // Feature engineering
        let fe = FeatureEngineering::new()
            .add_returns(10)
            .add_volatility(20)
            .add_rsi(14)
            .add_momentum(5);

        let features = fe.transform(&klines);

        // Prepare targets
        let returns: Vec<ndarray::Array1<f64>> = klines.windows(2)
            .skip(fe.required_lookback())
            .take(features.len())
            .map(|w| ndarray::Array1::from_vec(vec![(w[1].close / w[0].close).ln()]))
            .collect();

        let n = features.len().min(returns.len());

        // Train ESN
        let config = ESNConfig::new(features[0].len(), 1)
            .reservoir_size(300)
            .spectral_radius(0.95)
            .leaking_rate(0.3)
            .washout(50);

        let mut esn = EchoStateNetwork::new(config);
        esn.train(&features[..n], &returns[..n]);
        println!("Model trained!");

        esn
    };

    // Initialize feature engineering and signal generator
    let fe = FeatureEngineering::new()
        .add_returns(10)
        .add_volatility(20)
        .add_rsi(14)
        .add_momentum(5);

    let mut signal_generator = SignalGenerator::new()
        .with_smoothing(3)
        .with_confidence_threshold(0.1);

    // Warmup with recent data
    println!("\nWarming up ESN state...");
    let warmup_klines = client.get_klines(&args.symbol, "60", 100, None).await?;
    let warmup_features = fe.transform(&warmup_klines);

    esn.reset_state();
    for feature in &warmup_features {
        esn.step(feature);
    }
    println!("Warmup complete!");

    // Main loop
    println!("\n=== Starting Live Predictions ===\n");

    loop {
        // Fetch latest data
        let klines = client.get_klines(&args.symbol, "60", fe.required_lookback() + 5, None).await?;

        if klines.is_empty() {
            println!("No data received, retrying...");
            tokio::time::sleep(Duration::from_secs(10)).await;
            continue;
        }

        let features = fe.transform(&klines);

        if let Some(feature) = features.last() {
            // Get prediction
            let prediction = esn.step(feature);
            let (signal, confidence) = signal_generator.generate(&prediction);

            // Get current price
            let current_price = klines.first().map(|k| k.close).unwrap_or(0.0);
            let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S");

            // Display
            println!("[{}] {} @ ${:.2}", timestamp, args.symbol, current_price);
            println!("  Prediction: {:.6}", prediction[0]);
            println!("  Signal: {:?} (confidence: {:.2})", signal, confidence);
            println!("  Implied direction: {}",
                if prediction[0] > 0.0 { "UP" } else if prediction[0] < 0.0 { "DOWN" } else { "FLAT" }
            );
            println!();
        }

        // Wait for next update
        tokio::time::sleep(Duration::from_secs(args.interval)).await;
    }
}
