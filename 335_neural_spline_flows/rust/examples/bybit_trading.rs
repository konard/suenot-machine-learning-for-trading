//! Bybit Trading Example
//!
//! This example demonstrates how to use Neural Spline Flows for
//! cryptocurrency trading with real data from Bybit exchange.

use ndarray::Array2;
use neural_spline_flows::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Neural Spline Flow Bybit Trading Example ===\n");

    // Create Bybit client
    println!("1. Connecting to Bybit API...");
    let client = BybitClient::new();

    // Fetch historical data
    println!("\n2. Fetching historical klines for BTCUSDT...");
    let symbol = "BTCUSDT";
    let interval = "60"; // 1 hour
    let limit = 500;

    let candles = match client.get_klines(symbol, interval, limit).await {
        Ok(c) => {
            println!("   Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("   Error fetching data: {}", e);
            println!("   Using synthetic data for demonstration...");
            generate_synthetic_candles(500)
        }
    };

    // Print some candle info
    if let Some(first) = candles.first() {
        println!("   First candle: {} - Close: {:.2}", first.timestamp, first.close);
    }
    if let Some(last) = candles.last() {
        println!("   Last candle: {} - Close: {:.2}", last.timestamp, last.close);
    }

    // Extract features
    println!("\n3. Extracting features...");
    let lookback = 20;
    let features: Vec<FeatureVector> = candles_to_features(&candles, lookback);
    println!("   Extracted {} feature vectors", features.len());

    if let Some(f) = features.last() {
        println!("   Feature names: {:?}", f.names);
        println!("   Latest features: {:?}", f.values);
    }

    // Normalize features
    println!("\n4. Normalizing features...");
    let (normalized, mean, std) = normalize_features(&features);
    println!("   Mean: {:?}", mean.as_slice().unwrap());
    println!("   Std: {:?}", std.as_slice().unwrap());

    // Convert to matrix for training
    let feature_dim = normalized[0].len();
    let n_samples = normalized.len();
    let mut data_matrix = Array2::zeros((n_samples, feature_dim));
    for (i, f) in normalized.iter().enumerate() {
        for (j, v) in f.values.iter().enumerate() {
            data_matrix[[i, j]] = *v;
        }
    }

    // Create and train NSF model
    println!("\n5. Creating Neural Spline Flow model...");
    let config = NSFConfig::new(feature_dim)
        .with_num_layers(4)
        .with_hidden_dim(64)
        .with_num_bins(8);

    let mut model = NeuralSplineFlow::new(config);

    println!("   Training model (this may take a moment)...");
    match model.fit(&data_matrix) {
        Ok(stats) => {
            println!("   Training complete!");
            println!("   Epochs: {}", stats.epochs);
            println!("   Final loss: {:.4}", stats.final_loss);
        }
        Err(e) => {
            println!("   Training error: {}", e);
        }
    }

    // Create signal generator
    println!("\n6. Setting up trading signal generator...");
    let signal_config = SignalGeneratorConfig {
        return_feature_idx: 0, // First feature is 1-period return
        density_threshold: -15.0,
        confidence_threshold: 0.3,
        z_threshold: 0.5,
        num_samples: 1000,
    };

    let signal_generator = SignalGenerator::new(model.clone(), signal_config);

    // Generate signal for current market state
    println!("\n7. Generating trading signal for current market state...");
    if let Some(current_features) = normalized.last() {
        let signal = signal_generator.generate_signal(&current_features.to_array());

        println!("   Signal: {:.4}", signal.signal);
        println!("   Confidence: {:.4}", signal.confidence);
        println!("   Log Probability: {:.4}", signal.log_prob);
        println!("   In Distribution: {}", signal.in_distribution);
        println!("   Reason: {}", signal.reason);

        if let Some(exp_ret) = signal.expected_return {
            println!("   Expected Return: {:.4}", exp_ret);
        }
        if let Some(lat_ret) = signal.latent_return {
            println!("   Latent Return: {:.4}", lat_ret);
        }

        // Trading decision
        println!("\n   Trading Decision:");
        if signal.is_buy() {
            println!("   -> LONG with position size {:.2}", signal.position_size);
        } else if signal.is_sell() {
            println!("   -> SHORT with position size {:.2}", signal.position_size.abs());
        } else {
            println!("   -> NO TRADE ({})", signal.reason);
        }
    }

    // Risk analysis
    println!("\n8. Risk Analysis...");
    let risk_config = RiskManagerConfig {
        return_feature_idx: 0,
        confidence_level: 0.95,
        num_samples: 10000,
        max_position: 1.0,
        risk_scale: 1.0,
    };

    let risk_manager = RiskManager::new(model, risk_config);
    let metrics = risk_manager.compute_risk_metrics();

    println!("   VaR (95%): {:.4}", metrics.var);
    println!("   CVaR (95%): {:.4}", metrics.cvar);
    println!("   Expected Return: {:.4}", metrics.expected_return);
    println!("   Return Std: {:.4}", metrics.return_std);
    println!("   Skewness: {:.4}", metrics.skewness);
    println!("   Kurtosis: {:.4}", metrics.kurtosis);

    // Probability of positive return
    let prob_positive = risk_manager.prob_positive_return();
    println!("   P(return > 0): {:.2}%", prob_positive * 100.0);

    // Get current ticker
    println!("\n9. Current Market Data...");
    match client.get_ticker(symbol).await {
        Ok(ticker) => {
            println!("   Symbol: {}", ticker.symbol);
            println!("   Last Price: {:.2}", ticker.last_price);
            println!("   24h Change: {:.2}%", ticker.price_change_24h * 100.0);
            println!("   24h High: {:.2}", ticker.high_24h);
            println!("   24h Low: {:.2}", ticker.low_24h);
            println!("   24h Volume: {:.2}", ticker.volume_24h);
        }
        Err(e) => {
            println!("   Could not fetch ticker: {}", e);
        }
    }

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Generate synthetic candles for demonstration when API is unavailable
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use chrono::Utc;
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let now = Utc::now();
    let mut price = 50000.0; // Starting BTC price

    (0..n)
        .map(|i| {
            // Random walk
            let change = rng.gen_range(-0.02..0.02);
            price *= 1.0 + change;

            let high = price * (1.0 + rng.gen_range(0.0..0.01));
            let low = price * (1.0 - rng.gen_range(0.0..0.01));
            let open = price * (1.0 + rng.gen_range(-0.005..0.005));

            Candle {
                timestamp: now - chrono::Duration::hours((n - i) as i64),
                open,
                high,
                low,
                close: price,
                volume: rng.gen_range(100.0..1000.0),
            }
        })
        .collect()
}

fn candles_to_features(candles: &[Candle], lookback: usize) -> Vec<FeatureVector> {
    let mut features = Vec::new();

    for i in lookback..candles.len() {
        let window = &candles[(i - lookback)..i];
        features.push(extract_features(window));
    }

    features
}
