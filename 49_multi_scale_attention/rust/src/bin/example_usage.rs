//! Complete example usage of Multi-Scale Attention for financial time series.
//!
//! This example demonstrates:
//! 1. Fetching real data from Bybit API
//! 2. Creating multi-scale features
//! 3. Building and running the model
//! 4. Backtesting a trading strategy
//! 5. Comparing different scale strategies

use multi_scale_attention::{
    fetch_bybit_klines,
    create_multi_scale_features,
    create_sequences,
    generate_synthetic_data,
    MultiScaleAttention,
    MultiScaleAttentionConfig,
    ScaleConfig,
    backtest_multi_scale_strategy,
    BacktestConfig,
};
use std::collections::HashMap;
use std::env;

fn run_with_bybit_data(symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("   MULTI-SCALE ATTENTION - BYBIT {} EXAMPLE", symbol);
    println!("================================================================================");

    // Configuration
    let scales = ["1min", "5min", "1H"];
    let lookbacks: HashMap<String, usize> = [
        ("1min".to_string(), 60),
        ("5min".to_string(), 24),
        ("1H".to_string(), 12),
    ]
    .into_iter()
    .collect();

    // Step 1: Fetch data from Bybit
    println!("\n1. Fetching {} data from Bybit...", symbol);
    let klines = match fetch_bybit_klines(symbol, "1", 1000) {
        Ok(k) => k,
        Err(e) => {
            println!("   Error fetching data: {}", e);
            println!("   Falling back to synthetic example...");
            return run_with_synthetic_data();
        }
    };

    if klines.len() < 500 {
        println!("   Not enough data ({}), falling back to synthetic...", klines.len());
        return run_with_synthetic_data();
    }

    println!("   Fetched {} klines", klines.len());
    println!("   Price range: {:.2} - {:.2}",
        klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max)
    );

    // Step 2: Create multi-scale features
    println!("\n2. Creating multi-scale features...");
    let multi_scale_data = create_multi_scale_features(&klines, &scales);

    for (scale, features) in &multi_scale_data.features {
        println!("   Scale {}: {} samples, {} features", scale, features.nrows(), features.ncols());
    }

    // Step 3: Create sequences
    println!("\n3. Creating sequences...");
    let (sequences, targets, timestamps) = create_sequences(&multi_scale_data, &lookbacks, 1);

    if sequences.is_empty() || targets.is_empty() {
        println!("   Not enough data to create sequences");
        return run_with_synthetic_data();
    }

    println!("   Created {} sequences", targets.len());
    for (scale, seq) in &sequences {
        println!("   Scale {}: {:?}", scale, seq.dim());
    }

    // Step 4: Create model
    println!("\n4. Creating multi-scale attention model...");
    let config = MultiScaleAttentionConfig {
        scale_configs: scales
            .iter()
            .map(|&s| {
                let n_features = sequences.get(s).map(|seq| seq.dim().2).unwrap_or(15);
                let seq_len = lookbacks.get(s).copied().unwrap_or(10);
                ScaleConfig {
                    name: s.to_string(),
                    input_dim: n_features,
                    seq_len,
                }
            })
            .collect(),
        d_model: 64,
        n_heads: 4,
        n_encoder_layers: 2,
        dropout: 0.1,
        output_dim: 1,
    };

    let model = MultiScaleAttention::new(config);
    println!("   Model created successfully");

    // Step 5: Make predictions
    println!("\n5. Making predictions...");
    let predictions = model.forward(&sequences);

    println!("   Direction predictions: {}", predictions.direction.nrows());
    println!("   Sample predictions:");
    for i in 0..5.min(predictions.direction.nrows()) {
        let ts = if i < timestamps.len() {
            timestamps[i]
        } else {
            0
        };
        println!(
            "   [{}] Dir: {:.3}, Short: {:.6}, Uncertainty: {:.4}",
            ts,
            predictions.direction[[i, 0]],
            predictions.short_term[[i, 0]],
            predictions.uncertainty[[i, 0]]
        );
    }

    // Step 6: Get scale importance
    println!("\n6. Scale importance weights:");
    let importance = model.get_scale_importance();
    for (scale, weight) in &importance {
        println!("   {}: {:.4}", scale, weight);
    }

    // Step 7: Backtest
    println!("\n7. Backtesting strategy...");
    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let test_prices = if prices.len() > targets.len() {
        &prices[prices.len() - targets.len()..]
    } else {
        &prices
    };

    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        transaction_cost: 0.001,
        confidence_threshold: 0.55,
        max_position_size: 1.0,
        stop_loss: None,
        take_profit: None,
    };

    let result = backtest_multi_scale_strategy(&predictions, test_prices, &backtest_config);
    println!("{}", result.report());

    println!("================================================================================");
    println!("   EXAMPLE COMPLETED SUCCESSFULLY");
    println!("================================================================================");

    Ok(())
}

fn run_with_synthetic_data() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("   MULTI-SCALE ATTENTION - SYNTHETIC DATA EXAMPLE");
    println!("================================================================================");

    // Configuration
    let scales = ["1min", "5min", "1H"];
    let lookbacks: HashMap<String, usize> = [
        ("1min".to_string(), 60),
        ("5min".to_string(), 24),
        ("1H".to_string(), 12),
    ]
    .into_iter()
    .collect();
    let n_features = 15;
    let n_samples = 500;

    // Step 1: Generate synthetic data
    println!("\n1. Generating synthetic data...");
    let (sequences, targets) = generate_synthetic_data(n_samples, &scales, n_features, &lookbacks);

    println!("   Generated {} samples", n_samples);
    for scale in &scales {
        if let Some(seq) = sequences.get(*scale) {
            println!("   Scale {}: {:?}", scale, seq.dim());
        }
    }

    // Step 2: Create model
    println!("\n2. Creating multi-scale attention model...");
    let config = MultiScaleAttentionConfig {
        scale_configs: vec![
            ScaleConfig {
                name: "1min".to_string(),
                input_dim: n_features,
                seq_len: 60,
            },
            ScaleConfig {
                name: "5min".to_string(),
                input_dim: n_features,
                seq_len: 24,
            },
            ScaleConfig {
                name: "1H".to_string(),
                input_dim: n_features,
                seq_len: 12,
            },
        ],
        d_model: 64,
        n_heads: 4,
        n_encoder_layers: 2,
        dropout: 0.1,
        output_dim: 1,
    };

    let model = MultiScaleAttention::new(config);
    println!("   Model created with {} scales", model.scale_names().len());

    // Step 3: Make predictions
    println!("\n3. Making predictions...");
    let predictions = model.forward(&sequences);

    println!("   Predictions: {}", predictions.direction.nrows());
    println!("   Sample predictions:");
    for i in 0..5.min(predictions.direction.nrows()) {
        println!(
            "   [{}] Dir: {:.3}, Short: {:.6}, Uncertainty: {:.4}",
            i,
            predictions.direction[[i, 0]],
            predictions.short_term[[i, 0]],
            predictions.uncertainty[[i, 0]]
        );
    }

    // Step 4: Scale importance
    println!("\n4. Scale importance weights:");
    let importance = model.get_scale_importance();
    for (scale, weight) in &importance {
        println!("   {}: {:.4}", scale, weight);
    }

    // Step 5: Backtest
    println!("\n5. Backtesting strategy...");
    let mut prices = Vec::with_capacity(n_samples);
    let mut price = 100.0;
    for target in targets.iter() {
        price *= 1.0 + target;
        prices.push(price);
    }

    let backtest_config = BacktestConfig::default();
    let result = backtest_multi_scale_strategy(&predictions, &prices, &backtest_config);
    println!("{}", result.report());

    println!("================================================================================");
    println!("   EXAMPLE COMPLETED SUCCESSFULLY");
    println!("================================================================================");

    Ok(())
}

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("synthetic");
    let symbol = args.get(2).map(|s| s.as_str()).unwrap_or("BTCUSDT");

    println!("\nMulti-Scale Attention for Financial Time Series");
    println!("Mode: {}", mode);

    let result = match mode {
        "bybit" => run_with_bybit_data(symbol),
        _ => run_with_synthetic_data(),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
