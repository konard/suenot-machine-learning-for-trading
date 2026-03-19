//! # Quantization Trading Example
//!
//! Fetches BTCUSDT data from Bybit, trains an FP32 model, quantizes to INT8 and INT4,
//! then compares accuracy, model size, and inference speed at each precision level.

use quantization_trading_models::*;
use rand::Rng;
use std::time::Instant;

fn main() {
    println!("=== Quantization Trading Models ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -----------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT kline data from Bybit...");

    let candles = match fetch_bybit_klines("BTCUSDT", "5", 200) {
        Ok(c) => {
            println!("    Fetched {} candles", c.len());
            if let (Some(first), Some(last)) = (c.first(), c.last()) {
                println!(
                    "    Price range: {:.2} - {:.2}",
                    first.open, last.close
                );
            }
            c
        }
        Err(e) => {
            println!("    Failed to fetch data: {}. Using synthetic data.", e);
            generate_synthetic_candles(200)
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Prepare features and targets
    // -----------------------------------------------------------------------
    println!("\n[2] Preparing features...");
    let (mut features, targets) = candles_to_features(&candles);
    normalize_features(&mut features);
    println!("    Training samples: {}", features.len());
    println!("    Features per sample: {}", features[0].len());

    // Split into train and test sets
    let split = (features.len() as f32 * 0.8) as usize;
    let train_features = &features[..split];
    let train_targets = &targets[..split];
    let test_features = &features[split..];
    let test_targets = &targets[split..];
    println!("    Train: {}, Test: {}", train_features.len(), test_features.len());

    // -----------------------------------------------------------------------
    // Step 3: Train FP32 model
    // -----------------------------------------------------------------------
    println!("\n[3] Training FP32 model...");
    let mut fp32_model = TradingModel::new(&[3, 16, 8, 1]);
    let train_start = Instant::now();
    fp32_model.train(train_features, train_targets, 3, 0.001);
    let train_time = train_start.elapsed();
    println!("    Training time: {:.2?}", train_time);
    println!("    FP32 model size: {} bytes", fp32_model.size_bytes());

    // -----------------------------------------------------------------------
    // Step 4: Evaluate FP32 model
    // -----------------------------------------------------------------------
    println!("\n[4] Evaluating FP32 model...");
    let fp32_predictions = benchmark_inference(&fp32_model, test_features);
    let fp32_mse = compute_prediction_mse(&fp32_predictions, test_targets);
    println!("    FP32 MSE: {:.8}", fp32_mse);

    // -----------------------------------------------------------------------
    // Step 5: Quantize to INT8
    // -----------------------------------------------------------------------
    println!("\n[5] Quantizing to INT8 (symmetric per-channel)...");
    let int8_model = QuantizedTradingModel::from_fp32(
        &fp32_model,
        QuantizationScheme::SymmetricPerChannel,
    );
    println!("    INT8 model size: {} bytes", int8_model.size_bytes());
    println!(
        "    Compression ratio: {:.1}x",
        fp32_model.size_bytes() as f32 / int8_model.size_bytes() as f32
    );

    // Evaluate INT8
    let int8_predictions = benchmark_quantized_inference(&int8_model, test_features);
    let int8_mse = compute_prediction_mse(&int8_predictions, test_targets);
    println!("    INT8 MSE: {:.8}", int8_mse);

    // Compute quantization error vs FP32
    let int8_vs_fp32_mse = compute_prediction_mse(&int8_predictions, &fp32_predictions_as_targets(&fp32_predictions));
    println!("    INT8 vs FP32 MSE: {:.10}", int8_vs_fp32_mse);

    // -----------------------------------------------------------------------
    // Step 6: Quantize to INT4
    // -----------------------------------------------------------------------
    println!("\n[6] Quantizing to INT4...");
    let int4_model = Int4TradingModel::from_fp32(&fp32_model);
    println!("    INT4 model size: {} bytes", int4_model.size_bytes());
    println!(
        "    Compression ratio: {:.1}x",
        fp32_model.size_bytes() as f32 / int4_model.size_bytes() as f32
    );

    // Evaluate INT4
    let int4_predictions = benchmark_int4_inference(&int4_model, test_features);
    let int4_mse = compute_prediction_mse(&int4_predictions, test_targets);
    println!("    INT4 MSE: {:.8}", int4_mse);

    let int4_vs_fp32_mse = compute_prediction_mse(&int4_predictions, &fp32_predictions_as_targets(&fp32_predictions));
    println!("    INT4 vs FP32 MSE: {:.10}", int4_vs_fp32_mse);

    // -----------------------------------------------------------------------
    // Step 7: Speed comparison
    // -----------------------------------------------------------------------
    println!("\n[7] Inference speed comparison (1000 runs)...");

    let n_runs = 1000;
    let test_input = &test_features[0];

    // FP32 speed
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = fp32_model.forward(test_input);
    }
    let fp32_time = start.elapsed();

    // INT8 speed
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = int8_model.forward(test_input);
    }
    let int8_time = start.elapsed();

    // INT4 speed
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = int4_model.forward(test_input);
    }
    let int4_time = start.elapsed();

    println!("    FP32: {:.2?} ({:.2} us/inference)", fp32_time, fp32_time.as_micros() as f64 / n_runs as f64);
    println!("    INT8: {:.2?} ({:.2} us/inference)", int8_time, int8_time.as_micros() as f64 / n_runs as f64);
    println!("    INT4: {:.2?} ({:.2} us/inference)", int4_time, int4_time.as_micros() as f64 / n_runs as f64);

    // -----------------------------------------------------------------------
    // Step 8: Weight quantization error analysis
    // -----------------------------------------------------------------------
    println!("\n[8] Weight quantization error analysis...");
    for (i, layer) in fp32_model.layers.iter().enumerate() {
        let weights_flat: Vec<f32> = layer.weights.iter().cloned().collect();

        // INT8 error
        let params8 = compute_params_symmetric(&weights_flat, Precision::Int8);
        let q8 = quantize_to_int8(&weights_flat, &params8);
        let dq8 = dequantize_int8(&q8, &params8);
        let mse8 = compute_mse(&weights_flat, &dq8);
        let snr8 = compute_snr(&weights_flat, &dq8);

        // INT4 error
        let params4 = compute_params_symmetric(&weights_flat, Precision::Int4);
        let q4 = quantize_to_int4(&weights_flat, &params4);
        let dq4 = dequantize_int4(&q4, weights_flat.len(), &params4);
        let mse4 = compute_mse(&weights_flat, &dq4);
        let snr4 = compute_snr(&weights_flat, &dq4);

        println!("    Layer {} ({}x{}):", i, layer.out_features, layer.in_features);
        println!("      INT8 - MSE: {:.8}, SNR: {:.2} dB", mse8, snr8);
        println!("      INT4 - MSE: {:.8}, SNR: {:.2} dB", mse4, snr4);
    }

    // -----------------------------------------------------------------------
    // Step 9: Compare quantization schemes
    // -----------------------------------------------------------------------
    println!("\n[9] Comparing quantization schemes (INT8)...");
    let schemes = [
        ("Symmetric Per-Tensor", QuantizationScheme::SymmetricPerTensor),
        ("Asymmetric Per-Tensor", QuantizationScheme::AsymmetricPerTensor),
        ("Symmetric Per-Channel", QuantizationScheme::SymmetricPerChannel),
        ("Asymmetric Per-Channel", QuantizationScheme::AsymmetricPerChannel),
    ];

    for (name, scheme) in &schemes {
        let q_model = QuantizedTradingModel::from_fp32(&fp32_model, *scheme);
        let preds: Vec<Vec<f32>> = test_features
            .iter()
            .map(|f| q_model.forward(f))
            .collect();
        let mse = compute_prediction_mse(&preds, &fp32_predictions_as_targets(&fp32_predictions));
        println!("    {}: MSE vs FP32 = {:.10}", name, mse);
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("+-----------+----------+-------------+----------+");
    println!("| Precision | Size (B) | Compression | MSE      |");
    println!("+-----------+----------+-------------+----------+");
    println!(
        "| FP32      | {:>8} | {:>11} | {:.6} |",
        fp32_model.size_bytes(),
        "1.0x",
        fp32_mse
    );
    println!(
        "| INT8      | {:>8} | {:>10.1}x | {:.6} |",
        int8_model.size_bytes(),
        fp32_model.size_bytes() as f32 / int8_model.size_bytes() as f32,
        int8_mse
    );
    println!(
        "| INT4      | {:>8} | {:>10.1}x | {:.6} |",
        int4_model.size_bytes(),
        fp32_model.size_bytes() as f32 / int4_model.size_bytes() as f32,
        int4_mse
    );
    println!("+-----------+----------+-------------+----------+");
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn benchmark_inference(model: &TradingModel, test_features: &[Vec<f32>]) -> Vec<Vec<f32>> {
    test_features.iter().map(|f| model.forward(f)).collect()
}

fn benchmark_quantized_inference(
    model: &QuantizedTradingModel,
    test_features: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    test_features.iter().map(|f| model.forward(f)).collect()
}

fn benchmark_int4_inference(
    model: &Int4TradingModel,
    test_features: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    test_features.iter().map(|f| model.forward(f)).collect()
}

fn compute_prediction_mse(predictions: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
    let n = predictions.len() as f32;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            p.iter()
                .zip(t.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        })
        .sum::<f32>()
        / n
}

fn fp32_predictions_as_targets(preds: &[Vec<f32>]) -> Vec<Vec<f32>> {
    preds.to_vec()
}

fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f32;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let change = rng.gen_range(-0.02_f32..0.02);
        let open = price;
        let close = open * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: (i * 300) as u64,
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }
    candles
}
