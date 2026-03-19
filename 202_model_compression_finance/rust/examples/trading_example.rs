//! Trading Example: Model Compression for Price Prediction
//!
//! This example demonstrates:
//! 1. Fetching BTCUSDT data from Bybit
//! 2. Training a neural network for price prediction
//! 3. Applying compression techniques (pruning, quantization, low-rank factorization)
//! 4. Benchmarking accuracy, model size, and inference speed

use model_compression_finance::*;
use std::time::Instant;

fn main() {
    println!("=== Model Compression for Trading ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -------------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT data from Bybit...");

    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("    Fetched {} candles", c.len());
            if let (Some(first), Some(last)) = (c.first(), c.last()) {
                println!(
                    "    Price range: {:.2} - {:.2}",
                    first.close, last.close
                );
            }
            c
        }
        Err(e) => {
            println!("    Warning: Could not fetch live data: {}", e);
            println!("    Using synthetic data instead...");
            generate_synthetic_candles(200)
        }
    };

    // -------------------------------------------------------------------------
    // Step 2: Prepare training data
    // -------------------------------------------------------------------------
    println!("\n[2] Preparing training data...");
    let lookback = 5;
    let input_size = lookback * 5; // 5 features per candle
    let (inputs, targets) = prepare_training_data(&candles, lookback);

    if inputs.is_empty() {
        println!("    Not enough data for training. Exiting.");
        return;
    }

    // Split into train/test (80/20 chronological split)
    let split_idx = (inputs.len() as f64 * 0.8) as usize;
    let train_inputs = &inputs[..split_idx];
    let train_targets = &targets[..split_idx];
    let test_inputs = &inputs[split_idx..];
    let test_targets = &targets[split_idx..];

    println!("    Training samples: {}", train_inputs.len());
    println!("    Test samples: {}", test_inputs.len());

    // -------------------------------------------------------------------------
    // Step 3: Train full model
    // -------------------------------------------------------------------------
    println!("\n[3] Training full model...");
    let mut full_model = NeuralNetwork::new(&[input_size, 64, 32, 1])
        .expect("Failed to create network");

    let start = Instant::now();
    full_model.train(train_inputs, train_targets, 0.001, 50);
    let train_time = start.elapsed();

    let full_mse = evaluate_mse(&full_model, test_inputs, test_targets);
    let full_dir_acc = evaluate_directional_accuracy(&full_model, test_inputs, test_targets);

    println!("    Training time: {:.2?}", train_time);
    println!("    Parameters: {}", full_model.param_count());
    println!("    Memory: {} bytes", full_model.memory_bytes());
    println!("    Test MSE: {:.8}", full_mse);
    println!("    Directional accuracy: {:.2}%", full_dir_acc * 100.0);

    let sample_input = &test_inputs[0];

    // -------------------------------------------------------------------------
    // Step 4: Apply compression techniques
    // -------------------------------------------------------------------------
    println!("\n{}", "=".repeat(60));
    println!("COMPRESSION BENCHMARKS");
    println!("{}", "=".repeat(60));

    // --- Magnitude Pruning ---
    for sparsity in [0.3, 0.5, 0.7, 0.9] {
        println!(
            "\n--- Magnitude Pruning (sparsity={:.0}%) ---",
            sparsity * 100.0
        );
        let pruned = magnitude_prune(&full_model, sparsity);

        let mse = evaluate_mse(&pruned, test_inputs, test_targets);
        let dir_acc = evaluate_directional_accuracy(&pruned, test_inputs, test_targets);
        let metrics = compute_metrics(&full_model, &pruned);
        let avg_inference = benchmark_inference(&pruned, sample_input, 1000);

        println!("    Nonzero params: {}", pruned.nonzero_param_count());
        println!("    Compression ratio: {:.2}x", metrics.compression_ratio);
        println!("    Actual sparsity: {:.2}%", metrics.sparsity * 100.0);
        println!("    Test MSE: {:.8}", mse);
        println!("    Directional accuracy: {:.2}%", dir_acc * 100.0);
        println!(
            "    MSE change: {:+.4}%",
            if full_mse > 0.0 {
                (mse - full_mse) / full_mse * 100.0
            } else {
                0.0
            }
        );
        println!("    Avg inference: {:.2} us", avg_inference);
    }

    // --- INT8 Quantization ---
    println!("\n--- INT8 Quantization ---");
    let quantized = quantize_network(&full_model);

    let quant_mse = evaluate_mse_quantized(&quantized, test_inputs, test_targets);
    let quant_inference = benchmark_inference_quantized(&quantized, sample_input, 1000);
    let quant_memory = quantized.memory_bytes();
    let full_memory = full_model.memory_bytes();

    println!("    Original memory: {} bytes", full_memory);
    println!("    Quantized memory: {} bytes", quant_memory);
    println!(
        "    Memory compression: {:.2}x",
        full_memory as f64 / quant_memory as f64
    );
    println!("    Test MSE: {:.8}", quant_mse);
    println!(
        "    MSE change: {:+.4}%",
        if full_mse > 0.0 {
            (quant_mse - full_mse) / full_mse * 100.0
        } else {
            0.0
        }
    );
    println!("    Avg inference: {:.2} us", quant_inference);

    // --- Low-Rank Factorization ---
    for rank_frac in [0.75, 0.5, 0.25] {
        println!(
            "\n--- Low-Rank Factorization (rank={:.0}%) ---",
            rank_frac * 100.0
        );
        let factorized = low_rank_factorize(&full_model, rank_frac);

        let mse = evaluate_mse(&factorized, test_inputs, test_targets);
        let dir_acc = evaluate_directional_accuracy(&factorized, test_inputs, test_targets);
        let avg_inference = benchmark_inference(&factorized, sample_input, 1000);

        println!("    Test MSE: {:.8}", mse);
        println!("    Directional accuracy: {:.2}%", dir_acc * 100.0);
        println!(
            "    MSE change: {:+.4}%",
            if full_mse > 0.0 {
                (mse - full_mse) / full_mse * 100.0
            } else {
                0.0
            }
        );
        println!("    Avg inference: {:.2} us", avg_inference);
    }

    // -------------------------------------------------------------------------
    // Step 5: Summary comparison
    // -------------------------------------------------------------------------
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));

    let full_inference = benchmark_inference(&full_model, sample_input, 1000);
    println!("\n{:<30} {:>12} {:>12} {:>12}", "Method", "MSE", "Dir Acc%", "Inf (us)");
    println!("{}", "-".repeat(66));
    println!(
        "{:<30} {:>12.8} {:>11.2}% {:>12.2}",
        "Full Model", full_mse, full_dir_acc * 100.0, full_inference
    );

    let pruned_50 = magnitude_prune(&full_model, 0.5);
    let p50_mse = evaluate_mse(&pruned_50, test_inputs, test_targets);
    let p50_acc = evaluate_directional_accuracy(&pruned_50, test_inputs, test_targets);
    let p50_inf = benchmark_inference(&pruned_50, sample_input, 1000);
    println!(
        "{:<30} {:>12.8} {:>11.2}% {:>12.2}",
        "Pruned (50%)", p50_mse, p50_acc * 100.0, p50_inf
    );

    println!(
        "{:<30} {:>12.8} {:>12} {:>12.2}",
        "Quantized (INT8)", quant_mse, "N/A", quant_inference
    );

    let lr_50 = low_rank_factorize(&full_model, 0.5);
    let lr_mse = evaluate_mse(&lr_50, test_inputs, test_targets);
    let lr_acc = evaluate_directional_accuracy(&lr_50, test_inputs, test_targets);
    let lr_inf = benchmark_inference(&lr_50, sample_input, 1000);
    println!(
        "{:<30} {:>12.8} {:>11.2}% {:>12.2}",
        "Low-Rank (50%)", lr_mse, lr_acc * 100.0, lr_inf
    );

    println!("\nDone!");
}

/// Generate synthetic candle data when the Bybit API is unavailable.
fn generate_synthetic_candles(count: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(count);

    for i in 0..count {
        let ret = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: i as u64 * 3600000,
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
