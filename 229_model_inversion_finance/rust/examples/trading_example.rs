use model_inversion_finance::*;
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    println!("=== Chapter 229: Model Inversion Finance ===\n");

    // ─── Step 1: Fetch market data from Bybit ───
    println!("[1] Fetching BTCUSDT klines from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "15", 200) {
        Ok(c) => {
            println!("    Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("    Warning: could not fetch from Bybit ({e}), generating synthetic data");
            generate_synthetic_candles(200)
        }
    };

    // ─── Step 2: Compute private trading features ───
    println!("\n[2] Computing private trading features...");
    let features = compute_private_features(&candles);
    println!("    Generated {} feature vectors (dim=5)", features.len());
    if features.is_empty() {
        println!("    Not enough data, exiting.");
        return Ok(());
    }

    // Show first feature vector
    println!("    Sample features: {:?}", features[0].as_slice().unwrap());

    // ─── Step 3: Train a private model ───
    println!("\n[3] Training private trading model...");
    let mut model = LinearModel::new_random(5, 1);

    // Create targets: next-bar return sign (simplified)
    let targets: Vec<Array1<f64>> = features
        .windows(2)
        .map(|w| {
            let next_return = w[1][0]; // lr1 of next bar
            Array1::from_vec(vec![if next_return > 0.0 { 1.0 } else { -1.0 }])
        })
        .collect();
    let train_features = &features[..features.len() - 1];

    model.train(train_features, &targets, 0.01, 200);
    println!("    Model trained on {} samples", train_features.len());

    // ─── Step 4: White-box inversion attack ───
    println!("\n[4] White-box inversion attack...");
    let test_idx = 0;
    let original = &features[test_idx];
    let model_output = model.predict(original);
    println!("    Original features:      {:?}", fmt_vec(original));
    println!("    Model output:           {:?}", fmt_arr(&model_output));

    let reconstructed_wb = white_box_inversion(&model, &model_output, 2000, 0.01, 0.001);
    let wb_mse = mse(original, &reconstructed_wb);
    let wb_cos = cosine_similarity(original, &reconstructed_wb);
    println!("    Reconstructed (WB):     {:?}", fmt_vec(&reconstructed_wb));
    println!("    MSE:  {wb_mse:.6}");
    println!("    Cosine similarity: {wb_cos:.4}");

    // ─── Step 5: Black-box inversion attack ───
    println!("\n[5] Black-box inversion attack...");
    let model_clone = model.clone();
    let target_for_bb = model_output.clone();
    let query_fn = |x: &Array1<f64>| model_clone.predict(x);
    let reconstructed_bb = black_box_inversion(
        query_fn,
        &target_for_bb,
        5,    // input_dim
        500,  // iterations
        0.01, // learning rate
        0.001, // epsilon for finite differences
        0.001, // regularization
    );
    let bb_mse = mse(original, &reconstructed_bb);
    let bb_cos = cosine_similarity(original, &reconstructed_bb);
    println!("    Reconstructed (BB):     {:?}", fmt_vec(&reconstructed_bb));
    println!("    MSE:  {bb_mse:.6}");
    println!("    Cosine similarity: {bb_cos:.4}");

    // ─── Step 6: Defense — Output Perturbation ───
    println!("\n[6] Defense: Output Perturbation");
    for noise_std in [0.01, 0.05, 0.1, 0.5] {
        let noisy_output = model.predict_with_perturbation(original, noise_std);
        let rec = white_box_inversion(&model, &noisy_output, 2000, 0.01, 0.001);
        let def_cos = cosine_similarity(original, &rec);
        let def_mse = mse(original, &rec);
        println!(
            "    noise_std={:.2}: MSE={:.6}, cos_sim={:.4}",
            noise_std, def_mse, def_cos
        );
    }

    // ─── Step 7: Defense — Confidence Limiting ───
    println!("\n[7] Defense: Confidence Limiting");
    for max_abs in [1.0, 0.5, 0.1, 0.01] {
        let limited_output = model.predict_with_confidence_limit(original, max_abs);
        let rec = white_box_inversion(&model, &limited_output, 2000, 0.01, 0.001);
        let def_cos = cosine_similarity(original, &rec);
        let def_mse = mse(original, &rec);
        println!(
            "    max_abs={:.2}: MSE={:.6}, cos_sim={:.4}",
            max_abs, def_mse, def_cos
        );
    }

    // ─── Step 8: Privacy-utility tradeoff ───
    println!("\n[8] Privacy-Utility Tradeoff Analysis");
    println!("    {:>12} {:>12} {:>12} {:>12}", "noise_std", "avg_MSE", "avg_cos", "pred_error");
    let eval_features = if features.len() > 20 {
        &features[..20]
    } else {
        &features[..]
    };

    for noise_std in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0] {
        let (avg_mse, avg_cos) = if noise_std < 1e-12 {
            measure_privacy_leakage(&model, eval_features, 1000, 0.01, 0.001)
        } else {
            measure_leakage_with_perturbation(
                &model, eval_features, noise_std, 1000, 0.01, 0.001,
            )
        };

        // Measure prediction accuracy degradation
        let mut pred_error = 0.0;
        for x in eval_features {
            let clean = model.predict(x);
            let noisy = if noise_std < 1e-12 {
                clean.clone()
            } else {
                model.predict_with_perturbation(x, noise_std)
            };
            pred_error += mse(&clean, &noisy);
        }
        pred_error /= eval_features.len() as f64;

        println!(
            "    {:>12.3} {:>12.6} {:>12.4} {:>12.6}",
            noise_std, avg_mse, avg_cos, pred_error
        );
    }

    println!("\n=== Done ===");
    Ok(())
}

/// Generate synthetic candle data when Bybit is unavailable.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(n);
    for i in 0..n {
        let ret = rng.gen_range(-0.02..0.02);
        let close = price * (1.0 + ret);
        let high = close * (1.0 + rng.gen_range(0.0..0.01));
        let low = close * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..1000.0);
        candles.push(Candle {
            timestamp: 1700000000000 + (i as u64) * 900000,
            open: price,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }
    candles
}

fn fmt_vec(arr: &ndarray::Array1<f64>) -> Vec<String> {
    arr.iter().map(|v| format!("{v:.6}")).collect()
}

fn fmt_arr(arr: &ndarray::Array1<f64>) -> Vec<String> {
    arr.iter().map(|v| format!("{v:.6}")).collect()
}
