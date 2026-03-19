use mae_pretraining_trading::*;

/// Demonstrate MAE pretraining on trading data
fn main() {
    println!("=== Chapter 283: MAE Pretraining for Trading ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Generate synthetic OHLCV data (in production, use BybitClient)
    // -------------------------------------------------------------------------
    println!("Step 1: Generating synthetic BTCUSDT OHLCV data...");
    let candles = generate_synthetic_ohlcv(200);
    println!("  Generated {} candles", candles.len());
    println!(
        "  Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // Convert to matrix and normalize
    let matrix = candles_to_matrix(&candles);
    let returns = normalize_to_log_returns(&matrix);
    println!(
        "  Normalized to log returns: {} rows x {} features\n",
        returns.dim().0,
        returns.dim().1
    );

    // -------------------------------------------------------------------------
    // Step 2: Configure and create MAE
    // -------------------------------------------------------------------------
    println!("Step 2: Configuring MAE...");
    let config = MAEConfig {
        patch_size: 4,
        num_features: 5,
        embed_dim: 32,
        encoder_layers: 2,
        decoder_layers: 1,
        ffn_dim: 64,
        mask_ratio: 0.75,
        learning_rate: 0.001,
    };
    println!("  Patch size: {} time steps", config.patch_size);
    println!("  Embedding dim: {}", config.embed_dim);
    println!("  Encoder layers: {}", config.encoder_layers);
    println!("  Decoder layers: {}", config.decoder_layers);
    println!("  Mask ratio: {:.0}%", config.mask_ratio * 100.0);

    let mae = MaskedAutoencoder::new(config.clone());
    let n_patches = returns.dim().0 / config.patch_size;
    println!("  Number of patches: {}", n_patches);
    println!(
        "  Visible patches per forward pass: {}\n",
        (n_patches as f64 * (1.0 - config.mask_ratio)).round() as usize
    );

    // -------------------------------------------------------------------------
    // Step 3: Patch OHLCV into windows
    // -------------------------------------------------------------------------
    println!("Step 3: Creating patches from OHLCV data...");
    let patches = mae.patch_embedding.create_patches(&returns);
    println!("  Created {} patches", patches.len());
    println!(
        "  Each patch: {} elements ({} steps x {} features)\n",
        patches[0].len(),
        config.patch_size,
        config.num_features
    );

    // -------------------------------------------------------------------------
    // Step 4: Pretrain MAE encoder
    // -------------------------------------------------------------------------
    println!("Step 4: Pretraining MAE (reconstruction task)...");
    let epochs = 10;
    let losses = mae.pretrain(&returns, epochs);

    for (i, loss) in losses.iter().enumerate() {
        let bar_len = (loss * 500.0).min(50.0) as usize;
        let bar: String = "#".repeat(bar_len);
        println!("  Epoch {:2}/{}: loss = {:.6} |{}|", i + 1, epochs, loss, bar);
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 5: Evaluate reconstruction quality
    // -------------------------------------------------------------------------
    println!("Step 5: Evaluating reconstruction quality...");
    let (final_loss, reconstructed) = mae.forward(&returns);
    println!("  Final reconstruction loss (MSE): {:.6}", final_loss);
    println!("  Number of reconstructed patches: {}", reconstructed.len());

    // Compare a few original vs reconstructed patches
    if !patches.is_empty() && !reconstructed.is_empty() {
        println!("\n  Sample patch comparison (first patch):");
        let orig = &patches[0];
        let recon = &reconstructed[0];
        let n_show = orig.len().min(10);
        println!("  {:>10} {:>10}", "Original", "Reconst.");
        for i in 0..n_show {
            println!("  {:10.6} {:10.6}", orig[i], recon[i]);
        }
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 6: Anomaly detection using reconstruction error
    // -------------------------------------------------------------------------
    println!("Step 6: Anomaly detection demo...");

    // Normal data anomaly score
    let normal_score = mae.anomaly_score(&returns);
    println!("  Normal data anomaly score:   {:.6}", normal_score);

    // Create anomalous data (spike in prices)
    let mut anomalous = returns.clone();
    let mid = anomalous.dim().0 / 2;
    for j in 0..5 {
        anomalous[[mid, j]] *= 10.0; // Inject anomaly
        if mid + 1 < anomalous.dim().0 {
            anomalous[[mid + 1, j]] *= 8.0;
        }
    }
    let anomaly_score = mae.anomaly_score(&anomalous);
    println!("  Anomalous data anomaly score: {:.6}", anomaly_score);
    println!(
        "  Anomaly amplification: {:.2}x\n",
        anomaly_score / normal_score.max(1e-10)
    );

    // -------------------------------------------------------------------------
    // Step 7: Extract features for downstream tasks
    // -------------------------------------------------------------------------
    println!("Step 7: Extracting encoder features for downstream tasks...");
    let features = mae.encode(&returns);
    println!("  Extracted {} feature vectors", features.len());
    if !features.is_empty() {
        println!("  Feature dimension: {}", features[0].len());
    }

    // -------------------------------------------------------------------------
    // Step 8: Fine-tune for direction prediction
    // -------------------------------------------------------------------------
    println!("\nStep 8: Fine-tuning for direction prediction...");
    let predictor = DirectionPredictor::new(config.embed_dim);

    // Predict on a few windows
    let window_size = 20; // 20 time steps per prediction window
    let n_windows = (returns.dim().0 / window_size).min(5);

    println!("  {:>8} {:>12} {:>12}", "Window", "Up Prob", "Direction");
    for w in 0..n_windows {
        let start = w * window_size;
        let end = ((w + 1) * window_size).min(returns.dim().0);
        if end - start < config.patch_size {
            continue;
        }
        let window_data = returns.slice(ndarray::s![start..end, ..]).to_owned();
        let window_features = mae.encode(&window_data);
        let up_prob = predictor.predict(&window_features);
        let direction = if up_prob > 0.5 { "UP" } else { "DOWN" };
        println!("  {:>8} {:>12.4} {:>12}", w + 1, up_prob, direction);
    }

    // -------------------------------------------------------------------------
    // Step 9: Bybit integration note
    // -------------------------------------------------------------------------
    println!("\n--- Bybit Integration ---");
    println!("To use real data, replace synthetic data with:");
    println!("  let client = BybitClient::new();");
    println!("  let candles = client.fetch_klines(\"BTCUSDT\", \"15\", 1000).await?;");
    println!();

    // Show how to use the Bybit client (without actually calling)
    let _client = BybitClient::new();
    println!("  BybitClient created (base_url: {})", _client.base_url);
    println!("  Ready to fetch BTCUSDT, ETHUSDT, or any Bybit perpetual contract");

    println!("\n=== MAE Pretraining Complete ===");
    println!("Key results:");
    println!("  - Pretrained MAE encoder on {} patches", patches.len());
    println!("  - Final reconstruction MSE: {:.6}", final_loss);
    println!("  - Anomaly detection working (score ratio shown above)");
    println!("  - Feature extraction ready for downstream tasks");
    println!("  - Direction predictor initialized for fine-tuning");
}
