//! Trading Example: Contrastive Self-Supervised Pretraining with Bybit Data
//!
//! This example demonstrates:
//! 1. Fetching BTCUSDT kline data from Bybit
//! 2. Creating augmented positive pairs
//! 3. Training a SimCLR-style encoder
//! 4. Using a linear probe for regime detection

use contrastive_pretraining_trading::*;
use ndarray::Array1;

/// Assign a simple regime label based on price change.
/// 0 = bearish, 1 = sideways, 2 = bullish
fn label_regime(window: &Array1<f64>) -> usize {
    let first = window[0];
    let last = window[window.len() - 1];
    if first == 0.0 {
        return 1;
    }
    let pct_change = (last - first) / first.abs();
    if pct_change > 0.005 {
        2 // bullish
    } else if pct_change < -0.005 {
        0 // bearish
    } else {
        1 // sideways
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Contrastive Pretraining for Trading ===\n");

    // --------------------------------------------------------
    // Step 1: Fetch BTCUSDT data from Bybit
    // --------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT 15m klines from Bybit...");
    let client = BybitClient::new();
    let candles = client.fetch_klines("BTCUSDT", "15", 200).await?;
    println!("  Fetched {} candles", candles.len());

    if candles.len() < 30 {
        println!("  Not enough candles for demonstration. Using synthetic data.");
        run_with_synthetic_data();
        return Ok(());
    }

    // Convert to close prices
    let close_prices = candles_to_close_prices(&candles);
    println!(
        "  Price range: {:.2} - {:.2}",
        close_prices.iter().cloned().fold(f64::INFINITY, f64::min),
        close_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // --------------------------------------------------------
    // Step 2: Create windows and normalize
    // --------------------------------------------------------
    println!("\nStep 2: Creating overlapping windows...");
    let window_size = 20;
    let stride = 5;
    let raw_windows = create_windows(&close_prices, window_size, stride);
    let windows: Vec<Array1<f64>> = raw_windows.iter().map(|w| zscore_normalize(w)).collect();
    println!("  Created {} normalized windows of size {}", windows.len(), window_size);

    // --------------------------------------------------------
    // Step 3: Create augmented pairs and train SimCLR encoder
    // --------------------------------------------------------
    println!("\nStep 3: Training SimCLR encoder...");
    let augmenter = Augmenter::new(AugmentationConfig::default());
    let encoder = Encoder::new(window_size, 64, 32);
    let projection_head = ProjectionHead::new(32, 32, 16);
    let temperature = 0.07;

    // Simulate training steps (forward pass only -- no autograd in pure Rust)
    let num_epochs = 5;
    let batch_size = 8;
    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        for batch_start in (0..windows.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(windows.len());
            let batch = &windows[batch_start..batch_end];
            let loss = simclr_step(batch, &augmenter, &encoder, &projection_head, temperature);
            epoch_loss += loss;
            num_batches += 1;
        }

        let avg_loss = epoch_loss / num_batches as f64;
        println!("  Epoch {}/{}: avg NT-Xent loss = {:.4}", epoch + 1, num_epochs, avg_loss);
    }

    // --------------------------------------------------------
    // Step 4: Encode all windows and run linear probe
    // --------------------------------------------------------
    println!("\nStep 4: Linear probe for regime detection...");

    // Encode all windows
    let representations: Vec<Array1<f64>> = windows.iter().map(|w| encoder.forward(w)).collect();

    // Label regimes from raw (un-normalized) windows
    let labels: Vec<usize> = raw_windows.iter().map(|w| label_regime(w)).collect();

    let label_counts = [
        labels.iter().filter(|&&l| l == 0).count(),
        labels.iter().filter(|&&l| l == 1).count(),
        labels.iter().filter(|&&l| l == 2).count(),
    ];
    println!(
        "  Regime distribution: bearish={}, sideways={}, bullish={}",
        label_counts[0], label_counts[1], label_counts[2]
    );

    // Train linear probe
    let mut probe = LinearProbe::new(32, 3, 0.01);
    let train_size = (representations.len() * 3) / 4;
    let train_reprs = &representations[..train_size];
    let train_labels = &labels[..train_size];
    let test_reprs = &representations[train_size..];
    let test_labels = &labels[train_size..];

    println!("  Training linear probe on {} samples...", train_size);
    for epoch in 0..50 {
        let mut epoch_loss = 0.0;
        for (repr, &label) in train_reprs.iter().zip(train_labels.iter()) {
            epoch_loss += probe.train_step(repr, label);
        }
        if (epoch + 1) % 10 == 0 {
            let acc = probe.evaluate(train_reprs, train_labels);
            println!(
                "  Probe epoch {}/50: loss={:.4}, train_acc={:.2}%",
                epoch + 1,
                epoch_loss / train_size as f64,
                acc * 100.0
            );
        }
    }

    if !test_reprs.is_empty() {
        let test_acc = probe.evaluate(test_reprs, test_labels);
        println!(
            "\n  Test accuracy: {:.2}% ({} samples)",
            test_acc * 100.0,
            test_reprs.len()
        );
    }

    // --------------------------------------------------------
    // Step 5: Anomaly detection demo
    // --------------------------------------------------------
    println!("\nStep 5: Anomaly detection...");
    let reference_reprs = &representations[..representations.len() / 2];
    println!("  Reference set: {} windows", reference_reprs.len());

    for (i, repr) in representations.iter().enumerate().skip(representations.len() / 2) {
        let min_dist = reference_reprs
            .iter()
            .map(|r| {
                let diff = repr - r;
                diff.dot(&diff).sqrt()
            })
            .fold(f64::INFINITY, f64::min);

        if min_dist > 5.0 {
            println!("  Window {} flagged as anomalous (dist={:.3})", i, min_dist);
        }
    }

    println!("\n=== Done! ===");
    Ok(())
}

/// Fallback: run demonstration with synthetic data if Bybit API is unavailable.
fn run_with_synthetic_data() {
    println!("\n--- Running with synthetic data ---\n");
    let mut rng = rand::thread_rng();

    // Generate synthetic price series: random walk with drift
    let n = 200;
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0_f64;
    for _ in 0..n {
        use rand::Rng;
        price += rng.gen_range(-100.0..100.0);
        prices.push(price);
    }
    let data = Array1::from_vec(prices);
    println!("Generated {} synthetic prices", n);

    let window_size = 20;
    let stride = 5;
    let raw_windows = create_windows(&data, window_size, stride);
    let windows: Vec<Array1<f64>> = raw_windows.iter().map(|w| zscore_normalize(w)).collect();
    println!("Created {} windows", windows.len());

    let augmenter = Augmenter::new(AugmentationConfig::default());
    let encoder = Encoder::new(window_size, 64, 32);
    let projection_head = ProjectionHead::new(32, 32, 16);

    for epoch in 0..3 {
        let loss = simclr_step(&windows[..8.min(windows.len())], &augmenter, &encoder, &projection_head, 0.07);
        println!("Epoch {}: NT-Xent loss = {:.4}", epoch + 1, loss);
    }

    // Linear probe
    let representations: Vec<Array1<f64>> = windows.iter().map(|w| encoder.forward(w)).collect();
    let labels: Vec<usize> = raw_windows.iter().map(|w| label_regime(w)).collect();

    let mut probe = LinearProbe::new(32, 3, 0.01);
    for _ in 0..30 {
        for (repr, &label) in representations.iter().zip(labels.iter()) {
            probe.train_step(repr, label);
        }
    }
    let acc = probe.evaluate(&representations, &labels);
    println!("Linear probe accuracy: {:.2}%", acc * 100.0);
    println!("\n--- Synthetic data demo complete ---");
}
