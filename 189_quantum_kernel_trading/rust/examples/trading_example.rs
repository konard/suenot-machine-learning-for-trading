//! Quantum Kernel Trading Example
//!
//! Fetches BTCUSDT data from Bybit, engineers features,
//! computes a quantum kernel matrix, trains a kernel perceptron
//! to detect market regimes, and reports classification accuracy.

use quantum_kernel_trading::*;

fn main() -> anyhow::Result<()> {
    println!("=== Quantum Kernel Trading ===");
    println!();

    // -----------------------------------------------------------------------
    // 1. Fetch data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT 1h candles from Bybit...");
    let candles = match fetch_bybit_candles("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("  Warning: Could not fetch live data: {}", e);
            println!("  Generating synthetic data for demonstration...");
            generate_synthetic_candles(200)
        }
    };

    if candles.len() < 30 {
        println!("Not enough candle data. Need at least 30 candles.");
        return Ok(());
    }

    println!(
        "  Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles
            .iter()
            .map(|c| c.high)
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // -----------------------------------------------------------------------
    // 2. Engineer features
    // -----------------------------------------------------------------------
    let window = 20;
    let (features, labels) = engineer_features(&candles, window);
    let n_samples = features.len();
    println!("Engineered features for {} samples", n_samples);
    println!(
        "  Features per sample: {} (returns, volatility, rel_volume, momentum)",
        features[0].len()
    );

    let n_high: usize = labels.iter().filter(|&&l| l > 0.0).count();
    let n_low = n_samples - n_high;
    println!(
        "  Regime distribution: {} high-vol, {} low-vol",
        n_high, n_low
    );
    println!();

    // -----------------------------------------------------------------------
    // 3. Build quantum kernel matrix
    // -----------------------------------------------------------------------
    let num_qubits = features[0].len(); // 4 qubits for 4 features
    println!(
        "Building quantum kernel matrix ({0}x{0}) with {1} qubits...",
        n_samples, num_qubits
    );

    let feature_map = QuantumFeatureMap::new(num_qubits);
    let kernel_matrix = build_kernel_matrix(&feature_map, &features);

    // Print some kernel statistics
    let mut k_sum = 0.0;
    let mut k_min = f64::INFINITY;
    let mut k_max = f64::NEG_INFINITY;
    let mut count = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let k = kernel_matrix[[i, j]];
            k_sum += k;
            k_min = k_min.min(k);
            k_max = k_max.max(k);
            count += 1;
        }
    }
    let k_mean = if count > 0 { k_sum / count as f64 } else { 0.0 };
    println!("  Off-diagonal kernel stats:");
    println!("    Mean: {:.4}", k_mean);
    println!("    Min:  {:.4}", k_min);
    println!("    Max:  {:.4}", k_max);

    // Kernel alignment
    let ka = kernel_alignment(&kernel_matrix, &labels);
    println!("  Kernel alignment with labels: {:.4}", ka);
    println!();

    // -----------------------------------------------------------------------
    // 4. Train kernel perceptron
    // -----------------------------------------------------------------------
    println!("Training kernel perceptron...");
    let epochs = 100;
    let mut classifier = KernelPerceptron::new(n_samples);
    classifier.train(&kernel_matrix, &labels, epochs);

    let train_accuracy = classifier.training_accuracy(&kernel_matrix, &labels);
    println!("  Training accuracy: {:.2}%", train_accuracy * 100.0);

    // Count support vectors (non-zero alphas)
    let n_sv = classifier.alphas.iter().filter(|&&a| a > 0.0).count();
    println!("  Support vectors: {} / {}", n_sv, n_samples);
    println!();

    // -----------------------------------------------------------------------
    // 5. Demonstrate prediction on last few samples
    // -----------------------------------------------------------------------
    println!("Last 5 predictions vs actual:");
    let states: Vec<QuantumState> = features.iter().map(|x| feature_map.encode(x)).collect();

    let start = if n_samples > 5 { n_samples - 5 } else { 0 };
    for i in start..n_samples {
        let kernel_row = compute_kernel_row(&feature_map, &states, &features[i]);
        let pred = classifier.predict(&kernel_row, &labels);
        let actual = labels[i];
        let correct = if (pred - actual).abs() < 1e-10 {
            "OK"
        } else {
            "MISS"
        };
        let regime_pred = if pred > 0.0 { "HIGH-VOL" } else { "LOW-VOL " };
        let regime_actual = if actual > 0.0 { "HIGH-VOL" } else { "LOW-VOL " };
        println!(
            "  Sample {}: predicted={} actual={} [{}]",
            i, regime_pred, regime_actual, correct
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. Print kernel matrix snippet
    // -----------------------------------------------------------------------
    let snippet_size = 5.min(n_samples);
    println!("Kernel matrix (top-left {}x{}):", snippet_size, snippet_size);
    for i in 0..snippet_size {
        let row: Vec<String> = (0..snippet_size)
            .map(|j| format!("{:.3}", kernel_matrix[[i, j]]))
            .collect();
        println!("  [{}]", row.join(", "));
    }
    println!();

    println!("=== Done ===");
    Ok(())
}

/// Generate synthetic candle data for demonstration when API is unavailable.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        // Alternate between low-vol and high-vol regimes
        let regime_vol = if (i / 50) % 2 == 0 { 0.002 } else { 0.008 };
        let ret: f64 = rng.gen::<f64>() * 2.0 * regime_vol - regime_vol;
        let close = price * (1.0 + ret);
        let high = close * (1.0 + rng.gen::<f64>() * regime_vol);
        let low = price * (1.0 - rng.gen::<f64>() * regime_vol);
        let volume = 100.0 + rng.gen::<f64>() * 200.0;

        candles.push(Candle {
            timestamp: 1700000000 + (i as u64) * 3600,
            open: price,
            high,
            low: low.min(price).min(close),
            close,
            volume,
        });

        price = close;
    }

    candles
}
