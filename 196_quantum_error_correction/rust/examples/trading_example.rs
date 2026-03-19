//! Trading Example: Quantum Error Correction for Kernel-Based Classification
//!
//! This example:
//! 1. Fetches BTCUSDT kline data from Bybit
//! 2. Computes features and labels for binary classification
//! 3. Runs quantum kernel computation under different noise conditions
//! 4. Compares classification accuracy: ideal vs noisy vs corrected vs mitigated

use quantum_error_correction::*;

fn main() -> anyhow::Result<()> {
    println!("=== Quantum Error Correction for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -----------------------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT data from Bybit...");

    let klines = match fetch_bybit_klines("BTCUSDT", "60", 100) {
        Ok(k) if !k.is_empty() => {
            println!("  Fetched {} klines from Bybit", k.len());
            k
        }
        Ok(_) | Err(_) => {
            println!("  Could not fetch from Bybit, using synthetic data");
            generate_synthetic_klines(100)
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Feature engineering
    // -----------------------------------------------------------------------
    println!("\nStep 2: Computing features...");

    let returns = compute_returns(&klines);
    let volatility = compute_volatility(&returns, 5);

    if volatility.len() < 20 {
        println!("  Not enough data points, using synthetic data");
        let klines = generate_synthetic_klines(100);
        let returns = compute_returns(&klines);
        let volatility = compute_volatility(&returns, 5);
        run_classification_experiment(&returns, &volatility);
        return Ok(());
    }

    println!("  Returns: {} data points", returns.len());
    println!("  Volatility features: {} data points", volatility.len());

    run_classification_experiment(&returns, &volatility);

    Ok(())
}

fn run_classification_experiment(returns: &[f64], volatility: &[f64]) {
    // Use volatility as features and next-period return sign as labels
    let n = volatility.len().min(returns.len() - 5);
    if n < 10 {
        println!("  Insufficient data for classification");
        return;
    }

    let features: Vec<f64> = volatility[..n].to_vec();
    let labels: Vec<i32> = returns[5..5 + n]
        .iter()
        .map(|&r| if r > 0.0 { 1 } else { -1 })
        .collect();

    // Split into train/test (70/30)
    let split = (n as f64 * 0.7) as usize;
    let train_features = &features[..split];
    let train_labels = &labels[..split];
    let test_features = &features[split..];
    let test_labels = &labels[split..];

    println!(
        "  Train: {} samples, Test: {} samples",
        train_features.len(),
        test_features.len()
    );

    // -----------------------------------------------------------------------
    // Step 3: Quantum kernel classification under different noise conditions
    // -----------------------------------------------------------------------
    println!("\nStep 3: Quantum kernel classification\n");

    let noise_prob = 0.05; // 5% depolarizing noise rate

    // --- Ideal (no noise) ---
    let ideal_preds = kernel_classify(
        train_features,
        train_labels,
        test_features,
        &|x, y| quantum_kernel_ideal(x, y),
    );
    let ideal_acc = accuracy(&ideal_preds, test_labels);

    // --- Noisy ---
    let noisy_preds = kernel_classify(
        train_features,
        train_labels,
        test_features,
        &|x, y| quantum_kernel_noisy(x, y, noise_prob),
    );
    let noisy_acc = accuracy(&noisy_preds, test_labels);

    // --- Error-corrected ---
    let corrected_preds = kernel_classify(
        train_features,
        train_labels,
        test_features,
        &|x, y| quantum_kernel_corrected(x, y, noise_prob),
    );
    let corrected_acc = accuracy(&corrected_preds, test_labels);

    // --- Zero-noise extrapolation (mitigated) ---
    let mitigated_preds = kernel_classify(
        train_features,
        train_labels,
        test_features,
        &|x, y| {
            let e1 = quantum_kernel_noisy(x, y, noise_prob);
            let e2 = quantum_kernel_noisy(x, y, 2.0 * noise_prob);
            let e3 = quantum_kernel_noisy(x, y, 3.0 * noise_prob);
            zero_noise_extrapolation(&[1.0, 2.0, 3.0], &[e1, e2, e3])
        },
    );
    let mitigated_acc = accuracy(&mitigated_preds, test_labels);

    // -----------------------------------------------------------------------
    // Step 4: Results
    // -----------------------------------------------------------------------
    println!("  Classification Results (noise_prob = {}):", noise_prob);
    println!("  +-----------------------+----------+");
    println!("  | Method                | Accuracy |");
    println!("  +-----------------------+----------+");
    println!("  | Ideal (no noise)      |  {:.1}%   |", ideal_acc * 100.0);
    println!("  | Noisy (depolarizing)  |  {:.1}%   |", noisy_acc * 100.0);
    println!("  | Error-corrected       |  {:.1}%   |", corrected_acc * 100.0);
    println!("  | ZNE mitigated         |  {:.1}%   |", mitigated_acc * 100.0);
    println!("  +-----------------------+----------+");

    // -----------------------------------------------------------------------
    // Step 5: Detailed kernel comparison for a sample pair
    // -----------------------------------------------------------------------
    println!("\nStep 4: Detailed kernel comparison (sample feature pair)\n");

    if features.len() >= 2 {
        let x = features[0];
        let y = features[1];
        let comp = compare_kernel_methods(x, y, noise_prob, 1000);

        println!("  Features: x = {:.6}, y = {:.6}", x, y);
        println!("  Ideal kernel:     {:.6}", comp.ideal);
        println!("  Noisy kernel:     {:.6} (error: {:.6})", comp.noisy, comp.noisy_error);
        println!(
            "  Corrected kernel: {:.6} (error: {:.6})",
            comp.corrected, comp.corrected_error
        );
        println!(
            "  Mitigated kernel: {:.6} (error: {:.6})",
            comp.mitigated, comp.mitigated_error
        );
        println!(
            "\n  Error reduction (correction): {:.1}x",
            if comp.corrected_error > 0.0 {
                comp.noisy_error / comp.corrected_error
            } else {
                f64::INFINITY
            }
        );
        println!(
            "  Error reduction (mitigation): {:.1}x",
            if comp.mitigated_error > 0.0 {
                comp.noisy_error / comp.mitigated_error
            } else {
                f64::INFINITY
            }
        );
    }

    // -----------------------------------------------------------------------
    // Step 6: Error correction demo on quantum states
    // -----------------------------------------------------------------------
    println!("\nStep 5: Error correction demo on quantum states\n");

    let original = QuantumState::new(0.6, 0.8);
    println!(
        "  Original state: {:.4}|0> + {:.4}|1>",
        original.amplitudes[0], original.amplitudes[1]
    );

    // Run many trials of error correction
    let num_trials = 1000;
    let mut corrected_fidelity_sum = 0.0;
    let mut uncorrected_fidelity_sum = 0.0;

    for _ in 0..num_trials {
        // Without error correction: just apply noise
        let noisy = bit_flip_channel(&original, noise_prob);
        uncorrected_fidelity_sum += noisy.fidelity(&original);

        // With error correction
        let corrected = run_error_correction_pipeline(&original, noise_prob);
        corrected_fidelity_sum += corrected.fidelity(&original);
    }

    let avg_uncorrected = uncorrected_fidelity_sum / num_trials as f64;
    let avg_corrected = corrected_fidelity_sum / num_trials as f64;

    println!(
        "  Average fidelity without EC: {:.4} ({} trials)",
        avg_uncorrected, num_trials
    );
    println!(
        "  Average fidelity with EC:    {:.4} ({} trials)",
        avg_corrected, num_trials
    );
    println!(
        "  Fidelity improvement:        {:.4}",
        avg_corrected - avg_uncorrected
    );

    println!("\n=== Done ===");
}

/// Generate synthetic kline data for testing when API is unavailable
fn generate_synthetic_klines(n: usize) -> Vec<Kline> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut klines = Vec::with_capacity(n);

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        klines.push(Kline {
            timestamp: (1700000000 + i as u64 * 3600) * 1000,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    klines
}
