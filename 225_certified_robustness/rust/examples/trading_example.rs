//! Trading Example: Certified Robustness with Bybit Data
//!
//! Fetches BTCUSDT kline data from Bybit, trains a base classifier,
//! applies randomized smoothing certification, and compares certified
//! vs empirical robustness.

use certified_robustness::*;
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    println!("=== Certified Robustness for Trading ===\n");

    // ─── Step 1: Fetch data from Bybit ──────────────────────────────────
    println!("[1] Fetching BTCUSDT klines from Bybit...");
    let klines = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(k) => {
            println!("    Fetched {} klines", k.len());
            k
        }
        Err(e) => {
            println!("    Warning: Could not fetch from Bybit ({}), using synthetic data", e);
            generate_synthetic_klines(200)
        }
    };

    if klines.len() < 20 {
        println!("    Not enough klines, using synthetic data");
        let klines = generate_synthetic_klines(200);
        return run_pipeline(&klines);
    }

    run_pipeline(&klines)
}

fn run_pipeline(klines: &[Kline]) -> anyhow::Result<()> {
    // ─── Step 2: Extract features ───────────────────────────────────────
    println!("\n[2] Extracting features...");
    let lookback = 10;
    let raw_features = extract_features(klines, lookback);
    let labels = generate_labels(klines, lookback);

    let n = raw_features.len().min(labels.len());
    let raw_features = &raw_features[..n];
    let labels = &labels[..n];

    println!("    {} samples with {} features each", n, if n > 0 { raw_features[0].len() } else { 0 });

    if n < 10 {
        println!("    Not enough samples for training");
        return Ok(());
    }

    // Normalize features
    let (features, mean, std) = normalize_features(raw_features);
    println!("    Feature mean: {:?}", mean.as_slice().unwrap());
    println!("    Feature std:  {:?}", std.as_slice().unwrap());

    // Split into train/test
    let split = (n as f64 * 0.7) as usize;
    let train_features = &features[..split];
    let train_labels = &labels[..split];
    let test_features = &features[split..];
    let test_labels = &labels[split..];

    println!("    Train: {} samples, Test: {} samples", split, n - split);

    // ─── Step 3: Train base classifier ──────────────────────────────────
    println!("\n[3] Training base classifier...");
    let mut net = SimpleNeuralNetwork::new(&[4, 16, 8, 2]);
    net.train(train_features, train_labels, 5000, 0.01);

    // Evaluate base classifier
    let base_correct: usize = test_features
        .iter()
        .zip(test_labels)
        .filter(|(f, &l)| net.predict(f) == l)
        .count();
    let base_accuracy = base_correct as f64 / test_labels.len() as f64;
    println!("    Base classifier accuracy: {:.2}%", base_accuracy * 100.0);

    // ─── Step 4: Apply randomized smoothing ─────────────────────────────
    println!("\n[4] Applying randomized smoothing certification...");

    let sigma_values = [0.1, 0.25, 0.5, 1.0];

    for &sigma in &sigma_values {
        println!("\n  --- sigma = {:.2} ---", sigma);
        let sc = SmoothedClassifier::new(net.clone(), sigma);

        let n_predict = 100;
        let n_certify = 500;
        let alpha = 0.001;

        let mut results: Vec<CertificationResult> = Vec::new();
        let mut smoothed_correct = 0usize;

        for (i, (f, &l)) in test_features.iter().zip(test_labels).enumerate() {
            let result = sc.certify(f, n_predict, n_certify, alpha);
            if result.predicted_class == l {
                smoothed_correct += 1;
            }
            results.push(result);

            if i < 3 {
                let r = &results[i];
                println!(
                    "    Sample {}: pred={}, true={}, radius={:.4}, p_lower={:.4}, certified={}",
                    i, r.predicted_class, l, r.certified_radius, r.p_lower, r.is_certified
                );
            }
        }

        let smoothed_accuracy = smoothed_correct as f64 / test_labels.len() as f64;
        println!("    Smoothed accuracy: {:.2}%", smoothed_accuracy * 100.0);

        // Certification rates at various radii
        let radii = vec![0.0, 0.1, 0.25, 0.5, 0.75, 1.0];
        let rates = certification_rate_at_radii(&results, &radii);
        println!("    Certified accuracy at radii:");
        for (r, rate) in &rates {
            println!("      r={:.2}: {:.2}%", r, rate * 100.0);
        }
    }

    // ─── Step 5: IBP Certification ──────────────────────────────────────
    println!("\n[5] IBP (Interval Bound Propagation) certification...");
    let ibp = IBPCertifier::new(&net);

    let epsilon_values = [0.01, 0.05, 0.1, 0.25];
    for &eps in &epsilon_values {
        let mut verified_count = 0;
        for f in test_features.iter() {
            let result = ibp.certify(f, eps);
            if result.is_verified {
                verified_count += 1;
            }
        }
        let ibp_rate = verified_count as f64 / test_features.len() as f64;
        println!(
            "    epsilon={:.2}: IBP verified {:.2}% ({}/{})",
            eps,
            ibp_rate * 100.0,
            verified_count,
            test_features.len()
        );
    }

    // ─── Step 6: Compare certified vs empirical robustness ──────────────
    println!("\n[6] Certified vs Empirical Robustness Comparison...");

    let sigma = 0.25;
    let sc = SmoothedClassifier::new(net.clone(), sigma);
    let eps_test = 0.1;

    let mut empirical_robust = 0;
    let mut certified_robust = 0;
    let n_test = test_features.len().min(30); // Limit for speed

    for i in 0..n_test {
        let f = &test_features[i];
        let base_pred = net.predict(f);

        // Empirical: test random perturbations
        let mut is_empirically_robust = true;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for _ in 0..50 {
            let perturbed = Array1::from_shape_fn(f.len(), |j| {
                f[j] + rng.gen_range(-eps_test..eps_test)
            });
            if net.predict(&perturbed) != base_pred {
                is_empirically_robust = false;
                break;
            }
        }
        if is_empirically_robust {
            empirical_robust += 1;
        }

        // Certified: check if certified radius >= eps_test
        let cert = sc.certify(f, 50, 200, 0.001);
        if cert.is_certified && cert.certified_radius >= eps_test {
            certified_robust += 1;
        }
    }

    println!(
        "    Empirical robustness (eps={:.2}): {:.2}% ({}/{})",
        eps_test,
        empirical_robust as f64 / n_test as f64 * 100.0,
        empirical_robust,
        n_test
    );
    println!(
        "    Certified robustness (eps={:.2}): {:.2}% ({}/{})",
        eps_test,
        certified_robust as f64 / n_test as f64 * 100.0,
        certified_robust,
        n_test
    );
    println!("    Note: Certified <= Empirical (certified is a lower bound)");

    println!("\n=== Done ===");
    Ok(())
}

/// Generate synthetic kline data for when API is unavailable.
fn generate_synthetic_klines(n: usize) -> Vec<Kline> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut klines = Vec::with_capacity(n);

    for i in 0..n {
        let ret = rng.gen_range(-0.02..0.02);
        let close = price * (1.0 + ret);
        let high = close * (1.0 + rng.gen_range(0.0..0.01));
        let low = close * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..1000.0);

        klines.push(Kline {
            timestamp: 1700000000000 + i as u64 * 3600000,
            open: price,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    klines
}
