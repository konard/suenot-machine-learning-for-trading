//! Full Variational Quantum Classifier Trading Pipeline
//!
//! This example demonstrates the complete workflow:
//! 1. Fetch market data from Bybit (or generate synthetic data)
//! 2. Engineer financial features
//! 3. Label market regimes
//! 4. Normalize features for quantum encoding
//! 5. Train/test split
//! 6. Train VQC using parameter-shift gradient descent
//! 7. Evaluate predictions and display results

use variational_quantum_classifier::*;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Variational Quantum Classifier (VQC) — Trading Pipeline");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Step 1: Load data ─────────────────────────────────────────────────
    println!("Step 1: Loading market data...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) if !c.is_empty() => {
            println!("  Fetched {} candles from Bybit (BTCUSDT, 1h)", c.len());
            c
        }
        _ => {
            println!("  Bybit API unavailable, generating synthetic data...");
            let c = generate_synthetic_candles(200, 42);
            println!("  Generated {} synthetic candles", c.len());
            c
        }
    };
    println!(
        "  Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // ── Step 2: Feature engineering ───────────────────────────────────────
    println!("Step 2: Engineering features...");
    let window = 10;
    let (features, indices) = engineer_features(&candles, window);
    println!(
        "  Computed {} feature vectors (5 features each, window={})",
        features.len(),
        window
    );
    println!();

    // ── Step 3: Label market regimes ──────────────────────────────────────
    println!("Step 3: Labeling market regimes...");
    let forward = 5;
    let threshold = 0.005;
    let regimes = label_regimes(&candles, forward, threshold);
    let binary = to_binary_labels(&regimes);

    // Align labels with features
    let n = features.len().min(binary.len().saturating_sub(indices[0]));
    let aligned_labels: Vec<f64> = (0..n)
        .filter_map(|i| {
            let candle_idx = indices[i];
            if candle_idx < binary.len() {
                Some(binary[candle_idx])
            } else {
                None
            }
        })
        .collect();
    let aligned_features: Vec<Vec<f64>> = features[..aligned_labels.len()].to_vec();

    let n_bull = aligned_labels.iter().filter(|&&l| l > 0.0).count();
    let n_bear = aligned_labels.len() - n_bull;
    println!(
        "  Labeled {} samples: {} bull (+1), {} bear/side (-1)",
        aligned_labels.len(),
        n_bull,
        n_bear
    );
    println!(
        "  Forward window: {} candles, threshold: {:.1}%",
        forward,
        threshold * 100.0
    );
    println!();

    // ── Step 4: Normalize features ────────────────────────────────────────
    println!("Step 4: Normalizing features to [0, pi]...");
    let normed = normalize_features(&aligned_features);
    println!("  Normalized {} feature vectors", normed.len());
    println!();

    // ── Step 5: Train/test split ──────────────────────────────────────────
    println!("Step 5: Splitting into train/test sets...");
    let split = (normed.len() as f64 * 0.8) as usize;
    let train_data = &normed[..split];
    let train_labels = &aligned_labels[..split];
    let test_data = &normed[split..];
    let test_labels = &aligned_labels[split..];
    println!(
        "  Training: {} samples, Testing: {} samples (80/20 split)",
        train_data.len(),
        test_data.len()
    );
    println!();

    // ── Step 6: Configure VQC ─────────────────────────────────────────────
    // Use 2 qubits (first 2 features) for tractable classical simulation
    let n_qubits = 2;
    let n_layers = 2;
    let learning_rate = 0.15;
    let epochs = 50;

    // Truncate features to n_qubits dimensions
    let train_2q: Vec<Vec<f64>> = train_data.iter().map(|f| f[..n_qubits].to_vec()).collect();
    let test_2q: Vec<Vec<f64>> = test_data.iter().map(|f| f[..n_qubits].to_vec()).collect();

    println!("Step 6: Training VQC...");
    println!("  Qubits: {}", n_qubits);
    println!("  Variational layers: {}", n_layers);
    println!(
        "  Parameters: {} (3 rotations x {} qubits x {} layers)",
        total_params(n_qubits, n_layers),
        n_qubits,
        n_layers
    );
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", epochs);
    println!();

    let mut model = VQCModel::new_identity_init(n_qubits, n_layers, learning_rate, 42);
    model.train(&train_2q, train_labels, epochs);
    println!();

    // ── Step 7: Evaluate on training set ──────────────────────────────────
    println!("Step 7: Evaluating on training set...");
    let train_preds = model.predict_batch(&train_2q, 0.0);
    let train_acc = accuracy(&train_preds, train_labels);
    println!("  Training accuracy: {:.2}%", train_acc);
    print_confusion_matrix(&train_preds, train_labels);
    println!();

    // ── Step 8: Evaluate on test set ──────────────────────────────────────
    println!("Step 8: Evaluating on test set...");
    let test_preds = model.predict_batch(&test_2q, 0.0);
    let test_acc = accuracy(&test_preds, test_labels);
    println!("  Test accuracy: {:.2}%", test_acc);
    print_confusion_matrix(&test_preds, test_labels);
    println!();

    // ── Step 9: Sample predictions ────────────────────────────────────────
    println!("Step 9: Sample predictions (last 10 test points)...");
    let start = if test_2q.len() > 10 { test_2q.len() - 10 } else { 0 };
    println!("  {:>4}  {:>8}  {:>8}  {:>10}", "Idx", "Actual", "Pred", "Raw Score");
    for i in start..test_2q.len() {
        let raw = model.predict_raw(&test_2q[i]);
        let pred = if raw > 0.0 { "BULL" } else { "BEAR" };
        let actual = if test_labels[i] > 0.0 { "BULL" } else { "BEAR" };
        println!(
            "  {:>4}  {:>8}  {:>8}  {:>+10.4}",
            i, actual, pred, raw
        );
    }
    println!();

    // ── Step 10: VQC analysis ─────────────────────────────────────────────
    println!("Step 10: VQC circuit analysis...");
    println!("  Trained parameters:");
    for (i, &p) in model.params.iter().enumerate() {
        let layer = i / params_per_layer(n_qubits);
        let local = i % params_per_layer(n_qubits);
        let qubit = local / 3;
        let gate_names = ["Rz₁", "Ry", "Rz₂"];
        let gate = gate_names[local % 3];
        println!(
            "    Layer {}, Qubit {}, {}: {:.4} rad ({:.1}°)",
            layer, qubit, gate, p, p.to_degrees()
        );
    }
    println!();

    // ── Step 11: Training convergence ─────────────────────────────────────
    println!("Step 11: Training convergence...");
    if model.training_costs.len() >= 2 {
        println!(
            "  Initial cost: {:.6}",
            model.training_costs[0]
        );
        println!(
            "  Final cost:   {:.6}",
            model.training_costs.last().unwrap()
        );
        let reduction = (1.0 - model.training_costs.last().unwrap() / model.training_costs[0]) * 100.0;
        println!("  Cost reduction: {:.1}%", reduction);
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  VQC Trading Pipeline Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
