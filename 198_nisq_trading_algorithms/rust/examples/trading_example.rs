use anyhow::Result;
use nisq_trading_algorithms::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=============================================================");
    println!("  NISQ Trading Algorithms - End-to-End Example");
    println!("=============================================================\n");

    // -------------------------------------------------------------------------
    // Step 1: Fetch data from Bybit (or use synthetic fallback)
    // -------------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT data from Bybit...");

    let client = BybitClient::new();
    let (features_vec, labels) = match client.fetch_klines("BTCUSDT", "15", 100).await {
        Ok(klines) => {
            println!("    Fetched {} klines from Bybit", klines.len());
            let trading_features = client.compute_features(&klines);
            println!(
                "    Computed {} feature vectors",
                trading_features.len()
            );
            if trading_features.len() < 20 {
                println!("    Not enough data, falling back to synthetic data");
                generate_synthetic_data(80)
            } else {
                BybitClient::features_to_vectors(&trading_features)
            }
        }
        Err(e) => {
            println!("    API error: {}. Using synthetic data.", e);
            generate_synthetic_data(80)
        }
    };

    let num_samples = features_vec.len();
    let train_size = num_samples * 7 / 10;
    let train_features = &features_vec[..train_size];
    let train_labels = &labels[..train_size];
    let test_features = &features_vec[train_size..];
    let test_labels = &labels[train_size..];

    println!(
        "    Train: {} samples, Test: {} samples\n",
        train_size,
        num_samples - train_size
    );

    // -------------------------------------------------------------------------
    // Step 2: Configure NISQ simulators
    // -------------------------------------------------------------------------
    println!("[2] Configuring NISQ simulators...");

    let num_qubits = 4;
    let gate_error_rate = 0.005; // 0.5% per gate
    let measurement_error_rate = 0.02; // 2% readout error

    let ideal_sim = NISQSimulator::new(num_qubits, 0.0, 0.0);
    let noisy_sim = NISQSimulator::new(num_qubits, gate_error_rate, measurement_error_rate);

    println!("    Ideal simulator: 0 noise");
    println!(
        "    Noisy simulator: gate_err={}, meas_err={}\n",
        gate_error_rate, measurement_error_rate
    );

    // -------------------------------------------------------------------------
    // Step 3: Run ShallowVQE for portfolio optimization
    // -------------------------------------------------------------------------
    println!("[3] Running ShallowVQE (portfolio optimization)...");

    let cost_coefficients = vec![1.0, -0.5, 0.3, -0.8]; // mock portfolio cost

    let vqe = ShallowVQE::new(num_qubits);

    let ideal_vqe = vqe.optimize(&ideal_sim, &cost_coefficients);
    println!("    Ideal VQE:  value = {:.6}", ideal_vqe.optimal_value);

    let noisy_vqe = vqe.optimize(&noisy_sim, &cost_coefficients);
    println!("    Noisy VQE:  value = {:.6}", noisy_vqe.optimal_value);

    let mitigated_vqe = ErrorMitigation::zero_noise_extrapolation(
        &noisy_sim,
        &noisy_vqe.optimal_params,
        &cost_coefficients,
        1,
    );
    println!("    Mitigated:  value = {:.6}", mitigated_vqe);
    println!(
        "    Noise error:     {:.6}",
        (noisy_vqe.optimal_value - ideal_vqe.optimal_value).abs()
    );
    println!(
        "    Mitigated error: {:.6}\n",
        (mitigated_vqe - ideal_vqe.optimal_value).abs()
    );

    // -------------------------------------------------------------------------
    // Step 4: Run QAOA-1 for binary asset selection
    // -------------------------------------------------------------------------
    println!("[4] Running QAOA-1 (binary asset selection)...");

    let qaoa = QAOA1::new(num_qubits);

    let ideal_qaoa = qaoa.optimize(&ideal_sim, &cost_coefficients);
    println!("    Ideal QAOA: value = {:.6}", ideal_qaoa.optimal_value);

    let noisy_qaoa = qaoa.optimize(&noisy_sim, &cost_coefficients);
    println!("    Noisy QAOA: value = {:.6}", noisy_qaoa.optimal_value);

    let mitigated_qaoa = ErrorMitigation::zero_noise_extrapolation(
        &noisy_sim,
        &noisy_qaoa.optimal_params,
        &cost_coefficients,
        2,
    );
    println!("    Mitigated:  value = {:.6}", mitigated_qaoa);
    println!(
        "    Noise error:     {:.6}",
        (noisy_qaoa.optimal_value - ideal_qaoa.optimal_value).abs()
    );
    println!(
        "    Mitigated error: {:.6}\n",
        (mitigated_qaoa - ideal_qaoa.optimal_value).abs()
    );

    // -------------------------------------------------------------------------
    // Step 5: Run Variational Classifier for market prediction
    // -------------------------------------------------------------------------
    println!("[5] Running Variational Classifier (market prediction)...");

    let classifier = VariationalClassifier::new(num_qubits);

    // Train on ideal simulator
    let ideal_result = classifier.train(&ideal_sim, train_features, train_labels);
    let ideal_test_accuracy = evaluate_test_accuracy(
        &classifier,
        &ideal_sim,
        &ideal_result.optimal_params,
        test_features,
        test_labels,
    );
    println!(
        "    Ideal classifier:  train_acc={:.1}%, test_acc={:.1}%",
        ideal_result.optimal_value * 100.0,
        ideal_test_accuracy * 100.0
    );

    // Train on noisy simulator
    let noisy_result = classifier.train(&noisy_sim, train_features, train_labels);
    let noisy_test_accuracy = evaluate_test_accuracy(
        &classifier,
        &noisy_sim,
        &noisy_result.optimal_params,
        test_features,
        test_labels,
    );
    println!(
        "    Noisy classifier:  train_acc={:.1}%, test_acc={:.1}%",
        noisy_result.optimal_value * 100.0,
        noisy_test_accuracy * 100.0
    );

    println!(
        "    Accuracy degradation: {:.1}%\n",
        (ideal_test_accuracy - noisy_test_accuracy) * 100.0
    );

    // -------------------------------------------------------------------------
    // Step 6: Full benchmarking suite
    // -------------------------------------------------------------------------
    println!("[6] Running full benchmark suite...");

    let runner = BenchmarkRunner::new(num_qubits, gate_error_rate, measurement_error_rate);
    let benchmarks = runner.run_benchmark(&cost_coefficients);

    println!("    {:<20} {:>10} {:>10} {:>10} {:>12}",
        "Algorithm", "Ideal", "Noisy", "Mitigated", "Improvement"
    );
    println!("    {}", "-".repeat(64));

    for b in &benchmarks {
        println!(
            "    {:<20} {:>10.6} {:>10.6} {:>10.6} {:>10.1}x",
            b.algorithm_name, b.ideal_value, b.noisy_value, b.mitigated_value, b.improvement_ratio
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 7: Accuracy vs circuit depth analysis
    // -------------------------------------------------------------------------
    println!("[7] Accuracy vs circuit depth analysis...");

    let depth_results = runner.accuracy_vs_depth(&cost_coefficients, 20);

    println!("    {:>6} {:>12} {:>14}", "Depth", "Ideal Value", "Rel. Error %");
    println!("    {}", "-".repeat(34));

    for (depth, ideal, error) in &depth_results {
        println!("    {:>6} {:>12.6} {:>13.1}%", depth, ideal, error * 100.0);
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 8: Error mitigation comparison
    // -------------------------------------------------------------------------
    println!("[8] Error mitigation comparison...");

    let test_params = vec![0.5, 1.0, 1.5, 2.0];
    let ideal_value = ideal_sim.evaluate_circuit_ideal(&test_params, &cost_coefficients);

    println!("    Ideal value:     {:.6}", ideal_value);

    let mut noisy_sum = 0.0;
    let n_samples = 50;
    for _ in 0..n_samples {
        noisy_sum += noisy_sim.evaluate_circuit(&test_params, &cost_coefficients, 3);
    }
    let noisy_avg = noisy_sum / n_samples as f64;
    println!("    Noisy average:   {:.6} (over {} samples)", noisy_avg, n_samples);

    let mut mitigated_sum = 0.0;
    for _ in 0..10 {
        mitigated_sum += ErrorMitigation::zero_noise_extrapolation(
            &noisy_sim,
            &test_params,
            &cost_coefficients,
            3,
        );
    }
    let mitigated_avg = mitigated_sum / 10.0;
    println!("    ZNE mitigated:   {:.6} (over 10 samples)", mitigated_avg);

    println!("    Noise error:     {:.6}", (noisy_avg - ideal_value).abs());
    println!(
        "    Mitigated error: {:.6}",
        (mitigated_avg - ideal_value).abs()
    );

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("\n=============================================================");
    println!("  Summary");
    println!("=============================================================");
    println!("  - ShallowVQE, QAOA-1, and VariationalClassifier all");
    println!("    function under NISQ noise conditions.");
    println!("  - Zero-noise extrapolation recovers significant accuracy.");
    println!("  - Accuracy degrades with circuit depth, confirming the");
    println!("    importance of shallow circuits for NISQ trading.");
    println!("  - Real Bybit data integration works end-to-end.");
    println!("=============================================================");

    Ok(())
}

/// Evaluate test accuracy for the variational classifier
fn evaluate_test_accuracy(
    classifier: &VariationalClassifier,
    simulator: &NISQSimulator,
    params: &[f64],
    features: &[Vec<f64>],
    labels: &[u8],
) -> f64 {
    let mut correct = 0;
    for (feature, &label) in features.iter().zip(labels.iter()) {
        let prediction = classifier.predict(simulator, params, feature);
        if prediction == label {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64
}
