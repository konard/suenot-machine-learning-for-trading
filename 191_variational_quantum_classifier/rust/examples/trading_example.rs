//! Trading example: Train a Variational Quantum Classifier to predict
//! BTCUSDT price direction using data from Bybit.

use variational_quantum_classifier::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Variational Quantum Classifier for Trading ===\n");

    // ── Fetch data from Bybit ──────────────────────────────
    println!("Fetching BTCUSDT 15m candles from Bybit...");
    let candles = fetch_bybit_klines("BTCUSDT", "15", 500).await?;
    println!("Fetched {} candles", candles.len());
    println!(
        "Date range: {} -> {}",
        candles.first().unwrap().timestamp,
        candles.last().unwrap().timestamp
    );
    println!(
        "Price range: {:.2} -> {:.2}\n",
        candles.first().unwrap().close,
        candles.last().unwrap().close
    );

    // ── Feature engineering ────────────────────────────────
    println!("Engineering features...");
    let (raw_features, labels) = engineer_features(&candles);
    println!("Generated {} samples with {} features each", raw_features.len(), 4);

    // Scale features to [-pi, pi] for angle encoding
    let features = scale_features(&raw_features);

    // Count label distribution
    let n_up: usize = labels.iter().filter(|&&l| l == 1.0).count();
    let n_down = labels.len() - n_up;
    println!("Label distribution: UP={}, DOWN={}", n_up, n_down);

    // ── Train/test split ───────────────────────────────────
    let (train_features, train_labels, test_features, test_labels) =
        train_test_split(&features, &labels, 0.8);
    println!(
        "Train: {} samples, Test: {} samples\n",
        train_features.len(),
        test_features.len()
    );

    // ── Create and train VQC ───────────────────────────────
    let num_qubits = 4; // one per feature
    let num_layers = 2;
    let vqc = VQC::new(num_qubits, num_layers);
    println!(
        "VQC: {} qubits, {} layers, {} parameters",
        num_qubits,
        num_layers,
        vqc.num_params()
    );

    let initial_params = vqc.init_params();
    let learning_rate = 0.3;
    let num_epochs = 50;

    // Use a subset for training if dataset is large (VQC is slow on large batches)
    let max_train = 50.min(train_features.len());
    let train_subset: Vec<Vec<f64>> = train_features[..max_train].to_vec();
    let labels_subset: Vec<f64> = train_labels[..max_train].to_vec();

    println!(
        "Training on {} samples for {} epochs (lr={})...\n",
        max_train, num_epochs, learning_rate
    );

    let (final_params, loss_history) = vqc.train(
        &train_subset,
        &labels_subset,
        &initial_params,
        learning_rate,
        num_epochs,
    );

    // ── Evaluate ───────────────────────────────────────────
    println!("\n=== Results ===");
    println!(
        "Initial loss: {:.6}",
        loss_history.first().unwrap_or(&0.0)
    );
    println!(
        "Final loss:   {:.6}",
        loss_history.last().unwrap_or(&0.0)
    );

    let train_acc = vqc.accuracy(&train_subset, &labels_subset, &final_params);
    println!("Train accuracy: {:.2}%", train_acc * 100.0);

    if !test_features.is_empty() {
        let max_test = 50.min(test_features.len());
        let test_subset: Vec<Vec<f64>> = test_features[..max_test].to_vec();
        let test_labels_subset: Vec<f64> = test_labels[..max_test].to_vec();
        let test_acc = vqc.accuracy(&test_subset, &test_labels_subset, &final_params);
        println!("Test accuracy:  {:.2}%", test_acc * 100.0);

        // Show some predictions
        println!("\n=== Sample Predictions (first 10 test samples) ===");
        let n_show = 10.min(test_subset.len());
        let probas = vqc.predict_proba(&test_subset[..n_show].to_vec(), &final_params);
        let preds = vqc.predict(&test_subset[..n_show].to_vec(), &final_params);

        println!("{:<8} {:<10} {:<10} {:<8}", "Sample", "P(UP)", "Pred", "Actual");
        println!("{}", "-".repeat(40));
        for i in 0..n_show {
            let actual = if test_labels_subset[i] == 1.0 {
                "UP"
            } else {
                "DOWN"
            };
            let pred = if preds[i] == 1 { "UP" } else { "DOWN" };
            println!(
                "{:<8} {:<10.4} {:<10} {:<8}",
                i + 1,
                probas[i],
                pred,
                actual
            );
        }
    }

    println!("\n=== Final Parameters ===");
    for (i, p) in final_params.iter().enumerate() {
        print!("{:.4} ", p);
        if (i + 1) % 8 == 0 {
            println!();
        }
    }
    println!();

    Ok(())
}
