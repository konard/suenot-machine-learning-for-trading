use hybrid_quantum_classical::{
    compute_features, generate_synthetic_candles, BybitClient, ClassicalModel, HybridModel,
};

fn main() {
    println!("=== Hybrid Quantum-Classical Trading Example ===\n");

    // Try fetching real data from Bybit, fall back to synthetic
    let candles = match BybitClient::new().fetch_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("Fetched {} candles from Bybit (BTCUSDT, 1h)", c.len());
            c
        }
        Err(e) => {
            println!("Could not fetch from Bybit ({}), using synthetic data", e);
            generate_synthetic_candles(200)
        }
    };

    // Feature engineering
    let (features, labels) = compute_features(&candles);
    println!(
        "Computed {} samples with {} features each",
        features.len(),
        features[0].len()
    );

    let up_count = labels.iter().filter(|&&l| l == 1.0).count();
    println!(
        "Label distribution: {:.1}% up, {:.1}% down\n",
        up_count as f64 / labels.len() as f64 * 100.0,
        (labels.len() - up_count) as f64 / labels.len() as f64 * 100.0
    );

    // Split into train/test (80/20)
    let split_idx = (features.len() as f64 * 0.8) as usize;
    let train_features = &features[..split_idx];
    let train_labels = &labels[..split_idx];
    let test_features = &features[split_idx..];
    let test_labels = &labels[split_idx..];

    println!(
        "Train size: {}, Test size: {}\n",
        train_features.len(),
        test_features.len()
    );

    // --- Train Hybrid Model ---
    println!("--- Training Hybrid Quantum-Classical Model ---");
    println!("Architecture: Classical Preprocessing -> 4-qubit Quantum Circuit (2 layers) -> Classical Postprocessing\n");

    let mut hybrid_model = HybridModel::new(4, 4, 2);
    hybrid_model.train(train_features, train_labels, 50);

    let hybrid_train_acc = hybrid_model.evaluate_accuracy(train_features, train_labels);
    let hybrid_test_acc = hybrid_model.evaluate_accuracy(test_features, test_labels);

    println!("\nHybrid Model Results:");
    println!("  Train accuracy: {:.2}%", hybrid_train_acc * 100.0);
    println!("  Test accuracy:  {:.2}%\n", hybrid_test_acc * 100.0);

    // --- Train Pure Classical Model ---
    println!("--- Training Pure Classical Model (Logistic Regression) ---\n");

    let mut classical_model = ClassicalModel::new(4);
    classical_model.train(train_features, train_labels, 50);

    let classical_train_acc = classical_model.evaluate_accuracy(train_features, train_labels);
    let classical_test_acc = classical_model.evaluate_accuracy(test_features, test_labels);

    println!("\nClassical Model Results:");
    println!("  Train accuracy: {:.2}%", classical_train_acc * 100.0);
    println!("  Test accuracy:  {:.2}%\n", classical_test_acc * 100.0);

    // --- Comparison ---
    println!("=== Model Comparison ===");
    println!(
        "Hybrid  - Train: {:.2}%, Test: {:.2}%",
        hybrid_train_acc * 100.0,
        hybrid_test_acc * 100.0
    );
    println!(
        "Classical - Train: {:.2}%, Test: {:.2}%",
        classical_train_acc * 100.0,
        classical_test_acc * 100.0
    );

    let diff = hybrid_test_acc - classical_test_acc;
    if diff > 0.0 {
        println!(
            "\nHybrid model outperforms classical by {:.2} percentage points on test set",
            diff * 100.0
        );
    } else if diff < 0.0 {
        println!(
            "\nClassical model outperforms hybrid by {:.2} percentage points on test set",
            -diff * 100.0
        );
    } else {
        println!("\nBoth models perform equally on the test set");
    }

    // --- Sample predictions ---
    println!("\n=== Sample Predictions (first 5 test samples) ===");
    println!("{:<10} {:<15} {:<15} {:<10}", "Actual", "Hybrid Pred", "Classical Pred", "Direction");
    for i in 0..5.min(test_features.len()) {
        let hybrid_pred = hybrid_model.predict_single(&test_features[i]);
        let classical_pred = classical_model.predict_single(&test_features[i]);
        let direction = if test_labels[i] == 1.0 { "UP" } else { "DOWN" };
        println!(
            "{:<10} {:<15.4} {:<15.4} {:<10}",
            test_labels[i], hybrid_pred, classical_pred, direction
        );
    }

    println!("\n=== Done ===");
}
