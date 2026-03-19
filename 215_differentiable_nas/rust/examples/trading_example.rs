//! Trading Example: DARTS Architecture Search with Bybit Data
//!
//! This example demonstrates:
//! 1. Fetching BTCUSDT kline data from Bybit
//! 2. Defining a trading-specific operation search space
//! 3. Running DARTS architecture search
//! 4. Extracting and evaluating the discovered architecture
//! 5. Comparing against a hand-designed baseline

use differentiable_nas::{
    candles_to_features, create_samples, darts_search, evaluate_architecture,
    fetch_bybit_klines, train_baseline, train_val_split, DARTSConfig, OpType,
};
use ndarray::Array1;

fn generate_synthetic_data(n: usize, feature_dim: usize) -> Vec<(Array1<f64>, Array1<f64>)> {
    let mut samples = Vec::new();
    for i in 0..n {
        let t = i as f64 * 0.1;
        let x = Array1::from(
            (0..feature_dim)
                .map(|f| (t + f as f64 * 0.3).sin() * 0.5 + 0.5)
                .collect::<Vec<f64>>(),
        );
        let y = Array1::from(
            (0..feature_dim)
                .map(|f| ((t + 0.1) + f as f64 * 0.3).sin() * 0.5 + 0.5)
                .collect::<Vec<f64>>(),
        );
        samples.push((x, y));
    }
    samples
}

fn main() {
    println!("==============================================");
    println!(" DARTS Architecture Search for Trading");
    println!("==============================================\n");

    // --- Step 1: Fetch data from Bybit ---
    println!("[1] Fetching BTCUSDT data from Bybit...");
    let (train_data, val_data, test_data, feature_dim) = match fetch_bybit_klines("BTCUSDT", "15", 200) {
        Ok(candles) => {
            println!("    Fetched {} candles from Bybit", candles.len());
            let features = candles_to_features(&candles);
            let samples = create_samples(&features);
            println!("    Created {} samples (input->target pairs)", samples.len());

            let (train_val, test) = train_val_split(&samples, 0.8);
            let (train, val) = train_val_split(&train_val, 0.75);
            println!(
                "    Split: {} train, {} val, {} test",
                train.len(),
                val.len(),
                test.len()
            );
            (train, val, test, 5usize)
        }
        Err(e) => {
            println!("    Warning: Could not fetch from Bybit: {}", e);
            println!("    Using synthetic trading data instead.\n");
            let feature_dim = 5;
            let all_samples = generate_synthetic_data(200, feature_dim);
            let (train_val, test) = train_val_split(&all_samples, 0.8);
            let (train, val) = train_val_split(&train_val, 0.75);
            (train, val, test, feature_dim)
        }
    };

    // --- Step 2: Define search space ---
    println!("\n[2] Defining operation search space...");
    let op_types = OpType::all();
    for op in &op_types {
        println!("    - {}", op.name());
    }

    // --- Step 3: Run DARTS search ---
    println!("\n[3] Running DARTS architecture search...");
    let config = DARTSConfig {
        num_intermediate_nodes: 2,
        feature_dim,
        search_epochs: 50,
        weight_lr: 0.005,
        arch_lr: 0.01,
        op_types: op_types.clone(),
    };

    let result = darts_search(&config, &train_data, &val_data);

    // --- Step 4: Extract and evaluate architecture ---
    println!("\n[4] Discovered architecture:");
    result.cell.print_architecture();

    let darts_test_loss = evaluate_architecture(&result.cell, &test_data);
    println!("\n    DARTS test loss (MSE): {:.6}", darts_test_loss);

    // --- Step 5: Compare with baseline ---
    println!("\n[5] Training hand-designed baseline (single dense layer)...");
    let baseline_test_loss = train_baseline(
        feature_dim,
        &train_data,
        &test_data,
        config.search_epochs * 10, // give baseline more training steps
        config.weight_lr,
    );
    println!("    Baseline test loss (MSE): {:.6}", baseline_test_loss);

    // --- Summary ---
    println!("\n==============================================");
    println!(" Results Summary");
    println!("==============================================");
    println!("  DARTS searched architecture MSE:  {:.6}", darts_test_loss);
    println!("  Hand-designed baseline MSE:       {:.6}", baseline_test_loss);
    if darts_test_loss < baseline_test_loss {
        println!("  >> DARTS architecture outperforms the baseline!");
    } else {
        println!("  >> Baseline performs similarly or better (try more search epochs).");
    }

    println!("\n  Architecture operations selected:");
    for node_edges in &result.architecture {
        for (i, j, op_type) in node_edges {
            println!("    Edge ({} -> {}): {}", i, j, op_type.name());
        }
    }

    println!("\n  Training loss trajectory (first 5 / last 5):");
    let n = result.train_losses.len();
    for i in 0..5.min(n) {
        println!("    Epoch {:>3}: {:.6}", i + 1, result.train_losses[i]);
    }
    if n > 5 {
        println!("    ...");
        for i in (n - 5)..n {
            println!("    Epoch {:>3}: {:.6}", i + 1, result.train_losses[i]);
        }
    }

    println!("\nDone.");
}
