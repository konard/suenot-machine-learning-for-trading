//! # One-Shot NAS Trading Example
//!
//! Fetches BTCUSDT data from Bybit, builds and trains a supernet,
//! samples and evaluates 50 random subnets, selects top 3,
//! retrains them standalone, and compares performance.

use one_shot_nas::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== One-Shot NAS for Trading ===\n");

    // ─── Step 1: Fetch market data from Bybit ───
    println!("[1] Fetching BTCUSDT 15-minute candles from Bybit...");
    let candles = fetch_bybit_klines("BTCUSDT", "15", 1000).await?;
    println!("    Fetched {} candles", candles.len());
    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        println!(
            "    Time range: {} -> {}",
            first.timestamp, last.timestamp
        );
        println!(
            "    Price range: {:.2} -> {:.2}",
            first.close, last.close
        );
    }

    // ─── Step 2: Build features ───
    println!("\n[2] Building features...");
    let (features, targets) = build_features(&candles);
    println!(
        "    Feature matrix: {} samples x {} features",
        features.nrows(),
        features.ncols()
    );

    if features.nrows() < 20 {
        println!("    ERROR: Not enough data to train. Need at least 60 candles.");
        return Ok(());
    }

    // Train/val split (70/30)
    let (train_x, train_y, val_x, val_y) = train_val_split(&features, &targets, 0.7);
    println!(
        "    Train: {} samples, Val: {} samples",
        train_x.nrows(),
        val_x.nrows()
    );

    // ─── Step 3: Define search space ───
    println!("\n[3] Defining search space...");
    let search_space = SearchSpace::default();
    println!("    Total architectures in search space: {}", search_space.size());
    println!("    Layer types: {:?}", search_space.layer_types);
    println!("    Hidden dims: {:?}", search_space.hidden_dims);
    println!("    Num layers: {:?}", search_space.num_layers_options);
    println!("    Activations: {:?}", search_space.activations);
    println!("    Dropout rates: {:?}", search_space.dropout_rates);

    // ─── Step 4: Train supernet ───
    println!("\n[4] Training supernet (500 steps with path sampling)...");
    let input_dim = features.ncols();
    let mut supernet = Supernet::new(input_dim, search_space);
    let losses = supernet.train(&train_x, &train_y, 500, 0.01);

    // Print loss at intervals
    for &i in &[0, 49, 99, 199, 299, 399, 499] {
        if i < losses.len() {
            println!("    Step {:>3}: loss = {:.8}", i + 1, losses[i]);
        }
    }

    // ─── Step 5: Evaluate 50 random subnets ───
    println!("\n[5] Evaluating 50 random subnets on validation data...");
    let results = supernet.evaluate_random_subnets(50, &val_x, &val_y);

    println!("    Top 10 subnets by validation MSE:");
    for (rank, (arch, mse)) in results.iter().take(10).enumerate() {
        println!("      #{:>2}: MSE={:.8}  {}", rank + 1, mse, arch);
    }

    println!("\n    Bottom 5 subnets (worst):");
    for (arch, mse) in results.iter().rev().take(5) {
        println!("      MSE={:.8}  {}", mse, arch);
    }

    // ─── Step 6: Select top 3 and retrain standalone ───
    let top_k = 3;
    let top_archs = Supernet::select_top_k(&results, top_k);
    println!("\n[6] Retraining top {} architectures standalone...", top_k);

    let mut standalone_results = Vec::new();

    for (i, arch) in top_archs.iter().enumerate() {
        println!("\n    --- Architecture #{} ---", i + 1);
        println!("    Config: {}", arch);

        let mut model = StandaloneModel::new(input_dim, arch.clone());
        let sa_losses = model.train(&train_x, &train_y, 300, 0.01);

        let train_mse = model.evaluate(&train_x, &train_y);
        let val_mse = model.evaluate(&val_x, &val_y);

        let val_pred = model.forward(&val_x);
        let dir_acc = directional_accuracy(&val_pred, &val_y);

        println!("    Train MSE: {:.8}", train_mse);
        println!("    Val MSE:   {:.8}", val_mse);
        println!("    Directional accuracy: {:.2}%", dir_acc * 100.0);
        println!(
            "    Loss trajectory: {:.8} -> {:.8}",
            sa_losses.first().unwrap_or(&0.0),
            sa_losses.last().unwrap_or(&0.0)
        );

        standalone_results.push((arch.clone(), val_mse, dir_acc));
    }

    // ─── Step 7: Compare supernet-ranked vs standalone performance ───
    println!("\n[7] Comparison: Supernet ranking vs. Standalone performance\n");
    println!(
        "    {:<5} {:<45} {:<15} {:<15} {:<12}",
        "Rank", "Architecture", "Supernet MSE", "Standalone MSE", "Dir. Acc."
    );
    println!("    {}", "-".repeat(92));

    for (i, (arch, sa_mse, dir_acc)) in standalone_results.iter().enumerate() {
        let supernet_mse = results[i].1;
        println!(
            "    {:<5} {:<45} {:<15.8} {:<15.8} {:.2}%",
            i + 1,
            format!("{}", arch),
            supernet_mse,
            sa_mse,
            dir_acc * 100.0
        );
    }

    // ─── Summary ───
    println!("\n=== Summary ===");
    println!(
        "  Search space size: {} architectures",
        supernet.search_space.size()
    );
    println!("  Subnets evaluated: 50");
    println!("  Top architectures retrained: {}", top_k);

    if let Some((best_arch, best_mse, best_acc)) = standalone_results
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
        println!("\n  Best standalone model:");
        println!("    Architecture: {}", best_arch);
        println!("    Validation MSE: {:.8}", best_mse);
        println!("    Directional accuracy: {:.2}%", best_acc * 100.0);
    }

    println!("\n  One-Shot NAS evaluated 50 architectures using shared weights");
    println!("  instead of training each one from scratch, saving ~47x compute.");

    Ok(())
}
