use pruning_trading_models::*;

fn main() {
    println!("=======================================================");
    println!("  Chapter 203: Pruning Trading Models");
    println!("  Trading Example with Bybit BTCUSDT Data");
    println!("=======================================================\n");

    // -------------------------------------------------------------------------
    // Step 1: Fetch data from Bybit (fall back to synthetic if unavailable)
    // -------------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT data from Bybit...");
    let candles = match fetch_bybit_candles("BTCUSDT", "5", 200) {
        Ok(c) => {
            println!("    Fetched {} candles from Bybit API", c.len());
            println!(
                "    Price range: {:.2} - {:.2}",
                c.iter().map(|x| x.low).fold(f64::MAX, f64::min),
                c.iter().map(|x| x.high).fold(f64::MIN, f64::max)
            );
            c
        }
        Err(e) => {
            println!("    Could not fetch from Bybit: {}", e);
            println!("    Using synthetic data instead.");
            generate_synthetic_candles(200)
        }
    };

    // -------------------------------------------------------------------------
    // Step 2: Prepare features and labels
    // -------------------------------------------------------------------------
    println!("\n[2] Preparing features and labels...");
    let (features, labels) = prepare_features(&candles);
    println!("    Samples: {}", features.nrows());
    println!("    Features per sample: {}", features.ncols());
    let positive_ratio = labels.iter().filter(|&&l| l == 1.0).count() as f64 / labels.len() as f64;
    println!("    Positive label ratio: {:.1}%", positive_ratio * 100.0);

    // -------------------------------------------------------------------------
    // Step 3: Create and train dense model
    // -------------------------------------------------------------------------
    println!("\n[3] Training dense model (4 -> 32 -> 16 -> 1)...");
    let mut dense_model = PrunableNetwork::new(&[4, 32, 16, 1]);

    let initial_acc = dense_model.accuracy(&features, &labels);
    println!("    Initial accuracy: {:.1}%", initial_acc * 100.0);

    dense_model.train(&features, &labels, 20, 0.01);

    let trained_acc = dense_model.accuracy(&features, &labels);
    let trained_loss = dense_model.loss(&features, &labels);
    println!("    Trained accuracy: {:.1}%", trained_acc * 100.0);
    println!("    Trained loss:     {:.4}", trained_loss);

    let report = dense_model.sparsity_report();
    println!("\n    Dense model report:");
    println!("    Total weights: {}", report.total_weights);
    println!("    Sparsity: {:.1}%", report.sparsity * 100.0);

    // -------------------------------------------------------------------------
    // Step 4: One-shot pruning at various sparsity levels
    // -------------------------------------------------------------------------
    println!("\n[4] One-shot magnitude pruning at various sparsity levels...");
    println!("    {:<12} {:<12} {:<15} {:<12}", "Sparsity", "Accuracy", "Active Weights", "Compression");
    println!("    {}", "-".repeat(51));

    let sparsity_levels = vec![0.0, 0.50, 0.75, 0.90, 0.95];
    let results = dense_model.evaluate_sparsity_levels(&features, &labels, &sparsity_levels);

    for r in &results {
        let report_clone = {
            let mut m = dense_model.clone();
            m.magnitude_prune(r.sparsity);
            m.sparsity_report()
        };
        println!(
            "    {:<12.1}% {:<12.1}% {:<15} {:.2}x",
            r.sparsity * 100.0,
            r.accuracy * 100.0,
            format!("{}/{}", r.active_weights, r.total_weights),
            report_clone.compression_ratio
        );
    }

    // -------------------------------------------------------------------------
    // Step 5: Iterative magnitude pruning (IMP)
    // -------------------------------------------------------------------------
    println!("\n[5] Iterative Magnitude Pruning (IMP)...");
    let mut imp_model = dense_model.clone();
    let imp_results = imp_model.iterative_prune(
        &features,
        &labels,
        6,    // 6 pruning steps
        0.3,  // Remove 30% of remaining weights each step
        5,    // Retrain for 5 epochs
        0.005,
    );

    println!("\n    IMP Results Summary:");
    println!("    {:<12} {:<12} {:<15}", "Sparsity", "Accuracy", "Active Weights");
    println!("    {}", "-".repeat(39));
    for r in &imp_results {
        println!(
            "    {:<12.1}% {:<12.1}% {}",
            r.sparsity * 100.0,
            r.accuracy * 100.0,
            r.active_weights
        );
    }

    // -------------------------------------------------------------------------
    // Step 6: Structured pruning
    // -------------------------------------------------------------------------
    println!("\n[6] Structured pruning (neuron removal)...");
    let mut struct_model = dense_model.clone();

    println!("    Before structured pruning:");
    let acc_before = struct_model.accuracy(&features, &labels);
    println!("    Accuracy: {:.1}%", acc_before * 100.0);
    println!("    {}", struct_model.sparsity_report());

    // Remove 16 of 32 neurons from layer 0, and 8 of 16 from layer 1
    struct_model.structured_prune(0, 16);
    struct_model.structured_prune(1, 8);

    println!("    After structured pruning (removed 16/32 + 8/16 neurons):");
    let acc_after = struct_model.accuracy(&features, &labels);
    println!("    Accuracy: {:.1}%", acc_after * 100.0);
    println!("    {}", struct_model.sparsity_report());

    // -------------------------------------------------------------------------
    // Step 7: Inference speed comparison
    // -------------------------------------------------------------------------
    println!("\n[7] Inference speed comparison...");
    let single_input = features.slice(ndarray::s![0..1, ..]).to_owned();
    let iterations = 10000;

    let dense_us = dense_model.measure_inference_us(&single_input, iterations);
    println!("    Dense model:  {:.2} us/inference", dense_us);

    // Pruned at 90% sparsity
    let mut fast_model = dense_model.clone();
    fast_model.magnitude_prune(0.9);
    let pruned_us = fast_model.measure_inference_us(&single_input, iterations);
    println!("    Pruned (90%): {:.2} us/inference", pruned_us);

    // Structured pruned
    let struct_us = struct_model.measure_inference_us(&single_input, iterations);
    println!("    Structured:   {:.2} us/inference", struct_us);

    if dense_us > 0.0 {
        println!("\n    Speedup (pruned vs dense):     {:.2}x", dense_us / pruned_us.max(0.01));
        println!("    Speedup (structured vs dense): {:.2}x", dense_us / struct_us.max(0.01));
    }

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("\n=======================================================");
    println!("  Summary");
    println!("=======================================================");
    println!("  - Dense model: {} weights, {:.1}% accuracy",
        report.total_weights, trained_acc * 100.0);
    if let Some(last_imp) = imp_results.last() {
        println!("  - IMP final:   {} active weights ({:.1}% sparsity), {:.1}% accuracy",
            last_imp.active_weights, last_imp.sparsity * 100.0, last_imp.accuracy * 100.0);
    }
    println!("  - Structured:  {:.1}% sparsity, {:.1}% accuracy",
        struct_model.sparsity_report().sparsity * 100.0, acc_after * 100.0);
    println!("  - Key insight: pruning can maintain accuracy while");
    println!("    dramatically reducing model size and inference latency.");
    println!("=======================================================\n");
}
