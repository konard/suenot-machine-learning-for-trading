use lobframe_benchmark::*;

fn main() {
    println!("LOBFrame Benchmark Trading Example");
    println!("===================================\n");

    // --- Step 1: Try to fetch live data from Bybit ---
    println!("Step 1: Fetching BTCUSDT orderbook from Bybit...");
    let live_snapshot = match fetch_bybit_orderbook("BTCUSDT", 50) {
        Ok(snap) => {
            println!(
                "  Fetched orderbook: {} bids, {} asks",
                snap.bids.len(),
                snap.asks.len()
            );
            if let Some(mid) = snap.mid_price() {
                println!("  Mid-price: {:.2}", mid);
            }
            if let Some(spread) = snap.spread() {
                println!("  Spread: {:.2}", spread);
            }
            if let Some(oi) = snap.order_imbalance(0) {
                println!("  Order imbalance (level 0): {:.4}", oi);
            }
            println!();
            Some(snap)
        }
        Err(e) => {
            println!("  Could not fetch live data: {}", e);
            println!("  Continuing with synthetic data only.\n");
            None
        }
    };

    // --- Step 2: Generate synthetic data for benchmarking ---
    println!("Step 2: Generating synthetic LOB data...");
    let levels = 10;
    let base_price = if let Some(ref snap) = live_snapshot {
        snap.mid_price().unwrap_or(50000.0)
    } else {
        50000.0
    };

    let train_snapshots = generate_synthetic_snapshots(500, levels, base_price, 42);
    let test_snapshots = generate_synthetic_snapshots(200, levels, base_price, 99);
    println!(
        "  Training samples: {}, Test samples: {}",
        train_snapshots.len(),
        test_snapshots.len()
    );

    // --- Step 3: Demonstrate normalization ---
    println!("\nStep 3: Normalizing features...");
    let mut normalizer = LobNormalizer::new(5.0);
    let runner_temp = BenchmarkRunner::new(levels, 5, 0.0001);
    let train_features = runner_temp.prepare_features(&train_snapshots);
    let normalized = normalizer.fit_transform(&train_features);
    println!(
        "  Feature matrix shape: {}x{}",
        normalized.nrows(),
        normalized.ncols()
    );
    println!(
        "  Normalized mean (feature 0): {:.6}",
        normalized.column(0).mean().unwrap_or(0.0)
    );
    println!(
        "  Normalized std  (feature 0): {:.6}",
        normalized.column(0).mapv(|x| x * x).mean().unwrap_or(0.0).sqrt()
    );

    // --- Step 4: Construct labels ---
    println!("\nStep 4: Constructing direction labels...");
    let test_mid_prices: Vec<f64> = test_snapshots
        .iter()
        .filter_map(|s| s.mid_price())
        .collect();
    let labels = construct_labels(&test_mid_prices, 5, 0.0001);

    let n_up = labels.iter().filter(|l| **l == Direction::Up).count();
    let n_flat = labels.iter().filter(|l| **l == Direction::Flat).count();
    let n_down = labels.iter().filter(|l| **l == Direction::Down).count();
    println!(
        "  Label distribution: Up={}, Flat={}, Down={} (total={})",
        n_up,
        n_flat,
        n_down,
        labels.len()
    );

    // --- Step 5: Create models ---
    println!("\nStep 5: Creating benchmark models...");
    let input_size = 4 * levels; // bid_prices + bid_vols + ask_prices + ask_vols
    let lstm = BaselineLstm::new(input_size, 64, 42);
    let cnn = BaselineCnn::new(3, 42);
    let random = RandomBaseline::new(42);
    println!("  Models: LSTM, CNN, Random Baseline");

    // --- Step 6: Run benchmark ---
    println!("\nStep 6: Running LOBFrame benchmark...");
    let mut runner = BenchmarkRunner::new(levels, 5, 0.0001);
    let results = runner.run_all(
        &[
            &lstm as &dyn BenchmarkModel,
            &cnn as &dyn BenchmarkModel,
            &random as &dyn BenchmarkModel,
        ],
        &train_snapshots,
        &test_snapshots,
    );

    // --- Step 7: Print results ---
    print_results_table(&results);

    // --- Step 8: Detailed analysis ---
    println!("Detailed Analysis:");
    println!("{:-<60}", "");
    for r in &results {
        println!("\n  {} (N={}):", r.model_name, r.num_samples);
        println!("    Accuracy:  {:.4}", r.metrics.accuracy);
        println!("    F1 Macro:  {:.4}", r.metrics.f1_macro);
        println!("    MCC:       {:.4}", r.metrics.mcc);
        println!("    Per-class precision: Down={:.4}, Flat={:.4}, Up={:.4}",
            r.metrics.per_class_precision[0],
            r.metrics.per_class_precision[1],
            r.metrics.per_class_precision[2],
        );
        println!("    Per-class recall:    Down={:.4}, Flat={:.4}, Up={:.4}",
            r.metrics.per_class_recall[0],
            r.metrics.per_class_recall[1],
            r.metrics.per_class_recall[2],
        );
    }

    // --- Step 9: Show live data features if available ---
    if let Some(ref snap) = live_snapshot {
        println!("\n\nLive Bybit BTCUSDT Orderbook Features:");
        println!("{:-<60}", "");
        let features = snap.to_feature_vector(5);
        println!("  Feature vector (5 levels): {:?}", features.as_slice().unwrap());

        // Normalize using fitted normalizer
        let live_features = ndarray::Array2::from_shape_vec(
            (1, features.len()),
            features.to_vec(),
        ).unwrap();

        // Pad or truncate to match training feature size
        let n_cols = normalized.ncols();
        let mut padded = ndarray::Array2::zeros((1, n_cols));
        let copy_len = features.len().min(n_cols);
        for j in 0..copy_len {
            padded[[0, j]] = features[j];
        }
        let live_norm = normalizer.transform(&padded);
        println!("  Normalized features: {:?}", live_norm.row(0).as_slice().unwrap());
    }

    println!("\nBenchmark complete.");
}
