use tick_data_forecasting::*;
use ndarray::Array1;
use rand::Rng;

fn main() {
    println!("=== Tick Data Forecasting for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Load or generate tick data
    // -----------------------------------------------------------------------
    println!("Step 1: Loading tick data...");
    let ticks = match load_bybit_ticks() {
        Ok(t) => {
            println!("  Loaded {} ticks from Bybit (BTCUSDT 1m -> synthetic ticks)\n", t.len());
            t
        }
        Err(e) => {
            println!("  Bybit fetch failed ({}), generating synthetic ticks\n", e);
            generate_synthetic_ticks(2000, 5.0, 1.5, 3.0, 50000.0, 0.0005)
        }
    };

    println!("  Total ticks: {}", ticks.len());
    if let (Some(first), Some(last)) = (ticks.first(), ticks.last()) {
        println!("  Price range: {:.2} - {:.2}", first.price, last.price);
        println!(
            "  Time span: {:.1}s",
            last.timestamp - first.timestamp
        );
    }

    // -----------------------------------------------------------------------
    // Step 2: Fit Hawkes process to tick arrivals
    // -----------------------------------------------------------------------
    println!("\nStep 2: Fitting Hawkes process to tick arrivals...");
    let event_times: Vec<f64> = ticks.iter().map(|t| t.timestamp).collect();
    let t_end = event_times.last().copied().unwrap_or(1.0);
    let t_start = event_times.first().copied().unwrap_or(0.0);
    let relative_times: Vec<f64> = event_times.iter().map(|t| t - t_start).collect();
    let duration = t_end - t_start;

    let hawkes = HawkesProcess::fit(&relative_times, duration);
    println!("  Estimated parameters:");
    println!("    mu (baseline):    {:.4}", hawkes.mu);
    println!("    alpha (excite):   {:.4}", hawkes.alpha);
    println!("    beta (decay):     {:.4}", hawkes.beta);
    println!(
        "    Branching ratio:  {:.4} (< 1.0 = stationary)",
        hawkes.branching_ratio()
    );
    println!(
        "    Log-likelihood:   {:.2}",
        hawkes.log_likelihood(&relative_times, duration)
    );

    // Show intensity at a few points
    println!("\n  Intensity samples:");
    let sample_points = [0.1, 0.25, 0.5, 0.75, 0.9];
    for &frac in &sample_points {
        let t = duration * frac;
        let lam = hawkes.intensity(t, &relative_times);
        println!("    t={:.1}s: lambda = {:.4}", t, lam);
    }

    // -----------------------------------------------------------------------
    // Step 3: Extract features and prepare training data
    // -----------------------------------------------------------------------
    println!("\nStep 3: Extracting features and preparing training data...");
    let window_size = 50;
    let mut extractor = TickFeatureExtractor::new(window_size);

    let mut features_list: Vec<Array1<f64>> = Vec::new();
    let mut targets_list: Vec<usize> = Vec::new();

    // Feed ticks and collect features + labels
    for i in 0..ticks.len() {
        extractor.push(ticks[i].clone());

        if i >= window_size && i + 1 < ticks.len() {
            let feat = extractor.extract_features();

            // Label: direction of next tick
            let dp = ticks[i + 1].price - ticks[i].price;
            let target = if dp > 1e-10 {
                2 // Up
            } else if dp < -1e-10 {
                0 // Down
            } else {
                1 // Flat
            };

            features_list.push(feat);
            targets_list.push(target);
        }
    }

    println!("  Training samples: {}", features_list.len());

    // Class distribution
    let n_down = targets_list.iter().filter(|&&t| t == 0).count();
    let n_flat = targets_list.iter().filter(|&&t| t == 1).count();
    let n_up = targets_list.iter().filter(|&&t| t == 2).count();
    println!(
        "  Class distribution: Down={} ({:.1}%), Flat={} ({:.1}%), Up={} ({:.1}%)",
        n_down,
        n_down as f64 / features_list.len() as f64 * 100.0,
        n_flat,
        n_flat as f64 / features_list.len() as f64 * 100.0,
        n_up,
        n_up as f64 / features_list.len() as f64 * 100.0,
    );

    // Split into train/test
    let split_idx = features_list.len() * 3 / 4;
    let train_features = &features_list[..split_idx];
    let train_targets = &targets_list[..split_idx];
    let test_features = &features_list[split_idx..];
    let test_targets = &targets_list[split_idx..];

    println!(
        "  Train: {}, Test: {}",
        train_features.len(),
        test_features.len()
    );

    // -----------------------------------------------------------------------
    // Step 4: Train the tick forecaster
    // -----------------------------------------------------------------------
    println!("\nStep 4: Training tick forecaster...");
    let input_dim = 8;
    let hidden_dim = 32;
    let n_epochs = 10;

    let mut model = TickForecaster::new(input_dim, hidden_dim);
    model.learning_rate = 0.005;

    for epoch in 0..n_epochs {
        let loss = model.train_batch(train_features, train_targets);

        if (epoch + 1) % 2 == 0 || epoch == 0 {
            // Compute train accuracy
            let train_correct = train_features
                .iter()
                .zip(train_targets.iter())
                .filter(|(x, &t)| model.predict(x) as usize == t)
                .count();
            let train_acc = train_correct as f64 / train_features.len() as f64;

            println!(
                "  Epoch {}/{}: loss = {:.4}, train_acc = {:.2}%",
                epoch + 1,
                n_epochs,
                loss,
                train_acc * 100.0,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Evaluate on test set
    // -----------------------------------------------------------------------
    println!("\nStep 5: Evaluating on test set...");

    let mut correct = 0;
    let mut confusion = [[0usize; 3]; 3]; // [actual][predicted]

    for (x, &target) in test_features.iter().zip(test_targets.iter()) {
        let pred = model.predict(x) as usize;
        if pred == target {
            correct += 1;
        }
        confusion[target][pred] += 1;
    }

    let accuracy = correct as f64 / test_features.len() as f64;
    println!("  Test accuracy: {:.2}% ({}/{})", accuracy * 100.0, correct, test_features.len());

    println!("\n  Confusion matrix:");
    println!("  {:>8} | {:>6} {:>6} {:>6}", "", "P:Down", "P:Flat", "P:Up");
    println!("  {}", "-".repeat(35));
    let labels = ["A:Down", "A:Flat", "A:Up"];
    for (i, label) in labels.iter().enumerate() {
        println!(
            "  {:>8} | {:>6} {:>6} {:>6}",
            label, confusion[i][0], confusion[i][1], confusion[i][2]
        );
    }

    // -----------------------------------------------------------------------
    // Step 6: Backtest trading strategy
    // -----------------------------------------------------------------------
    println!("\nStep 6: Backtesting tick forecasting strategy...");
    let test_tick_start = ticks.len() - test_features.len() - 1;

    let mut backtester = TickForecastBacktester::new(0.00005); // 0.5 bps cost

    for (i, x) in test_features.iter().enumerate() {
        let (predicted, confidence) = model.predict_with_confidence(x);
        let tick_idx = test_tick_start + i + 1;
        let actual_dp = if tick_idx + 1 < ticks.len() {
            ticks[tick_idx + 1].price - ticks[tick_idx].price
        } else {
            0.0
        };
        let actual = if actual_dp > 1e-10 {
            TickDirection::Up
        } else if actual_dp < -1e-10 {
            TickDirection::Down
        } else {
            TickDirection::Flat
        };

        backtester.update(predicted, actual, ticks[tick_idx].price, confidence);
    }

    println!("  Directional accuracy: {:.2}%", backtester.accuracy() * 100.0);
    println!("  Total return:         {:.4}%", backtester.total_return() * 100.0);
    println!("  Max drawdown:         {:.4}%", backtester.max_drawdown * 100.0);
    println!("  Number of trades:     {}", backtester.trades);
    println!(
        "  Return / Drawdown:    {:.4}",
        backtester.return_over_drawdown()
    );

    // -----------------------------------------------------------------------
    // Step 7: Analyze feature importance (via permutation)
    // -----------------------------------------------------------------------
    println!("\nStep 7: Feature importance analysis...");
    let feature_names = [
        "Trade Imbalance",
        "VWAP Deviation",
        "Tick Intensity",
        "Realized Vol",
        "Mean Inter-Arrival",
        "Momentum (3)",
        "Momentum (5)",
        "Momentum (10)",
    ];

    let baseline_acc = accuracy;
    let mut importances: Vec<(usize, f64)> = Vec::new();

    for feat_idx in 0..input_dim {
        // Permute feature and measure accuracy drop
        let mut permuted_correct = 0;
        let mut rng = rand::thread_rng();

        for (i, x) in test_features.iter().enumerate() {
            let mut x_perm = x.clone();
            // Replace with random other sample's feature value
            let rand_idx = rng.gen_range(0..test_features.len());
            x_perm[feat_idx] = test_features[rand_idx][feat_idx];

            let pred = model.predict(&x_perm) as usize;
            if pred == test_targets[i] {
                permuted_correct += 1;
            }
        }

        let perm_acc = permuted_correct as f64 / test_features.len() as f64;
        let importance = baseline_acc - perm_acc;
        importances.push((feat_idx, importance));
    }

    importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  {:>20} | {:>12}", "Feature", "Importance");
    println!("  {}", "-".repeat(37));
    for (idx, imp) in &importances {
        let bar_len = ((imp.abs() * 200.0) as usize).min(20);
        let bar: String = if *imp > 0.0 {
            "+".repeat(bar_len)
        } else {
            "-".repeat(bar_len)
        };
        println!(
            "  {:>20} | {:>+10.4}% {}",
            feature_names[*idx],
            imp * 100.0,
            bar,
        );
    }

    // -----------------------------------------------------------------------
    // Step 8: Show sample predictions
    // -----------------------------------------------------------------------
    println!("\nStep 8: Sample predictions on recent ticks...");
    let n_samples = 10.min(test_features.len());
    println!(
        "  {:>4} | {:>8} | {:>8} | {:>6} | {:>8}",
        "Tick", "Pred", "Actual", "Conf%", "Correct"
    );
    println!("  {}", "-".repeat(50));

    for i in 0..n_samples {
        let (pred, conf) = model.predict_with_confidence(&test_features[i]);
        let actual = match test_targets[i] {
            0 => TickDirection::Down,
            1 => TickDirection::Flat,
            _ => TickDirection::Up,
        };
        let dir_name = |d: &TickDirection| match d {
            TickDirection::Down => "Down",
            TickDirection::Flat => "Flat",
            TickDirection::Up => "Up",
        };
        let correct = if pred == actual { "Y" } else { "N" };
        println!(
            "  {:>4} | {:>8} | {:>8} | {:>5.1} | {:>8}",
            i + 1,
            dir_name(&pred),
            dir_name(&actual),
            conf * 100.0,
            correct,
        );
    }

    println!("\n=== Done ===");
}

/// Attempt to load tick data from Bybit
fn load_bybit_ticks() -> anyhow::Result<Vec<TickData>> {
    let client = BybitClient::new();
    let ticks = client.fetch_ticks("BTCUSDT", "1", 200, 10)?;
    if ticks.is_empty() {
        anyhow::bail!("No data returned");
    }
    Ok(ticks)
}
