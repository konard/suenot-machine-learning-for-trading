//! AutoML Trading Example
//!
//! Fetches BTCUSDT data from Bybit, runs the full AutoML pipeline,
//! and compares the AutoML ensemble against a manual baseline.

use automl_finance::*;

fn main() -> anyhow::Result<()> {
    println!("=== AutoML Finance - Trading Example ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT 1h candles from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("Fetched {} candles", c.len());
            if c.len() < 100 {
                println!("Not enough candles from API, generating synthetic data...");
                generate_synthetic_candles(200)
            } else {
                c
            }
        }
        Err(e) => {
            println!("API error: {}. Using synthetic data for demonstration.", e);
            generate_synthetic_candles(200)
        }
    };

    println!(
        "Data range: {} candles, first close={:.2}, last close={:.2}\n",
        candles.len(),
        candles.first().map(|c| c.close).unwrap_or(0.0),
        candles.last().map(|c| c.close).unwrap_or(0.0),
    );

    // -----------------------------------------------------------------------
    // Step 2: Generate features
    // -----------------------------------------------------------------------
    let forecast_horizon = 1; // predict next-bar return
    println!("Generating features (forecast horizon = {} bars)...", forecast_horizon);
    let fm = generate_features(&candles, forecast_horizon)?;
    println!(
        "Feature matrix: {} samples x {} features\n",
        fm.data.nrows(),
        fm.data.ncols()
    );

    // -----------------------------------------------------------------------
    // Step 3: Run manual baseline (simple linear model with all features)
    // -----------------------------------------------------------------------
    println!("--- Manual Baseline ---");
    let cv = WalkForwardCV::new(3, 0.7, 2);
    let all_features: Vec<usize> = (0..fm.data.ncols()).collect();
    let baseline_params = HyperParams::new(); // default alpha=1.0
    let (baseline_mean, baseline_std, _) =
        cv.evaluate(&fm, &all_features, ModelType::Linear, &baseline_params);
    println!(
        "Linear baseline (all features): mean_score={:.6}, std={:.6}\n",
        baseline_mean, baseline_std
    );

    // -----------------------------------------------------------------------
    // Step 4: Run AutoML pipeline
    // -----------------------------------------------------------------------
    println!("--- AutoML Pipeline ---");
    let mut pipeline = AutoMLPipeline::new(
        0.001, // MI threshold (low for financial data)
        0.85,  // correlation threshold
        3,     // CV splits
        2,     // embargo periods
        SearchStrategy::BayesianInspired,
        10,    // HPO iterations per model type
        3,     // ensemble top-k
    );

    let (results, ensemble_weights) = pipeline.run(&fm)?;

    // -----------------------------------------------------------------------
    // Step 5: Report results
    // -----------------------------------------------------------------------
    println!("\n=== Results Summary ===\n");

    // Best individual model
    let best = results
        .iter()
        .max_by(|a, b| a.mean_score.partial_cmp(&b.mean_score).unwrap())
        .unwrap();
    println!(
        "Best individual model: {} (score={:.6}, std={:.6})",
        best.model_type, best.mean_score, best.std_score
    );
    println!("  Hyperparameters: {:?}", best.hyperparams.params);
    println!(
        "  Selected features: {:?}",
        best.selected_features
            .iter()
            .map(|&i| &fm.feature_names[i])
            .collect::<Vec<_>>()
    );

    // Ensemble prediction
    let ensemble_pred =
        pipeline
            .ensemble_builder
            .predict_ensemble(&fm, &results, &ensemble_weights);
    let ensemble_mse = (&ensemble_pred - &fm.target)
        .mapv(|x| x.powi(2))
        .mean()
        .unwrap_or(f64::MAX);
    println!("\nEnsemble in-sample MSE: {:.8}", ensemble_mse);

    // Comparison
    println!("\n--- Comparison ---");
    println!(
        "Manual baseline score: {:.6}",
        baseline_mean
    );
    println!("AutoML best score:    {:.6}", best.mean_score);
    let improvement = if baseline_mean != 0.0 {
        ((best.mean_score - baseline_mean) / baseline_mean.abs()) * 100.0
    } else {
        0.0
    };
    println!("Improvement:          {:.1}%", improvement);

    // Model diversity
    println!("\n--- Model Diversity in Search ---");
    for model_type in [ModelType::Linear, ModelType::DecisionTree, ModelType::NeuralNet] {
        let type_results: Vec<&PipelineResult> =
            results.iter().filter(|r| r.model_type == model_type).collect();
        if !type_results.is_empty() {
            let best_score = type_results
                .iter()
                .map(|r| r.mean_score)
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "  {}: {} configs evaluated, best score={:.6}",
                model_type,
                type_results.len(),
                best_score
            );
        }
    }

    println!("\n=== AutoML pipeline complete ===");
    Ok(())
}

/// Generate synthetic candles for demonstration when API is unavailable
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        let ret = rng.gen_range(-0.02..0.02);
        let close = price * (1.0 + ret);
        let high = close * (1.0 + rng.gen_range(0.0..0.01));
        let low = close * (1.0 - rng.gen_range(0.0..0.01));
        let open = price;
        let volume = rng.gen_range(100.0..1000.0);

        candles.push(Candle {
            open_time: 1700000000000 + (i as u64) * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }
    candles
}
