use vae_portfolio_generation::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== VAE Portfolio Generation - Trading Example ===\n");

    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"];
    let asset_names: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
    let n_assets = symbols.len();

    // ── Step 1: Fetch live data from Bybit ──────────────────────────
    println!("[1] Fetching market data from Bybit V5 API...\n");

    let client = BybitClient::new();

    let returns = match client
        .get_multi_asset_returns(&symbols, "60", 100)
        .await
    {
        Ok(r) => {
            println!("  Fetched returns for {} assets", r.len());
            for (i, symbol) in symbols.iter().enumerate() {
                if let Some(rets) = r.get(i) {
                    let mean: f64 =
                        rets.iter().sum::<f64>() / rets.len().max(1) as f64;
                    println!(
                        "  {}: {} periods, mean return = {:.4}%",
                        symbol,
                        rets.len(),
                        mean * 100.0
                    );
                }
            }
            r
        }
        Err(e) => {
            println!(
                "  Could not fetch live data: {}. Using synthetic data.\n",
                e
            );
            generate_synthetic_returns(n_assets, 99)
        }
    };

    // ── Step 2: Generate training portfolios ─────────────────────────
    println!("\n[2] Generating training portfolios with Dirichlet sampling...\n");

    let training_data = generate_training_portfolios(n_assets, 500);
    println!("  Generated {} training portfolios", training_data.len());

    // Show a few examples
    for (i, p) in training_data.iter().take(3).enumerate() {
        let weights_str: Vec<String> = p.iter().map(|w| format!("{:.1}%", w * 100.0)).collect();
        println!("  Example {}: [{}]", i + 1, weights_str.join(", "));
    }

    // ── Step 3: Train the VAE ─────────────────────────────────────────
    println!("\n[3] Training Portfolio VAE...\n");

    let mut generator = PortfolioGenerator::new(
        asset_names.clone(),
        3,    // latent_dim
        0.1,  // beta
        0.6,  // max_weight
        0.05, // min_weight
    );

    let losses = generator.train(&training_data, 50, 0.001);
    println!(
        "  Training complete. Final loss: {:.6}",
        losses.last().unwrap_or(&0.0)
    );
    println!(
        "  Loss reduction: {:.6} -> {:.6}",
        losses.first().unwrap_or(&0.0),
        losses.last().unwrap_or(&0.0)
    );

    // ── Step 4: Generate new portfolios ────────────────────────────────
    println!("\n[4] Generating portfolios from trained VAE...\n");

    let evaluator = PortfolioEvaluator::new(returns.clone());

    let generated = generator.generate_many(5);
    for (i, portfolio) in generated.iter().enumerate() {
        println!("  Portfolio {}:", i + 1);
        println!("{}", portfolio.display());
        let metrics = evaluator.evaluate(&portfolio.weights);
        println!("{}", metrics.display());
        println!();
    }

    // ── Step 5: Portfolio interpolation ─────────────────────────────────
    println!("[5] Interpolating between concentrated and balanced portfolios...\n");

    let concentrated = vec![0.7, 0.15, 0.1, 0.05];
    let balanced = vec![0.25, 0.25, 0.25, 0.25];

    let interpolated = generator.interpolate(&concentrated, &balanced, 4);
    for (i, portfolio) in interpolated.iter().enumerate() {
        let alpha = i as f64 / 4.0;
        println!(
            "  alpha = {:.2} ({}% concentrated, {}% balanced):",
            alpha,
            ((1.0 - alpha) * 100.0) as u32,
            (alpha * 100.0) as u32
        );
        println!("{}", portfolio.display());
        let metrics = evaluator.evaluate(&portfolio.weights);
        println!("    Sharpe: {:.4}, MaxDD: {:.2}%", metrics.sharpe_ratio, metrics.max_drawdown * 100.0);
        println!();
    }

    // ── Step 6: Find best portfolio ──────────────────────────────────
    println!("[6] Searching for best Sharpe ratio portfolio...\n");

    let candidates = generator.generate_many(100);
    let mut best_sharpe = f64::NEG_INFINITY;
    let mut best_portfolio = &candidates[0];

    for portfolio in &candidates {
        let sharpe = evaluator.sharpe_ratio(&portfolio.weights);
        if sharpe > best_sharpe {
            best_sharpe = sharpe;
            best_portfolio = portfolio;
        }
    }

    println!("  Best portfolio (out of 100 candidates):");
    println!("{}", best_portfolio.display());
    let best_metrics = evaluator.evaluate(&best_portfolio.weights);
    println!("{}", best_metrics.display());

    // ── Step 7: Compare with equal-weight baseline ────────────────────
    println!("\n[7] Comparison with equal-weight baseline...\n");

    let equal_weights = vec![1.0 / n_assets as f64; n_assets];
    let baseline_metrics = evaluator.evaluate(&equal_weights);

    println!("  Equal-weight portfolio:");
    println!("{}", baseline_metrics.display());
    println!();
    println!("  VAE best portfolio:");
    println!("{}", best_metrics.display());
    println!();

    if best_metrics.sharpe_ratio > baseline_metrics.sharpe_ratio {
        println!(
            "  VAE portfolio outperforms by {:.4} Sharpe ratio points",
            best_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio
        );
    } else {
        println!(
            "  Equal-weight portfolio leads by {:.4} Sharpe ratio points",
            baseline_metrics.sharpe_ratio - best_metrics.sharpe_ratio
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
