//! # DARTS Trading Example
//!
//! Demonstrates the full DARTS pipeline for trading:
//! 1. Fetches BTCUSDT kline data from Bybit
//! 2. Computes features (log returns, normalized)
//! 3. Runs DARTS architecture search
//! 4. Extracts and displays discovered architecture
//! 5. Tests on held-out data
//! 6. Compares with hand-designed baseline

use darts_trading::*;

fn main() {
    println!("=== DARTS Trading Architecture Search ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -----------------------------------------------------------------------
    println!("[1/6] Fetching BTCUSDT 1h klines from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 1000) {
        Ok(c) => {
            println!("      Fetched {} candles", c.len());
            println!(
                "      Date range: {} -> {}",
                c.first().map(|x| x.timestamp).unwrap_or(0),
                c.last().map(|x| x.timestamp).unwrap_or(0)
            );
            c
        }
        Err(e) => {
            println!("      Failed to fetch from Bybit: {}", e);
            println!("      Using synthetic data for demonstration...\n");
            generate_synthetic_data(500)
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Feature engineering
    // -----------------------------------------------------------------------
    println!("\n[2/6] Computing features...");
    let (features, closes) = candles_to_features(&candles);
    println!("      Feature length: {}", features.len());
    println!(
        "      Price range: {:.2} - {:.2}",
        closes.iter().cloned().fold(f64::INFINITY, f64::min),
        closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Create targets: next-step returns (shifted by 1)
    let targets: Vec<f64> = if features.len() > 1 {
        features[1..].to_vec()
    } else {
        vec![0.0]
    };
    let features = &features[..targets.len()];

    // Split chronologically: 60% train, 20% val, 20% test
    let (train_feat, val_feat, test_feat) = split_data(features, 0.6, 0.2);
    let (train_tgt, val_tgt, test_tgt) = split_data(&targets, 0.6, 0.2);

    println!("      Train: {} samples", train_feat.len());
    println!("      Val:   {} samples", val_feat.len());
    println!("      Test:  {} samples", test_feat.len());

    // -----------------------------------------------------------------------
    // Step 3: Run DARTS architecture search
    // -----------------------------------------------------------------------
    println!("\n[3/6] Running DARTS architecture search...");
    let config = SearchConfig {
        num_nodes: 3,
        num_epochs: 15,
        weight_lr: 0.005,
        arch_lr: 0.002,
        initial_temperature: 1.0,
        final_temperature: 0.1,
        warmup_epochs: 3,
        min_entropy: 0.05,
        skip_penalty: 0.01,
    };

    println!("      Config: {} nodes, {} epochs, warmup = {}",
             config.num_nodes, config.num_epochs, config.warmup_epochs);

    let result = darts_search(train_feat, val_feat, train_tgt, val_tgt, &config);

    println!("      Search complete!");
    println!("      Epochs run: {}", result.train_losses.len());
    println!(
        "      Final train loss: {:.6}",
        result.train_losses.last().unwrap_or(&0.0)
    );
    println!(
        "      Final val loss:   {:.6}",
        result.val_losses.last().unwrap_or(&0.0)
    );
    println!(
        "      Final entropy:    {:.4}",
        result.entropies.last().unwrap_or(&0.0)
    );
    if result.stopped_early {
        println!("      WARNING: Search stopped early (entropy collapse detected)");
    }

    // -----------------------------------------------------------------------
    // Step 4: Display discovered architecture
    // -----------------------------------------------------------------------
    println!("\n[4/6] Discovered Architecture:");
    println!("{}", result.architecture);

    // Show architecture weight distribution for each edge
    println!("      Architecture weight details:");
    for (i, alpha) in result.cell.alphas.iter().enumerate() {
        let probs = alpha.probabilities();
        let ops = default_op_space();
        let best = alpha.best_op();
        println!(
            "      Edge {}: best = {} ({:.2}%)",
            i,
            ops[best],
            probs[best] * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // Step 5: Test on held-out data
    // -----------------------------------------------------------------------
    println!("\n[5/6] Testing on held-out data...");

    // Assess the discrete architecture
    let darts_test_loss =
        assess_discrete(&result.architecture, &result.cell, test_feat, test_tgt);
    println!("      DARTS architecture test MSE: {:.6}", darts_test_loss);

    // Also check the mixed (soft) architecture
    let mixed_out = result.cell.forward_deterministic(test_feat, test_feat);
    let mixed_test_loss = mse_loss(&mixed_out, test_tgt);
    println!("      Mixed (soft) arch test MSE:  {:.6}", mixed_test_loss);

    // -----------------------------------------------------------------------
    // Step 6: Compare with baseline
    // -----------------------------------------------------------------------
    println!("\n[6/6] Comparing with baselines...");

    // Baseline 1: Moving average predictor
    let ma5_pred = baseline_moving_average(test_feat, 5);
    let ma5_loss = mse_loss(&ma5_pred, test_tgt);
    println!("      Moving Avg (w=5) test MSE:   {:.6}", ma5_loss);

    let ma10_pred = baseline_moving_average(test_feat, 10);
    let ma10_loss = mse_loss(&ma10_pred, test_tgt);
    println!("      Moving Avg (w=10) test MSE:  {:.6}", ma10_loss);

    // Baseline 2: Persistence (predict last value)
    let persistence_loss = if test_feat.len() > 1 {
        let persistence: Vec<f64> = test_feat[..test_feat.len() - 1].to_vec();
        let tgt_trimmed = &test_tgt[1..];
        mse_loss(&persistence, tgt_trimmed)
    } else {
        f64::NAN
    };
    println!("      Persistence test MSE:        {:.6}", persistence_loss);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("DARTS automatically discovered an architecture with {} active edges.",
             result.architecture.edges.len());
    println!("Operations used:");
    let mut op_counts = std::collections::HashMap::new();
    for edge in &result.architecture.edges {
        *op_counts.entry(format!("{}", edge.op)).or_insert(0) += 1;
    }
    for (op, count) in &op_counts {
        println!("  - {}: {} edges", op, count);
    }

    let best_baseline = ma5_loss.min(ma10_loss).min(persistence_loss);
    if darts_test_loss < best_baseline {
        println!(
            "\nDART architecture OUTPERFORMS best baseline by {:.2}%",
            (1.0 - darts_test_loss / best_baseline) * 100.0
        );
    } else {
        println!(
            "\nBaseline is better by {:.2}% (search may need more epochs/data)",
            (1.0 - best_baseline / darts_test_loss) * 100.0
        );
    }
    println!("\nDone.");
}

/// Generate synthetic candle data for demonstration when Bybit is unavailable.
fn generate_synthetic_data(n: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let returns = rng.gen_range(-0.02..0.02_f64);
        // Add some trend and mean reversion
        let trend = 0.0001 * (i as f64 * 0.01).sin();
        price *= 1.0 + returns + trend;

        let high = price * (1.0 + rng.gen_range(0.0..0.01));
        let low = price * (1.0 - rng.gen_range(0.0..0.01));
        let open = price * (1.0 + rng.gen_range(-0.005..0.005));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1700000000 + (i as u64) * 3600,
            open,
            high,
            low,
            close: price,
            volume,
        });
    }
    candles
}
