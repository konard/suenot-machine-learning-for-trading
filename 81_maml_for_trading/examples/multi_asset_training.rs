//! Multi-asset MAML training example.
//!
//! This example demonstrates:
//! - Training MAML on multiple cryptocurrency assets
//! - Using simulated data for different market conditions
//! - Evaluating cross-asset generalization

use maml_trading::{
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    maml::algorithm::{MAMLTrainer, TaskData},
    model::network::TradingModel,
};

/// Asset configuration for simulation
struct AssetConfig {
    name: &'static str,
    base_price: f64,
    volatility: f64,
    trend: f64, // Positive = bullish, negative = bearish
}

fn main() {
    println!("=== Multi-Asset MAML Training Example ===\n");

    // Define multiple assets with different characteristics
    let assets = vec![
        AssetConfig {
            name: "BTC",
            base_price: 50000.0,
            volatility: 0.02,
            trend: 0.0001,
        },
        AssetConfig {
            name: "ETH",
            base_price: 3000.0,
            volatility: 0.025,
            trend: 0.0002,
        },
        AssetConfig {
            name: "SOL",
            base_price: 100.0,
            volatility: 0.035,
            trend: -0.0001,
        },
        AssetConfig {
            name: "AVAX",
            base_price: 40.0,
            volatility: 0.03,
            trend: 0.00015,
        },
        AssetConfig {
            name: "DOGE",
            base_price: 0.1,
            volatility: 0.05,
            trend: 0.0,
        },
    ];

    println!("Training assets:");
    for asset in &assets {
        println!(
            "  {} - Base: ${:.2}, Vol: {:.1}%, Trend: {:.2}%",
            asset.name,
            asset.base_price,
            asset.volatility * 100.0,
            asset.trend * 100.0
        );
    }
    println!();

    // Generate tasks for each asset
    let feature_gen = FeatureGenerator::default_window();
    let mut tasks = Vec::new();

    println!("Generating training data...");
    for asset in &assets {
        let klines = SimulatedDataGenerator::generate_trending_klines(
            300,
            asset.base_price,
            asset.volatility,
            asset.trend,
        );

        let features = feature_gen.compute_features(&klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        // Split 60/40 for support/query
        let split_idx = (features.len() as f64 * 0.6) as usize;
        let support_features = features[..split_idx].to_vec();
        let support_labels = returns[..split_idx].to_vec();
        let query_features = features[split_idx..].to_vec();
        let query_labels = returns[split_idx..returns.len().min(split_idx + query_features.len())]
            .to_vec();

        if support_labels.len() == support_features.len()
            && query_labels.len() == query_features.len()
            && !query_labels.is_empty()
        {
            tasks.push(TaskData {
                support_features,
                support_labels,
                query_features,
                query_labels,
            });
            println!("  {} task created ({} samples)", asset.name, features.len());
        }
    }
    println!();

    // Create and train MAML
    let model = TradingModel::new(11, 64, 1);
    let mut trainer = MAMLTrainer::new(model, 0.01, 0.001, 5, true);

    println!("Meta-training on {} assets...", tasks.len());
    let num_epochs = 20;
    let mut loss_history = Vec::new();

    for epoch in 0..num_epochs {
        let meta_loss = trainer.meta_train_step(&tasks);
        loss_history.push(meta_loss);

        if (epoch + 1) % 5 == 0 {
            println!("  Epoch {:2}: Meta-loss = {:.6}", epoch + 1, meta_loss);
        }
    }

    // Show training progress
    println!("\nTraining summary:");
    println!("  Initial loss: {:.6}", loss_history[0]);
    println!("  Final loss:   {:.6}", loss_history[loss_history.len() - 1]);
    let improvement = (loss_history[0] - loss_history[loss_history.len() - 1]) / loss_history[0] * 100.0;
    println!("  Improvement:  {:.1}%", improvement);
    println!();

    // Test on a new asset
    println!("Testing on NEW asset (MATIC - not in training set)...");
    let test_asset = AssetConfig {
        name: "MATIC",
        base_price: 1.0,
        volatility: 0.04,
        trend: 0.00005,
    };

    let test_klines = SimulatedDataGenerator::generate_trending_klines(
        150,
        test_asset.base_price,
        test_asset.volatility,
        test_asset.trend,
    );

    let test_features = feature_gen.compute_features(&test_klines);
    let test_closes: Vec<f64> = test_klines.iter().map(|k| k.close).collect();
    let test_returns: Vec<f64> = test_closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

    // Few-shot adaptation
    let k_shot = 20;
    let adapt_features = test_features[..k_shot].to_vec();
    let adapt_labels = test_returns[..k_shot].to_vec();

    println!("  Adapting with {} examples...", k_shot);
    let adapted = trainer.adapt(&adapt_features, &adapt_labels, Some(10));

    // Evaluate
    let mut mse_before = 0.0;
    let mut mse_after = 0.0;
    let mut correct_before = 0;
    let mut correct_after = 0;
    let mut count = 0;

    let base_model = trainer.base_model();

    for i in k_shot..test_features.len().min(test_returns.len()) {
        let pred_before = base_model.predict(&test_features[i]);
        let pred_after = adapted.predict(&test_features[i]);
        let actual = test_returns[i];

        mse_before += (pred_before - actual).powi(2);
        mse_after += (pred_after - actual).powi(2);

        if (pred_before > 0.0) == (actual > 0.0) {
            correct_before += 1;
        }
        if (pred_after > 0.0) == (actual > 0.0) {
            correct_after += 1;
        }
        count += 1;
    }

    if count > 0 {
        mse_before /= count as f64;
        mse_after /= count as f64;

        println!("\nResults on {} (evaluated on {} points):", test_asset.name, count);
        println!("  Before adaptation:");
        println!("    MSE: {:.8}", mse_before);
        println!("    Direction accuracy: {:.1}%", (correct_before as f64 / count as f64) * 100.0);
        println!("  After adaptation:");
        println!("    MSE: {:.8}", mse_after);
        println!("    Direction accuracy: {:.1}%", (correct_after as f64 / count as f64) * 100.0);

        if mse_after < mse_before {
            println!("\n  Adaptation improved MSE by {:.1}%!", (1.0 - mse_after / mse_before) * 100.0);
        }
    }

    println!("\n=== Multi-Asset Training Complete ===");
}
