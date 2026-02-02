//! Multi-asset meta-volatility example.
//!
//! Demonstrates creating tasks from multiple simulated assets
//! and training a meta-learner across them.

use meta_volatility::prelude::*;
use meta_volatility::meta::maml::VolatilityTask;
use meta_volatility::data::features::Features;
use rand::Rng;

/// Simulate asset price series with given parameters.
fn simulate_asset(
    n: usize,
    base_price: f64,
    base_vol: f64,
    _seed_offset: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut prices = vec![base_price];
    let mut returns = Vec::new();

    for _ in 0..n {
        let r = rng.gen::<f64>() * 2.0 * base_vol - base_vol;
        returns.push(r);
        let new_price = prices.last().unwrap() * (1.0 + r);
        prices.push(new_price.max(0.01));
    }

    (prices, returns)
}

/// Create tasks from simulated returns.
fn create_tasks_from_returns(
    returns: &[f64],
    support_size: usize,
    query_size: usize,
    forecast_horizon: usize,
) -> Vec<VolatilityTask> {
    let total = support_size + query_size;
    let mut tasks = Vec::new();

    let mut start = 20; // Skip warmup
    while start + total + forecast_horizon <= returns.len() {
        let mut support_x = Vec::new();
        let mut support_y = Vec::new();
        let mut query_x = Vec::new();
        let mut query_y = Vec::new();

        for j in 0..total {
            let idx = start + j;
            let r = returns[idx];

            // Simple features
            let rv5: f64 = if idx >= 5 {
                let s = &returns[idx - 5..idx];
                let m = s.iter().sum::<f64>() / 5.0;
                (s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 5.0).sqrt()
            } else {
                0.02
            };

            let rv10: f64 = if idx >= 10 {
                let s = &returns[idx - 10..idx];
                let m = s.iter().sum::<f64>() / 10.0;
                (s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 10.0).sqrt()
            } else {
                0.02
            };

            let rv20: f64 = if idx >= 20 {
                let s = &returns[idx - 20..idx];
                let m = s.iter().sum::<f64>() / 20.0;
                (s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 20.0).sqrt()
            } else {
                0.02
            };

            let features = vec![
                r, r.abs(), r * r,
                rv5, rv10, rv20,
                1.0, 0.0, 50.0, 0.02,
            ];

            // Target: realized vol over forecast horizon
            let end = (idx + forecast_horizon).min(returns.len());
            let fut = &returns[idx..end];
            let target_vol = if !fut.is_empty() {
                let m = fut.iter().sum::<f64>() / fut.len() as f64;
                (fut.iter().map(|x| (x - m).powi(2)).sum::<f64>() / fut.len() as f64).sqrt()
            } else {
                0.02
            };

            if j < support_size {
                support_x.push(features);
                support_y.push(target_vol);
            } else {
                query_x.push(features);
                query_y.push(target_vol);
            }
        }

        tasks.push(VolatilityTask {
            support_x,
            support_y,
            query_x,
            query_y,
        });

        start += total;
    }

    tasks
}

fn main() {
    println!("=== Meta-Volatility: Multi-Asset Training ===\n");

    // Simulate multiple assets with different characteristics
    let assets = vec![
        ("BTC-like", 40000.0, 0.03),
        ("ETH-like", 2500.0, 0.04),
        ("SPY-like", 450.0, 0.01),
        ("AAPL-like", 180.0, 0.015),
    ];

    let mut all_tasks = Vec::new();

    for (name, price, vol) in &assets {
        let (_, returns) = simulate_asset(500, *price, *vol, 0);
        let tasks = create_tasks_from_returns(&returns, 10, 5, 5);
        println!("  {} -> {} tasks", name, tasks.len());
        all_tasks.extend(tasks);
    }

    println!("\nTotal tasks: {}", all_tasks.len());

    // Create and train meta-learner
    let net = MetaVolatilityNet::new(Features::NUM_FEATURES, 32);
    let mut trainer = MAMLTrainer::new(net, 0.01, 0.001, 3);

    println!("\nMeta-training across all assets...");
    let batch_size = 4;
    let num_epochs = 10;

    for epoch in 0..num_epochs {
        let start = (epoch * batch_size) % all_tasks.len();
        let end = (start + batch_size).min(all_tasks.len());
        let batch: Vec<&VolatilityTask> = all_tasks[start..end].iter().collect();

        let loss = trainer.meta_train_step(&batch);
        println!("  Epoch {:>3}, Loss: {:.6}", epoch, loss);
    }

    // Test adaptation on held-out tasks
    println!("\nTesting adaptation:");
    if let Some(test_task) = all_tasks.last() {
        let preds = trainer.adapt_and_predict(test_task);
        println!("  Predictions: {:?}", preds);
        println!("  Targets:     {:?}", test_task.query_y);
    }

    println!("\nDONE");
}
