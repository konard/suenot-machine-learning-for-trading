//! Basic meta-volatility prediction example.
//!
//! Demonstrates creating a meta-volatility network, constructing tasks,
//! and running MAML training on synthetic data.

use meta_volatility::prelude::*;
use meta_volatility::meta::maml::VolatilityTask;
use meta_volatility::data::features::Features;

fn create_synthetic_tasks(num_tasks: usize) -> Vec<VolatilityTask> {
    let mut tasks = Vec::new();

    for t in 0..num_tasks {
        let base_vol = 0.01 + (t as f64 * 0.005);

        let support_x: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let r = (i as f64 * 0.01) * (-1.0_f64).powi(i as i32);
                vec![
                    r,
                    r.abs(),
                    r * r,
                    base_vol * 0.9,
                    base_vol * 0.95,
                    base_vol,
                    1.0 + (i as f64 * 0.05),
                    0.0,
                    50.0 + (i as f64),
                    0.02,
                ]
            })
            .collect();

        let support_y: Vec<f64> = (0..10)
            .map(|i| base_vol + (i as f64 * 0.001))
            .collect();

        let query_x: Vec<Vec<f64>> = (0..5)
            .map(|i| {
                let r = (i as f64 * 0.015) * (-1.0_f64).powi(i as i32);
                vec![
                    r,
                    r.abs(),
                    r * r,
                    base_vol * 1.05,
                    base_vol * 1.02,
                    base_vol * 1.01,
                    1.1,
                    0.0,
                    52.0,
                    0.022,
                ]
            })
            .collect();

        let query_y: Vec<f64> = (0..5)
            .map(|i| base_vol * 1.05 + (i as f64 * 0.001))
            .collect();

        tasks.push(VolatilityTask {
            support_x,
            support_y,
            query_x,
            query_y,
        });
    }

    tasks
}

fn main() {
    println!("=== Meta-Volatility Prediction: Basic Example ===\n");

    // Create network
    let net = MetaVolatilityNet::new(Features::NUM_FEATURES, 32);
    println!(
        "Created MetaVolatilityNet with {} parameters",
        net.num_params()
    );

    // Create MAML trainer
    let mut trainer = MAMLTrainer::new(net, 0.01, 0.001, 3);

    // Create synthetic tasks
    let tasks = create_synthetic_tasks(8);
    println!("Created {} synthetic volatility tasks\n", tasks.len());

    // Meta-training loop
    println!("Starting meta-training...");
    for epoch in 0..20 {
        let batch: Vec<&VolatilityTask> = tasks.iter().take(4).collect();
        let loss = trainer.meta_train_step(&batch);
        if epoch % 5 == 0 {
            println!("  Epoch {:>3}, Meta-Loss: {:.6}", epoch, loss);
        }
    }
    println!("Meta-training complete.\n");

    // Adapt to a new task and predict
    let test_task = &tasks[tasks.len() - 1];
    let predictions = trainer.adapt_and_predict(test_task);

    println!("Adaptation and prediction on new task:");
    println!("  Predicted volatilities: {:?}", predictions);
    println!("  Actual volatilities:    {:?}", test_task.query_y);

    let mse: f64 = predictions
        .iter()
        .zip(&test_task.query_y)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;
    println!("  MSE: {:.6}", mse);

    println!("\nDONE");
}
