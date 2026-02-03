//! Multi-scale analysis example: demonstrates how different hierarchy levels
//! capture patterns at different temporal resolutions.

use hierarchical_ssm::data::bybit::generate_synthetic_candles;
use hierarchical_ssm::data::features::{compute_features, normalize_features};
use hierarchical_ssm::model::{HiSSConfig, HierarchicalSSM};

fn main() {
    println!("=== HiSS Multi-Scale Analysis ===\n");

    // Compare models with different numbers of hierarchy levels
    let configs = vec![
        ("Single-scale (1 level)", HiSSConfig {
            input_dim: 8,
            hidden_dim: 32,
            num_levels: 1,
            downsample_factors: vec![],
            ssm_state_dim: 8,
            ..Default::default()
        }),
        ("Two-scale (2 levels, 4x)", HiSSConfig {
            input_dim: 8,
            hidden_dim: 32,
            num_levels: 2,
            downsample_factors: vec![4],
            ssm_state_dim: 8,
            ..Default::default()
        }),
        ("Three-scale (3 levels, 4x4x)", HiSSConfig {
            input_dim: 8,
            hidden_dim: 32,
            num_levels: 3,
            downsample_factors: vec![4, 4],
            ssm_state_dim: 8,
            ..Default::default()
        }),
    ];

    // Generate data
    println!("Generating synthetic market data...");
    let candles = generate_synthetic_candles(300, 50000.0);
    let features = compute_features(&candles);
    let normalized = normalize_features(&features);
    println!("  {} feature rows available\n", normalized.len());

    let window_size = 64;
    let num_predictions = normalized.len().saturating_sub(window_size);

    for (name, config) in &configs {
        println!("--- {} ---", name);
        println!("  Levels: {}, Downsample: {:?}", config.num_levels, config.downsample_factors);

        let model = HierarchicalSSM::new(config.clone());

        let mut return_sum = 0.0;
        let mut vol_sum = 0.0;
        let mut dir_sum = 0.0;

        for i in 0..num_predictions {
            let window = &normalized[i..i + window_size];
            let pred = model.predict(window);
            return_sum += pred.predicted_return.abs();
            vol_sum += pred.predicted_volatility;
            dir_sum += pred.direction_probability;
        }

        let n = num_predictions.max(1) as f64;
        println!("  Avg |predicted return|: {:.6}", return_sum / n);
        println!("  Avg predicted vol:      {:.6}", vol_sum / n);
        println!("  Avg direction prob:     {:.4}", dir_sum / n);

        // Analyze how predictions vary with different effective time scales
        if num_predictions >= 20 {
            let mut recent_returns: Vec<f64> = Vec::new();
            let mut older_returns: Vec<f64> = Vec::new();

            for i in (num_predictions - 10)..num_predictions {
                let window = &normalized[i..i + window_size];
                let pred = model.predict(window);
                recent_returns.push(pred.predicted_return);
            }
            for i in 0..10 {
                let window = &normalized[i..i + window_size];
                let pred = model.predict(window);
                older_returns.push(pred.predicted_return);
            }

            let recent_std = std_dev(&recent_returns);
            let older_std = std_dev(&older_returns);
            println!(
                "  Prediction variability (recent vs older): {:.6} vs {:.6}",
                recent_std, older_std
            );
        }
        println!();
    }

    println!("Multi-scale models capture patterns at different temporal resolutions.");
    println!("More hierarchy levels allow the model to reason about both");
    println!("fine-grained and coarse-grained market dynamics simultaneously.");
}

fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}
