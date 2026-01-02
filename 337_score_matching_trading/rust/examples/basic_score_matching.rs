//! # Basic Score Matching Example
//!
//! Demonstrates the core score matching concepts with synthetic data.
//!
//! Run with: cargo run --example basic_score_matching

use ndarray::{Array1, Array2};
use score_matching_trading::{
    score::{
        DenoisingScoreMatchingTrainer, LangevinConfig, LangevinDynamics,
        MultiScaleScoreNetwork, ScoreNetwork, TrainingConfig,
    },
};

fn main() -> anyhow::Result<()> {
    println!("=== Score Matching Trading Demo ===\n");

    // 1. Create a simple score network
    println!("1. Creating Score Network...");
    let input_dim = 10;
    let hidden_dim = 64;
    let num_layers = 3;

    let network = ScoreNetwork::new(input_dim, hidden_dim, num_layers);
    println!(
        "   Network created with {} parameters",
        network.num_parameters()
    );

    // 2. Test forward pass
    println!("\n2. Testing Forward Pass...");
    let test_input = Array1::from_vec(vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.0, -0.3, 0.4]);
    let score = network.forward(&test_input);
    println!("   Input:  {:?}", test_input.as_slice().unwrap());
    println!("   Score:  {:?}", score.as_slice().unwrap());

    // 3. Create multi-scale network
    println!("\n3. Creating Multi-Scale Score Network...");
    let ms_network = MultiScaleScoreNetwork::new(
        input_dim,
        hidden_dim,
        num_layers,
        10,   // num_noise_levels
        0.01, // sigma_min
        1.0,  // sigma_max
    );
    println!("   Noise levels: {:?}", ms_network.sigmas.as_slice().unwrap());

    // 4. Test multi-scale forward pass
    println!("\n4. Testing Multi-Scale Forward Pass...");
    for level in [0, 4, 9] {
        let score = ms_network.forward(&test_input, level);
        let sigma = ms_network.get_sigma(level);
        println!(
            "   Level {}: sigma={:.4}, score_norm={:.4}",
            level,
            sigma,
            score.iter().map(|x| x * x).sum::<f64>().sqrt()
        );
    }

    // 5. Generate synthetic data
    println!("\n5. Generating Synthetic Data...");
    let n_samples = 1000;
    let train_data = generate_mixture_of_gaussians(n_samples, input_dim, 3);
    let val_data = generate_mixture_of_gaussians(200, input_dim, 3);
    println!("   Generated {} training samples", n_samples);
    println!("   Generated {} validation samples", val_data.nrows());

    // 6. Create trainer and compute initial loss
    println!("\n6. Training Setup...");
    let config = TrainingConfig {
        learning_rate: 0.001,
        weight_decay: 0.0001,
        batch_size: 64,
        epochs: 5, // Short for demo
        grad_clip: 1.0,
        verbose: true,
    };

    let ms_network = MultiScaleScoreNetwork::new(input_dim, 32, 2, 5, 0.01, 1.0);
    let trainer = DenoisingScoreMatchingTrainer::new(ms_network, config);

    let initial_loss = trainer.compute_loss(&train_data);
    println!("   Initial DSM Loss: {:.4}", initial_loss);

    // 7. Demonstrate Langevin dynamics
    println!("\n7. Langevin Dynamics Demo...");
    let network = MultiScaleScoreNetwork::new(input_dim, 32, 2, 5, 0.01, 1.0);
    let langevin_config = LangevinConfig {
        n_steps_per_level: 10,
        step_size: 0.01,
        use_annealing: true,
        final_noise: 0.0,
    };
    let langevin = LangevinDynamics::new(&network, langevin_config);

    // Denoise a noisy sample
    let noisy_sample = Array1::from_vec(vec![2.0, -1.5, 0.8, 0.3, -0.5, 1.2, -0.8, 0.1, 0.6, -0.3]);
    let denoised = langevin.denoise(&noisy_sample, 20);
    println!("   Noisy:    {:?}", &noisy_sample.as_slice().unwrap()[..5]);
    println!("   Denoised: {:?}", &denoised.as_slice().unwrap()[..5]);

    // 8. Trading signal generation
    println!("\n8. Trading Signal Demo...");
    let market_state = Array1::from_vec(vec![
        0.02,  // return_1
        0.05,  // return_5
        0.08,  // return_10
        0.12,  // return_20
        0.015, // volatility
        1.2,   // volatility_ratio
        0.10,  // momentum
        1.5,   // volume_ratio
        0.7,   // price_position
        0.6,   // rsi
    ]);

    let (signal, confidence) = langevin.get_trading_signal(&market_state, 0);
    println!("   Market State Summary:");
    println!("     - Returns: 1d={:.2}%, 5d={:.2}%, 20d={:.2}%",
             market_state[0] * 100.0,
             market_state[1] * 100.0,
             market_state[3] * 100.0);
    println!("     - Volatility: {:.4}", market_state[4]);
    println!("     - Momentum: {:.2}%", market_state[6] * 100.0);
    println!("   Signal: {:.2} ({})",
             signal,
             if signal > 0.0 { "LONG" } else if signal < 0.0 { "SHORT" } else { "NEUTRAL" });
    println!("   Confidence: {:.2}%", confidence * 100.0);

    // 9. Probability estimation
    println!("\n9. Distribution Analysis...");
    let log_density = langevin.estimate_log_density(&market_state, 50);
    let is_in_dist = langevin.is_in_distribution(&market_state, -5.0);
    println!("   Log Density (relative): {:.4}", log_density);
    println!("   In Distribution: {}", is_in_dist);

    // 10. Summary
    println!("\n=== Summary ===");
    println!("Score Matching allows us to:");
    println!("  1. Learn the distribution of market states");
    println!("  2. Estimate score (gradient of log probability)");
    println!("  3. Use score to predict likely market movements");
    println!("  4. Detect unusual (out-of-distribution) market conditions");
    println!("  5. Generate trading signals with confidence levels");

    println!("\nFor real trading, you would:");
    println!("  1. Fetch data from Bybit (see bybit_trading example)");
    println!("  2. Train the network on historical data");
    println!("  3. Generate signals in real-time");
    println!("  4. Backtest the strategy (see backtest example)");

    Ok(())
}

/// Generate synthetic data from mixture of Gaussians
fn generate_mixture_of_gaussians(n_samples: usize, dim: usize, n_modes: usize) -> Array2<f64> {
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let mut data = Array2::zeros((n_samples, dim));

    // Generate mode centers
    let center_normal = Normal::new(0.0, 2.0).unwrap();
    let mut centers = Vec::new();
    for _ in 0..n_modes {
        let center: Vec<f64> = (0..dim).map(|_| center_normal.sample(&mut rng)).collect();
        centers.push(center);
    }

    // Sample from mixture
    let mode_normal = Normal::new(0.0, 0.5).unwrap();

    for i in 0..n_samples {
        let mode = rng.gen_range(0..n_modes);
        let center = &centers[mode];

        for j in 0..dim {
            data[[i, j]] = center[j] + mode_normal.sample(&mut rng);
        }
    }

    data
}
