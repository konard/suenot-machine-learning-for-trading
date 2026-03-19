//! # Cross-Silo Federated Learning Trading Example
//!
//! Simulates 3 institutional participants (hedge funds) collaborating via
//! federated learning on Bybit market data. Falls back to synthetic data
//! if the Bybit API is unreachable.
//!
//! Usage:
//!   cargo run --example trading_example

use fl_cross_silo_trading::*;

fn main() {
    println!("==========================================================");
    println!("  Cross-Silo Federated Learning for Trading");
    println!("  Simulating 3 institutional participants");
    println!("==========================================================\n");

    let feature_dim = 3;
    let num_participants = 3;
    let num_rounds = 20;

    // Define institutional participants with different trading pairs
    let institutions = vec![
        ("AlphaCapital", "BTCUSDT", "60"),
        ("BetaHedge", "ETHUSDT", "60"),
        ("GammaQuant", "SOLUSDT", "60"),
    ];

    // Differential privacy configuration
    // clip_norm = 1.0, noise_sigma = 0.05 (moderate privacy)
    let dp = DifferentialPrivacy::new(1.0, 0.05);

    // Create institutional clients
    let mut clients: Vec<InstitutionalClient> = institutions
        .iter()
        .enumerate()
        .map(|(id, (name, _symbol, _interval))| {
            InstitutionalClient::new(
                id,
                name.to_string(),
                feature_dim,
                0.005, // learning rate
                5,     // local epochs
                dp.clone(),
            )
        })
        .collect();

    // Fetch data from Bybit for each participant
    println!("--- Data Acquisition Phase ---\n");

    for (i, (_name, symbol, interval)) in institutions.iter().enumerate() {
        print!(
            "  [{}] Fetching {} klines for {}... ",
            clients[i].name, symbol, interval
        );

        match fetch_bybit_klines(symbol, interval, 100) {
            Ok(candles) if !candles.is_empty() => {
                let dataset = candles_to_dataset(&candles);
                println!(
                    "OK - {} candles -> {} training samples",
                    candles.len(),
                    dataset.len()
                );
                clients[i].data = dataset;
            }
            Ok(_) => {
                println!("empty response, using synthetic data");
                clients[i].data =
                    generate_synthetic_data(80 + i * 20, feature_dim, (i as u64 + 1) * 777);
                println!(
                    "         Generated {} synthetic samples",
                    clients[i].data.len()
                );
            }
            Err(e) => {
                println!("FAILED ({}), using synthetic data", e);
                clients[i].data =
                    generate_synthetic_data(80 + i * 20, feature_dim, (i as u64 + 1) * 777);
                println!(
                    "         Generated {} synthetic samples",
                    clients[i].data.len()
                );
            }
        }
    }

    // Print data distribution summary
    println!("\n--- Data Distribution ---\n");
    let total_samples: usize = clients.iter().map(|c| c.data_size()).sum();
    for client in &clients {
        let pct = 100.0 * client.data_size() as f64 / total_samples as f64;
        println!(
            "  [{}]: {} samples ({:.1}% of total)",
            client.name,
            client.data_size(),
            pct
        );
    }
    println!("  Total: {} samples across {} silos\n", total_samples, num_participants);

    // Demonstrate secure aggregation
    println!("--- Secure Aggregation Verification ---\n");
    let sa = SecureAggregator::new(num_participants);
    let test_updates: Vec<ndarray::Array1<f64>> = (0..num_participants)
        .map(|i| ndarray::Array1::from(vec![i as f64 * 0.1; feature_dim + 1]))
        .collect();

    let masked: Vec<ndarray::Array1<f64>> = test_updates
        .iter()
        .enumerate()
        .map(|(i, u)| sa.mask(i, u))
        .collect();

    let agg_sum = sa.aggregate(&masked);
    let true_sum: ndarray::Array1<f64> = test_updates
        .iter()
        .fold(ndarray::Array1::zeros(feature_dim + 1), |acc, u| acc + u);

    let max_error = (&agg_sum - &true_sum)
        .iter()
        .map(|x| x.abs())
        .fold(0.0f64, f64::max);
    println!("  Masks cancel correctly: max error = {:.2e}", max_error);
    println!(
        "  Verification: {}\n",
        if max_error < 1e-10 {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    // Run federated training
    println!("--- Federated Training ({} rounds) ---\n", num_rounds);

    let mut coordinator = FederatedCoordinator::new(feature_dim, num_participants, num_rounds);
    let losses = coordinator.train(&mut clients);

    // Summary
    println!("\n--- Training Summary ---\n");
    if let (Some(first), Some(last)) = (losses.first(), losses.last()) {
        println!("  Initial loss:  {:.6}", first);
        println!("  Final loss:    {:.6}", last);
        if *first > 0.0 {
            let reduction = (first - last) / first * 100.0;
            println!("  Loss reduction: {:.1}%", reduction);
        }
    }

    // Show final model parameters
    println!("\n--- Global Model Parameters ---\n");
    println!("  Weights: {:?}", coordinator.global_model.weights.to_vec());
    println!("  Bias:    {:.6}", coordinator.global_model.bias);

    // Make a sample prediction
    println!("\n--- Sample Prediction ---\n");
    let sample_features = ndarray::Array1::from(vec![0.01, 0.02, 1.0]);
    let prediction = coordinator.global_model.predict(&sample_features);
    println!(
        "  Features: return=0.01, volatility=0.02, norm_volume=1.0"
    );
    println!("  Predicted next-period return: {:.6}", prediction);

    println!("\n==========================================================");
    println!("  Federated training complete.");
    println!("  {} institutions collaborated without sharing raw data.", num_participants);
    println!("==========================================================");
}
