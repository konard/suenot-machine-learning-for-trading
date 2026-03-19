//! EXAMM Trading Example
//!
//! Demonstrates evolving recurrent neural network architectures for
//! BTCUSDT price prediction using data from Bybit.
//!
//! Usage: cargo run --example trading_example

use examm_trading::*;
use ndarray::Array1;
use rand::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== EXAMM Trading: Evolving Recurrent Architectures for BTC/USDT ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // -----------------------------------------------------------------------
    // Step 1: Fetch data from Bybit (fall back to synthetic if API unavailable)
    // -----------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT 1h kline data from Bybit...");

    let klines = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(k) if !k.is_empty() => {
            println!("    Fetched {} klines from Bybit API.", k.len());
            println!(
                "    Price range: {:.2} - {:.2}",
                k.iter().map(|x| x.low).fold(f64::INFINITY, f64::min),
                k.iter().map(|x| x.high).fold(f64::NEG_INFINITY, f64::max),
            );
            k
        }
        _ => {
            println!("    Bybit API unavailable, using synthetic data...");
            let k = generate_synthetic_data(200, &mut rng);
            println!("    Generated {} synthetic klines.", k.len());
            k
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Prepare training data
    // -----------------------------------------------------------------------
    println!("\n[2] Preparing training data...");
    let all_data = prepare_training_data(&klines);
    let split = (all_data.len() as f64 * 0.7) as usize;
    let train_data = &all_data[..split];
    let test_data = &all_data[split..];
    println!("    Training samples: {}", train_data.len());
    println!("    Test samples: {}", test_data.len());
    println!("    Features per sample: {}", if !all_data.is_empty() { all_data[0].0.len() } else { 0 });

    // -----------------------------------------------------------------------
    // Step 3: Initialize EXAMM with islands using different cell types
    // -----------------------------------------------------------------------
    println!("\n[3] Initializing EXAMM engine...");
    let config = ExammConfig {
        num_islands: 4,
        population_per_island: 15,
        num_generations: 50,
        tournament_size: 3,
        migration_interval: 10,
        crossover_prob: 0.2,
        hidden_size: 8,
        mutation_config: MutationConfig {
            add_node_prob: 0.08,
            add_edge_prob: 0.12,
            change_cell_type_prob: 0.05,
            modify_weights_prob: 0.85,
            toggle_edge_prob: 0.02,
            weight_perturbation_scale: 0.3,
        },
        species_threshold: 3.0,
        c1: 1.0,
        c2: 1.0,
        c3: 0.4,
    };

    let input_size = 2; // log_return + volume
    let output_size = 1; // predicted next log_return

    let mut engine = ExammEngine::new(input_size, output_size, config.clone(), &mut rng);

    println!("    Islands: {}", config.num_islands);
    println!("    Population per island: {}", config.population_per_island);
    println!("    Island cell types:");
    for (i, island) in engine.islands.iter().enumerate() {
        println!("      Island {}: {:?}", i, island.default_cell_type);
    }

    // -----------------------------------------------------------------------
    // Step 4: Evolve for N generations with migration
    // -----------------------------------------------------------------------
    println!("\n[4] Running evolution ({} generations)...", config.num_generations);
    println!("    Migration every {} generations", config.migration_interval);
    println!();

    engine.run(&train_data.to_vec(), &mut rng);

    // -----------------------------------------------------------------------
    // Step 5: Show best evolved architecture
    // -----------------------------------------------------------------------
    println!("\n[5] Best evolved architecture:");
    println!("{}", engine.describe_best());

    // -----------------------------------------------------------------------
    // Step 6: Evaluate best genome on test data
    // -----------------------------------------------------------------------
    println!("\n[6] Evaluating on test data...");

    let best_genome = engine.best_genome.as_ref().unwrap();
    let test_fitness = evaluate_fitness(best_genome, &test_data.to_vec(), config.hidden_size, &mut rng);
    let test_mse = -test_fitness;
    println!("    Evolved architecture test MSE: {:.6}", test_mse);

    // -----------------------------------------------------------------------
    // Step 7: Compare with a standard LSTM baseline
    // -----------------------------------------------------------------------
    println!("\n[7] Comparing with standard LSTM baseline...");

    reset_innovation_counter();
    let lstm_genome = Genome::new_minimal(input_size, output_size, CellType::LSTM, &mut rng);
    let lstm_fitness = evaluate_fitness(&lstm_genome, &test_data.to_vec(), config.hidden_size, &mut rng);
    let lstm_mse = -lstm_fitness;
    println!("    Standard LSTM (minimal) test MSE: {:.6}", lstm_mse);

    // Also compare with a standard GRU
    reset_innovation_counter();
    let gru_genome = Genome::new_minimal(input_size, output_size, CellType::GRU, &mut rng);
    let gru_fitness = evaluate_fitness(&gru_genome, &test_data.to_vec(), config.hidden_size, &mut rng);
    let gru_mse = -gru_fitness;
    println!("    Standard GRU (minimal) test MSE:  {:.6}", gru_mse);

    // Simple RNN baseline
    reset_innovation_counter();
    let rnn_genome = Genome::new_minimal(input_size, output_size, CellType::SimpleRNN, &mut rng);
    let rnn_fitness = evaluate_fitness(&rnn_genome, &test_data.to_vec(), config.hidden_size, &mut rng);
    let rnn_mse = -rnn_fitness;
    println!("    Standard SimpleRNN (minimal) test MSE: {:.6}", rnn_mse);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("  Evolved architecture MSE: {:.6}", test_mse);
    println!("  Standard LSTM MSE:        {:.6}", lstm_mse);
    println!("  Standard GRU MSE:         {:.6}", gru_mse);
    println!("  Standard SimpleRNN MSE:   {:.6}", rnn_mse);

    let best_standard = lstm_mse.min(gru_mse).min(rnn_mse);
    if test_mse < best_standard {
        let improvement = ((best_standard - test_mse) / best_standard * 100.0).abs();
        println!("\n  The evolved architecture outperformed the best standard baseline by {:.1}%!", improvement);
    } else {
        println!("\n  The standard baselines performed comparably or better on this run.");
        println!("  Tip: Try increasing generations, population size, or running multiple seeds.");
    }

    println!("\n=== Done ===");
    Ok(())
}
