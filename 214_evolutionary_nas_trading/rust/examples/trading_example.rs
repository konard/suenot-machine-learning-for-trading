//! Evolutionary NAS Trading Example
//!
//! Fetches BTCUSDT data from Bybit, initializes a population of 20 random
//! architectures, evolves for 50 generations, and shows fitness progression.
//!
//! Usage:
//!   cargo run --example trading_example

use evolutionary_nas_trading::*;
use rand::SeedableRng;

fn main() {
    println!("=== Evolutionary NAS for Trading ===\n");

    // ─── Step 1: Fetch data from Bybit ───
    println!("[1/5] Fetching BTCUSDT data from Bybit...");

    let trading_data = match fetch_bybit_data() {
        Ok(data) => {
            println!(
                "  Fetched {} samples with {} features each.",
                data.features.len(),
                data.features.first().map(|f| f.len()).unwrap_or(0)
            );
            data
        }
        Err(e) => {
            println!("  Warning: Could not fetch Bybit data: {}", e);
            println!("  Falling back to synthetic data...");
            generate_synthetic_data(200, 5, 42)
        }
    };

    // ─── Step 2: Split data ───
    let (train_data, val_data, test_data) = trading_data.split(0.6, 0.2);
    println!(
        "  Train: {} samples, Validation: {} samples, Test: {} samples\n",
        train_data.features.len(),
        val_data.features.len(),
        test_data.features.len()
    );

    let input_dim = train_data.features.first().map(|f| f.len()).unwrap_or(5);

    // ─── Step 3: Initialize evolution engine ───
    println!("[2/5] Initializing population of 20 random architectures...");

    let config = EvolutionConfig {
        population_size: 20,
        num_generations: 50,
        tournament_size: 3,
        mutation_rate: 0.3,
        crossover_rate: 0.7,
        crossover_strategy: CrossoverStrategy::Uniform,
        use_aging: true,
        input_dim,
        output_dim: 1,
    };

    let mut engine = EvolutionEngine::new(config, 42);
    println!("  Population initialized with {} genomes.\n", engine.population.len());

    // ─── Step 4: Evolve ───
    println!("[3/5] Evolving for 50 generations...\n");

    for gen in 0..50 {
        engine.step(&train_data, &val_data);

        if gen % 10 == 0 || gen == 49 {
            let stats = &engine.history[engine.history.len() - 1];
            println!(
                "  Gen {:3} | Best: {:.4} | Mean: {:.4} | Diversity: {:.2} | Pop: {}",
                stats.generation,
                stats.best_fitness,
                stats.mean_fitness,
                stats.diversity,
                stats.population_size,
            );
        }
    }

    // ─── Step 5: Show results ───
    println!("\n[4/5] Fitness Progression:");
    engine.print_convergence();

    println!("\n[5/5] Top 5 Architectures:\n");

    let top5 = engine.top_k(5);
    for (rank, genome) in top5.iter().enumerate() {
        let arch: Vec<String> = genome
            .layers
            .iter()
            .map(|l| format!("{}({:?})", l.size, l.activation))
            .collect();

        println!(
            "  Rank {} | Fitness: {:.4} | Accuracy: {:.4} | Latency: {:.4} | Robustness: {:.4}",
            rank + 1,
            genome.fitness.unwrap_or(0.0),
            genome.accuracy_score,
            genome.latency_score,
            genome.robustness_score,
        );
        println!(
            "         | Pareto Rank: {} | Params: {} | Layers: [{}]",
            genome.pareto_rank,
            genome.param_count(input_dim, 1),
            arch.join(" -> "),
        );
        println!();
    }

    // ─── Evaluate best on test data ───
    println!("=== Test Set Evaluation ===\n");

    if let Some(best) = engine.best_genome() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (test_accuracy, test_latency, test_robustness) =
            evaluate_genome(best, &train_data, &test_data, input_dim, &mut rng);

        println!("  Best architecture on TEST data:");
        println!("    Accuracy:   {:.4}", test_accuracy);
        println!("    Latency:    {:.4}", test_latency);
        println!("    Robustness: {:.4}", test_robustness);
        println!(
            "    Composite:  {:.4}",
            0.6 * test_accuracy + 0.2 * test_latency + 0.2 * test_robustness
        );
    }

    println!("\nDone!");
}

/// Attempt to fetch BTCUSDT kline data from Bybit.
fn fetch_bybit_data() -> anyhow::Result<TradingData> {
    let client = BybitClient::new();
    let klines = client.fetch_klines("BTCUSDT", "60", 200)?;

    if klines.len() < 20 {
        anyhow::bail!("Insufficient data: only {} klines fetched", klines.len());
    }

    println!(
        "  Received {} klines from {} to {}",
        klines.len(),
        klines.first().map(|k| k.timestamp).unwrap_or(0),
        klines.last().map(|k| k.timestamp).unwrap_or(0),
    );

    let data = klines_to_trading_data(&klines, 10);

    if data.features.is_empty() {
        anyhow::bail!("No features generated from kline data");
    }

    Ok(data)
}
