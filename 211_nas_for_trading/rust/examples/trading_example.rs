use anyhow::Result;
use nas_for_trading::*;

fn main() -> Result<()> {
    println!("=== Neural Architecture Search for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch BTCUSDT data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT klines from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            eprintln!("Failed to fetch from Bybit: {}. Using synthetic data.", e);
            generate_synthetic_candles(200)
        }
    };

    if candles.is_empty() {
        eprintln!("No candle data available.");
        return Ok(());
    }

    println!(
        "Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max),
    );

    // -----------------------------------------------------------------------
    // Step 2: Prepare features and labels
    // -----------------------------------------------------------------------
    let lookback = 10;
    let (features, labels) = prepare_trading_data(&candles, lookback);
    println!(
        "Prepared {} samples with {} features each\n",
        features.nrows(),
        features.ncols()
    );

    let pos_count = labels.iter().filter(|&&l| l == 1.0).count();
    println!(
        "Label distribution: {:.1}% up, {:.1}% down\n",
        pos_count as f64 / labels.len() as f64 * 100.0,
        (labels.len() - pos_count) as f64 / labels.len() as f64 * 100.0,
    );

    // -----------------------------------------------------------------------
    // Step 3: Define trading-specific search space
    // -----------------------------------------------------------------------
    let space = SearchSpace::trading();
    println!("Search space: max_layers={}, hidden_sizes={:?}", space.max_layers, space.hidden_sizes);
    println!(
        "Layer types: {:?}",
        space.layer_types
    );
    println!();

    // -----------------------------------------------------------------------
    // Step 4: Run random search baseline
    // -----------------------------------------------------------------------
    println!("--- Random Search Baseline (50 architectures) ---");
    let random_results = random_search(50, &space, &features, &labels, 42);
    println!(
        "Best random architecture fitness: {:.4}",
        random_results[0].fitness.unwrap_or(0.0)
    );
    println!("{}\n", random_results[0].summary());

    // -----------------------------------------------------------------------
    // Step 5: Run evolutionary NAS
    // -----------------------------------------------------------------------
    println!("--- Evolutionary NAS ---");
    let config = NasConfig {
        population_size: 20,
        generations: 15,
        tournament_k: 3,
        mutation_prob: 0.8,
        crossover_prob: 0.2,
    };
    println!(
        "Config: pop={}, gens={}, tournament_k={}\n",
        config.population_size, config.generations, config.tournament_k
    );

    let evo_results = run_evolutionary_nas(&config, &space, &features, &labels, 123);

    // -----------------------------------------------------------------------
    // Step 6: Show top architectures
    // -----------------------------------------------------------------------
    println!("\n--- Top 5 Architectures Found ---");
    for (i, arch) in evo_results.iter().take(5).enumerate() {
        println!("\n#{}", i + 1);
        println!("{}", arch.summary());
    }

    // -----------------------------------------------------------------------
    // Step 7: Pareto front analysis
    // -----------------------------------------------------------------------
    println!("\n--- Pareto Front (Accuracy vs Model Size) ---");
    let pareto_points: Vec<ParetoPoint> = evo_results
        .iter()
        .enumerate()
        .map(|(i, a)| ParetoPoint {
            accuracy: a.fitness.unwrap_or(0.0),
            param_count: a.param_count.unwrap_or(0),
            arch_index: i,
        })
        .collect();
    let front_indices = pareto_front(&pareto_points);
    println!("Pareto front contains {} architectures:", front_indices.len());
    for &idx in &front_indices {
        let p = &pareto_points[idx];
        println!(
            "  Arch #{}: accuracy={:.4}, params={}",
            p.arch_index + 1,
            p.accuracy,
            p.param_count
        );
    }

    // -----------------------------------------------------------------------
    // Step 8: Evaluate best architecture in detail
    // -----------------------------------------------------------------------
    println!("\n--- Best Architecture Detailed Evaluation ---");
    let best = &evo_results[0];
    println!("{}", best.summary());
    println!(
        "Estimated FLOPs: {:.0}",
        best.estimated_flops.unwrap_or(0.0)
    );

    // Compare evolutionary vs random
    let evo_best = evo_results[0].fitness.unwrap_or(0.0);
    let rand_best = random_results[0].fitness.unwrap_or(0.0);
    println!("\n--- Comparison ---");
    println!("Evolutionary NAS best: {:.4}", evo_best);
    println!("Random search best:    {:.4}", rand_best);
    println!(
        "Improvement: {:.2}%",
        if rand_best > 0.0 {
            (evo_best - rand_best) / rand_best * 100.0
        } else {
            0.0
        }
    );

    println!("\nDone!");
    Ok(())
}

/// Generate synthetic candles for testing when API is unavailable.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let ret = (rng.gen::<f64>() - 0.48) * 0.02; // slight upward bias
        let open = price;
        price *= 1.0 + ret;
        let close = price;
        let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.005);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.005);
        let volume = 100.0 + rng.gen::<f64>() * 500.0;

        candles.push(Candle {
            timestamp: 1700000000 + i as u64 * 3600,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    candles
}
