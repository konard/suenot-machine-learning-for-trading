//! Trading example: Hyperparameter optimization for a momentum strategy on BTCUSDT.
//!
//! Fetches data from Bybit, defines a search space, and compares grid search,
//! random search, and Bayesian optimization.
//!
//! Usage:
//!   cargo run --example trading_example

use hyperparameter_optimization::*;
use rand::Rng;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Hyperparameter Optimization for Trading ===\n");

    // -----------------------------------------------------------------------
    // 1. Fetch BTCUSDT data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT daily data from Bybit...");
    let client = BybitClient::new();
    let candles = match client.fetch_klines("BTCUSDT", "D", 365) {
        Ok(c) => {
            println!("Fetched {} candles from Bybit.\n", c.len());
            c
        }
        Err(e) => {
            println!("Could not fetch from Bybit ({}). Using synthetic data.\n", e);
            generate_synthetic_candles(365)
        }
    };

    if candles.len() < 50 {
        println!("Not enough candles for meaningful optimization. Using synthetic data.");
        let candles = generate_synthetic_candles(365);
        run_optimization(&candles);
    } else {
        run_optimization(&candles);
    }

    Ok(())
}

fn run_optimization(candles: &[Candle]) {
    // -----------------------------------------------------------------------
    // 2. Define the hyperparameter search space
    // -----------------------------------------------------------------------
    let mut space = SearchSpace::new();
    space.add_discrete("lookback", vec![5, 10, 20, 30, 50]);
    space.add_continuous("threshold", 0.001, 0.1, true);
    space.add_continuous("position_size", 0.1, 1.0, false);
    space.add_continuous("stop_loss", 0.01, 0.1, true);

    println!("Search space:");
    for param in &space.params {
        println!("  - {}: {:?}", param.name, param.param_type);
    }
    println!();

    // -----------------------------------------------------------------------
    // 3. Define objective: walk-forward validation with momentum strategy
    // -----------------------------------------------------------------------
    let n_folds = 3;

    let objective = |config: &Config| -> f64 {
        walk_forward_objective(candles, config, n_folds, &momentum_strategy)
    };

    // -----------------------------------------------------------------------
    // 4. Run Grid Search
    // -----------------------------------------------------------------------
    println!("--- Grid Search ---");
    let start = Instant::now();
    let grid_history = grid_search(&space, &objective, 3);
    let grid_time = start.elapsed();

    let grid_best = grid_history.best().unwrap();
    println!(
        "Trials evaluated: {}",
        grid_history.len()
    );
    println!("Best score: {:.4}", grid_best.score);
    print_config(&grid_best.config);
    println!("Time: {:.2?}\n", grid_time);

    // -----------------------------------------------------------------------
    // 5. Run Random Search
    // -----------------------------------------------------------------------
    println!("--- Random Search ---");
    let start = Instant::now();
    let random_history = random_search(&space, &objective, 50, 42);
    let random_time = start.elapsed();

    let random_best = random_history.best().unwrap();
    println!(
        "Trials evaluated: {}",
        random_history.len()
    );
    println!("Best score: {:.4}", random_best.score);
    print_config(&random_best.config);
    println!("Time: {:.2?}\n", random_time);

    // -----------------------------------------------------------------------
    // 6. Run Bayesian Optimization
    // -----------------------------------------------------------------------
    println!("--- Bayesian Optimization (GP + EI) ---");
    let start = Instant::now();
    let bo_history = bayesian_optimization(&space, &objective, 30, 5, 42);
    let bo_time = start.elapsed();

    let bo_best = bo_history.best().unwrap();
    println!(
        "Trials evaluated: {}",
        bo_history.len()
    );
    println!("Best score: {:.4}", bo_best.score);
    print_config(&bo_best.config);
    println!("Time: {:.2?}\n", bo_time);

    // -----------------------------------------------------------------------
    // 7. Comparison Summary
    // -----------------------------------------------------------------------
    println!("=== Comparison Summary ===");
    println!(
        "{:<25} {:>10} {:>12} {:>10}",
        "Method", "Trials", "Best Score", "Time"
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:<25} {:>10} {:>12.4} {:>10.2?}",
        "Grid Search",
        grid_history.len(),
        grid_best.score,
        grid_time
    );
    println!(
        "{:<25} {:>10} {:>12.4} {:>10.2?}",
        "Random Search",
        random_history.len(),
        random_best.score,
        random_time
    );
    println!(
        "{:<25} {:>10} {:>12.4} {:>10.2?}",
        "Bayesian Optimization",
        bo_history.len(),
        bo_best.score,
        bo_time
    );
    println!();

    // -----------------------------------------------------------------------
    // 8. Show convergence: best score at each step for BO
    // -----------------------------------------------------------------------
    println!("=== Bayesian Optimization Convergence ===");
    let mut running_best = f64::NEG_INFINITY;
    for (i, trial) in bo_history.trials.iter().enumerate() {
        if trial.score > running_best {
            running_best = trial.score;
        }
        if (i + 1) % 5 == 0 || i == 0 {
            println!("  Trial {:>3}: current = {:.4}, best so far = {:.4}", i + 1, trial.score, running_best);
        }
    }
    println!();

    println!("=== Best Hyperparameters (Bayesian Optimization) ===");
    print_config_detailed(&bo_best.config);
}

fn print_config(config: &Config) {
    let mut parts: Vec<String> = config
        .iter()
        .map(|(k, v)| match v {
            HyperparameterValue::Continuous(x) => format!("{}={:.6}", k, x),
            HyperparameterValue::Discrete(x) => format!("{}={}", k, x),
            HyperparameterValue::Categorical(x) => format!("{}={}", k, x),
        })
        .collect();
    parts.sort();
    println!("Config: {{{}}}", parts.join(", "));
}

fn print_config_detailed(config: &Config) {
    let mut keys: Vec<&String> = config.keys().collect();
    keys.sort();
    for key in keys {
        match config.get(key).unwrap() {
            HyperparameterValue::Continuous(x) => println!("  {}: {:.6}", key, x),
            HyperparameterValue::Discrete(x) => println!("  {}: {}", key, x),
            HyperparameterValue::Categorical(x) => println!("  {}: {}", key, x),
        }
    }
}

/// Generate synthetic candle data with trend and noise for offline testing.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

    let mut price = 30000.0_f64;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        // Trending + mean-reverting + noise
        let trend = 0.0002 * (i as f64 * 0.02).sin();
        let noise: f64 = (rng.gen::<f64>() - 0.5) * 0.04;
        let ret = trend + noise;

        let open = price;
        price *= 1.0 + ret;
        let close = price;
        let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.01);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.01);
        let volume = 1000.0 + rng.gen::<f64>() * 5000.0;

        candles.push(Candle {
            timestamp: (1700000000000 + i as u64 * 86400000),
            open,
            high,
            low,
            close,
            volume,
        });
    }

    candles
}
