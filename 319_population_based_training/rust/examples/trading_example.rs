use population_based_training::{
    manual_tuning, random_search, BybitClient, ExploitStrategy, PbtConfig, Population,
    TradingEnvironment,
};

fn main() {
    println!("=== Chapter 319: Population Based Training for Trading ===\n");

    // --- Step 1: Fetch data from Bybit or use synthetic fallback ---
    let env = match fetch_bybit_data() {
        Ok(env) => {
            println!(
                "Loaded {} prices for {} from Bybit\n",
                env.prices.len(),
                env.symbol
            );
            env
        }
        Err(e) => {
            println!("Could not fetch Bybit data ({}), using synthetic data\n", e);
            TradingEnvironment::synthetic(500, 40000.0)
        }
    };

    let (train_env, test_env) = env.train_test_split(0.8);
    println!(
        "Train set: {} prices, Test set: {} prices\n",
        train_env.prices.len(),
        test_env.prices.len()
    );

    // --- Step 2: Run PBT ---
    println!("--- Population Based Training ---");
    let pbt_config = PbtConfig {
        population_size: 15,
        weight_dim: 20,
        exploit_strategy: ExploitStrategy::TruncationSelection { top_fraction: 0.2 },
        steps_between_eval: 10,
        warmup_steps: 20,
    };

    let mut population = Population::new(pbt_config);
    let num_generations = 20;

    for gen in 0..num_generations {
        population.step(&train_env);
        let stats = population.stats();

        if gen % 5 == 0 || gen == num_generations - 1 {
            println!(
                "Gen {:3}: best={:+.4}, mean={:+.4}, std={:.4}, worst={:+.4}",
                gen, stats.best, stats.mean, stats.std, stats.worst
            );
        }
    }

    // Evaluate best agent on test set
    let mut best_agent = population.best_agent().clone();
    let pbt_test_perf = best_agent.evaluate(&test_env);
    println!("\nPBT best agent test performance (Sharpe): {:.4}", pbt_test_perf);
    println!(
        "  Learning rate: {:.6}, Lookback: {}, Batch size: {}, Risk penalty: {:.4}, Momentum: {:.4}",
        best_agent.hyperparams.learning_rate,
        best_agent.hyperparams.lookback_window,
        best_agent.hyperparams.batch_size,
        best_agent.hyperparams.risk_penalty,
        best_agent.hyperparams.momentum,
    );

    // --- Step 3: Random Search baseline ---
    println!("\n--- Random Search Baseline ---");
    let total_budget = 15 * 10 * num_generations; // same total training steps as PBT
    let (rs_perf, rs_hp) = random_search(&train_env, 15, total_budget / 15);
    println!("Random search best performance (Sharpe): {:.4}", rs_perf);
    println!(
        "  Learning rate: {:.6}, Lookback: {}, Batch size: {}",
        rs_hp.learning_rate, rs_hp.lookback_window, rs_hp.batch_size
    );

    // --- Step 4: Manual Tuning baseline ---
    println!("\n--- Manual Tuning Baseline ---");
    let (mt_perf, mt_hp) = manual_tuning(&train_env, total_budget);
    println!("Manual tuning performance (Sharpe): {:.4}", mt_perf);
    println!(
        "  Learning rate: {:.6}, Lookback: {}, Batch size: {}",
        mt_hp.learning_rate, mt_hp.lookback_window, mt_hp.batch_size
    );

    // --- Step 5: Summary comparison ---
    println!("\n--- Comparison Summary ---");
    println!("{:<20} {:>15}", "Method", "Train Sharpe");
    println!("{:-<35}", "");
    println!(
        "{:<20} {:>+15.4}",
        "PBT",
        population.best_agent().performance
    );
    println!("{:<20} {:>+15.4}", "Random Search", rs_perf);
    println!("{:<20} {:>+15.4}", "Manual Tuning", mt_perf);

    // --- Step 6: Show hyperparameter evolution ---
    println!("\n--- Hyperparameter Evolution (Best Agent) ---");
    let best = population.best_agent();
    if !best.lr_history.is_empty() {
        println!("Learning rate schedule:");
        for (i, lr) in best.lr_history.iter().enumerate() {
            let bar_len = ((lr.log10() + 5.0) * 10.0).max(0.0) as usize;
            let bar: String = "#".repeat(bar_len.min(40));
            println!("  Step {:3}: {:.6} |{}", i, lr, bar);
        }
    }

    if !best.performance_history.is_empty() {
        println!("\nPerformance history:");
        for (i, perf) in best.performance_history.iter().enumerate() {
            println!("  Step {:3}: {:+.4}", i, perf);
        }
    }

    println!("\nDone! PBT optimized {} agents over {} generations.", 15, num_generations);
}

fn fetch_bybit_data() -> anyhow::Result<TradingEnvironment> {
    let client = BybitClient::new();
    println!("Fetching BTCUSDT 1h klines from Bybit...");
    client.fetch_trading_env("BTCUSDT", "60", 200)
}
