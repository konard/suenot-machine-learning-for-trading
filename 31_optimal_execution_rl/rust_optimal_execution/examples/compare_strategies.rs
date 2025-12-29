//! Сравнение различных стратегий исполнения
//!
//! ```bash
//! cargo run --example compare_strategies
//! ```

use rust_optimal_execution::{
    agent::{DQNAgent, DQNConfig, Agent},
    baselines::{AlmgrenChrissExecutor, TWAPExecutor, VWAPExecutor},
    environment::{EnvConfig, ExecutionAction, ExecutionEnv},
    impact::ImpactParams,
};

const NUM_EPISODES: usize = 100;

fn run_strategy<F>(env: &mut ExecutionEnv, name: &str, strategy: F) -> Vec<f64>
where
    F: Fn(usize, &EnvConfig) -> f64,
{
    let config = env.config().clone();
    let mut shortfalls = Vec::with_capacity(NUM_EPISODES);

    for _ in 0..NUM_EPISODES {
        let _state = env.reset();
        let mut done = false;
        let mut step = 0;

        while !done && step < config.max_steps {
            let fraction = strategy(step, &config);
            let action = ExecutionAction::Continuous(fraction);
            let result = env.step(action);
            done = result.done;
            step += 1;
        }

        shortfalls.push(env.cumulative_shortfall());
    }

    shortfalls
}

fn main() {
    println!("=== Strategy Comparison ===\n");
    println!("Running {} episodes per strategy...\n", NUM_EPISODES);

    let config = EnvConfig {
        total_quantity: 1000.0,
        max_steps: 30,
        num_actions: 11,
        ..Default::default()
    };

    let mut env = ExecutionEnv::synthetic(50000.0, 0.02, config.clone());

    // TWAP
    let twap = TWAPExecutor::new(config.total_quantity, config.max_steps);
    let twap_schedule = twap.generate_schedule();
    let twap_shortfalls = run_strategy(&mut env, "TWAP", |step, _| {
        twap_schedule.fraction_at(step)
    });

    // VWAP
    let vwap = VWAPExecutor::with_u_shape(config.total_quantity, config.max_steps);
    let vwap_schedule = vwap.generate_schedule();
    let vwap_shortfalls = run_strategy(&mut env, "VWAP", |step, _| {
        vwap_schedule.fraction_at(step)
    });

    // Almgren-Chriss
    let params = ImpactParams::crypto_default();
    let ac = AlmgrenChrissExecutor::from_params(
        config.total_quantity,
        config.max_steps,
        config.risk_aversion,
        &params,
    );
    let ac_schedule = ac.generate_schedule();
    let ac_shortfalls = run_strategy(&mut env, "Almgren-Chriss", |step, _| {
        ac_schedule.fraction_at(step)
    });

    // Aggressive (front-loaded)
    let aggressive_shortfalls = run_strategy(&mut env, "Aggressive", |step, cfg| {
        let remaining = cfg.max_steps - step;
        if remaining > 0 {
            2.0 / (remaining + 1) as f64
        } else {
            1.0
        }
    });

    // Passive (back-loaded)
    let passive_shortfalls = run_strategy(&mut env, "Passive", |step, cfg| {
        let progress = step as f64 / cfg.max_steps as f64;
        0.5 * progress.powi(2) + 0.05
    });

    // Random DQN (untrained, for baseline)
    let dqn_config = DQNConfig {
        state_dim: env.state_dim(),
        num_actions: env.action_dim(),
        ..Default::default()
    };
    let agent = DQNAgent::new(dqn_config);

    let mut random_shortfalls = Vec::with_capacity(NUM_EPISODES);
    for _ in 0..NUM_EPISODES {
        let mut state = env.reset();
        let mut done = false;

        while !done {
            let action = agent.select_action(&state, 1.0); // Full exploration
            let result = env.step(action);
            state = result.state;
            done = result.done;
        }
        random_shortfalls.push(env.cumulative_shortfall());
    }

    // Calculate statistics
    fn stats(shortfalls: &[f64]) -> (f64, f64) {
        let mean = shortfalls.iter().sum::<f64>() / shortfalls.len() as f64;
        let std = (shortfalls.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / shortfalls.len() as f64).sqrt();
        (mean, std)
    }

    let strategies = vec![
        ("TWAP", stats(&twap_shortfalls)),
        ("VWAP (U-shape)", stats(&vwap_shortfalls)),
        ("Almgren-Chriss", stats(&ac_shortfalls)),
        ("Aggressive", stats(&aggressive_shortfalls)),
        ("Passive", stats(&passive_shortfalls)),
        ("Random", stats(&random_shortfalls)),
    ];

    // Print results
    println!("{:<20} {:>15} {:>15} {:>12}", "Strategy", "Mean IS ($)", "Std IS ($)", "Mean (bps)");
    println!("{}", "-".repeat(65));

    let arrival_value = config.total_quantity * 50000.0;

    for (name, (mean, std)) in &strategies {
        let bps = mean / arrival_value * 10000.0;
        println!("{:<20} {:>15.4} {:>15.4} {:>12.4}", name, mean, std, bps);
    }

    // Find best
    let (best_name, (best_mean, _)) = strategies.iter()
        .min_by(|(_, (a, _)), (_, (b, _))| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("\nBest strategy: {} (mean shortfall: ${:.4})", best_name, best_mean);

    // Win rates vs TWAP
    println!("\nWin rates vs TWAP:");
    let twap_mean = stats(&twap_shortfalls).0;

    for (name, shortfalls) in [
        ("VWAP", &vwap_shortfalls),
        ("Almgren-Chriss", &ac_shortfalls),
        ("Aggressive", &aggressive_shortfalls),
        ("Passive", &passive_shortfalls),
    ] {
        let wins = shortfalls.iter()
            .zip(twap_shortfalls.iter())
            .filter(|(s, t)| s < t)
            .count();
        let win_rate = wins as f64 / NUM_EPISODES as f64 * 100.0;
        let (mean, _) = stats(shortfalls);
        let improvement = (twap_mean - mean) / twap_mean.abs() * 100.0;

        println!(
            "  {:<20} Win Rate: {:5.1}%, Improvement: {:+.2}%",
            name, win_rate, improvement
        );
    }

    println!("\n=== Comparison Complete ===");
}
