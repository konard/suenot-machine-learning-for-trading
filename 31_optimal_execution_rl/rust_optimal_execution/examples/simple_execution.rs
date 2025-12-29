//! Простой пример исполнения ордера
//!
//! ```bash
//! cargo run --example simple_execution
//! ```

use rust_optimal_execution::{
    baselines::{AlmgrenChrissExecutor, TWAPExecutor, VWAPExecutor},
    environment::{EnvConfig, ExecutionAction, ExecutionEnv},
    impact::ImpactParams,
};

fn main() {
    println!("=== Simple Execution Example ===\n");

    // Создаём среду с синтетическими данными
    let config = EnvConfig {
        total_quantity: 1000.0,
        max_steps: 20,
        num_actions: 11,
        ..Default::default()
    };

    let mut env = ExecutionEnv::synthetic(50000.0, 0.02, config.clone());

    // 1. TWAP Execution
    println!("1. TWAP Execution");
    println!("-----------------");

    let twap = TWAPExecutor::new(config.total_quantity, config.max_steps);
    let twap_schedule = twap.generate_schedule();

    let _state = env.reset();
    let arrival_price = env.arrival_price();
    println!("Arrival Price: ${:.2}", arrival_price);

    for step in 0..config.max_steps {
        let fraction = twap_schedule.fraction_at(step);
        let action = ExecutionAction::Continuous(fraction);
        let result = env.step(action);

        if step < 5 || step >= config.max_steps - 2 {
            println!(
                "  Step {:2}: Execute {:.2}% -> Reward: {:.4}",
                step + 1,
                fraction * 100.0,
                result.reward
            );
        } else if step == 5 {
            println!("  ...");
        }

        if result.done {
            break;
        }
    }

    println!("TWAP Shortfall: ${:.4}\n", env.cumulative_shortfall());

    // 2. VWAP Execution
    println!("2. VWAP Execution (U-shape)");
    println!("---------------------------");

    let vwap = VWAPExecutor::with_u_shape(config.total_quantity, config.max_steps);
    let vwap_schedule = vwap.generate_schedule();

    let _state = env.reset();

    for step in 0..config.max_steps {
        let fraction = vwap_schedule.fraction_at(step);
        let action = ExecutionAction::Continuous(fraction);
        let result = env.step(action);

        if step < 5 || step >= config.max_steps - 2 {
            println!(
                "  Step {:2}: Execute {:.2}% -> Reward: {:.4}",
                step + 1,
                fraction * 100.0,
                result.reward
            );
        } else if step == 5 {
            println!("  ...");
        }

        if result.done {
            break;
        }
    }

    println!("VWAP Shortfall: ${:.4}\n", env.cumulative_shortfall());

    // 3. Almgren-Chriss Execution
    println!("3. Almgren-Chriss Optimal Execution");
    println!("------------------------------------");

    let params = ImpactParams::crypto_default();
    let ac = AlmgrenChrissExecutor::from_params(
        config.total_quantity,
        config.max_steps,
        config.risk_aversion,
        &params,
    );
    let ac_schedule = ac.generate_schedule();

    println!("Kappa (urgency): {:.4}", ac.kappa());
    println!("Interpretation: {}", ac.urgency_interpretation());

    let _state = env.reset();

    for step in 0..config.max_steps {
        let fraction = ac_schedule.fraction_at(step);
        let action = ExecutionAction::Continuous(fraction);
        let result = env.step(action);

        if step < 5 || step >= config.max_steps - 2 {
            println!(
                "  Step {:2}: Execute {:.2}% -> Reward: {:.4}",
                step + 1,
                fraction * 100.0,
                result.reward
            );
        } else if step == 5 {
            println!("  ...");
        }

        if result.done {
            break;
        }
    }

    println!("A-C Shortfall: ${:.4}\n", env.cumulative_shortfall());

    println!("=== Example Complete ===");
}
