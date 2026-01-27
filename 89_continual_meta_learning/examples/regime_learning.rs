//! Regime learning example demonstrating sequential market regime acquisition.
//!
//! This example shows:
//! - How EWC prevents forgetting of previously learned regimes
//! - Replay buffer mixing past and current regime data
//! - Quantitative forgetting metrics across regimes

use continual_meta_learning::{
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    continual::algorithm::{ContinualMAMLTrainer, TaskData},
    model::network::TradingModel,
};

fn create_regime_tasks(
    feature_gen: &FeatureGenerator,
    base_price: f64,
    volatility: f64,
    trend: f64,
    regime_id: usize,
    num_tasks: usize,
) -> Vec<TaskData> {
    let mut tasks = Vec::new();

    for _ in 0..num_tasks {
        let klines = SimulatedDataGenerator::generate_trending_klines(
            200, base_price, volatility, trend,
        );
        let features = feature_gen.compute_features(&klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        if features.len() < 40 || returns.len() < 40 {
            continue;
        }

        let split = features.len() / 2;
        let support_features = features[..split].to_vec();
        let support_labels = returns[..split].to_vec();
        let query_features = features[split..].to_vec();
        let query_len = query_features.len().min(returns.len() - split);
        let query_labels = returns[split..split + query_len].to_vec();

        if support_features.len() == support_labels.len()
            && query_features.len() == query_labels.len()
        {
            tasks.push(TaskData::new(
                support_features,
                support_labels,
                query_features,
                query_labels,
                regime_id,
            ));
        }
    }

    tasks
}

fn main() {
    println!("=== Sequential Regime Learning with Continual MAML ===\n");

    let feature_gen = FeatureGenerator::default_window();

    // Define 5 sequential market regimes
    let regimes: Vec<(&str, f64, f64, f64)> = vec![
        ("Bull Market",     50000.0, 0.015, 0.0003),
        ("Bear Market",     55000.0, 0.02,  -0.0003),
        ("Sideways",        48000.0, 0.008, 0.0),
        ("High Volatility", 49000.0, 0.035, 0.0),
        ("Recovery",        45000.0, 0.015, 0.0002),
    ];

    // Generate tasks for each regime
    println!("Generating tasks for {} regimes...", regimes.len());
    let mut all_tasks: Vec<Vec<TaskData>> = Vec::new();
    for (i, (name, price, vol, trend)) in regimes.iter().enumerate() {
        let tasks = create_regime_tasks(&feature_gen, *price, *vol, *trend, i, 5);
        println!("  Regime {}: {} - {} tasks", i, name, tasks.len());
        all_tasks.push(tasks);
    }
    println!();

    // Create trainer
    let model = TradingModel::new(11, 32, 1);
    let mut trainer = ContinualMAMLTrainer::new(
        model, 0.01, 0.001, 5, true, 100.0, 50,
    );

    // Learn regimes one by one, tracking performance on all
    println!("=== Sequential Learning ===\n");
    for regime_idx in 0..regimes.len() {
        let (name, _, _, _) = regimes[regime_idx];
        println!("--- Learning Regime {}: {} ---", regime_idx, name);

        let losses = trainer.learn_regime(
            &all_tasks[regime_idx],
            regime_idx,
            15,
            5,
        );
        println!("  Training losses: first={:.6} -> last={:.6}",
            losses.first().unwrap_or(&0.0),
            losses.last().unwrap_or(&0.0)
        );

        // Evaluate on ALL regimes (including future ones as baseline)
        println!("  Performance on all regimes after learning {}:", name);
        for (eval_idx, (eval_name, _, _, _)) in regimes.iter().enumerate() {
            if !all_tasks[eval_idx].is_empty() {
                let (loss, accuracy) = trainer.evaluate_regime(&all_tasks[eval_idx]);
                let marker = if eval_idx == regime_idx {
                    " <-- current"
                } else if eval_idx < regime_idx {
                    " (past)"
                } else {
                    " (future)"
                };
                println!(
                    "    Regime {} ({}): Loss={:.6}, Accuracy={:.1}%{}",
                    eval_idx, eval_name, loss, accuracy * 100.0, marker
                );
            }
        }
        println!("  Replay buffer: {} tasks, {} regimes\n",
            trainer.replay_buffer_size(),
            trainer.regimes_seen().len()
        );
    }

    // Final summary
    println!("=== Final Evaluation ===");
    println!("Regimes learned: {:?}", trainer.regimes_seen());
    println!("Replay buffer: {} tasks\n", trainer.replay_buffer_size());

    for (i, (name, _, _, _)) in regimes.iter().enumerate() {
        let (loss, accuracy) = trainer.evaluate_regime(&all_tasks[i]);
        println!(
            "  Regime {} ({}): Loss={:.6}, Accuracy={:.1}%",
            i, name, loss, accuracy * 100.0
        );
    }

    println!("\n=== Example Complete ===");
}
