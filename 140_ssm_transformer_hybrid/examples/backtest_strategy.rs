//! Example: Full backtesting of the SSM-Transformer Hybrid trading strategy.

use ssm_transformer_hybrid::prelude::*;
use ssm_transformer_hybrid::data::bybit::generate_synthetic_klines;
use ssm_transformer_hybrid::data::features::FeatureRow;
use ssm_transformer_hybrid::trading::strategy::StrategyConfig;

fn main() {
    tracing_subscriber::fmt::init();

    println!("=== SSM-Transformer Hybrid: Backtest Strategy ===\n");

    // Generate data
    let klines = generate_synthetic_klines(500);
    println!("Data: {} klines (synthetic hourly)\n", klines.len());

    // Create model
    let model = SSMTransformerModel::new(
        FeatureRow::feature_count(),
        32, 4, 1, 8, 4,
    );
    println!("{}\n", model.summary());

    // Run backtest with different configurations
    let configs = vec![
        ("Conservative", StrategyConfig {
            direction_threshold: 0.60,
            max_position: 0.5,
            base_size: 0.05,
            lookback: 50,
        }),
        ("Moderate", StrategyConfig {
            direction_threshold: 0.55,
            max_position: 1.0,
            base_size: 0.1,
            lookback: 50,
        }),
        ("Aggressive", StrategyConfig {
            direction_threshold: 0.52,
            max_position: 2.0,
            base_size: 0.2,
            lookback: 50,
        }),
    ];

    let engine = BacktestEngine::new(100_000.0, 0.001);

    for (name, config) in configs {
        match engine.run_with_config(&model, &klines, config) {
            Ok(results) => {
                println!("--- {} Strategy ---", name);
                println!("{}\n", results);
            }
            Err(e) => {
                println!("--- {} Strategy ---", name);
                println!("Error: {}\n", e);
            }
        }
    }

    // Detailed backtest with moderate strategy
    let config = StrategyConfig {
        direction_threshold: 0.55,
        max_position: 1.0,
        base_size: 0.1,
        lookback: 50,
    };

    if let Ok(results) = engine.run_with_config(&model, &klines, config) {
        println!("--- Equity Curve (sampled) ---");
        let curve = &results.equity_curve;
        let step = curve.len() / 10;
        for i in (0..curve.len()).step_by(step.max(1)) {
            let pct = (curve[i] / curve[0] - 1.0) * 100.0;
            println!("  Step {:4}: ${:.2} ({:+.2}%)", i, curve[i], pct);
        }
    }

    println!("\nDone!");
}
