//! Complete adaptive trading strategy example.
//!
//! This example demonstrates:
//! - Full end-to-end MAML-based trading
//! - Adaptive strategy with market regime detection
//! - Backtesting with realistic metrics
//! - Performance evaluation

use maml_trading::{
    backtest::engine::{BacktestConfig, BacktestEngine},
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    maml::algorithm::{MAMLTrainer, TaskData},
    model::network::TradingModel,
    trading::strategy::TradingStrategy,
};

fn main() {
    println!("=== Adaptive MAML Trading Strategy Example ===\n");

    // Phase 1: Generate diverse market regimes for meta-training
    println!("Phase 1: Generating market regime data...");
    let feature_gen = FeatureGenerator::default_window();

    let regimes = vec![
        ("Bull Market", 0.015, 0.0003),
        ("Bear Market", 0.02, -0.0003),
        ("Sideways Low Vol", 0.008, 0.0),
        ("Sideways High Vol", 0.025, 0.0),
        ("Recovery", 0.018, 0.0002),
        ("Crash", 0.04, -0.001),
    ];

    let mut tasks = Vec::new();
    for (name, vol, trend) in &regimes {
        let klines = SimulatedDataGenerator::generate_trending_klines(400, 50000.0, *vol, *trend);
        let features = feature_gen.compute_features(&klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        let split = features.len() / 2;
        let support_features = features[..split].to_vec();
        let support_labels = returns[..split].to_vec();
        let query_features = features[split..].to_vec();
        let query_labels = returns[split..returns.len().min(split + query_features.len())].to_vec();

        if support_features.len() == support_labels.len()
            && query_features.len() == query_labels.len()
            && !query_labels.is_empty()
        {
            tasks.push(TaskData {
                support_features,
                support_labels,
                query_features,
                query_labels,
            });
            println!("  {} regime added", name);
        }
    }
    println!();

    // Phase 2: Meta-train MAML
    println!("Phase 2: Meta-training MAML...");
    let model = TradingModel::new(11, 64, 1);
    let mut trainer = MAMLTrainer::new(model, 0.01, 0.001, 5, true);

    for epoch in 0..30 {
        let loss = trainer.meta_train_step(&tasks);
        if (epoch + 1) % 10 == 0 {
            println!("  Epoch {:2}: Loss = {:.6}", epoch + 1, loss);
        }
    }
    println!();

    // Phase 3: Simulate live trading scenario
    println!("Phase 3: Simulating live trading...\n");

    // Generate test market (mixed conditions)
    let test_data = SimulatedDataGenerator::generate_regime_changing_klines(1000, 50000.0);
    println!("Generated {} candles of test data\n", test_data.len());

    // Create trading strategy (shown for reference - backtesting uses trainer directly)
    let _strategy = TradingStrategy::new(
        trainer.base_model().clone(),
        0.01,  // inner_lr
        0.001, // outer_lr
        5,     // inner_steps
        true,  // first_order (FOMAML)
    )
    .with_thresholds(0.001, 0.005)
    .with_risk_management(0.02, 0.03);

    // Phase 4: Backtest with adaptation
    println!("Phase 4: Running backtest...\n");

    let config = BacktestConfig {
        initial_capital: 10000.0,
        transaction_cost: 0.001,
        slippage: 0.0005,
        threshold: 0.001,
        adaptation_window: 30,
        adaptation_steps: 5,
    };

    let engine = BacktestEngine::new(config);
    let results = engine.run(&trainer, &test_data);

    // Phase 5: Display results
    println!("=== Backtest Results ===\n");
    println!("Capital:");
    println!("  Initial:  ${:.2}", results.initial_capital);
    println!("  Final:    ${:.2}", results.final_capital);
    println!();

    println!("Returns:");
    println!("  Total Return:     {:+.2}%", results.total_return * 100.0);
    println!("  Annualized:       {:+.2}%", results.annualized_return * 100.0);
    println!("  Volatility (ann): {:.2}%", results.annualized_volatility * 100.0);
    println!();

    println!("Risk Metrics:");
    println!("  Sharpe Ratio:  {:.3}", results.sharpe_ratio);
    println!("  Sortino Ratio: {:.3}", results.sortino_ratio);
    println!("  Max Drawdown:  {:.2}%", results.max_drawdown * 100.0);
    println!();

    println!("Trading Statistics:");
    println!("  Total Trades:  {}", results.num_trades);
    println!("  Win Rate:      {:.1}%", results.win_rate * 100.0);
    println!(
        "  Profit Factor: {}",
        if results.profit_factor.is_finite() {
            format!("{:.2}", results.profit_factor)
        } else {
            "Inf (no losses)".to_string()
        }
    );
    println!();

    // Show sample trades
    if !results.trades.is_empty() {
        println!("Sample Trades (first 5):");
        for (i, trade) in results.trades.iter().take(5).enumerate() {
            let direction = if trade.direction > 0 { "LONG" } else { "SHORT" };
            println!(
                "  {}. {} Entry: ${:.2} -> Exit: ${:.2} | P&L: {:+.2}%",
                i + 1,
                direction,
                trade.entry_price,
                trade.exit_price,
                trade.pnl_pct * 100.0
            );
        }
        println!();
    }

    // Phase 6: Compare with buy-and-hold
    println!("=== Comparison with Buy & Hold ===\n");

    let first_price = test_data.first().map(|k| k.close).unwrap_or(50000.0);
    let last_price = test_data.last().map(|k| k.close).unwrap_or(50000.0);
    let buy_hold_return = (last_price / first_price) - 1.0;

    println!("  MAML Strategy: {:+.2}%", results.total_return * 100.0);
    println!("  Buy & Hold:    {:+.2}%", buy_hold_return * 100.0);

    let outperformance = results.total_return - buy_hold_return;
    if outperformance > 0.0 {
        println!(
            "\n  MAML outperformed by {:.2}%!",
            outperformance * 100.0
        );
    } else {
        println!(
            "\n  Buy & Hold outperformed by {:.2}%",
            (-outperformance) * 100.0
        );
    }

    // Risk-adjusted comparison
    if results.max_drawdown > 0.0 {
        let calmar = results.annualized_return / results.max_drawdown;
        println!("  Calmar Ratio (Risk-Adjusted): {:.2}", calmar);
    }

    println!("\n=== Trading Strategy Example Complete ===");
}
