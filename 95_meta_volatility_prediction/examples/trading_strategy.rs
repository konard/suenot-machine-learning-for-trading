//! Trading strategy example using meta-volatility predictions.
//!
//! Demonstrates running a backtest with volatility-adaptive position sizing
//! driven by the meta-volatility predictor.

use meta_volatility::backtest::engine::{BacktestConfig, BacktestEngine};
use meta_volatility::trading::strategy::StrategyConfig;
use rand::Rng;

/// Generate synthetic price series with regime changes.
fn generate_regime_prices(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut prices = vec![100.0_f64];
    let mut realized_vols = Vec::new();

    for i in 0..n {
        // Regime switching: calm -> volatile -> calm
        let vol = if i < n / 3 {
            0.01 // Low vol regime
        } else if i < 2 * n / 3 {
            0.04 // High vol regime
        } else {
            0.015 // Medium vol regime
        };

        let r: f64 = rng.gen::<f64>() * 2.0 * vol - vol;
        let new_price = prices.last().unwrap() * (1.0 + r);
        prices.push(new_price.max(0.01));
        realized_vols.push(vol);
    }

    (prices, realized_vols)
}

fn main() {
    println!("=== Volatility-Adaptive Trading Strategy ===\n");

    // Generate price series with regime changes
    let n = 500;
    let (prices, realized_vols) = generate_regime_prices(n);
    println!("Generated {} price observations with regime changes", n);

    // Simulate predicted volatilities (slightly noisy version of realized)
    let mut rng = rand::thread_rng();
    let predicted_vols: Vec<f64> = realized_vols
        .iter()
        .map(|&rv| {
            let noise: f64 = rng.gen::<f64>() * 0.005 - 0.0025;
            (rv + noise).max(0.001)
        })
        .collect();

    // Run backtest
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        transaction_cost_bps: 10.0,
        slippage_bps: 5.0,
        strategy_config: StrategyConfig {
            target_daily_risk: 0.02,
            max_position_pct: 0.5,
            min_position_pct: 0.01,
        },
    };

    let engine = BacktestEngine::new(config);
    let result = engine.run(&prices, &predicted_vols, &realized_vols);

    println!();
    BacktestEngine::print_summary(&result);

    // Compare with fixed position sizing
    let fixed_vols = vec![0.02; n]; // Constant vol assumption
    let fixed_config = BacktestConfig::default();
    let fixed_engine = BacktestEngine::new(fixed_config);
    let fixed_result = fixed_engine.run(&prices, &fixed_vols, &realized_vols);

    println!("\n--- Comparison: Fixed vs Adaptive ---");
    println!(
        "  Adaptive Total Return: {:.2}%",
        result.total_return * 100.0
    );
    println!(
        "  Fixed Total Return:    {:.2}%",
        fixed_result.total_return * 100.0
    );
    println!(
        "  Adaptive Sharpe:       {:.3}",
        result.sharpe_ratio
    );
    println!(
        "  Fixed Sharpe:          {:.3}",
        fixed_result.sharpe_ratio
    );
    println!(
        "  Adaptive Max DD:       {:.2}%",
        result.max_drawdown * 100.0
    );
    println!(
        "  Fixed Max DD:          {:.2}%",
        fixed_result.max_drawdown * 100.0
    );

    println!("\nDONE");
}
