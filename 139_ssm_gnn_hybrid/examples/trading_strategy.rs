//! Full SSM-GNN trading strategy with backtesting.
//!
//! Demonstrates the complete pipeline: data loading, feature engineering,
//! signal generation, and backtesting with performance metrics.

use ssm_gnn_hybrid::backtest::engine::{run_backtest, BacktestConfig};
use ssm_gnn_hybrid::data::bybit::generate_simulated_klines;
use ssm_gnn_hybrid::trading::strategy::{SsmGnnStrategy, StrategyConfig};

fn main() {
    println!("=== SSM-GNN Hybrid - Trading Strategy Backtest ===\n");

    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"];

    // Generate simulated data
    let klines: Vec<Vec<_>> = symbols
        .iter()
        .map(|s| generate_simulated_klines(s, 200))
        .collect();

    println!("Generated {} candles for {} assets", 200, symbols.len());

    // Create strategy
    let config = StrategyConfig::default();
    let strategy = SsmGnnStrategy::new(config, 5);

    // Generate rolling signals
    let n_periods = klines[0].len();
    let n_assets = symbols.len();
    let window = 80;

    let mut all_signals = vec![vec![0.0; n_assets]; n_periods];
    let mut all_confidences = vec![vec![1.0; n_assets]; n_periods];

    println!("Generating rolling signals...");
    for t in window..n_periods {
        let windowed_klines: Vec<Vec<_>> = klines
            .iter()
            .map(|k| k[t - window..t].to_vec())
            .collect();

        let weights = strategy.generate_weights(&windowed_klines);
        for a in 0..n_assets {
            all_signals[t][a] = if weights[a] > 0.1 {
                1.0
            } else if weights[a] < -0.1 {
                -1.0
            } else {
                0.0
            };
            all_confidences[t][a] = weights[a].abs().min(1.0);
        }
    }

    // Prepare prices for backtest
    let prices: Vec<Vec<f64>> = (0..n_periods)
        .map(|t| (0..n_assets).map(|a| klines[a][t].close).collect())
        .collect();

    // Run backtest
    println!("Running backtest...\n");
    let bt_config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: 0.001,
        slippage: 0.0005,
        risk_free_rate: 0.02,
        periods_per_year: 8760.0, // hourly data
    };

    let result = run_backtest(&bt_config, &all_signals, &prices, Some(&all_confidences));

    println!("--- Backtest Results ---");
    println!("Total Return:      {:.4}%", result.total_return * 100.0);
    println!("Annualized Return: {:.4}%", result.annualized_return * 100.0);
    println!("Sharpe Ratio:      {:.4}", result.sharpe_ratio);
    println!("Sortino Ratio:     {:.4}", result.sortino_ratio);
    println!("Max Drawdown:      {:.4}%", result.max_drawdown * 100.0);
    println!("Win Rate:          {:.4}%", result.win_rate * 100.0);
    println!("Profit Factor:     {:.4}", result.profit_factor);
    println!("Calmar Ratio:      {:.4}", result.calmar_ratio);
    println!("Number of Trades:  {}", result.n_trades);

    println!("\nDone!");
}
