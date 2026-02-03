//! Full trading strategy example: runs a backtest on synthetic market data
//! using the Hierarchical SSM for prediction and signal generation.

use hierarchical_ssm::backtest::{BacktestEngine, BacktestResult};
use hierarchical_ssm::data::bybit::generate_synthetic_candles;
use hierarchical_ssm::model::HiSSConfig;
use hierarchical_ssm::trading::strategy::StrategyConfig;

fn main() {
    println!("=== HiSS Trading Strategy Backtest ===\n");

    // Model configuration
    let model_config = HiSSConfig {
        input_dim: 8,
        hidden_dim: 32,
        output_dim: 3,
        num_levels: 3,
        downsample_factors: vec![4, 4],
        ssm_state_dim: 8,
        dropout: 0.1,
    };

    // Strategy configuration
    let strategy_config = StrategyConfig {
        max_position_pct: 0.2,
        stop_loss_multiplier: 2.0,
        take_profit_multiplier: 3.0,
        max_drawdown_limit: 0.15,
        transaction_cost: 0.001,
    };

    let initial_capital = 100_000.0;
    let window_size = 64;

    println!("Strategy Parameters:");
    println!("  Initial capital:     ${:.0}", initial_capital);
    println!("  Max position:        {:.0}%", strategy_config.max_position_pct * 100.0);
    println!("  Stop-loss:           {:.1}x volatility", strategy_config.stop_loss_multiplier);
    println!("  Take-profit:         {:.1}x volatility", strategy_config.take_profit_multiplier);
    println!("  Max drawdown limit:  {:.0}%", strategy_config.max_drawdown_limit * 100.0);
    println!("  Transaction cost:    {:.1}%", strategy_config.transaction_cost * 100.0);
    println!("  Window size:         {} bars", window_size);
    println!();

    // Create backtest engine
    let engine = BacktestEngine::new(
        model_config,
        strategy_config,
        window_size,
        initial_capital,
    );

    // Run on different synthetic markets
    let scenarios = vec![
        ("BTC-like (high vol)", 50000.0, 500),
        ("ETH-like (medium vol)", 3000.0, 500),
        ("Stable asset", 100.0, 500),
    ];

    for (name, base_price, n_candles) in &scenarios {
        println!("--- {} ({} candles, base ${:.0}) ---", name, n_candles, base_price);

        let candles = generate_synthetic_candles(*n_candles, *base_price);
        let result = engine.run(&candles);

        print_result(&result);
        println!();
    }

    println!("Note: Results use randomly initialized model weights on synthetic data.");
    println!("Train the model on real data for meaningful trading signals.");
}

fn print_result(result: &BacktestResult) {
    println!("  Total Return:    {:+.2}%", result.total_return * 100.0);
    println!("  Sharpe Ratio:    {:.2}", result.sharpe_ratio);
    println!("  Sortino Ratio:   {:.2}", result.sortino_ratio);
    println!("  Max Drawdown:    {:.2}%", result.max_drawdown * 100.0);
    println!("  Win Rate:        {:.1}%", result.win_rate * 100.0);

    if result.profit_factor.is_finite() {
        println!("  Profit Factor:   {:.2}", result.profit_factor);
    } else {
        println!("  Profit Factor:   inf (no losses)");
    }

    println!("  Total Trades:    {}", result.num_trades);

    if let (Some(first), Some(last)) = (result.equity_curve.first(), result.equity_curve.last()) {
        println!("  Equity:          ${:.0} -> ${:.0}", first, last);
    }
}
