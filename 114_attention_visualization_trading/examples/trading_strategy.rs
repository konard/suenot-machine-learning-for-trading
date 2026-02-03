//! Full trading strategy with attention-based confidence filtering.
//!
//! This example demonstrates:
//! 1. Loading/generating market data
//! 2. Running a backtest with attention-based trading
//! 3. Comparing different confidence thresholds

use attention_visualization_trading::{
    AttentionTransformer,
    Backtester,
    BybitClient,
    FeatureEngineering,
    TradingStrategy,
};
use attention_visualization_trading::model::TransformerConfig;
use attention_visualization_trading::trading::StrategyConfig;

fn main() {
    println!("=== Attention-Based Trading Strategy ===\n");

    // Create model
    let config = TransformerConfig {
        input_dim: 8,
        d_model: 64,
        n_heads: 4,
        n_layers: 3,
        ..Default::default()
    };
    let model = AttentionTransformer::new(config);
    println!("Created Transformer model");

    // Generate synthetic data (1000 candles = ~41 days of hourly data)
    let client = BybitClient::new("BTCUSDT", "60");
    let candles = client.generate_synthetic_data(1000);
    println!("Generated {} synthetic candles\n", candles.len());

    // Prepare features
    let features = FeatureEngineering::prepare_features(&candles);
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

    // Create sequences
    let seq_len = 60;
    let (sequences, indices) = FeatureEngineering::create_sequences(&features, seq_len);
    let test_prices: Vec<f64> = indices.iter().map(|&i| prices[i]).collect();
    println!("Created {} sequences for backtesting\n", sequences.len());

    // Test different confidence thresholds
    let thresholds = [0.0, 0.3, 0.5, 0.7];

    println!("=== Comparing Confidence Thresholds ===\n");
    println!("{:^12} {:^12} {:^12} {:^12} {:^12} {:^12}",
        "Threshold", "Return", "Sharpe", "MaxDD", "WinRate", "Trades");
    println!("{}", "-".repeat(72));

    for &threshold in &thresholds {
        // Create strategy with specific confidence threshold
        let strategy_config = StrategyConfig {
            confidence_threshold: threshold,
            ..Default::default()
        };
        let strategy = TradingStrategy::with_config(strategy_config);

        // Create backtester (need to clone model for each run)
        let model_clone = AttentionTransformer::new(TransformerConfig {
            input_dim: 8,
            d_model: 64,
            n_heads: 4,
            n_layers: 3,
            ..Default::default()
        });

        let backtester = Backtester::new(
            model_clone,
            strategy,
            10000.0,  // Initial capital
            0.001,    // Transaction cost (0.1%)
        );

        // Run backtest
        let (metrics, trades, _equity) = backtester.run(&sequences, &test_prices);

        println!(
            "{:^12.1} {:^11.2}% {:^12.2} {:^11.2}% {:^11.2}% {:^12}",
            threshold,
            metrics.total_return * 100.0,
            metrics.sharpe_ratio,
            metrics.max_drawdown * 100.0,
            metrics.win_rate * 100.0,
            metrics.num_trades
        );
    }

    // Detailed analysis with best threshold
    println!("\n=== Detailed Analysis (Threshold = 0.5) ===\n");

    let strategy_config = StrategyConfig {
        confidence_threshold: 0.5,
        ..Default::default()
    };
    let strategy = TradingStrategy::with_config(strategy_config);

    let model_final = AttentionTransformer::new(TransformerConfig {
        input_dim: 8,
        d_model: 64,
        n_heads: 4,
        n_layers: 3,
        ..Default::default()
    });

    let backtester = Backtester::new(model_final, strategy, 10000.0, 0.001);
    let (metrics, trades, equity_curve) = backtester.run(&sequences, &test_prices);

    // Print full metrics
    metrics.print_summary();

    // Print sample trades
    if !trades.is_empty() {
        println!("\n=== Sample Trades ===");
        for (i, trade) in trades.iter().take(5).enumerate() {
            println!(
                "Trade {}: {} | Entry: {:.2} @ idx {} | Exit: {:.2} @ idx {} | PnL: {:+.2}% | Conf: {:.2}",
                i + 1,
                trade.signal,
                trade.entry_price,
                trade.entry_idx,
                trade.exit_price,
                trade.exit_idx,
                trade.pnl * 100.0,
                trade.confidence
            );
        }
    }

    // Equity curve summary
    println!("\n=== Equity Curve Summary ===");
    println!("  Starting Capital: ${:.2}", equity_curve.first().unwrap_or(&10000.0));
    println!("  Final Capital:    ${:.2}", equity_curve.last().unwrap_or(&10000.0));
    println!("  Peak Capital:     ${:.2}", equity_curve.iter().cloned().fold(0.0_f64, f64::max));
    println!("  Min Capital:      ${:.2}", equity_curve.iter().cloned().fold(f64::MAX, f64::min));

    println!("\n=== Strategy Complete ===");
}
