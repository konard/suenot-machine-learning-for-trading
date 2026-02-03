//! Trading Example for Attention Rollout
//!
//! This example demonstrates how to use attention rollout for
//! interpretable trading analysis.

use attention_rollout_trading::{
    AttentionRollout, AttentionRolloutConfig, HeadFusion,
    TradingAttentionAnalyzer, AttentionRegime,
    data::fetch_bybit_candles,
    backtest::{calculate_sharpe, calculate_sortino, calculate_max_drawdown},
};
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Attention Rollout Trading Example ===\n");

    // Fetch market data from Bybit
    println!("Fetching BTC/USDT data from Bybit...");
    let candles = fetch_bybit_candles("BTCUSDT", "D", 100)?;
    println!("Fetched {} candles\n", candles.len());

    // Print first few candles
    println!("Sample data:");
    for (i, candle) in candles.iter().take(5).enumerate() {
        println!(
            "  {}: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}, Vol={:.2}",
            i, candle.open, candle.high, candle.low, candle.close, candle.volume
        );
    }
    println!();

    // Create simulated attention matrices (in practice, from transformer model)
    let seq_len = 20;
    let n_heads = 4;
    let n_layers = 3;

    println!("Creating simulated attention matrices...");
    println!("  Sequence length: {}", seq_len);
    println!("  Number of heads: {}", n_heads);
    println!("  Number of layers: {}", n_layers);
    println!();

    let attentions: Vec<Array3<f64>> = (0..n_layers)
        .map(|layer| {
            Array3::from_shape_fn((n_heads, seq_len, seq_len), |(_, _, j)| {
                // Simulate attention pattern with recency bias
                let recency_weight = (j as f64 + 1.0) / seq_len as f64;
                let layer_factor = 1.0 + layer as f64 * 0.1;
                let base = recency_weight * layer_factor;

                // Normalize
                base / (seq_len as f64 * 0.5)
            })
        })
        .collect();

    // Compute attention rollout
    println!("Computing attention rollout...");
    let config = AttentionRolloutConfig {
        head_fusion: HeadFusion::Mean,
        discard_ratio: 0.1,
        add_residual: true,
        residual_weight: 0.5,
    };

    let rollout = AttentionRollout::with_config(config);
    let rollout_matrix = rollout.compute_rollout(&attentions, 0)?;

    println!("Rollout matrix shape: {:?}", rollout_matrix.shape());

    // Get input attribution for last output position
    let attribution = rollout.get_input_attribution(&attentions, seq_len - 1)?;

    println!("\nInput Attribution Scores (most recent = rightmost):");
    println!("{:=<60}", "");
    for (i, &score) in attribution.iter().enumerate() {
        let bar_len = (score * 50.0) as usize;
        let bar: String = "#".repeat(bar_len.min(50));
        println!("t-{:>2}: {:50} {:.4}", seq_len - 1 - i, bar, score);
    }

    // Analyze trading regime
    println!("\n{:=<60}", "");
    println!("Trading Regime Analysis");
    println!("{:=<60}", "");

    let analyzer = TradingAttentionAnalyzer::new();
    let analysis = analyzer.analyze(&attentions)?;

    let regime_str = match analysis.regime {
        AttentionRegime::Momentum => "MOMENTUM (Recent price focus)",
        AttentionRegime::MeanReversion => "MEAN REVERSION (Historical focus)",
        AttentionRegime::Mixed => "MIXED (Balanced attention)",
    };

    println!("Detected Regime: {}", regime_str);
    println!("Recent Bias Score: {:.4}", analysis.recent_bias);
    println!("Attention Concentration: {:.4}", analysis.concentration_score);

    // Calculate backtest metrics using actual price data
    println!("\n{:=<60}", "");
    println!("Backtest Metrics");
    println!("{:=<60}", "");

    // Calculate returns from actual candle data
    let returns: Vec<f64> = candles
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect();

    let sharpe = calculate_sharpe(&returns, 0.02);
    let sortino = calculate_sortino(&returns, 0.02);

    let equity: Vec<f64> = returns
        .iter()
        .scan(100000.0, |capital, &r| {
            *capital *= 1.0 + r;
            Some(*capital)
        })
        .collect();

    let max_dd = calculate_max_drawdown(&equity);

    let total_return = if !equity.is_empty() {
        (equity.last().unwrap_or(&100000.0) - 100000.0) / 100000.0
    } else {
        0.0
    };

    println!("Total Return:  {:.2}%", total_return * 100.0);
    println!("Sharpe Ratio:  {:.4}", sharpe);
    println!("Sortino Ratio: {:.4}", sortino);
    println!("Max Drawdown:  {:.2}%", max_dd * 100.0);

    // Summary
    println!("\n{:=<60}", "");
    println!("Summary");
    println!("{:=<60}", "");
    println!("- The attention rollout analysis shows how the model would");
    println!("  distribute attention across the input sequence.");
    println!("- Higher scores indicate more important time periods.");
    println!("- The detected regime suggests the trading strategy type.");
    println!();
    println!("=== Example Complete ===");

    Ok(())
}
