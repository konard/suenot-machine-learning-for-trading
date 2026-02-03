//! Example: Fetch data from Bybit and generate trading signals.

use ssm_transformer_hybrid::prelude::*;
use ssm_transformer_hybrid::data::bybit::generate_synthetic_klines;
use ssm_transformer_hybrid::data::features::FeatureRow;
use ssm_transformer_hybrid::trading::strategy::StrategyConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== SSM-Transformer Hybrid: Bybit Trading Example ===\n");

    // Try to fetch from Bybit, fall back to synthetic data
    let client = BybitClient::new();
    let klines = match client.fetch_klines("BTCUSDT", "60", 500).await {
        Ok(data) if !data.is_empty() => {
            println!("Fetched {} klines from Bybit (BTCUSDT 1H)", data.len());
            data
        }
        _ => {
            println!("Using synthetic data (Bybit unavailable)");
            generate_synthetic_klines(500)
        }
    };

    println!(
        "Price range: {:.2} - {:.2}",
        klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max),
    );

    // Create model and strategy
    let model = SSMTransformerModel::new(
        FeatureRow::feature_count(),
        32, 4, 1, 8, 4,
    );

    let config = StrategyConfig {
        direction_threshold: 0.55,
        max_position: 1.0,
        base_size: 0.1,
        lookback: 50,
    };

    let strategy = TradingStrategy::new(model, config);

    // Generate signal for latest data
    if let Some(signal) = strategy.generate_signal(&klines) {
        println!("\n--- Latest Trading Signal ---");
        println!("Direction:     {:?}", signal.direction);
        println!("Position size: {:.4}", signal.position_size);
        println!("Confidence:    {:.1}%", signal.confidence * 100.0);
        println!("Pred. Vol:     {:.6}", signal.predicted_volatility);
        println!("Pred. Return:  {:.6}", signal.predicted_return);
    }

    // Generate all historical signals
    let signals = strategy.generate_all_signals(&klines);
    println!("\n--- Historical Signal Summary ---");
    println!("Total signals: {}", signals.len());

    let long_count = signals.iter()
        .filter(|(s, _)| matches!(s.direction, ssm_transformer_hybrid::trading::signals::SignalDirection::Long))
        .count();
    let short_count = signals.iter()
        .filter(|(s, _)| matches!(s.direction, ssm_transformer_hybrid::trading::signals::SignalDirection::Short))
        .count();
    let flat_count = signals.len() - long_count - short_count;

    println!("Long:  {} ({:.1}%)", long_count, long_count as f64 / signals.len() as f64 * 100.0);
    println!("Short: {} ({:.1}%)", short_count, short_count as f64 / signals.len() as f64 * 100.0);
    println!("Flat:  {} ({:.1}%)", flat_count, flat_count as f64 / signals.len() as f64 * 100.0);

    Ok(())
}
