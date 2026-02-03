//! Multi-asset SHAP analysis example.
//!
//! This example demonstrates:
//! 1. Fetching data for multiple assets from Bybit
//! 2. Computing SHAP explanations for each asset
//! 3. Comparing feature importance across assets

use shap_trading::data::bybit::BybitClient;
use shap_trading::model::shap::{LinearModel, ShapExplainer};
use shap_trading::trading::signals::SignalGenerator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Multi-Asset SHAP Analysis ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Assets to analyze
    let assets = vec!["BTCUSDT", "ETHUSDT"];

    // Feature names for our model
    let feature_names = vec![
        "returns_1".to_string(),
        "returns_5".to_string(),
        "returns_10".to_string(),
        "volatility".to_string(),
        "rsi".to_string(),
        "volume_ratio".to_string(),
        "price_ma_ratio".to_string(),
    ];

    // Create model
    let weights = vec![0.5, 0.3, 0.2, -0.4, 0.1, 0.15, 0.25];
    let model = LinearModel::new(weights, 0.0);

    // Signal generator for feature computation
    let signal_gen = SignalGenerator::new(20, 0.1);

    for asset in assets {
        println!("--- {} ---", asset);

        // Fetch market data
        let data = client.fetch_klines(asset, "60", 200).await?;
        println!("Data source: {}", data.source);
        println!("Candles: {}", data.candles.len());

        // Get prices and volumes
        let prices = data.close_prices();
        let volumes = data.volumes();

        if prices.len() < 30 {
            println!("Not enough data for {}\n", asset);
            continue;
        }

        // Generate background data from historical features
        let mut background_data = Vec::new();
        for i in 30..prices.len().saturating_sub(10) {
            let features = signal_gen.compute_features(
                &prices[..=i],
                &volumes[..=i],
            );
            if features.len() == 7 {
                background_data.push(features);
            }
        }

        if background_data.is_empty() {
            println!("Could not generate background data for {}\n", asset);
            continue;
        }

        // Compute base value
        let base_value: f64 = background_data
            .iter()
            .map(|x| model.predict(x))
            .sum::<f64>()
            / background_data.len() as f64;

        println!("Base value: {:.4}", base_value);

        // Create SHAP explainer
        let explainer = ShapExplainer::new(
            feature_names.clone(),
            base_value,
            background_data,
            50,
        );

        // Compute features for latest candle
        let latest_features = signal_gen.compute_features(&prices, &volumes);
        if latest_features.len() != 7 {
            println!("Could not compute features for {}\n", asset);
            continue;
        }

        // Get SHAP explanation
        let explanation = explainer.explain(|x| model.predict(x), &latest_features);

        println!("Latest prediction: {:.4}", explanation.prediction);
        println!(
            "Signal: {}",
            if explanation.prediction > 0.55 {
                "BULLISH"
            } else if explanation.prediction < 0.45 {
                "BEARISH"
            } else {
                "NEUTRAL"
            }
        );

        println!("\nTop feature contributions:");
        for (name, value) in explanation.shap_values.top_contributors(5) {
            let direction = if value > 0.0 { "+" } else { "" };
            println!("  {}: {}{:.4}", name, direction, value);
        }

        // Show latest price info
        if let Some(candle) = data.candles.last() {
            println!("\nLatest candle:");
            println!("  Close: {:.2}", candle.close);
            println!("  Return: {:.2}%", candle.return_pct() * 100.0);
            println!("  Volume: {:.2}", candle.volume);
        }

        println!();
    }

    println!("=== Analysis Complete ===");
    Ok(())
}
