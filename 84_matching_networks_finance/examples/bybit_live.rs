//! Bybit Live Data Example
//!
//! This example demonstrates how to fetch real cryptocurrency data from
//! Bybit and use it with the Matching Network for pattern classification.
//!
//! Note: This example requires network access to the Bybit API.

use matching_networks_finance::{
    MatchingNetwork,
    MarketPattern,
    data::{BybitClient, KlineInterval, FeatureExtractor, StockDataLoader},
};
use ndarray::{Array1, Array2};

#[tokio::main]
async fn main() {
    println!("=== Bybit Live Data Example ===\n");

    // Initialize tracing for better debugging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Create Bybit client
    let client = BybitClient::new();

    println!("Fetching available symbols...");
    match client.get_symbols().await {
        Ok(symbols) => {
            let usdt_pairs: Vec<_> = symbols
                .iter()
                .filter(|s| s.ends_with("USDT"))
                .take(5)
                .collect();
            println!("Sample USDT pairs: {:?}", usdt_pairs);
        }
        Err(e) => {
            println!("Warning: Could not fetch symbols: {}", e);
            println!("Continuing with default symbol...");
        }
    }

    // Fetch BTCUSDT kline data
    let symbol = "BTCUSDT";
    let interval = KlineInterval::Hour1;

    println!("\nFetching {} {} klines...", symbol, "1H");

    match client.fetch_klines(symbol, interval, 200, None, None).await {
        Ok(bars) => {
            println!("Fetched {} bars", bars.len());

            if bars.len() < 50 {
                println!("Not enough data for analysis (need at least 50 bars)");
                return;
            }

            // Show recent price action
            println!("\nRecent Price Action:");
            for bar in bars.iter().rev().take(5).rev() {
                let change = (bar.close - bar.open) / bar.open * 100.0;
                let direction = if change >= 0.0 { "↑" } else { "↓" };
                println!(
                    "  {} O:{:.2} H:{:.2} L:{:.2} C:{:.2} ({}{:.2}%)",
                    chrono::DateTime::from_timestamp_millis(bar.timestamp)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_default(),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    direction,
                    change.abs()
                );
            }

            // Create support set from historical data
            // In a real application, you would have labeled historical patterns
            println!("\nCreating support set from synthetic labeled data...");

            let loader = StockDataLoader::new();
            let extractor = FeatureExtractor::new();

            // Generate labeled synthetic data (in practice, use actual labeled data)
            let (support_features, support_labels) = create_support_set(&loader, &extractor);

            if support_features.nrows() == 0 {
                println!("Error: Could not create support set");
                return;
            }

            println!("Support set: {} samples", support_features.nrows());

            // Create matching network
            let network = MatchingNetwork::new(
                support_features.ncols(),
                64,
                32,
                true,
            );

            // Extract features from live data
            println!("\nAnalyzing current market conditions...");
            let window_size = 50;

            for i in (bars.len() - 5)..bars.len() {
                if i < window_size {
                    continue;
                }

                let window = &bars[i - window_size..=i];
                if let Some(features) = extractor.extract(window) {
                    let query = features.features.insert_axis(ndarray::Axis(0));

                    let (predictions, confidences) = network.predict_with_confidence(
                        &support_features,
                        &support_labels,
                        &query,
                    );

                    let pattern = MarketPattern::from_index(predictions[0])
                        .unwrap_or(MarketPattern::Consolidation);
                    let confidence = confidences[0];

                    let timestamp = chrono::DateTime::from_timestamp_millis(bars[i].timestamp)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_default();

                    println!(
                        "  {}: {} (confidence: {:.1}%)",
                        timestamp,
                        pattern,
                        confidence * 100.0
                    );

                    // Suggest action based on pattern
                    let action = pattern.typical_action();
                    println!("      Suggested action: {}", action);
                }
            }

            // Get current ticker
            println!("\nCurrent {} price:", symbol);
            match client.get_ticker(symbol).await {
                Ok(price) => println!("  ${:.2}", price),
                Err(e) => println!("  Error fetching ticker: {}", e),
            }
        }
        Err(e) => {
            println!("Error fetching klines: {}", e);
            println!("\nNote: This example requires network access to Bybit API.");
            println!("If you're behind a firewall or the API is unavailable,");
            println!("the example will not be able to fetch live data.");

            // Fall back to demonstration with synthetic data
            println!("\nDemonstrating with synthetic data instead...");
            demo_with_synthetic_data();
        }
    }

    println!("\n=== Example Complete ===");
}

/// Create a support set from synthetic data
fn create_support_set(
    loader: &StockDataLoader,
    extractor: &FeatureExtractor,
) -> (Array2<f64>, Array1<usize>) {
    let window_size = 50;
    let step = 10;

    let patterns = [
        ("uptrend", MarketPattern::TrendContinuation),
        ("downtrend", MarketPattern::TrendReversal),
        ("sideways", MarketPattern::Consolidation),
        ("breakout", MarketPattern::Breakout),
        ("reversal", MarketPattern::FalseBreakout),
    ];

    let mut all_features = Vec::new();
    let mut all_labels = Vec::new();

    for (pattern_name, pattern) in &patterns {
        let data = loader.generate_pattern_data(pattern_name, 100, 100.0);
        let features = extractor.extract_windows(&data, window_size, step);

        for row in features.rows() {
            all_features.push(row.to_vec());
            all_labels.push(pattern.to_index());
        }
    }

    if all_features.is_empty() {
        return (Array2::zeros((0, 0)), Array1::zeros(0));
    }

    let feature_dim = all_features[0].len();
    let num_samples = all_features.len();
    let flat: Vec<f64> = all_features.into_iter().flatten().collect();

    (
        Array2::from_shape_vec((num_samples, feature_dim), flat).unwrap(),
        Array1::from_vec(all_labels),
    )
}

/// Demonstration with synthetic data when API is unavailable
fn demo_with_synthetic_data() {
    let loader = StockDataLoader::new();
    let extractor = FeatureExtractor::new();

    // Create support set
    let (support_features, support_labels) = create_support_set(&loader, &extractor);

    if support_features.nrows() == 0 {
        println!("Could not create support set");
        return;
    }

    // Create network
    let network = MatchingNetwork::new(
        support_features.ncols(),
        64,
        32,
        true,
    );

    // Generate test scenario
    let test_data = loader.generate_pattern_data("breakout", 100, 100.0);

    println!("\nAnalyzing synthetic breakout pattern:");
    let window_size = 50;

    for i in [50, 60, 70, 80, 90] {
        if i >= test_data.len() {
            continue;
        }

        let window = &test_data[i - window_size..=i];
        if let Some(features) = extractor.extract(window) {
            let query = features.features.insert_axis(ndarray::Axis(0));

            let (predictions, confidences) = network.predict_with_confidence(
                &support_features,
                &support_labels,
                &query,
            );

            let pattern = MarketPattern::from_index(predictions[0])
                .unwrap_or(MarketPattern::Consolidation);

            println!(
                "  Bar {}: {} (confidence: {:.1}%)",
                i,
                pattern,
                confidences[0] * 100.0
            );
        }
    }
}
