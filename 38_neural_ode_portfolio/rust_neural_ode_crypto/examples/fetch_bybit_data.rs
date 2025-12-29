//! # Fetch Bybit Data Example
//!
//! Demonstrates how to fetch cryptocurrency data from Bybit exchange.
//!
//! Run with:
//! ```bash
//! cargo run --example fetch_bybit_data
//! ```

use anyhow::Result;
use neural_ode_crypto::data::{BybitClient, TechnicalIndicators, CandleData, Timeframe};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== Bybit Data Fetcher ===");

    // Create client
    let client = BybitClient::new();

    // Define symbols to fetch
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in &symbols {
        info!("Fetching {} hourly data...", symbol);

        // Fetch 500 hourly candles
        let candles = client.get_klines(symbol, "60", 500).await?;

        info!("  Received {} candles", candles.len());

        if !candles.is_empty() {
            let first = &candles[0];
            let last = candles.last().unwrap();

            info!("  Time range: {} to {}",
                first.open_datetime().format("%Y-%m-%d %H:%M"),
                last.open_datetime().format("%Y-%m-%d %H:%M")
            );
            info!("  Price range: ${:.2} to ${:.2}",
                first.close, last.close
            );

            // Create CandleData and calculate features
            let candle_data = CandleData::new(
                symbol.to_string(),
                Timeframe::Hour1,
                candles,
            );

            // Calculate volatility
            let volatility = candle_data.volatility();
            info!("  Volatility: {:.4}%", volatility * 100.0);

            // Calculate technical indicators
            let indicators = TechnicalIndicators::default();
            let features = indicators.calculate_all(&candle_data);

            info!("  Features calculated: {} features", features.n_features);

            // Print some key indicators
            if let Some(asset_features) = features.get_asset(0) {
                for (name, value) in features.names.iter().zip(asset_features.iter()) {
                    info!("    {}: {:.4}", name, value);
                }
            }
        }

        info!("");
    }

    // Fetch ticker data
    info!("=== Current Ticker Data ===");
    for symbol in &symbols {
        match client.get_ticker(symbol).await {
            Ok(ticker) => {
                info!("{}: ${} ({:+.2}%)",
                    ticker.symbol,
                    ticker.last_price,
                    ticker.price_change_24h() * 100.0
                );
            }
            Err(e) => {
                info!("{}: Error fetching ticker: {}", symbol, e);
            }
        }
    }

    info!("\nDone!");
    Ok(())
}
