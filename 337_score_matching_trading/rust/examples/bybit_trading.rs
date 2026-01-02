//! # Bybit Trading Example
//!
//! Demonstrates fetching data from Bybit and generating trading signals.
//!
//! Run with: cargo run --example bybit_trading

use score_matching_trading::{
    api::BybitClient,
    score::MultiScaleScoreNetwork,
    trading::{ScoreMatchingTrader, TraderConfig},
    utils::compute_market_state,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Score Matching Trading with Bybit ===\n");

    // 1. Create Bybit client
    println!("1. Connecting to Bybit API...");
    let client = BybitClient::new();

    // 2. Fetch available symbols
    println!("\n2. Fetching Available Symbols...");
    match client.get_symbols().await {
        Ok(symbols) => {
            println!("   Found {} perpetual contracts", symbols.len());
            println!("   Examples: {:?}", &symbols[..5.min(symbols.len())]);
        }
        Err(e) => println!("   Warning: Could not fetch symbols: {}", e),
    }

    // 3. Fetch BTCUSDT ticker
    println!("\n3. Fetching BTCUSDT Ticker...");
    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!("   Symbol: {}", ticker.symbol);
            println!("   Last Price: ${}", ticker.last_price);
            println!("   24h Change: {:.2}%", ticker.price_change_pct());
            println!("   24h High: ${}", ticker.high_price_24h);
            println!("   24h Low: ${}", ticker.low_price_24h);
            println!("   24h Volume: {} BTC", ticker.volume_24h);
        }
        Err(e) => println!("   Warning: Could not fetch ticker: {}", e),
    }

    // 4. Fetch historical klines
    println!("\n4. Fetching Historical Data...");
    let symbol = "BTCUSDT";
    let interval = "60"; // 1 hour
    let limit = 200;

    match client.get_klines(symbol, interval, limit).await {
        Ok(candles) => {
            println!("   Fetched {} hourly candles for {}", candles.len(), symbol);

            if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
                println!(
                    "   Date range: {} to {}",
                    first.timestamp.format("%Y-%m-%d %H:%M"),
                    last.timestamp.format("%Y-%m-%d %H:%M")
                );
                println!("   Price range: ${:.2} to ${:.2}",
                    candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
                    candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
                );
            }

            // 5. Compute market state
            println!("\n5. Computing Market State...");
            let state = compute_market_state(&candles, 20);
            println!("   Features (10-dimensional vector):");
            println!("     return_1:        {:.4}%", state[0] * 100.0);
            println!("     return_5:        {:.4}%", state[1] * 100.0);
            println!("     return_10:       {:.4}%", state[2] * 100.0);
            println!("     return_20:       {:.4}%", state[3] * 100.0);
            println!("     volatility:      {:.6}", state[4]);
            println!("     volatility_ratio:{:.4}", state[5]);
            println!("     momentum:        {:.4}%", state[6] * 100.0);
            println!("     volume_ratio:    {:.4}", state[7]);
            println!("     price_position:  {:.4}", state[8]);
            println!("     rsi:             {:.4}", state[9]);

            // 6. Create score network and trader
            println!("\n6. Creating Score Network...");
            let network = MultiScaleScoreNetwork::new(
                10,   // input_dim (matching our features)
                64,   // hidden_dim
                3,    // num_layers
                10,   // num_noise_levels
                0.01, // sigma_min
                1.0,  // sigma_max
            );
            println!("   Network parameters: {}", network.num_parameters());

            let config = TraderConfig {
                return_feature_idx: 0, // Use return_1 for signal
                confidence_threshold: 0.3,
                density_threshold: -5.0,
                denoise_steps: 20,
                lookback: 20,
            };

            let trader = ScoreMatchingTrader::with_config(network, config);

            // 7. Generate trading signal
            println!("\n7. Generating Trading Signal...");
            if let Some(signal) = trader.generate_signal(&candles) {
                println!("   Direction:       {}", signal.direction());
                println!("   Signal Strength: {:.4}", signal.signal);
                println!("   Confidence:      {:.2}%", signal.confidence * 100.0);
                println!("   In Distribution: {}", signal.in_distribution);

                if signal.should_trade(0.3) {
                    let position = signal.position_size(1.0);
                    println!("   Recommended Position: {:.2}%", position * 100.0);
                } else {
                    println!("   Recommendation: No trade (low confidence or out of distribution)");
                }

                if let Some(denoised) = &signal.denoised_state {
                    println!("\n   Denoised State (first 5 features):");
                    for (i, &v) in denoised.iter().take(5).enumerate() {
                        println!("     Feature {}: {:.4}", i, v);
                    }
                }
            }

            // 8. Analyze multiple symbols
            println!("\n8. Multi-Symbol Analysis...");
            let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

            for sym in symbols {
                match client.get_klines(sym, "60", 50).await {
                    Ok(sym_candles) if sym_candles.len() >= 30 => {
                        let sym_state = compute_market_state(&sym_candles, 20);
                        let last_price = sym_candles.last().map(|c| c.close).unwrap_or(0.0);
                        let return_24h = sym_state[3]; // return_20 as proxy

                        println!(
                            "   {}: ${:.2} | 24h: {:+.2}% | Vol Ratio: {:.2} | RSI: {:.2}",
                            sym,
                            last_price,
                            return_24h * 100.0,
                            sym_state[7],
                            sym_state[9]
                        );
                    }
                    Ok(_) => println!("   {}: Insufficient data", sym),
                    Err(e) => println!("   {}: Error - {}", sym, e),
                }
            }
        }
        Err(e) => {
            println!("   Error fetching klines: {}", e);
            println!("\n   Note: Make sure you have internet connectivity.");
            println!("   The Bybit API is freely accessible for market data.");
        }
    }

    // 9. Example: Simulate real-time signal generation
    println!("\n9. Real-Time Signal Generation Pattern...");
    println!("   In production, you would:");
    println!("   1. Connect to Bybit WebSocket for live data");
    println!("   2. Update candle buffer on each new candle");
    println!("   3. Compute market state");
    println!("   4. Generate signal from trained network");
    println!("   5. Execute trades via Bybit API");
    println!("\n   Example loop:");
    println!("   ```");
    println!("   loop {{");
    println!("       let candles = fetch_latest_candles().await?;");
    println!("       let signal = trader.generate_signal(&candles);");
    println!("       if signal.should_trade(0.3) {{");
    println!("           execute_trade(signal).await?;");
    println!("       }}");
    println!("       tokio::time::sleep(Duration::from_secs(60)).await;");
    println!("   }}");
    println!("   ```");

    println!("\n=== Demo Complete ===");

    Ok(())
}
