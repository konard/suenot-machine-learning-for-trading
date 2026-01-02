//! Generate chart images from market data
//!
//! This example shows how to convert OHLCV data into various image formats.

use efficientnet_trading::api::BybitClient;
use efficientnet_trading::data::Candle;
use efficientnet_trading::imaging::{CandlestickRenderer, GasfRenderer, RecurrencePlot};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Chart Image Generator ===\n");

    // Create output directory
    let output_dir = Path::new("output");
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
    }

    // Fetch data from Bybit
    println!("Fetching market data...");
    let client = BybitClient::new();
    let candles = client.fetch_klines("BTCUSDT", "15", 100).await?;
    println!("Fetched {} candles\n", candles.len());

    // Extract close prices for GASF
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

    // 1. Generate candlestick chart
    println!("Generating candlestick chart...");
    let candlestick_renderer = CandlestickRenderer::new(224, 224);
    let candlestick_img = candlestick_renderer.render(&candles);
    candlestick_img.save(output_dir.join("candlestick.png"))?;
    println!("  Saved: output/candlestick.png");

    // 2. Generate candlestick with moving averages
    println!("Generating candlestick with MAs...");
    let candlestick_ma_img = candlestick_renderer.render_with_ma(&candles, &[7, 25, 50]);
    candlestick_ma_img.save(output_dir.join("candlestick_ma.png"))?;
    println!("  Saved: output/candlestick_ma.png");

    // 3. Generate GASF image
    println!("Generating GASF image...");
    let gasf_renderer = GasfRenderer::gasf(224, 224);
    let gasf_img = gasf_renderer.render(&closes);
    gasf_img.save(output_dir.join("gasf.png"))?;
    println!("  Saved: output/gasf.png");

    // 4. Generate GADF image
    println!("Generating GADF image...");
    let gadf_renderer = GasfRenderer::gadf(224, 224);
    let gadf_img = gadf_renderer.render(&closes);
    gadf_img.save(output_dir.join("gadf.png"))?;
    println!("  Saved: output/gadf.png");

    // 5. Generate Recurrence Plot
    println!("Generating recurrence plot...");
    let recurrence_renderer = RecurrencePlot::new(224, 224)
        .embedding_dim(3)
        .time_delay(1);
    let recurrence_img = recurrence_renderer.render(&closes);
    recurrence_img.save(output_dir.join("recurrence.png"))?;
    println!("  Saved: output/recurrence.png");

    // 6. Generate gradient recurrence plot
    println!("Generating gradient recurrence plot...");
    let recurrence_gradient = recurrence_renderer.render_gradient(&closes);
    recurrence_gradient.save(output_dir.join("recurrence_gradient.png"))?;
    println!("  Saved: output/recurrence_gradient.png");

    // 7. Generate different timeframe images
    println!("\nGenerating multi-resolution images...");
    for (size, name) in [(224, "b0"), (300, "b3"), (456, "b5")] {
        let renderer = CandlestickRenderer::new(size, size);
        let img = renderer.render(&candles);
        let filename = format!("candlestick_{}.png", name);
        img.save(output_dir.join(&filename))?;
        println!("  Saved: output/{} ({}x{})", filename, size, size);
    }

    // Print summary
    println!("\n=== Summary ===");
    println!("Generated images for EfficientNet variants:");
    println!("  - B0: 224x224 (5.3M params, fastest)");
    println!("  - B3: 300x300 (12M params, balanced)");
    println!("  - B5: 456x456 (30M params, most accurate)");
    println!("\nImage types:");
    println!("  - Candlestick: Traditional price chart");
    println!("  - GASF/GADF: Gramian Angular Field encoding");
    println!("  - Recurrence: Phase space trajectory visualization");

    println!("\nDone! Check the 'output' directory for generated images.");
    Ok(())
}
