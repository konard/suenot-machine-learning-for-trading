//! Example: Create GAF and MTF images from price data
//!
//! This example demonstrates how to convert time series data
//! into image representations (GASF, GADF, MTF) suitable for
//! EfficientNet processing.

use efficientnet_trading::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("EfficientNet Trading - Image Creation Example");
    println!("==============================================\n");

    // Generate sample price data (simulating 120 hourly candles)
    let mut prices = Vec::with_capacity(120);
    let mut price = 45000.0; // Starting BTC price

    for i in 0..120 {
        // Add some trend and noise
        let trend = 50.0 * (i as f64 * 0.05).sin();
        let noise = (i as f64 * 0.3).cos() * 100.0;
        price = (price + trend + noise).max(40000.0);
        prices.push(price);
    }

    println!("Generated {} price points", prices.len());
    println!("Price range: ${:.2} - ${:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Create GAF transformer
    let gaf = GAFTransformer::new(64);  // Using 64x64 for demo

    // Transform to GASF
    println!("\n1. Creating GASF (Gramian Angular Summation Field)...");
    let gasf = gaf.transform_gasf(&prices);
    println!("   Shape: {}x{}", gasf.dim().0, gasf.dim().1);
    println!("   Value range: [{:.3}, {:.3}]",
        gasf.iter().cloned().fold(f64::INFINITY, f64::min),
        gasf.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Transform to GADF
    println!("\n2. Creating GADF (Gramian Angular Difference Field)...");
    let gadf = gaf.transform_gadf(&prices);
    println!("   Shape: {}x{}", gadf.dim().0, gadf.dim().1);
    println!("   Value range: [{:.3}, {:.3}]",
        gadf.iter().cloned().fold(f64::INFINITY, f64::min),
        gadf.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Create MTF transformer
    println!("\n3. Creating MTF (Markov Transition Field)...");
    let mtf_transformer = MTFTransformer::new(8, 64);  // 8 bins, 64x64 output
    let mtf = mtf_transformer.transform(&prices);
    println!("   Shape: {}x{}", mtf.dim().0, mtf.dim().1);
    println!("   Value range: [{:.3}, {:.3}]",
        mtf.iter().cloned().fold(f64::INFINITY, f64::min),
        mtf.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Create multi-transform RGB stack
    println!("\n4. Creating Multi-Transform Stack (GASF/GADF/MTF as RGB)...");
    let generator = ImageStackGenerator::new(64);
    let rgb_image = generator.create_multi_transform_stack(&prices);
    let (h, w, c) = rgb_image.dim();
    println!("   Shape: {}x{}x{} (H x W x Channels)", h, w, c);

    // Print sample values from each channel
    println!("\n   Sample values from center pixel:");
    let mid = h / 2;
    println!("   Red (GASF):  {:.4}", rgb_image[[mid, mid, 0]]);
    println!("   Green (GADF): {:.4}", rgb_image[[mid, mid, 1]]);
    println!("   Blue (MTF):   {:.4}", rgb_image[[mid, mid, 2]]);

    // Simulate multi-timeframe data
    println!("\n5. Creating Multi-Timeframe Stack...");

    // 1-minute data (more points, more noise)
    let prices_1m: Vec<f64> = (0..120)
        .map(|i| 45000.0 + 100.0 * (i as f64 * 0.1).sin() + 50.0 * (i as f64 * 0.5).cos())
        .collect();

    // 5-minute data (aggregated, less noise)
    let prices_5m: Vec<f64> = (0..60)
        .map(|i| 45000.0 + 80.0 * (i as f64 * 0.08).sin())
        .collect();

    // 15-minute data (even more aggregated)
    let prices_15m: Vec<f64> = (0..30)
        .map(|i| 45000.0 + 50.0 * (i as f64 * 0.05).sin())
        .collect();

    let timeframe_stack = generator.create_timeframe_stack(&prices_1m, &prices_5m, &prices_15m);
    let (h, w, c) = timeframe_stack.dim();
    println!("   Shape: {}x{}x{}", h, w, c);
    println!("   Red: 1-min GASF, Green: 5-min GASF, Blue: 15-min GASF");

    // Convert to bytes (for saving as image)
    println!("\n6. Converting to bytes for image saving...");
    let bytes = generator.to_bytes(&rgb_image);
    println!("   Generated {} bytes", bytes.len());
    println!("   Expected: {} bytes ({}x{}x3)", h * w * 3, h, w);

    // Display ASCII representation of the image
    println!("\n7. ASCII Visualization of GASF (scaled):");
    print_ascii_image(&gasf);

    println!("\nImage creation complete!");
    println!("\nNote: To save actual image files, use the 'image' crate.");

    Ok(())
}

/// Print a simple ASCII visualization of a 2D array
fn print_ascii_image(arr: &ndarray::Array2<f64>) {
    let (h, w) = arr.dim();
    let scale_h = 10;
    let scale_w = 40;

    let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

    // Normalize to [0, 1]
    let min_val = arr.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    println!("   +" + &"-".repeat(scale_w) + "+");

    for i in 0..scale_h {
        let y = i * h / scale_h;
        print!("   |");
        for j in 0..scale_w {
            let x = j * w / scale_w;
            let val = if range > 0.0 {
                (arr[[y, x]] - min_val) / range
            } else {
                0.5
            };
            let idx = ((val * 9.0) as usize).min(9);
            print!("{}", chars[idx]);
        }
        println!("|");
    }

    println!("   +" + &"-".repeat(scale_w) + "+");
}
