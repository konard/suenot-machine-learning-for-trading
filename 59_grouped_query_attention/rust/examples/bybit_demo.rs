//! Bybit Data Demo
//!
//! This example demonstrates loading real cryptocurrency data from Bybit
//! and making predictions with the GQA model.

use gqa_trading::{
    data::{generate_synthetic_data, load_bybit_data, normalize_data},
    model::GQATrader,
    predict::{analyze_prediction, predict_next},
};

fn main() {
    env_logger::init();

    println!("GQA Trading - Bybit Data Demo");
    println!("==============================\n");

    // Try to load real data from Bybit
    println!("Attempting to load BTC/USDT data from Bybit...");
    let data = match load_bybit_data("BTCUSDT", "1h", 200) {
        Ok(data) => {
            println!("✓ Successfully loaded {} candles", data.len());
            println!("  Symbol: {}", data.symbol);
            println!("  Interval: {}", data.interval);
            println!("  Latest close: ${:.2}", data.latest_close());
            data
        }
        Err(e) => {
            println!("✗ Failed to load from Bybit: {}", e);
            println!("  Using synthetic data instead...\n");
            generate_synthetic_data(200, 50000.0, 0.02)
        }
    };

    // Display some price statistics
    let prices = data.close_prices();
    let min_price = prices.iter().fold(f32::MAX, |a, &b| a.min(b));
    let max_price = prices.iter().fold(f32::MIN, |a, &b| a.max(b));
    let avg_price: f32 = prices.iter().sum::<f32>() / prices.len() as f32;

    println!("\nPrice Statistics:");
    println!("  Min:  ${:.2}", min_price);
    println!("  Max:  ${:.2}", max_price);
    println!("  Avg:  ${:.2}", avg_price);
    println!("  Range: ${:.2} ({:.2}%)", max_price - min_price, (max_price - min_price) / avg_price * 100.0);

    // Normalize data
    println!("\nNormalizing data...");
    let (normalized, _params) = normalize_data(&data.data, "zscore").expect("Normalization failed");
    println!("  Normalized shape: {:?}", normalized.shape());

    // Create model
    println!("\nCreating GQA model...");
    let model = GQATrader::new(5, 64, 8, 2, 4);

    // Make predictions on recent data
    let seq_len: usize = 60;
    if data.len() >= seq_len {
        println!("\nMaking prediction on latest {} candles...", seq_len);

        let start_idx = normalized.shape()[0] - seq_len;
        let sequence = normalized.slice(ndarray::s![start_idx.., ..]).to_owned();
        let analysis = analyze_prediction(&model, &sequence);

        println!("\n┌─────────────────────────────────────┐");
        println!("│         PREDICTION ANALYSIS         │");
        println!("├─────────────────────────────────────┤");
        println!("│ Signal:      {:^21} │", format!("{}", analysis.signal));
        println!("│ Prediction:  {:^21} │", analysis.prediction_label);
        println!("│ Confidence:  {:^21} │", format!("{:.1}%", analysis.confidence * 100.0));
        println!("├─────────────────────────────────────┤");
        println!("│ Probabilities:                      │");
        println!("│   DOWN:      {:>18.1}%  │", analysis.probabilities.down * 100.0);
        println!("│   NEUTRAL:   {:>18.1}%  │", analysis.probabilities.neutral * 100.0);
        println!("│   UP:        {:>18.1}%  │", analysis.probabilities.up * 100.0);
        println!("├─────────────────────────────────────┤");
        println!("│ Entropy:     {:^21} │", format!("{:.3}", analysis.entropy));
        println!("│ Action:      {:^21} │", analysis.recommended_action);
        println!("└─────────────────────────────────────┘");

        // Make predictions at different points
        println!("\nPrediction history (last 5 sequences):");
        println!("  {:>5} │ {:>10} │ {:>10} │ {:>10}", "Idx", "Signal", "Conf", "Price");
        println!("  ──────┼────────────┼────────────┼────────────");

        for i in 0..5 {
            if data.len() >= seq_len + 5 {
                let offset = data.len() - seq_len - 5 + i;
                let seq = normalized.slice(ndarray::s![offset..offset + seq_len, ..]).to_owned();
                let result = predict_next(&model, &seq);
                let price = data.data[[offset + seq_len - 1, 3]];

                println!(
                    "  {:>5} │ {:>10} │ {:>9.1}% │ ${:>9.2}",
                    offset + seq_len,
                    format!("{}", result.signal),
                    result.confidence * 100.0,
                    price
                );
            }
        }
    }

    println!("\n✓ Demo completed successfully!");
}
