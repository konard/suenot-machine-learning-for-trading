//! DCT Demo Example
//!
//! Demonstrates the Deep Convolutional Transformer model with synthetic data.

use ndarray::Array1;
use rust_dct::{
    data::{DatasetConfig, OHLCV},
    model::{DCTConfig, DCTModel},
    strategy::{BacktestConfig, Backtester},
};

fn main() {
    println!("=== Deep Convolutional Transformer Demo ===\n");

    // Generate synthetic OHLCV data
    println!("1. Generating synthetic market data...");
    let ohlcv = generate_synthetic_data(500);
    println!("   Generated {} data points", ohlcv.len());

    // Prepare dataset
    println!("\n2. Preparing dataset...");
    let dataset_config = DatasetConfig::default();
    let dataset = rust_dct::data::prepare_dataset(&ohlcv, &dataset_config);

    match dataset {
        Some(ds) => {
            println!("   Train samples: {}", ds.x_train.dim().0);
            println!("   Validation samples: {}", ds.x_val.dim().0);
            println!("   Test samples: {}", ds.x_test.dim().0);
            println!("   Features: {}", ds.x_train.dim().2);

            // Count labels
            let mut train_labels = [0, 0, 0];
            for &label in &ds.y_train {
                train_labels[label as usize] += 1;
            }
            println!(
                "   Label distribution: Up={}, Down={}, Stable={}",
                train_labels[0], train_labels[1], train_labels[2]
            );

            // Create DCT model
            println!("\n3. Creating DCT model...");
            let model_config = DCTConfig::new(
                30,                     // seq_len
                ds.x_train.dim().2,     // input_features
                64,                     // d_model
                4,                      // num_heads
                2,                      // num_encoder_layers
            );
            let model = DCTModel::new(model_config);
            println!("   Model parameters: ~{}", model.num_parameters());

            // Run inference on test set
            println!("\n4. Running inference on test set...");
            let predictions = model.predict(&ds.x_test);
            println!("   Generated {} predictions", predictions.len());

            // Count predicted classes
            let mut pred_counts = [0, 0, 0];
            for pred in &predictions {
                pred_counts[pred.predicted_class] += 1;
            }
            println!(
                "   Predictions: Up={}, Down={}, Stable={}",
                pred_counts[0], pred_counts[1], pred_counts[2]
            );

            // Calculate average confidence
            let avg_confidence: f64 =
                predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64;
            println!("   Average confidence: {:.2}%", avg_confidence * 100.0);

            // Run backtest
            println!("\n5. Running backtest...");
            let backtest_config = BacktestConfig {
                initial_capital: 100000.0,
                position_size: 0.1,
                transaction_cost: 0.001,
                stop_loss: Some(0.03),
                take_profit: Some(0.06),
                confidence_threshold: 0.4,
            };
            let backtester = Backtester::new(backtest_config);

            // Get test prices
            let n_test = ds.x_test.dim().0;
            let test_prices: Vec<f64> = ohlcv
                .close
                .iter()
                .skip(ohlcv.len() - n_test)
                .cloned()
                .collect();

            let result = backtester.run(&model, &ds.x_test, &test_prices);

            println!("\n=== Backtest Results ===");
            result.print_report();

            // Show some trades
            if !result.trades.is_empty() {
                println!("\nSample trades:");
                for (i, trade) in result.trades.iter().take(5).enumerate() {
                    println!(
                        "   Trade {}: {:?} at {:.2} -> {:.2}, PnL: ${:.2} ({:.2}%)",
                        i + 1,
                        trade.signal,
                        trade.entry_price,
                        trade.exit_price,
                        trade.pnl,
                        trade.return_pct * 100.0
                    );
                }
            }
        }
        None => {
            println!("   Failed to prepare dataset (not enough data)");
        }
    }

    println!("\n=== Demo Complete ===");
}

/// Generate synthetic OHLCV data with realistic price patterns
fn generate_synthetic_data(n: usize) -> OHLCV {
    let mut close = Vec::with_capacity(n);
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    let mut timestamps = Vec::with_capacity(n);

    // Simple LCG for reproducibility
    let mut seed: u64 = 12345;
    let rand = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(1103515245).wrapping_add(12345) % (1 << 31);
        (*s as f64) / (1u64 << 31) as f64
    };

    // Generate random walk with trend and mean reversion
    let mut price = 100.0;
    let trend = 0.0002; // Slight upward trend
    let volatility = 0.02;

    for i in 0..n {
        // Random return with trend
        let r = (rand(&mut seed) - 0.5) * volatility + trend;
        price *= 1.0 + r;

        // Generate OHLC
        let c = price;
        let daily_vol = c * volatility * (rand(&mut seed) * 0.5 + 0.5);
        let o = c * (1.0 + (rand(&mut seed) - 0.5) * 0.01);
        let h = c.max(o) + daily_vol * rand(&mut seed);
        let l = c.min(o) - daily_vol * rand(&mut seed);

        close.push(c);
        open.push(o);
        high.push(h);
        low.push(l);
        volume.push(1_000_000.0 * (0.5 + rand(&mut seed)));
        timestamps.push(i as i64);
    }

    OHLCV {
        timestamps,
        open: Array1::from_vec(open),
        high: Array1::from_vec(high),
        low: Array1::from_vec(low),
        close: Array1::from_vec(close),
        volume: Array1::from_vec(volume),
    }
}
