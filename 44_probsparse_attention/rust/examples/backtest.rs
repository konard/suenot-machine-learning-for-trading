//! Example: Backtesting trading strategy with Informer predictions
//!
//! Run with: cargo run --example backtest

use informer_probsparse::{
    InformerConfig, InformerModel, DataLoader,
    SignalGenerator, TradingStrategy, TradingSignal,
    api::Kline,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Informer ProbSparse: Backtest Example ===\n");

    // Create synthetic price data
    println!("Generating synthetic market data...");
    let klines = create_synthetic_market(500, 0.001); // Slight uptrend

    println!("Generated {} hourly candles", klines.len());
    println!("Price range: ${:.2} to ${:.2}",
        klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max)
    );

    // Prepare data
    let loader = DataLoader::new();
    let seq_len = 48;
    let pred_len = 12;

    let dataset = loader.prepare_dataset(&klines, seq_len, pred_len)?;
    println!("\nDataset: {} samples", dataset.n_samples);

    // Create and configure model
    let config = InformerConfig {
        seq_len,
        pred_len,
        input_features: 6,
        d_model: 32,
        n_heads: 4,
        n_encoder_layers: 2,
        use_distilling: true,
        ..Default::default()
    };

    let model = InformerModel::new(config);
    println!("Model created with {} parameters", model.num_parameters());

    // Generate signals for backtest
    println!("\n--- Generating Trading Signals ---\n");

    let signal_generator = SignalGenerator::with_thresholds(0.0005, -0.0005);

    // Generate predictions and signals
    let skip_start = loader.prepare_inference(&klines[..150], seq_len)
        .map(|_| 150 + seq_len)
        .unwrap_or(200);

    let mut signals: Vec<TradingSignal> = vec![TradingSignal::Neutral; skip_start];

    let batch_size = 32;
    for start in (0..dataset.n_samples).step_by(batch_size) {
        let end = (start + batch_size).min(dataset.n_samples);
        let indices: Vec<usize> = (start..end).collect();
        let (batch_x, _) = dataset.get_batch(&indices);

        let predictions = model.predict(&batch_x);
        let batch_signals = signal_generator.generate_batch(&predictions);

        signals.extend(batch_signals);
    }

    // Pad signals to match klines length
    while signals.len() < klines.len() {
        signals.push(TradingSignal::Neutral);
    }

    // Analyze signal distribution
    let n_long = signals.iter().filter(|&&s| s == TradingSignal::Long).count();
    let n_short = signals.iter().filter(|&&s| s == TradingSignal::Short).count();
    let n_neutral = signals.iter().filter(|&&s| s == TradingSignal::Neutral).count();

    println!("Signal distribution:");
    println!("  Long: {} ({:.1}%)", n_long, 100.0 * n_long as f64 / signals.len() as f64);
    println!("  Short: {} ({:.1}%)", n_short, 100.0 * n_short as f64 / signals.len() as f64);
    println!("  Neutral: {} ({:.1}%)", n_neutral, 100.0 * n_neutral as f64 / signals.len() as f64);

    // Run backtest
    println!("\n--- Running Backtest ---\n");

    let strategy = TradingStrategy::new(signal_generator)
        .with_capital(10000.0)
        .with_commission(0.001);

    let result = strategy.backtest(&klines, &signals, 8760); // Hourly data

    println!("{}", result.summary());

    // Compare with buy-and-hold
    println!("\n--- Buy and Hold Comparison ---\n");

    let bh_return = (klines.last().unwrap().close / klines.first().unwrap().close) - 1.0;
    println!("Buy & Hold Return: {:.2}%", bh_return * 100.0);
    println!("Strategy Return: {:.2}%", result.total_return * 100.0);

    if result.total_return > bh_return {
        println!("Strategy OUTPERFORMED buy & hold by {:.2}%",
            (result.total_return - bh_return) * 100.0);
    } else {
        println!("Strategy UNDERPERFORMED buy & hold by {:.2}%",
            (bh_return - result.total_return) * 100.0);
    }

    // Equity curve statistics
    if !result.equity_curve.is_empty() {
        let min_equity = result.equity_curve.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_equity = result.equity_curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\nEquity Curve:");
        println!("  Initial: ${:.2}", result.equity_curve.first().unwrap_or(&10000.0));
        println!("  Final: ${:.2}", result.equity_curve.last().unwrap_or(&10000.0));
        println!("  Min: ${:.2}", min_equity);
        println!("  Max: ${:.2}", max_equity);
    }

    println!("\n=== Backtest Complete ===");

    Ok(())
}

/// Create synthetic market data with trend
fn create_synthetic_market(n: usize, drift: f64) -> Vec<Kline> {
    use std::f64::consts::PI;

    let mut price = 100.0;
    let volatility = 0.02;

    (0..n).map(|i| {
        // Random walk with drift
        let return_val = drift + rand_normal() * volatility;
        price *= 1.0 + return_val;

        // Add intraday variation
        let intraday_vol = volatility * 0.5;
        let high = price * (1.0 + rand_normal().abs() * intraday_vol);
        let low = price * (1.0 - rand_normal().abs() * intraday_vol);
        let open = price * (1.0 + rand_normal() * intraday_vol * 0.5);

        // Volume with pattern
        let base_volume = 1000.0;
        let volume = base_volume * (1.0 + 0.3 * (i as f64 * 2.0 * PI / 24.0).sin().abs()
            + rand_normal().abs() * 0.5);

        Kline {
            timestamp: i as u64 * 3600000,
            open,
            high: high.max(open).max(price),
            low: low.min(open).min(price),
            close: price,
            volume,
            turnover: price * volume,
        }
    }).collect()
}

/// Generate random normal number
fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
