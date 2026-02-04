//! Example: Trading signal generation with conformal prediction

use conformal_trading::conformal::predictor::{SplitConformalPredictor, PredictionInterval};
use conformal_trading::strategy::signal::{SignalGenerator, SignalType, PositionScaling};
use rand::Rng;

fn main() {
    println!("Conformal Prediction - Trading Signals Demo");
    println!("{}", "=".repeat(50));

    // Generate synthetic prediction intervals
    let mut rng = rand::thread_rng();

    // Create calibration scores (simulating model residuals)
    let calibration_scores: Vec<f64> = (0..200)
        .map(|_| rng.gen_range(0.0..0.05)) // Typical residuals up to 5%
        .collect();

    // Calibrate conformal predictor
    let mut cp = SplitConformalPredictor::new(0.1);
    cp.calibrate_from_scores(calibration_scores);

    println!("\nConformal Predictor Calibrated");
    println!("  Quantile: {:.4}", cp.quantile().unwrap());

    // Create signal generator
    let signal_gen = SignalGenerator::new(0.4, 1.0)
        .with_historical_volatility(0.02)
        .with_volatility_threshold(0.05)
        .with_position_scaling(PositionScaling::InverseWidth);

    // Generate predictions and signals
    println!("\n{}", "-".repeat(50));
    println!("Signal Generation Examples");
    println!("{}", "-".repeat(50));

    // Scenario 1: Strong bullish prediction
    println!("\n1. Strong Bullish (interval entirely positive):");
    let interval1 = cp.predict_interval(0.03).unwrap();
    let signal1 = signal_gen.generate(1000, &interval1);
    print_signal(&signal1, &interval1);

    // Scenario 2: Strong bearish prediction
    println!("\n2. Strong Bearish (interval entirely negative):");
    let interval2 = cp.predict_interval(-0.025).unwrap();
    let signal2 = signal_gen.generate(2000, &interval2);
    print_signal(&signal2, &interval2);

    // Scenario 3: Uncertain (interval crosses zero)
    println!("\n3. Uncertain (interval crosses zero):");
    let interval3 = cp.predict_interval(0.005).unwrap();
    let signal3 = signal_gen.generate(3000, &interval3);
    print_signal(&signal3, &interval3);

    // Scenario 4: Weak bullish
    println!("\n4. Weak Bullish (positive but interval crosses zero):");
    let interval4 = PredictionInterval::new(0.015, -0.01, 0.04, 0.1);
    let signal4 = signal_gen.generate(4000, &interval4);
    print_signal(&signal4, &interval4);

    // ==========================================================
    // Batch Signal Generation
    // ==========================================================
    println!("\n{}", "-".repeat(50));
    println!("Batch Signal Generation (100 samples)");
    println!("{}", "-".repeat(50));

    let point_predictions: Vec<f64> = (0..100)
        .map(|_| rng.gen_range(-0.05..0.05))
        .collect();

    let intervals: Vec<PredictionInterval> = point_predictions.iter()
        .map(|&pred| cp.predict_interval(pred).unwrap())
        .collect();

    let timestamps: Vec<i64> = (0..100).map(|i| i * 3600000).collect();

    let signals = signal_gen.generate_batch(&timestamps, &intervals);

    // Count signal types
    let long_count = signals.iter().filter(|s| s.signal_type == SignalType::Long).count();
    let short_count = signals.iter().filter(|s| s.signal_type == SignalType::Short).count();
    let hold_count = signals.iter().filter(|s| s.signal_type == SignalType::Hold).count();

    println!("\nSignal Distribution:");
    println!("  LONG:  {} ({:.1}%)", long_count, 100.0 * long_count as f64 / signals.len() as f64);
    println!("  SHORT: {} ({:.1}%)", short_count, 100.0 * short_count as f64 / signals.len() as f64);
    println!("  HOLD:  {} ({:.1}%)", hold_count, 100.0 * hold_count as f64 / signals.len() as f64);

    // Position size distribution
    let position_sizes: Vec<f64> = signals.iter()
        .map(|s| s.position_size.abs())
        .filter(|&p| p > 0.0)
        .collect();

    if !position_sizes.is_empty() {
        let avg_position = position_sizes.iter().sum::<f64>() / position_sizes.len() as f64;
        let max_position = position_sizes.iter().cloned().fold(0.0, f64::max);
        let min_position = position_sizes.iter().cloned().fold(f64::MAX, f64::min);

        println!("\nPosition Size Statistics (active trades only):");
        println!("  Average: {:.3}", avg_position);
        println!("  Min: {:.3}", min_position);
        println!("  Max: {:.3}", max_position);
    }

    // Confidence distribution
    let confidences: Vec<f64> = signals.iter()
        .filter(|s| s.signal_type != SignalType::Hold)
        .map(|s| s.confidence)
        .collect();

    if !confidences.is_empty() {
        let avg_conf = confidences.iter().sum::<f64>() / confidences.len() as f64;

        println!("\nConfidence Statistics (active signals only):");
        println!("  Average: {:.3}", avg_conf);
    }

    // ==========================================================
    // Different Position Scaling Methods
    // ==========================================================
    println!("\n{}", "-".repeat(50));
    println!("Position Scaling Comparison");
    println!("{}", "-".repeat(50));

    let test_interval = cp.predict_interval(0.02).unwrap();

    for scaling in [PositionScaling::Fixed, PositionScaling::InverseWidth, PositionScaling::Kelly] {
        let gen = SignalGenerator::new(0.4, 1.0)
            .with_historical_volatility(0.02)
            .with_position_scaling(scaling);

        let signal = gen.generate(0, &test_interval);

        println!("\n{:?} Scaling:", scaling);
        println!("  Signal: {}", signal.signal_type);
        println!("  Position Size: {:.4}", signal.position_size);
        println!("  Confidence: {:.4}", signal.confidence);
    }

    println!("\nDemo complete!");
}

fn print_signal(signal: &conformal_trading::strategy::signal::Signal, interval: &PredictionInterval) {
    println!("  Interval: [{:.4}, {:.4}]", interval.lower, interval.upper);
    println!("  Point prediction: {:.4}", interval.point);
    println!("  Width: {:.4}", interval.width);
    println!("  Signal type: {}", signal.signal_type);
    println!("  Position size: {:.4}", signal.position_size);
    println!("  Confidence: {:.4}", signal.confidence);
}
