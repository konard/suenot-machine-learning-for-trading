//! Example: Generate market scenarios using GLOW model
//!
//! This example demonstrates how to use a trained GLOW model
//! to generate realistic market scenarios for risk analysis.
//!
//! Run with: cargo run --example generate_scenarios

use anyhow::Result;
use glow_trading::{Checkpoint, GLOWTrader, TraderConfig};

fn main() -> Result<()> {
    println!("=== GLOW Trading: Scenario Generator ===\n");

    // Load model
    let model_file = "glow_model.bin";
    println!("Loading model from {}...", model_file);

    let checkpoint = match Checkpoint::load(model_file) {
        Ok(c) => c,
        Err(_) => {
            println!("Model file not found. Please train model first:");
            println!("  cargo run --example train_model");
            return Ok(());
        }
    };

    println!("Model loaded successfully!");

    // Create trader (for scenario generation)
    let trader_config = TraderConfig::default();
    let mut trader = GLOWTrader::new(checkpoint.model, trader_config);

    if let Some(normalizer) = checkpoint.normalizer {
        trader.set_normalizer(normalizer);
    }

    // Generate scenarios
    let num_scenarios = 1000;
    let temperature = 1.0;

    println!("\nGenerating {} scenarios with temperature {}...", num_scenarios, temperature);
    let scenarios = trader.generate_scenarios(num_scenarios, temperature);

    println!("Generated scenarios shape: {} x {}", scenarios.nrows(), scenarios.ncols());

    // Analyze return distribution (assuming first column is returns)
    let returns: Vec<f64> = scenarios.column(0).to_vec();

    // Basic statistics
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();
    let min = returns.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n{:=<60}", "");
    println!("=== SCENARIO ANALYSIS ===");
    println!("{:=<60}", "");

    println!("\n--- Basic Statistics ---");
    println!("Mean Return:         {:>12.6}", mean);
    println!("Std Deviation:       {:>12.6}", std);
    println!("Minimum:             {:>12.6}", min);
    println!("Maximum:             {:>12.6}", max);
    println!("Skewness:            {:>12.6}", compute_skewness(&returns, mean, std));
    println!("Kurtosis:            {:>12.6}", compute_kurtosis(&returns, mean, std));

    // Compute percentiles
    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("\n--- Percentiles ---");
    let percentiles = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0];
    for p in percentiles {
        let idx = ((p / 100.0) * num_scenarios as f64) as usize;
        let idx = idx.min(num_scenarios - 1);
        println!("{:>3.0}th percentile:    {:>12.6}", p, sorted[idx]);
    }

    // Risk metrics
    println!("\n--- Risk Metrics ---");

    // Value at Risk
    let var_95 = sorted[(num_scenarios as f64 * 0.05) as usize];
    let var_99 = sorted[(num_scenarios as f64 * 0.01) as usize];

    // Conditional Value at Risk (Expected Shortfall)
    let cvar_95_tail: Vec<f64> = sorted[..(num_scenarios as f64 * 0.05) as usize].to_vec();
    let cvar_95 = if !cvar_95_tail.is_empty() {
        cvar_95_tail.iter().sum::<f64>() / cvar_95_tail.len() as f64
    } else {
        var_95
    };

    let cvar_99_tail: Vec<f64> = sorted[..(num_scenarios as f64 * 0.01) as usize].to_vec();
    let cvar_99 = if !cvar_99_tail.is_empty() {
        cvar_99_tail.iter().sum::<f64>() / cvar_99_tail.len() as f64
    } else {
        var_99
    };

    println!("VaR (95%):           {:>12.6}", var_95);
    println!("VaR (99%):           {:>12.6}", var_99);
    println!("CVaR (95%):          {:>12.6}", cvar_95);
    println!("CVaR (99%):          {:>12.6}", cvar_99);

    // Distribution analysis
    println!("\n--- Distribution Analysis ---");
    let negative_count = returns.iter().filter(|&&r| r < 0.0).count();
    let positive_count = returns.iter().filter(|&&r| r > 0.0).count();
    let extreme_negative = returns.iter().filter(|&&r| r < -2.0 * std).count();
    let extreme_positive = returns.iter().filter(|&&r| r > 2.0 * std).count();

    println!("Positive returns:    {:>10} ({:.1}%)",
             positive_count, positive_count as f64 / num_scenarios as f64 * 100.0);
    println!("Negative returns:    {:>10} ({:.1}%)",
             negative_count, negative_count as f64 / num_scenarios as f64 * 100.0);
    println!("Extreme negative:    {:>10} ({:.1}%) [<-2 std]",
             extreme_negative, extreme_negative as f64 / num_scenarios as f64 * 100.0);
    println!("Extreme positive:    {:>10} ({:.1}%) [>+2 std]",
             extreme_positive, extreme_positive as f64 / num_scenarios as f64 * 100.0);

    // Generate scenarios at different temperatures
    println!("\n--- Temperature Sensitivity ---");
    for temp in [0.5, 1.0, 1.5, 2.0] {
        let temp_scenarios = trader.generate_scenarios(500, temp);
        let temp_returns: Vec<f64> = temp_scenarios.column(0).to_vec();
        let temp_std = {
            let mean = temp_returns.iter().sum::<f64>() / temp_returns.len() as f64;
            let var = temp_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / temp_returns.len() as f64;
            var.sqrt()
        };
        let temp_min = temp_returns.iter().cloned().fold(f64::INFINITY, f64::min);
        let temp_max = temp_returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("Temp={:.1}: std={:.6}, min={:.6}, max={:.6}",
                 temp, temp_std, temp_min, temp_max);
    }

    // Histogram
    println!("\n--- Return Distribution (ASCII Histogram) ---");
    print_histogram(&sorted, 20);

    println!("\n{:=<60}", "");

    // Save scenarios to CSV
    let output_file = "scenarios.csv";
    let mut wtr = csv::Writer::from_path(output_file)?;

    // Write header
    let mut header = Vec::new();
    for i in 0..scenarios.ncols() {
        header.push(format!("feature_{}", i));
    }
    wtr.write_record(&header)?;

    // Write data
    for i in 0..scenarios.nrows() {
        let row: Vec<String> = scenarios.row(i).iter().map(|v| v.to_string()).collect();
        wtr.write_record(&row)?;
    }
    wtr.flush()?;

    println!("\nScenarios saved to {}", output_file);

    Ok(())
}

fn compute_skewness(data: &[f64], mean: f64, std: f64) -> f64 {
    if std < 1e-10 || data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f64;
    let m3 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
    m3
}

fn compute_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
    if std < 1e-10 || data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f64;
    let m4 = data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
    m4 - 3.0 // Excess kurtosis
}

fn print_histogram(sorted_data: &[f64], num_bins: usize) {
    if sorted_data.is_empty() {
        return;
    }

    let min = sorted_data[0];
    let max = sorted_data[sorted_data.len() - 1];
    let range = max - min;

    if range < 1e-10 {
        println!("All values are the same: {:.6}", min);
        return;
    }

    let bin_width = range / num_bins as f64;
    let mut bins = vec![0usize; num_bins];

    for &val in sorted_data {
        let bin = ((val - min) / bin_width) as usize;
        let bin = bin.min(num_bins - 1);
        bins[bin] += 1;
    }

    let max_count = *bins.iter().max().unwrap_or(&1);
    let bar_width = 40;

    for (i, &count) in bins.iter().enumerate() {
        let bin_start = min + i as f64 * bin_width;
        let bin_end = bin_start + bin_width;
        let bar_len = (count as f64 / max_count as f64 * bar_width as f64) as usize;
        let bar: String = "=".repeat(bar_len);

        println!(
            "[{:>8.4}, {:>8.4}): {:>4} |{}",
            bin_start, bin_end, count, bar
        );
    }
}
