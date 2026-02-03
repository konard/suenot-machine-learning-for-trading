//! Multi-asset trading example with S4.
//!
//! Demonstrates using S4 models for multiple assets simultaneously.

use s4_trading::prelude::*;
use s4_trading::data::bybit::generate_synthetic_data;
use s4_trading::model::s4::{S4Config, S4Model};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== S4 Trading: Multi-Asset Example ===\n");

    // Define assets to trade
    let assets = vec![
        ("BTCUSDT", 50000.0),
        ("ETHUSDT", 3000.0),
        ("SOLUSDT", 100.0),
        ("BNBUSDT", 300.0),
    ];

    println!("Trading {} assets: {:?}\n",
        assets.len(),
        assets.iter().map(|(s, _)| *s).collect::<Vec<_>>());

    // Create a shared model configuration
    let config = S4Config {
        d_model: 64,
        d_state: 64,
        n_layers: 4,
        ..Default::default()
    };

    // Create model for each asset
    let models: Vec<(String, S4Model)> = assets
        .iter()
        .map(|(symbol, _)| (symbol.to_string(), S4Model::new(config.clone())))
        .collect();

    println!("Created {} S4 models\n", models.len());

    // Generate data and make predictions for each asset
    println!("Generating signals for each asset:\n");
    println!("{:<12} {:>10} {:>12} {:>10} {:>12}",
        "Asset", "Signal", "Confidence", "Direction", "Position");
    println!("{}", "-".repeat(58));

    let mut portfolio_signals = Vec::new();

    for ((symbol, base_price), (_, model)) in assets.iter().zip(models.iter()) {
        // Generate synthetic data
        let data = generate_synthetic_data(symbol, "60", 500);

        // Get features and predict
        let features = data.to_features();
        let signal = model.predict(&features);

        // Calculate position sizing
        let position_size = signal.position_size(0.25, 0.6);

        let direction = match signal.direction() {
            1 => "LONG",
            -1 => "SHORT",
            _ => "FLAT",
        };

        println!("{:<12} {:>10} {:>11.1}% {:>10} {:>11.2}%",
            symbol,
            format!("{:?}", if signal.is_buy() { "BUY" } else if signal.is_sell() { "SELL" } else { "HOLD" }),
            signal.confidence() * 100.0,
            direction,
            position_size * 100.0
        );

        portfolio_signals.push((symbol.to_string(), signal, position_size));
    }

    println!("\n{}", "-".repeat(58));

    // Portfolio allocation summary
    println!("\nPortfolio Allocation Summary:");

    let long_positions: Vec<_> = portfolio_signals.iter()
        .filter(|(_, s, _)| s.is_buy())
        .collect();

    let short_positions: Vec<_> = portfolio_signals.iter()
        .filter(|(_, s, _)| s.is_sell())
        .collect();

    let neutral_positions: Vec<_> = portfolio_signals.iter()
        .filter(|(_, s, _)| s.is_hold())
        .collect();

    println!("  Long positions: {} ({:?})",
        long_positions.len(),
        long_positions.iter().map(|(s, _, _)| s.as_str()).collect::<Vec<_>>());

    println!("  Short positions: {} ({:?})",
        short_positions.len(),
        short_positions.iter().map(|(s, _, _)| s.as_str()).collect::<Vec<_>>());

    println!("  Neutral: {} ({:?})",
        neutral_positions.len(),
        neutral_positions.iter().map(|(s, _, _)| s.as_str()).collect::<Vec<_>>());

    // Calculate net exposure
    let net_exposure: f64 = portfolio_signals.iter()
        .map(|(_, _, pos)| *pos)
        .sum();

    let gross_exposure: f64 = portfolio_signals.iter()
        .map(|(_, _, pos)| pos.abs())
        .sum();

    println!("\n  Net exposure: {:.1}%", net_exposure * 100.0);
    println!("  Gross exposure: {:.1}%", gross_exposure * 100.0);

    // Cross-asset correlation analysis (simplified)
    println!("\nCross-Asset State Correlation:");

    let states: Vec<(String, Vec<f64>)> = assets.iter()
        .zip(models.iter())
        .map(|((symbol, _), (_, model))| {
            let data = generate_synthetic_data(symbol, "60", 500);
            let features = data.to_features();
            (symbol.to_string(), model.get_state(&features))
        })
        .collect();

    // Simple correlation matrix
    println!("\n{:>12}", "");
    for (symbol, _) in &states {
        print!("{:>12}", &symbol[..symbol.len().min(10)]);
    }
    println!();

    for (i, (symbol_i, state_i)) in states.iter().enumerate() {
        print!("{:<12}", &symbol_i[..symbol_i.len().min(10)]);

        for (j, (_, state_j)) in states.iter().enumerate() {
            let corr = if i == j {
                1.0
            } else {
                pearson_correlation(state_i, state_j)
            };
            print!("{:>12.3}", corr);
        }
        println!();
    }

    println!("\n=== Multi-Asset Example Complete ===");
}

/// Calculate Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 1e-10 {
        cov / denom
    } else {
        0.0
    }
}
