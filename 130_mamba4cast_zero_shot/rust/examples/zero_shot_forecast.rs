//! Example: Zero-Shot Forecasting with Mamba4Cast
//!
//! This example demonstrates how to use Mamba4Cast for zero-shot
//! time series forecasting.

use mamba4cast::prelude::*;

fn main() {
    println!("Mamba4Cast Zero-Shot Forecasting Example");
    println!("========================================\n");

    // Create forecaster with default configuration
    let forecaster = Mamba4CastForecaster::new(
        64,   // d_model
        4,    // n_layers
        16,   // d_state
        96,   // max_horizon
    );

    println!("Model Configuration:");
    let config = forecaster.config();
    println!("  d_model:     {}", config.d_model);
    println!("  n_layers:    {}", config.n_layers);
    println!("  d_state:     {}", config.d_state);
    println!("  max_horizon: {}", config.max_horizon);
    println!();

    // Generate sample time series (simulating price data)
    println!("Generating sample time series...");
    let time_series: Vec<f32> = (0..200)
        .map(|i| {
            let trend = 100.0 + (i as f32) * 0.1;
            let seasonal = 5.0 * ((i as f32) * 0.1).sin();
            let noise = (rand::random::<f32>() - 0.5) * 2.0;
            trend + seasonal + noise
        })
        .collect();

    println!("  Time series length: {}", time_series.len());
    println!("  First value:  {:.2}", time_series.first().unwrap());
    println!("  Last value:   {:.2}", time_series.last().unwrap());
    println!();

    // Generate zero-shot forecast
    println!("Generating zero-shot forecast...");
    let context_length = 100;
    let horizon = 24;

    let forecast = forecaster.zero_shot_forecast(
        &time_series,
        context_length,
        horizon,
    );

    println!("  Context length: {}", context_length);
    println!("  Forecast horizon: {}", horizon);
    println!();

    // Display forecast results
    println!("Forecast Results:");
    println!("-----------------");
    for (i, value) in forecast.iter().enumerate() {
        if i < 5 || i >= horizon - 3 {
            println!("  Step {:>2}: {:.2}", i + 1, value);
        } else if i == 5 {
            println!("  ...");
        }
    }
    println!();

    // Calculate expected return
    let current_price = *time_series.last().unwrap();
    let final_price = *forecast.last().unwrap();
    let expected_return = (final_price - current_price) / current_price * 100.0;

    println!("Analysis:");
    println!("  Current Price:   {:.2}", current_price);
    println!("  Predicted Price: {:.2}", final_price);
    println!("  Expected Return: {:.2}%", expected_return);
    println!();

    // Generate trading signal
    let signal = forecaster.trading_signals(
        &time_series,
        context_length,
        horizon,
        0.01,  // 1% threshold
    );

    println!("Trading Signal:");
    println!("  Signal:          {}", signal.signal);
    println!("  Expected Return: {:.2}%", signal.expected_return * 100.0);
    println!("  Confidence:      {:.2}%", signal.confidence * 100.0);
    println!();

    println!("Example completed successfully!");
}
