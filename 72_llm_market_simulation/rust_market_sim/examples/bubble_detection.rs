//! Bubble Detection Example
//!
//! Demonstrates how momentum-heavy markets can form bubbles
//! and how to detect them using simulation metrics.

use llm_market_sim::{
    agents::{ValueInvestor, MomentumTrader, MarketMaker},
    simulation::{SimulationEngine, calculate_performance_metrics, detect_bubble},
};

fn main() {
    println!("Bubble Detection Simulation Example");
    println!("====================================");
    println!();

    println!("Scenario: Momentum-Heavy Market");
    println!("This configuration is prone to bubble formation due to");
    println!("the dominance of trend-following traders over value investors.");
    println!();

    // Create simulation with settings that encourage bubbles
    let mut engine = SimulationEngine::new(100.0, 100.0, 0.025);

    // Only 1 value investor (weak anchor to fundamentals)
    engine.add_agent(Box::new(
        ValueInvestor::new("value_1".to_string(), 100000.0, 100, 100.0)
            .with_thresholds(0.15, 0.15) // Very patient
    ));
    println!("Added: 1 Value Investor (threshold: 15%)");

    // Many aggressive momentum traders
    for i in 0..5 {
        engine.add_agent(Box::new(
            MomentumTrader::new(
                format!("momentum_{}", i + 1),
                100000.0,
                100,
            )
            .with_windows(3, 8) // Fast signals
            .with_threshold(0.01) // Low threshold
        ));
    }
    println!("Added: 5 Momentum Traders (aggressive settings)");

    // Market makers
    for i in 0..2 {
        engine.add_agent(Box::new(
            MarketMaker::new(format!("mm_{}", i + 1), 200000.0, 200)
        ));
    }
    println!("Added: 2 Market Makers");
    println!();

    println!("Running simulation for 300 steps...");
    let result = engine.run(300);
    println!("Done!");
    println!();

    // Analyze for bubbles
    println!("═══════════════════════════════════════");
    println!("              BUBBLE ANALYSIS          ");
    println!("═══════════════════════════════════════");
    println!();

    let bubble_info = detect_bubble(&result.price_history, &result.fundamental_history, 0.30);

    println!("Bubble Threshold: 30% above fundamental");
    println!("Bubble Detected: {}", bubble_info.bubble_detected);
    println!("Number of Bubbles: {}", bubble_info.num_bubbles);
    println!("Max Deviation: {:.1}%", bubble_info.max_deviation * 100.0);
    println!("Time in Bubble: {:.1}%", bubble_info.time_in_bubble_pct);

    if !bubble_info.bubble_periods.is_empty() {
        println!();
        println!("Bubble Periods:");
        for (i, period) in bubble_info.bubble_periods.iter().enumerate() {
            println!("  Period {}:", i + 1);
            println!("    Start: step {}", period.start);
            println!("    Peak: step {} ({:.1}% deviation)", period.peak, period.peak_deviation * 100.0);
            println!("    End: step {}", period.end);
            println!("    Duration: {} steps", period.duration);
            if period.peak < result.price_history.len() {
                println!(
                    "    Peak Price: ${:.2} vs Fundamental: ${:.2}",
                    result.price_history[period.peak],
                    result.fundamental_history[period.peak]
                );
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════");
    println!("            PRICE STATISTICS           ");
    println!("═══════════════════════════════════════");
    println!();

    let prices = &result.price_history;
    let fundamentals = &result.fundamental_history;

    let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let deviations: Vec<f64> = prices.iter()
        .zip(fundamentals.iter())
        .map(|(p, f)| (p - f) / f * 100.0)
        .collect();

    let min_dev = deviations.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_dev = deviations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_dev: f64 = deviations.iter().sum::<f64>() / deviations.len() as f64;

    println!("Price Range:");
    println!("  Minimum: ${:.2} ({:.1}% from fundamental)", min_price, min_dev);
    println!("  Maximum: ${:.2} ({:.1}% from fundamental)", max_price, max_dev);
    println!("  Final: ${:.2}", result.final_price);
    println!();
    println!("Deviation Statistics:");
    println!("  Mean: {:.1}%", mean_dev);
    let above_20 = deviations.iter().filter(|&&d| d > 20.0).count() as f64 / deviations.len() as f64 * 100.0;
    let below_20 = deviations.iter().filter(|&&d| d < -20.0).count() as f64 / deviations.len() as f64 * 100.0;
    println!("  Time Above +20%: {:.1}%", above_20);
    println!("  Time Below -20%: {:.1}%", below_20);

    println!();
    println!("═══════════════════════════════════════");
    println!("          PERFORMANCE METRICS          ");
    println!("═══════════════════════════════════════");
    println!();

    let metrics = calculate_performance_metrics(
        prices,
        Some(fundamentals),
        0.02,
        252.0,
    );

    println!("Total Return: {:.2}%", metrics.total_return_pct);
    println!("Volatility: {:.2}%", metrics.volatility * 100.0);
    if let Some(sharpe) = metrics.sharpe_ratio {
        println!("Sharpe Ratio: {:.3}", sharpe);
    }
    println!("Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    if let Some(corr) = metrics.fundamental_correlation {
        println!("Fundamental Correlation: {:.3}", corr);
    }

    println!();
    println!("Done!");
}
