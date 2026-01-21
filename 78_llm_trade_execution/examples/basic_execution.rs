//! Basic execution example demonstrating TWAP and VWAP strategies.
//!
//! This example shows how to use the execution engine with traditional
//! execution algorithms without LLM integration.

use llm_trade_execution::{
    ExecutionConfig, ExecutionEngine, ParentOrder, Side,
    TwapStrategy, VwapStrategy, VolumeProfile,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Basic Execution Example ===\n");

    // Create execution engine with default configuration
    let config = ExecutionConfig {
        min_slice_interval_ms: 100,  // Fast for demo
        max_slice_interval_ms: 500,
        verbose: true,
        ..Default::default()
    };

    let mut engine = ExecutionEngine::new(config);

    // Example 1: TWAP Execution
    println!("--- TWAP Execution ---");

    let order = ParentOrder::new(
        "BTCUSDT".to_string(),
        Side::Buy,
        0.5,    // 0.5 BTC
        10,     // Execute over 10 seconds (fast for demo)
    ).with_urgency(0.5);

    let strategy = Box::new(TwapStrategy::new(2)); // 2-second slices

    let result = engine.execute(order, strategy).await?;

    println!("TWAP Execution Result:");
    println!("  Symbol: {}", result.symbol);
    println!("  Side: {}", result.side);
    println!("  Total Quantity: {:.4}", result.total_quantity);
    println!("  Filled Quantity: {:.4}", result.filled_quantity);
    println!("  Child Orders: {}", result.child_order_count);
    println!("  Average Price: {:.2}", result.average_price);
    println!("  Arrival Price: {:.2}", result.arrival_price);
    println!("  Implementation Shortfall: {:.2} bps", result.implementation_shortfall);
    println!("  VWAP Slippage: {:.2} bps", result.vwap_slippage);
    println!("  Duration: {} seconds", result.duration_secs);
    println!();

    // Example 2: VWAP Execution
    println!("--- VWAP Execution ---");

    let mut engine2 = ExecutionEngine::new(ExecutionConfig {
        min_slice_interval_ms: 100,
        max_slice_interval_ms: 500,
        verbose: true,
        ..Default::default()
    });

    let order2 = ParentOrder::new(
        "ETHUSDT".to_string(),
        Side::Sell,
        2.0,    // 2 ETH
        10,     // Execute over 10 seconds
    ).with_urgency(0.3);  // Lower urgency

    // Use uniform volume profile for simplicity
    let vwap_strategy = Box::new(VwapStrategy::new(5)); // 5 periods

    let result2 = engine2.execute(order2, vwap_strategy).await?;

    println!("VWAP Execution Result:");
    println!("  Symbol: {}", result2.symbol);
    println!("  Side: {}", result2.side);
    println!("  Total Quantity: {:.4}", result2.total_quantity);
    println!("  Filled Quantity: {:.4}", result2.filled_quantity);
    println!("  Child Orders: {}", result2.child_order_count);
    println!("  Average Price: {:.2}", result2.average_price);
    println!("  Implementation Shortfall: {:.2} bps", result2.implementation_shortfall);
    println!();

    // Example 3: Compare TWAP vs VWAP
    println!("--- Strategy Comparison ---");
    println!("TWAP IS: {:.2} bps", result.implementation_shortfall);
    println!("VWAP IS: {:.2} bps", result2.implementation_shortfall);

    if result.implementation_shortfall.abs() < result2.implementation_shortfall.abs() {
        println!("TWAP performed better in this simulation.");
    } else {
        println!("VWAP performed better in this simulation.");
    }

    println!("\nNote: Results use simulated market data. Real execution will vary.");

    Ok(())
}
