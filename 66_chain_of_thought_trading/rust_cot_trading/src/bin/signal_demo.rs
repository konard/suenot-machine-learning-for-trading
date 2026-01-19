//! Signal Generation Demo
//!
//! Demonstrates multi-step signal generation with CoT reasoning.

use cot_trading::{SignalGenerator, Signal, PositionSizer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("====================================================");
    println!("Multi-Step Signal Generation Demo");
    println!("====================================================\n");

    let generator = SignalGenerator::new_mock();
    let position_sizer = PositionSizer::new(0.1, 0.02);

    // Test multiple scenarios
    let scenarios = vec![
        ("AAPL", 185.0, 45.0, 1.2, 0.8, 182.0, 178.0, 1.5, 3.0),   // Bullish
        ("MSFT", 420.0, 72.0, -0.5, -0.3, 425.0, 430.0, 0.8, 5.0), // Bearish
        ("NVDA", 880.0, 55.0, 0.2, 0.1, 875.0, 870.0, 1.0, 8.0),   // Neutral
    ];

    let portfolio_value = 100_000.0;

    for (symbol, price, rsi, macd, macd_sig, sma20, sma50, vol_ratio, atr) in scenarios {
        println!("----------------------------------------------------");
        println!("Analyzing {} at ${:.2}", symbol, price);
        println!("----------------------------------------------------\n");

        println!("Input Data:");
        println!("  RSI: {:.1}", rsi);
        println!("  MACD: {:.2} (Signal: {:.2})", macd, macd_sig);
        println!("  SMA20: ${:.2}, SMA50: ${:.2}", sma20, sma50);
        println!("  Volume Ratio: {:.1}x", vol_ratio);
        println!("  ATR: ${:.2}", atr);

        // Generate signal
        let signal = generator.generate(
            symbol,
            price,
            rsi,
            macd,
            macd_sig,
            sma20,
            sma50,
            vol_ratio,
            atr,
        ).await?;

        println!("\nSignal Generated:");
        println!("  Type: {:?}", signal.signal_type);
        println!("  Confidence: {:.0}%", signal.confidence * 100.0);
        println!("  Stop Loss: ${:.2}", signal.stop_loss);
        println!("  Take Profit: ${:.2}", signal.take_profit);

        let risk = (price - signal.stop_loss).abs();
        let reward = (signal.take_profit - price).abs();
        println!("  Risk/Reward: 1:{:.1}", reward / risk);

        println!("\nReasoning Chain:");
        for (i, reason) in signal.reasoning_chain.iter().enumerate() {
            println!("  {}. {}", i + 1, reason);
        }

        // Calculate position size
        if signal.signal_type != Signal::Hold {
            let position = position_sizer.calculate(
                signal.signal_type,
                signal.confidence,
                price,
                signal.stop_loss,
                portfolio_value,
                Some(atr / price),
            );

            println!("\nPosition Sizing:");
            println!("  Units: {:.4}", position.units);
            println!("  Value: ${:.2}", position.position_value);
            println!("  Risk Amount: ${:.2}", position.risk_amount);
            println!("  Portfolio %: {:.2}%", position.position_pct * 100.0);

            println!("\nPosition Reasoning:");
            for (i, reason) in position.reasoning_chain.iter().enumerate() {
                println!("  {}. {}", i + 1, reason);
            }
        }

        println!();
    }

    println!("====================================================");
    println!("Demo complete!");
    println!("====================================================");

    Ok(())
}
