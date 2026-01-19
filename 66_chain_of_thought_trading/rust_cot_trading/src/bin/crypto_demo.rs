//! Cryptocurrency Analysis Demo
//!
//! Demonstrates CoT trading analysis for cryptocurrencies using Bybit data.

use cot_trading::{
    SignalGenerator, PositionSizer, Signal,
    MockLoader, DataLoader, add_indicators,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("====================================================");
    println!("Chain-of-Thought Cryptocurrency Analysis Demo");
    println!("====================================================\n");

    let portfolio_value = 100_000.0;
    println!("Portfolio Value: ${:.2}\n", portfolio_value);

    // Analyze multiple cryptocurrencies
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    let generator = SignalGenerator::new_mock();
    let position_sizer = PositionSizer::new(0.1, 0.02)
        .with_kelly(0.55, 1.5);

    let mut total_position = 0.0;
    let mut total_risk = 0.0;
    let mut results = Vec::new();

    for symbol in &symbols {
        println!("----------------------------------------------------");
        println!("Analyzing {}", symbol);
        println!("----------------------------------------------------\n");

        // Load data (mock for demo)
        let loader = MockLoader::new(symbol.as_bytes().iter().map(|b| *b as u64).sum());
        let bars = loader.get_latest(symbol, 100).await?;

        if bars.is_empty() {
            println!("  No data available for {}", symbol);
            continue;
        }

        // Add technical indicators
        let with_indicators = add_indicators(&bars);
        let latest = with_indicators.last().unwrap();

        println!("Market Data:");
        println!("  Current Price:   ${:>12.2}", latest.bar.close);
        println!("  RSI (14):        {:>12.1}", latest.rsi);
        println!("  MACD:            {:>12.4}", latest.macd);
        println!("  SMA 20:          ${:>12.2}", latest.sma_20);
        println!("  SMA 50:          ${:>12.2}", latest.sma_50);
        println!("  ATR:             ${:>12.2}", latest.atr);

        // Calculate 24h change (approximate from last 2 bars)
        let prev_close = if bars.len() >= 2 {
            bars[bars.len() - 2].close
        } else {
            latest.bar.close
        };
        let change_24h = (latest.bar.close - prev_close) / prev_close * 100.0;
        println!("  24h Change:      {:>12.2}%", change_24h);

        // Generate signal
        let signal = generator.generate(
            symbol,
            latest.bar.close,
            latest.rsi,
            latest.macd,
            latest.macd_signal,
            latest.sma_20,
            latest.sma_50,
            1.0, // Volume ratio (mock)
            latest.atr,
        ).await?;

        println!("\nSignal:");
        println!("  Type:            {:?}", signal.signal_type);
        println!("  Confidence:      {:.0}%", signal.confidence * 100.0);
        println!("  Stop Loss:       ${:.2}", signal.stop_loss);
        println!("  Take Profit:     ${:.2}", signal.take_profit);

        // Calculate position size
        let volatility = latest.atr / latest.bar.close;

        if signal.signal_type != Signal::Hold {
            let position = position_sizer.calculate(
                signal.signal_type,
                signal.confidence,
                latest.bar.close,
                signal.stop_loss,
                portfolio_value,
                Some(volatility),
            );

            println!("\nPosition:");
            println!("  Size:            {:.6} units", position.units);
            println!("  Value:           ${:.2}", position.position_value);
            println!("  Risk Amount:     ${:.2}", position.risk_amount);
            println!("  Portfolio %:     {:.2}%", position.position_pct * 100.0);

            total_position += position.position_value;
            total_risk += position.risk_amount;

            results.push((
                symbol.to_string(),
                signal.signal_type,
                signal.confidence,
                position.position_value,
                position.risk_amount,
            ));
        } else {
            println!("\nPosition: No position (HOLD signal)");
            results.push((symbol.to_string(), signal.signal_type, signal.confidence, 0.0, 0.0));
        }

        println!("\nReasoning Chain:");
        for (i, reason) in signal.reasoning_chain.iter().take(4).enumerate() {
            println!("  {}. {}", i + 1, reason);
        }

        println!();
    }

    // Portfolio summary
    println!("====================================================");
    println!("Portfolio Summary");
    println!("====================================================\n");

    println!("{:<12} {:<12} {:<12} {:>15} {:>12}",
             "Symbol", "Signal", "Confidence", "Position", "Risk");
    println!("{}", "-".repeat(65));

    for (symbol, signal, confidence, position, risk) in &results {
        println!("{:<12} {:<12?} {:>10.0}% ${:>14.2} ${:>11.2}",
                 symbol, signal, confidence * 100.0, position, risk);
    }

    println!("{}", "-".repeat(65));
    println!("{:<12} {:<12} {:>12} ${:>14.2} ${:>11.2}",
             "TOTAL", "", "", total_position, total_risk);

    println!("\nAllocation: {:.1}% of portfolio", total_position / portfolio_value * 100.0);
    println!("Total Risk: {:.2}% of portfolio", total_risk / portfolio_value * 100.0);

    println!("\n====================================================");
    println!("Demo complete!");
    println!("====================================================");
    println!("\nNote: This uses mock data. For real crypto trading:");
    println!("  1. Use BybitLoader for real-time Bybit data");
    println!("  2. Configure API keys for authenticated endpoints");
    println!("  3. Always use appropriate risk management!");

    Ok(())
}
