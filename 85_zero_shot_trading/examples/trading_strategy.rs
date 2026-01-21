//! Trading strategy example: Live trading simulation
//!
//! This example shows how to use the ZeroShotTradingStrategy
//! for real-time trading decisions with position management.
//!
//! Run with: cargo run --example trading_strategy

use zero_shot_trading::prelude::*;
use zero_shot_trading::data::bybit::Kline;
use zero_shot_trading::data::attributes::{AssetType, MarketCapTier, VolatilityRegime, Sector};
use chrono::Utc;

fn main() -> Result<()> {
    println!("Zero-Shot Trading: Trading Strategy Example");
    println!("===========================================\n");

    // Create model and strategy
    let model_config = zero_shot_trading::model::network::ModelConfig::default();
    let model = ZeroShotModel::new(model_config);

    let strategy_config = StrategyConfig {
        symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
        interval: "15".to_string(),
        lookback: 200,
        max_position_size: 0.1,
        risk_per_trade: 0.02,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.04,
        min_confidence: 0.5,
    };

    let mut strategy = ZeroShotTradingStrategy::new(model, strategy_config);

    // Register custom asset attributes
    strategy.register_asset("BTCUSDT", AssetAttributes {
        asset_type: AssetType::Cryptocurrency,
        market_cap: MarketCapTier::Large,
        volatility: VolatilityRegime::High,
        sector: Sector::Layer1,
        volume_normalized: 0.95,
        benchmark_correlation: 1.0, // BTC is the benchmark
        avg_spread: 0.0001,
        trading_hours: 24.0,
    });

    strategy.register_asset("ETHUSDT", AssetAttributes {
        asset_type: AssetType::Cryptocurrency,
        market_cap: MarketCapTier::Large,
        volatility: VolatilityRegime::High,
        sector: Sector::Layer1,
        volume_normalized: 0.9,
        benchmark_correlation: 0.85,
        avg_spread: 0.0002,
        trading_hours: 24.0,
    });

    println!("Strategy Configuration:");
    println!("  Max Position Size: {:.1}%", strategy.config().max_position_size * 100.0);
    println!("  Stop Loss: {:.1}%", strategy.config().stop_loss_pct * 100.0);
    println!("  Take Profit: {:.1}%", strategy.config().take_profit_pct * 100.0);
    println!("  Min Confidence: {:.1}%", strategy.config().min_confidence * 100.0);
    println!();

    // Generate synthetic market data for BTC
    let btc_klines = generate_market_data(200, 50000.0, 0.001);
    let eth_klines = generate_market_data(200, 3000.0, 0.0015);

    // Portfolio value
    let portfolio_value = 100_000.0;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  ANALYZING BTCUSDT");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Analyze BTC
    match strategy.analyze(&btc_klines, "BTCUSDT") {
        Ok(signal) => {
            println!("  Signal: {}", signal.signal);
            println!("  Regime: {:?}", signal.regime);
            println!("  Confidence: {:.1}%", signal.confidence * 100.0);
            println!("  Price: ${:.2}", signal.price);

            // Try to open position
            if let Some(position) = strategy.open_position(&signal, portfolio_value) {
                println!("\n  📈 POSITION OPENED:");
                println!("     Side: {}", if position.is_long { "LONG" } else { "SHORT" });
                println!("     Entry: ${:.2}", position.entry_price);
                println!("     Size: ${:.2}", position.size);
                println!("     Stop Loss: ${:.2}", position.stop_loss);
                println!("     Take Profit: ${:.2}", position.take_profit);
            }
        }
        Err(e) => println!("  Analysis error: {}", e),
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  ANALYZING ETHUSDT");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Analyze ETH
    match strategy.analyze(&eth_klines, "ETHUSDT") {
        Ok(signal) => {
            println!("  Signal: {}", signal.signal);
            println!("  Regime: {:?}", signal.regime);
            println!("  Confidence: {:.1}%", signal.confidence * 100.0);
            println!("  Price: ${:.2}", signal.price);

            // Try to open position
            if let Some(position) = strategy.open_position(&signal, portfolio_value) {
                println!("\n  📈 POSITION OPENED:");
                println!("     Side: {}", if position.is_long { "LONG" } else { "SHORT" });
                println!("     Entry: ${:.2}", position.entry_price);
                println!("     Size: ${:.2}", position.size);
                println!("     Stop Loss: ${:.2}", position.stop_loss);
                println!("     Take Profit: ${:.2}", position.take_profit);
            }
        }
        Err(e) => println!("  Analysis error: {}", e),
    }

    // Display strategy summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  STRATEGY SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");

    let summary = strategy.summary();
    println!("{}", summary);

    // Show all positions
    println!("Open Positions:");
    for (symbol, position) in strategy.positions() {
        println!("  {} | {} | Entry: ${:.2} | Size: ${:.2}",
            symbol,
            if position.is_long { "LONG" } else { "SHORT" },
            position.entry_price,
            position.size
        );
    }

    // Show signal history
    println!("\nRecent Signals:");
    for signal in strategy.signal_history().iter().take(5) {
        println!("  {} | {} | {:?} | {:.1}% conf",
            signal.symbol,
            signal.signal,
            signal.regime,
            signal.confidence * 100.0
        );
    }

    // Simulate price updates and check exits
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  SIMULATING PRICE UPDATES");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Simulate a profitable move in BTC
    let mut prices = std::collections::HashMap::new();
    prices.insert("BTCUSDT".to_string(), 52000.0); // Price moved up
    prices.insert("ETHUSDT".to_string(), 3100.0);

    strategy.update_positions(&prices);

    // Check for exits
    let exits = strategy.check_exits(&prices);
    for (symbol, pnl, reason) in exits {
        println!("  CLOSED {} | PnL: ${:.2} | Reason: {}",
            symbol, pnl, reason);
    }

    // Final positions state
    println!("\nFinal Positions:");
    for (symbol, position) in strategy.positions() {
        println!("  {} | {} | PnL: ${:.2}",
            symbol,
            if position.is_long { "LONG" } else { "SHORT" },
            position.unrealized_pnl
        );
    }

    println!("\n✓ Strategy example completed!");
    Ok(())
}

/// Generate synthetic market data
fn generate_market_data(n: usize, start_price: f64, trend: f64) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(n);
    let mut price = start_price;

    for i in 0..n {
        let change = trend + (rand::random::<f64>() - 0.5) * 0.02;
        price *= 1.0 + change;

        let volatility = 0.01 + rand::random::<f64>() * 0.01;
        klines.push(Kline {
            timestamp: Utc::now() + chrono::Duration::minutes(i as i64 * 15),
            open: price * (1.0 - volatility * 0.5),
            high: price * (1.0 + volatility),
            low: price * (1.0 - volatility),
            close: price,
            volume: 100.0 * (1.0 + rand::random::<f64>()),
            turnover: price * 100.0,
        });
    }

    klines
}
