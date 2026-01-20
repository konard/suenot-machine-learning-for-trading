//! Cryptocurrency factor discovery example
//!
//! Demonstrates factor discovery specifically for crypto markets using Bybit.
//!
//! Run with: cargo run --example crypto_factors

use llm_factor_discovery::{
    data::{BybitClient, BybitConfig, DataBuffer, TimeFrame},
    evaluation::IcCalculator,
    parser::FactorExpr,
    strategy::{FactorSignal, SignalGenerator, TradingSignal},
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== Cryptocurrency Factor Discovery Example ===\n");

    // Define crypto-specific factors
    let crypto_factors = vec![
        (
            "btc_momentum_short",
            "rank(returns(close, 4))",
            "Short-term momentum (4-hour) often works in crypto",
        ),
        (
            "funding_proxy",
            "ts_rank(ts_delta(close, 8), 24)",
            "Proxy for funding rate direction in perpetuals",
        ),
        (
            "volume_spike",
            "div(volume, ts_mean(volume, 24))",
            "Volume relative to 24-period average",
        ),
        (
            "volatility_regime",
            "ts_rank(volatility(close, 24), 72)",
            "Volatility regime indicator",
        ),
        (
            "range_breakout",
            "div(sub(close, ts_min(low, 24)), sub(ts_max(high, 24), ts_min(low, 24)))",
            "Position within recent range",
        ),
        (
            "mean_reversion_extreme",
            "sub(0, ts_rank(close, 48))",
            "Counter-trend when at extremes",
        ),
    ];

    info!("Crypto-Specific Factors:\n");
    for (name, expr_str, description) in &crypto_factors {
        info!("Factor: {}", name);
        info!("  Expression: {}", expr_str);
        info!("  Rationale: {}", description);

        match FactorExpr::parse(expr_str) {
            Ok(expr) => {
                info!("  Complexity: {} levels", expr.depth());
                info!("  Variables: {:?}", expr.variables());
            }
            Err(e) => {
                info!("  Parse Error: {}", e);
            }
        }
        info!("");
    }

    // Demonstrate Bybit client usage
    info!("=== Bybit Data Client ===\n");

    // Check if we should use testnet
    let use_testnet = std::env::var("USE_TESTNET")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(true);

    let config = if use_testnet {
        info!("Using Bybit testnet (safe for testing)");
        BybitConfig::testnet()
    } else {
        info!("Using Bybit mainnet");
        BybitConfig::default()
    };

    info!("  Base URL: {}\n", config.base_url);

    // Try to fetch market data
    match BybitClient::new(config) {
        Ok(client) => {
            // Fetch ticker
            info!("Fetching BTCUSDT ticker...");
            match client.get_ticker("BTCUSDT").await {
                Ok(ticker) => {
                    info!("  Symbol: {}", ticker.symbol);
                    info!("  Last Price: ${:.2}", ticker.last_price);
                    info!("  24h High: ${:.2}", ticker.high_24h);
                    info!("  24h Low: ${:.2}", ticker.low_24h);
                    info!("  24h Volume: {:.2}", ticker.volume_24h);
                    info!("  24h Change: {:.2}%", ticker.price_change_24h * 100.0);
                    info!("  Spread: ${:.2} ({:.4}%)",
                        ticker.spread(),
                        ticker.spread_percent()
                    );
                }
                Err(e) => {
                    info!("  Failed to fetch ticker: {}", e);
                }
            }

            // Fetch klines
            info!("\nFetching BTCUSDT hourly klines...");
            match client.get_klines("BTCUSDT", TimeFrame::Hour1, Some(24), None, None).await {
                Ok(bars) => {
                    info!("  Received {} bars", bars.len());

                    if let Some(latest) = bars.last() {
                        info!("  Latest bar:");
                        info!("    Time: {}", latest.timestamp);
                        info!("    OHLC: {:.2}/{:.2}/{:.2}/{:.2}",
                            latest.open, latest.high, latest.low, latest.close);
                        info!("    Volume: {:.2}", latest.volume);
                    }

                    // Calculate some quick stats
                    if bars.len() >= 10 {
                        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
                        let returns: Vec<f64> = closes.windows(2)
                            .map(|w| (w[1] - w[0]) / w[0])
                            .collect();

                        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
                        let volatility = {
                            let var: f64 = returns.iter()
                                .map(|r| (r - mean_ret).powi(2))
                                .sum::<f64>() / returns.len() as f64;
                            var.sqrt()
                        };

                        info!("\n  Quick Stats (last {} hours):", bars.len());
                        info!("    Total Return: {:.2}%",
                            ((closes.last().unwrap() / closes.first().unwrap()) - 1.0) * 100.0);
                        info!("    Hourly Volatility: {:.4}%", volatility * 100.0);
                        info!("    Annualized Vol: {:.2}%", volatility * (24.0 * 365.0_f64).sqrt() * 100.0);
                    }
                }
                Err(e) => {
                    info!("  Failed to fetch klines: {}", e);
                }
            }

            // Fetch orderbook
            info!("\nFetching BTCUSDT orderbook...");
            match client.get_orderbook("BTCUSDT", Some(5)).await {
                Ok(book) => {
                    info!("  Bids (top 3):");
                    for (price, qty) in book.bids.iter().take(3) {
                        info!("    ${:.2} x {:.4}", price, qty);
                    }

                    info!("  Asks (top 3):");
                    for (price, qty) in book.asks.iter().take(3) {
                        info!("    ${:.2} x {:.4}", price, qty);
                    }

                    if let Some(mid) = book.mid_price() {
                        info!("  Mid Price: ${:.2}", mid);
                    }

                    info!("  Book Imbalance (5 levels): {:.4}", book.imbalance(5));
                }
                Err(e) => {
                    info!("  Failed to fetch orderbook: {}", e);
                }
            }
        }
        Err(e) => {
            info!("Failed to create Bybit client: {}", e);
            info!("Continuing with simulated data...\n");
        }
    }

    // Signal generation example
    info!("\n=== Signal Generation Example ===\n");

    // Create some factor signals
    let factor_signals = vec![
        FactorSignal::new("momentum_4h".to_string(), 1.8, 0.0, 1.0),
        FactorSignal::new("volume_spike".to_string(), 2.2, 1.0, 0.5),
        FactorSignal::new("mean_reversion".to_string(), -0.5, 0.0, 1.0),
    ];

    info!("Factor Signals:");
    for signal in &factor_signals {
        info!("  {}: value={:.2}, z={:.2}",
            signal.factor_name, signal.value, signal.z_score);
    }

    // Generate trading signal
    let generator = SignalGenerator::new();
    let weights = vec![1.0, 0.5, 0.3]; // Momentum > Volume > Mean Reversion

    if let Some(trading_signal) = generator.generate("BTCUSDT", &factor_signals, &weights) {
        info!("\nGenerated Trading Signal:");
        info!("  Symbol: {}", trading_signal.symbol);
        info!("  Direction: {:.2} ({})",
            trading_signal.direction,
            if trading_signal.is_long() { "LONG" } else { "SHORT" });
        info!("  Strength: {:.2}", trading_signal.strength);
        info!("  Confidence: {:.2}", trading_signal.confidence);

        if trading_signal.is_tradeable(0.3, 0.5) {
            info!("  Status: TRADEABLE");
            info!("  Suggested Size: {:.2}% of capital",
                trading_signal.suggested_position_size(0.1) * 100.0);
        } else {
            info!("  Status: Not tradeable (below thresholds)");
        }
    }

    // Data buffer example
    info!("\n=== Data Buffer Example ===\n");

    let mut buffer = DataBuffer::new(100);
    info!("Created buffer with capacity: {}", buffer.capacity());

    // Simulate streaming data
    for i in 0..10 {
        let bar = llm_factor_discovery::data::OhlcvBar {
            timestamp: chrono::Utc::now(),
            open: 50000.0 + i as f64 * 10.0,
            high: 50010.0 + i as f64 * 10.0,
            low: 49990.0 + i as f64 * 10.0,
            close: 50005.0 + i as f64 * 10.0,
            volume: 100.0 + i as f64 * 5.0,
            turnover: 5000000.0,
        };
        buffer.push(bar);
    }

    info!("Buffer size: {}", buffer.len());
    info!("Latest 3 closes: {:?}", &buffer.closes()[buffer.len()-3..]);

    let returns = buffer.returns(1);
    if !returns.is_empty() {
        info!("Recent returns: {:?}",
            returns.iter().take(3).map(|r| format!("{:.4}%", r * 100.0)).collect::<Vec<_>>());
    }

    info!("\n=== Cryptocurrency Example Complete ===");

    Ok(())
}
