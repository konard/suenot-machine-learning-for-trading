//! Example: Live trading simulation
//!
//! This example demonstrates a live trading simulation loop that:
//! 1. Fetches real-time data from Bybit
//! 2. Generates signals using the DSC model
//! 3. Simulates order execution
//!
//! Run with: cargo run --example live_trading
//!
//! NOTE: This is a SIMULATION only. It does not execute real trades.

use dsc_trading::{
    convolution::DepthwiseSeparableConv1d,
    data::{BybitClient, Candle},
    strategy::{Portfolio, Signal, SignalGenerator, TradingStrategy},
};
use std::error::Error;
use std::time::Duration;
use tokio::time::sleep;

/// Simple live trading simulator
struct LiveTrader {
    client: BybitClient,
    strategy: TradingStrategy,
    portfolio: Portfolio,
    symbol: String,
    candles: Vec<Candle>,
    window_size: usize,
}

impl LiveTrader {
    fn new(symbol: &str, initial_capital: f64) -> Result<Self, Box<dyn Error>> {
        // Create DSC model (22 features: 5 OHLCV + 17 indicators)
        let model = DepthwiseSeparableConv1d::new(22, 64, 3)?;

        let strategy = TradingStrategy::new(model)
            .with_window_size(100)
            .with_confidence_threshold(0.6);

        Ok(Self {
            client: BybitClient::new(),
            strategy,
            portfolio: Portfolio::new(initial_capital, 0.001),
            symbol: symbol.to_string(),
            candles: Vec::new(),
            window_size: 100,
        })
    }

    async fn initialize(&mut self) -> Result<(), Box<dyn Error>> {
        println!("Initializing with historical data...");

        // Fetch initial historical data
        self.candles = self.client.get_klines(&self.symbol, "5", 200).await?;

        println!("Loaded {} candles", self.candles.len());
        Ok(())
    }

    async fn update(&mut self) -> Result<Option<Signal>, Box<dyn Error>> {
        // Fetch latest candle
        let latest = self.client.get_klines(&self.symbol, "5", 1).await?;

        if let Some(candle) = latest.into_iter().next() {
            // Check if this is a new candle
            let is_new = self
                .candles
                .last()
                .map(|c| c.timestamp != candle.timestamp)
                .unwrap_or(true);

            if is_new {
                self.candles.push(candle);

                // Keep only necessary history
                if self.candles.len() > 500 {
                    self.candles.remove(0);
                }
            }
        }

        // Generate signal if we have enough data
        if self.candles.len() >= self.window_size {
            let features = self.strategy.prepare_features(&self.candles)?;
            let window = features
                .slice(ndarray::s![.., features.dim().1 - self.window_size..])
                .to_owned();

            let (signal, confidence) = self.strategy.generate_signal(&window)?;

            if confidence >= 0.6 {
                return Ok(Some(signal));
            }
        }

        Ok(None)
    }

    fn execute_signal(&mut self, signal: Signal, price: f64) {
        match signal {
            Signal::StrongBuy | Signal::Buy => {
                if self.portfolio.position.is_flat() {
                    let size = self.portfolio.cash * 0.9 / price;
                    self.portfolio.open_long(price, size, 0);
                    println!(
                        "  📈 OPENED LONG: {:.4} {} @ ${:.2}",
                        size, self.symbol, price
                    );
                }
            }
            Signal::StrongSell | Signal::Sell => {
                if self.portfolio.position.is_long() {
                    // Close position (simplified - no proper tracking)
                    println!("  📉 CLOSED LONG @ ${:.2}", price);
                    self.portfolio.position = dsc_trading::strategy::Position::Flat;
                }
            }
            Signal::Hold => {
                // Do nothing
            }
        }
    }

    fn status(&self) -> String {
        let last_price = self.candles.last().map(|c| c.close).unwrap_or(0.0);
        let equity = self.portfolio.equity(last_price);

        format!(
            "Equity: ${:.2} | Position: {:?} | Price: ${:.2}",
            equity, self.portfolio.position, last_price
        )
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("===========================================");
    println!("  Live Trading Simulation");
    println!("  ⚠️  SIMULATION MODE - NO REAL TRADES");
    println!("===========================================\n");

    let symbol = "BTCUSDT";
    let initial_capital = 10_000.0;

    println!("Symbol: {}", symbol);
    println!("Initial Capital: ${:.2}", initial_capital);
    println!();

    // Create trader
    let mut trader = LiveTrader::new(symbol, initial_capital)?;

    // Initialize with historical data
    trader.initialize().await?;

    println!("\nStarting live simulation...");
    println!("Press Ctrl+C to stop\n");

    // Simulation loop
    let mut iteration = 0;
    let max_iterations = 20; // Run for limited iterations in example

    loop {
        iteration += 1;

        if iteration > max_iterations {
            println!("\nReached maximum iterations ({})", max_iterations);
            break;
        }

        print!("[{}] ", iteration);

        // Update and get signal
        match trader.update().await {
            Ok(Some(signal)) => {
                let price = trader.candles.last().map(|c| c.close).unwrap_or(0.0);
                println!("Signal: {:?}", signal);
                trader.execute_signal(signal, price);
            }
            Ok(None) => {
                println!("No signal");
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }

        println!("     {}", trader.status());

        // Wait before next update (5 seconds for demo)
        sleep(Duration::from_secs(5)).await;
    }

    // Final summary
    println!("\n===========================================");
    println!("  Simulation Summary");
    println!("===========================================");

    let final_price = trader.candles.last().map(|c| c.close).unwrap_or(0.0);
    let final_equity = trader.portfolio.equity(final_price);
    let total_return = (final_equity - initial_capital) / initial_capital * 100.0;

    println!("Final Equity: ${:.2}", final_equity);
    println!("Total Return: {:.2}%", total_return);
    println!("Trades: {}", trader.portfolio.trades.len());

    Ok(())
}
