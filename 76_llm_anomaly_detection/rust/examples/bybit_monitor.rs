//! Real-time anomaly monitoring for Bybit markets.
//!
//! This example demonstrates continuous monitoring of cryptocurrency
//! markets for anomalies using data from Bybit.

use llm_anomaly_detection::{
    data_loader::{BybitLoader, FeatureCalculator, Ticker},
    detector::{AnomalyDetector, EnsembleDetector, IsolationDetector, StatisticalDetector, VotingMethod},
    signals::{SignalGenerator, SignalStrategy},
    AnomalyResult, Features, TradingSignal,
};
use std::collections::HashMap;
use tokio::time::{sleep, Duration};

/// Monitor state for a single symbol.
struct SymbolMonitor {
    symbol: String,
    detector: Box<dyn AnomalyDetector + Send + Sync>,
    signal_generator: SignalGenerator,
    last_features: Option<Features>,
    is_initialized: bool,
}

impl SymbolMonitor {
    fn new(symbol: &str, z_threshold: f64) -> Self {
        // Create ensemble detector for robust detection
        let ensemble = EnsembleDetector::new(VotingMethod::Soft, 0.5)
            .add_detector(StatisticalDetector::new(z_threshold))
            .add_detector(IsolationDetector::new(95.0));

        Self {
            symbol: symbol.to_string(),
            detector: Box::new(ensemble),
            signal_generator: SignalGenerator::with_strategy(SignalStrategy::Risk),
            last_features: None,
            is_initialized: false,
        }
    }
}

/// Multi-symbol anomaly monitor.
struct AnomalyMonitor {
    loader: BybitLoader,
    calculator: FeatureCalculator,
    monitors: HashMap<String, SymbolMonitor>,
}

impl AnomalyMonitor {
    fn new(symbols: &[&str], z_threshold: f64) -> Self {
        let monitors = symbols
            .iter()
            .map(|s| (s.to_string(), SymbolMonitor::new(s, z_threshold)))
            .collect();

        Self {
            loader: BybitLoader::new(),
            calculator: FeatureCalculator::new(20),
            monitors,
        }
    }

    async fn initialize(&mut self, history_limit: usize) -> anyhow::Result<()> {
        println!("Initializing anomaly monitors...");

        for (symbol, monitor) in &mut self.monitors {
            print!("  Loading history for {}...", symbol);

            match self.loader.get_klines(symbol, "15m", history_limit).await {
                Ok(candles) => {
                    let features = self.calculator.calculate_features(&candles);

                    if features.len() >= 50 {
                        if let Err(e) = monitor.detector.fit(&features) {
                            println!(" FAILED (fit error: {})", e);
                            continue;
                        }
                        monitor.is_initialized = true;
                        println!(" OK ({} candles)", candles.len());
                    } else {
                        println!(" FAILED (not enough data)");
                    }
                }
                Err(e) => {
                    println!(" FAILED ({})", e);
                }
            }
        }

        println!("Initialization complete!\n");
        Ok(())
    }

    async fn check_symbol(
        &self,
        symbol: &str,
    ) -> anyhow::Result<(AnomalyResult, TradingSignal, Option<Ticker>)> {
        let monitor = self
            .monitors
            .get(symbol)
            .ok_or_else(|| anyhow::anyhow!("Symbol not found: {}", symbol))?;

        if !monitor.is_initialized {
            return Err(anyhow::anyhow!("Monitor not initialized for {}", symbol));
        }

        // Get latest data
        let candles = self.loader.get_klines(symbol, "15m", 50).await?;
        let features = self.calculator.calculate_features(&candles);

        let latest = features
            .last()
            .ok_or_else(|| anyhow::anyhow!("No features calculated"))?;

        // Detect anomaly
        let result = monitor.detector.detect(latest)?;

        // Generate signal
        let signal = monitor.signal_generator.generate(&result, latest, 0.0);

        // Get ticker info
        let ticker = self.loader.get_ticker(symbol).await.ok();

        Ok((result, signal, ticker))
    }

    async fn run_monitoring_loop(
        &mut self,
        check_interval_secs: u64,
        max_checks: usize,
    ) -> anyhow::Result<()> {
        println!("{}", "=".repeat(60));
        println!("STARTING REAL-TIME MONITORING");
        println!("{}", "=".repeat(60));
        println!(
            "Symbols: {}",
            self.monitors.keys().cloned().collect::<Vec<_>>().join(", ")
        );
        println!("Check interval: {} seconds", check_interval_secs);
        println!("{}", "=".repeat(60));

        let symbols: Vec<String> = self.monitors.keys().cloned().collect();

        for check in 1..=max_checks {
            let now = chrono::Utc::now();
            println!(
                "\n--- Check #{} at {} ---",
                check,
                now.format("%Y-%m-%d %H:%M:%S")
            );

            for symbol in &symbols {
                match self.check_symbol(symbol).await {
                    Ok((result, signal, ticker)) => {
                        let status = if result.is_anomaly { "ANOMALY" } else { "NORMAL" };

                        // ANSI colors: red for anomaly, green for normal
                        let color = if result.is_anomaly { "\x1b[91m" } else { "\x1b[92m" };
                        let reset = "\x1b[0m";

                        println!("\n{}:", symbol);
                        println!("  Status: {}{}{}", color, status, reset);

                        if let Some(t) = ticker {
                            println!("  Price: ${:.2}", t.last_price);
                            println!("  24h Change: {:.2}%", t.price_change_pct);
                        }

                        if result.is_anomaly {
                            println!("  Anomaly Score: {:.3}", result.score);
                            println!("  Type: {}", result.anomaly_type);
                            println!("  Explanation: {}", result.explanation);
                            println!("  Signal: {}", signal.signal_type);
                            println!("  Signal Reason: {}", signal.reason);
                        }
                    }
                    Err(e) => {
                        println!("\n{}: Error - {}", symbol, e);
                    }
                }
            }

            if check < max_checks {
                println!("\nNext check in {} seconds...", check_interval_secs);
                sleep(Duration::from_secs(check_interval_secs)).await;
            }
        }

        println!("\n{}", "=".repeat(60));
        println!("Monitoring stopped");

        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("{}", "=".repeat(60));
    println!("Bybit Real-time Anomaly Monitor");
    println!("{}", "=".repeat(60));

    // Create monitor for multiple symbols
    let symbols = &["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let mut monitor = AnomalyMonitor::new(symbols, 2.5);

    // Initialize with historical data
    monitor.initialize(300).await?;

    // Run monitoring loop (3 checks with 30 second intervals)
    monitor.run_monitoring_loop(30, 3).await?;

    Ok(())
}
