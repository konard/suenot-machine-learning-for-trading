//! Rule Extraction Trading CLI
//!
//! A command-line tool for extracting trading rules from black-box models
//! and backtesting rule-based strategies.

use anyhow::Result;
use clap::{Parser, Subcommand};
use log::info;

mod data;
mod extraction;
mod rules;
mod trading;

use data::{BybitClient, Interval};

/// Rule Extraction Trading CLI
#[derive(Parser)]
#[command(name = "rule-extraction")]
#[command(about = "Extract interpretable trading rules from black-box models")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch market data from Bybit
    Fetch {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Candlestick interval (1, 5, 15, 60, D)
        #[arg(short, long, default_value = "60")]
        interval: String,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "100")]
        limit: u32,
    },

    /// Extract rules from sample data
    Extract {
        /// Maximum tree depth for rule extraction
        #[arg(short, long, default_value = "5")]
        max_depth: usize,

        /// Minimum samples per leaf
        #[arg(short = 's', long, default_value = "20")]
        min_samples: usize,
    },

    /// Run backtest with extracted rules
    Backtest {
        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        capital: f64,

        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            symbol,
            interval,
            limit,
        } => {
            info!("Fetching {} data from Bybit...", symbol);
            fetch_data(&symbol, &interval, limit)?;
        }
        Commands::Extract {
            max_depth,
            min_samples,
        } => {
            info!("Extracting rules with max_depth={}, min_samples={}", max_depth, min_samples);
            extract_rules(max_depth, min_samples)?;
        }
        Commands::Backtest { capital, symbol } => {
            info!("Running backtest for {} with ${} capital", symbol, capital);
            run_backtest(capital, &symbol)?;
        }
    }

    Ok(())
}

fn fetch_data(symbol: &str, interval: &str, limit: u32) -> Result<()> {
    let client = BybitClient::new();

    // Fetch klines
    let klines = client.get_klines(symbol, interval, Some(limit))?;

    println!("\n=== {} Kline Data ({} interval) ===", symbol, interval);
    println!("{:-<80}", "");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{:-<80}", "");

    for kline in klines.iter().take(10) {
        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            kline.timestamp.format("%Y-%m-%d %H:%M"),
            kline.open,
            kline.high,
            kline.low,
            kline.close,
            kline.volume
        );
    }

    if klines.len() > 10 {
        println!("... ({} more rows)", klines.len() - 10);
    }

    // Fetch ticker
    let ticker = client.get_ticker(symbol)?;
    println!("\n=== Current Ticker ===");
    println!("Last Price: ${:.2}", ticker.last_price);
    println!("24h Change: {:.2}%", ticker.price_change_24h * 100.0);
    println!("24h Volume: {:.2}", ticker.volume_24h);

    Ok(())
}

fn extract_rules(max_depth: usize, min_samples: usize) -> Result<()> {
    use extraction::TrepanExtractor;
    use rules::Condition;

    let feature_names = vec![
        "RSI".to_string(),
        "MACD".to_string(),
        "SMA_Ratio".to_string(),
        "Volatility".to_string(),
        "Volume_Change".to_string(),
    ];

    // Generate sample data
    println!("Generating sample market data...");
    let (features, labels) = generate_sample_data(500);

    // Extract rules
    println!("Extracting rules with TREPAN-like algorithm...");
    let extractor = TrepanExtractor::new(feature_names.clone())
        .with_max_depth(max_depth)
        .with_min_samples_leaf(min_samples);

    let ruleset = extractor.extract_from_predictions(&features, &labels);

    println!("\n=== Extracted Rules ===");
    println!("{:-<80}", "");

    for (i, rule) in ruleset.rules.iter().enumerate() {
        println!("Rule {}: {}", i + 1, rule.to_string(&feature_names));
    }

    println!("{:-<80}", "");
    println!("Total rules: {}", ruleset.rules.len());
    println!("Average complexity: {:.2}", ruleset.average_complexity());
    println!("Coverage: {:.1}%", ruleset.calculate_coverage(&features) * 100.0);
    println!(
        "Fidelity: {:.1}%",
        ruleset.calculate_fidelity(&features, &labels) * 100.0
    );

    Ok(())
}

fn run_backtest(initial_capital: f64, symbol: &str) -> Result<()> {
    use extraction::TrepanExtractor;
    use rules::{Condition, Operator, Rule};
    use trading::{backtest, buy_and_hold_benchmark, RuleBasedStrategy};

    let client = BybitClient::new();

    println!("Fetching {} data from Bybit...", symbol);
    let klines = client.get_klines(symbol, "60", Some(200))?;

    if klines.len() < 50 {
        anyhow::bail!("Not enough data for backtesting");
    }

    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();

    // Compute features
    let stock_data: Vec<data::stock::StockData> = klines
        .iter()
        .map(|k| data::stock::StockData {
            date: k.timestamp.to_string(),
            open: k.open,
            high: k.high,
            low: k.low,
            close: k.close,
            volume: k.volume,
        })
        .collect();

    let (features, feature_names) = data::stock::compute_features(&stock_data);
    let labels = data::stock::generate_labels(&prices);

    // Extract rules
    println!("Extracting trading rules...");
    let extractor = TrepanExtractor::new(feature_names.clone())
        .with_max_depth(5)
        .with_min_samples_leaf(20);

    let ruleset = extractor.extract_from_predictions(&features, &labels);

    println!("\nExtracted {} rules:", ruleset.rules.len());
    for (i, rule) in ruleset.rules.iter().take(5).enumerate() {
        println!("  {}. {}", i + 1, rule.to_string(&feature_names));
    }

    // Run backtest
    println!("\n=== Backtest Results ===");
    let strategy = RuleBasedStrategy::from_ruleset(ruleset);
    let result = backtest(&strategy, &features, &prices, initial_capital);

    println!("{:-<50}", "");
    println!("Rule-Based Strategy:");
    println!("  Total Return:    {:>10.2}%", result.total_return * 100.0);
    println!("  Sharpe Ratio:    {:>10.2}", result.sharpe_ratio);
    println!("  Max Drawdown:    {:>10.2}%", result.max_drawdown * 100.0);
    println!("  Number of Trades:{:>10}", result.n_trades);
    println!("  Win Rate:        {:>10.1}%", result.win_rate * 100.0);
    println!("  Avg Trade Return:{:>10.2}%", result.avg_trade_return * 100.0);

    // Compare with buy-and-hold
    let benchmark = buy_and_hold_benchmark(&prices, initial_capital);
    println!("\nBuy-and-Hold Benchmark:");
    println!("  Total Return:    {:>10.2}%", benchmark.total_return * 100.0);
    println!("  Sharpe Ratio:    {:>10.2}", benchmark.sharpe_ratio);
    println!("  Max Drawdown:    {:>10.2}%", benchmark.max_drawdown * 100.0);

    // Show some trades
    if !result.trades.is_empty() {
        println!("\n=== Sample Trades ===");
        for trade in result.trades.iter().take(5) {
            println!(
                "  [{}] {} at ${:.2}",
                trade.idx, trade.trade_type, trade.price
            );
            if let Some(reason) = trade.explanation.first() {
                println!("       Reason: {}", reason);
            }
        }
    }

    Ok(())
}

/// Generate sample market data for demonstration.
fn generate_sample_data(n: usize) -> (Vec<Vec<f64>>, Vec<i32>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut features = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for _ in 0..n {
        // Generate random features
        let rsi: f64 = rng.gen_range(20.0..80.0);
        let macd: f64 = rng.gen_range(-2.0..2.0);
        let sma_ratio: f64 = rng.gen_range(0.95..1.05);
        let volatility: f64 = rng.gen_range(0.005..0.05);
        let volume_change: f64 = rng.gen_range(-0.5..0.5);

        features.push(vec![rsi, macd, sma_ratio, volatility, volume_change]);

        // Generate label based on a simple rule (with noise)
        let signal = if rsi < 35.0 && macd > 0.0 {
            1 // BUY
        } else if rsi > 65.0 && macd < 0.0 {
            -1 // SELL
        } else if sma_ratio > 1.02 {
            1 // BUY
        } else if sma_ratio < 0.98 {
            -1 // SELL
        } else {
            0 // HOLD
        };

        // Add some noise
        let noisy_label = if rng.gen::<f64>() < 0.15 {
            if rng.gen::<bool>() {
                1
            } else {
                -1
            }
        } else {
            signal
        };

        labels.push(noisy_label);
    }

    (features, labels)
}
