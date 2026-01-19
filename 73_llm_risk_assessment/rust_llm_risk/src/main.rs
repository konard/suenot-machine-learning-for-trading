//! LLM Risk Assessment CLI
//!
//! Command-line interface for risk assessment using Large Language Models.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber;

use llm_risk_assessment::data::BybitClient;
use llm_risk_assessment::llm::{LLMClient, LLMProvider};
use llm_risk_assessment::risk::RiskAssessor;
use llm_risk_assessment::strategy::{BacktestConfig, Backtester, RiskBasedTrader};
use llm_risk_assessment::utils::load_config;

#[derive(Parser)]
#[command(name = "llm-risk")]
#[command(about = "LLM-based risk assessment for trading")]
#[command(version)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Assess risk from text input
    Assess {
        /// Text to analyze
        #[arg(short, long)]
        text: String,

        /// Symbol for context
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,
    },

    /// Fetch market data
    Fetch {
        /// Symbols to fetch (comma-separated)
        #[arg(short, long, default_value = "BTCUSDT,ETHUSDT")]
        symbols: String,

        /// Number of days
        #[arg(short, long, default_value = "7")]
        days: u32,

        /// Interval (1, 5, 15, 30, 60, 240, D)
        #[arg(short, long, default_value = "60")]
        interval: String,
    },

    /// Run a backtest
    Backtest {
        /// Symbol to backtest
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Number of days
        #[arg(short, long, default_value = "30")]
        days: u32,

        /// Initial capital
        #[arg(short, long, default_value = "10000")]
        capital: f64,
    },

    /// Generate sample configuration file
    Config {
        /// Output path
        #[arg(short, long, default_value = "config.toml")]
        output: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    // Load configuration
    let config = load_config(&cli.config)?;

    match cli.command {
        Commands::Assess { text, symbol } => {
            assess_risk(&config, &text, &symbol).await?;
        }
        Commands::Fetch {
            symbols,
            days,
            interval,
        } => {
            fetch_data(&symbols, days, &interval).await?;
        }
        Commands::Backtest {
            symbol,
            days,
            capital,
        } => {
            run_backtest(&symbol, days, capital).await?;
        }
        Commands::Config { output } => {
            generate_config(&output)?;
        }
    }

    Ok(())
}

async fn assess_risk(
    config: &llm_risk_assessment::utils::Config,
    text: &str,
    symbol: &str,
) -> Result<()> {
    info!("Assessing risk for {}", symbol);

    // Create LLM provider
    let provider = match config.llm.provider.as_str() {
        "openai" => {
            let api_key = config
                .get_api_key()
                .expect("API key required for OpenAI");
            LLMProvider::OpenAI {
                model: config.llm.model.clone(),
                api_key,
            }
        }
        "anthropic" => {
            let api_key = config
                .get_api_key()
                .expect("API key required for Anthropic");
            LLMProvider::Anthropic {
                model: config.llm.model.clone(),
                api_key,
            }
        }
        _ => LLMProvider::Local {
            model: config.llm.model.clone(),
            base_url: config
                .llm
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434".to_string()),
        },
    };

    let llm_client = LLMClient::new(provider)
        .with_temperature(config.llm.temperature)
        .with_max_tokens(config.llm.max_tokens);

    let assessor = RiskAssessor::new(llm_client);

    let score = assessor.assess_with_context(text, symbol, None).await?;

    println!("\n{}", score);
    println!("Recommended position multiplier: {:.2}", score.position_multiplier());

    Ok(())
}

async fn fetch_data(symbols: &str, days: u32, interval: &str) -> Result<()> {
    let symbol_list: Vec<&str> = symbols.split(',').map(|s| s.trim()).collect();

    info!("Fetching data for {} symbols", symbol_list.len());

    let client = BybitClient::new();
    let data = client
        .fetch_multi_symbol(&symbol_list, interval, days)
        .await?;

    println!("\n=== Market Data Summary ===\n");

    for symbol in data.symbols() {
        if let Some(ohlcv) = data.get(symbol) {
            println!("{}: {} candles", symbol, ohlcv.len());

            if let (Some(first), Some(last)) = (ohlcv.first(), ohlcv.last()) {
                let change = (last.close - first.close) / first.close * 100.0;
                println!("  Price: ${:.2} -> ${:.2} ({:+.2}%)", first.close, last.close, change);
            }
        }
    }

    Ok(())
}

async fn run_backtest(symbol: &str, days: u32, capital: f64) -> Result<()> {
    info!("Running backtest for {} ({} days)", symbol, days);

    // Fetch data
    let client = BybitClient::new();
    let price_data = client.fetch_historical_klines(symbol, "60", days).await?;

    if price_data.is_empty() {
        println!("No data available for backtest");
        return Ok(());
    }

    // Generate synthetic risk scores for demo
    let risk_scores: Vec<_> = price_data
        .iter()
        .map(|_| {
            use llm_risk_assessment::risk::RiskScore;
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let base = rng.gen_range(3.0..7.0);
            RiskScore::new(base, base, base, base, base, base)
        })
        .collect();

    // Configure and run backtest
    let mut trader = RiskBasedTrader::new();
    let backtester = Backtester::with_config(BacktestConfig {
        initial_capital: capital,
        ..Default::default()
    });

    let result = backtester.run(&price_data, &risk_scores, &mut trader);

    result.print_summary();

    Ok(())
}

fn generate_config(output: &str) -> Result<()> {
    info!("Generating sample configuration at {}", output);

    llm_risk_assessment::utils::Config::create_sample_config(output)?;

    println!("Sample configuration saved to {}", output);
    println!("\nEdit the file to configure:");
    println!("  - LLM provider (openai, anthropic, local)");
    println!("  - Trading parameters (thresholds, position size)");
    println!("  - Data sources (symbols, intervals)");

    Ok(())
}
