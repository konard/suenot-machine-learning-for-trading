//! Stochastic Volatility Model Example
//!
//! This example demonstrates how to estimate time-varying volatility
//! in cryptocurrency markets using a Bayesian stochastic volatility model.
//!
//! The model captures:
//! - Time-varying volatility (volatility clustering)
//! - Fat tails in return distributions
//! - Volatility persistence and mean reversion

use anyhow::Result;
use bayesian_crypto::bayesian::inference::MCMCConfig;
use bayesian_crypto::bayesian::volatility::StochasticVolatility;
use bayesian_crypto::data::{BybitClient, Returns, Symbol};
use clap::Parser;
use colored::Colorize;
use tabled::{Table, Tabled};

#[derive(Parser, Debug)]
#[command(name = "stochastic_volatility")]
#[command(about = "Bayesian stochastic volatility model for crypto")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Time interval (1, 5, 15, 30, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "300")]
    limit: u32,

    /// Number of MCMC samples
    #[arg(long, default_value = "2000")]
    samples: usize,

    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Tabled)]
struct VolatilityRow {
    #[tabled(rename = "Period")]
    period: String,
    #[tabled(rename = "Return")]
    return_pct: String,
    #[tabled(rename = "Volatility")]
    volatility: String,
    #[tabled(rename = "95% CI")]
    ci: String,
    #[tabled(rename = "Regime")]
    regime: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "\n═══════════════════════════════════════════════════════════════".magenta());
    println!("{}", "  Bayesian Stochastic Volatility Model".magenta().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════\n".magenta());

    // Parse symbol
    let symbol = Symbol::from_str(&args.symbol)
        .ok_or_else(|| anyhow::anyhow!("Unknown symbol: {}", args.symbol))?;

    println!("Symbol: {}", symbol.to_string().cyan());
    println!("Interval: {}", args.interval);
    println!();

    // Fetch data
    let client = BybitClient::new();

    println!("Fetching data from Bybit...");
    let klines = client.get_klines(symbol, &args.interval, args.limit).await?;
    println!("Fetched {} candles\n", klines.len());

    // Calculate returns
    let returns = Returns::from_klines(&klines);

    // Basic return statistics
    println!("{}", "Return Statistics:".yellow().bold());
    println!("{}", "─".repeat(50));
    println!("  Mean:           {:>10.4}%", returns.mean() * 100.0);
    println!("  Std Dev:        {:>10.4}%", returns.std() * 100.0);
    println!("  Skewness:       {:>10.4}", returns.skewness());
    println!("  Kurtosis:       {:>10.4}", returns.kurtosis());
    println!("  Min:            {:>10.4}%", returns.min() * 100.0);
    println!("  Max:            {:>10.4}%", returns.max() * 100.0);
    println!();

    // Evidence of volatility clustering
    println!("{}", "Volatility Clustering Analysis:".yellow().bold());
    println!("{}", "─".repeat(50));

    // Calculate rolling volatility
    let window = 20;
    let rolling_vol = returns.rolling_std(window);

    if rolling_vol.len() > 10 {
        let vol_mean: f64 = rolling_vol.iter().sum::<f64>() / rolling_vol.len() as f64;
        let vol_std: f64 = {
            let var: f64 = rolling_vol.iter().map(|v| (v - vol_mean).powi(2)).sum::<f64>()
                / (rolling_vol.len() - 1) as f64;
            var.sqrt()
        };

        // Autocorrelation of squared returns (evidence of ARCH effects)
        let squared_returns: Vec<f64> = returns.values.iter().map(|r| r.powi(2)).collect();
        let n = squared_returns.len();
        if n > 10 {
            let sq_mean: f64 = squared_returns.iter().sum::<f64>() / n as f64;
            let sq_var: f64 = squared_returns.iter().map(|r| (r - sq_mean).powi(2)).sum::<f64>() / n as f64;

            let mut autocorr = 0.0;
            for i in 0..(n - 1) {
                autocorr += (squared_returns[i] - sq_mean) * (squared_returns[i + 1] - sq_mean);
            }
            autocorr /= (n - 1) as f64 * sq_var;

            println!("  Rolling vol mean:       {:>10.4}%", vol_mean * 100.0);
            println!("  Rolling vol std:        {:>10.4}%", vol_std * 100.0);
            println!("  Squared returns AC(1):  {:>10.4}", autocorr);

            if autocorr > 0.1 {
                println!(
                    "\n  {} Volatility clustering detected!",
                    "Evidence:".green().bold()
                );
                println!("  Stochastic volatility model is appropriate.");
            } else {
                println!("\n  Weak evidence of volatility clustering.");
            }
        }
    }
    println!();

    // Fit stochastic volatility model
    println!("{}", "Fitting Stochastic Volatility Model...".yellow().bold());
    println!("  This may take a moment...\n");

    let model = StochasticVolatility::default();
    let config = MCMCConfig::new(args.samples)
        .with_warmup(args.samples / 2)
        .with_seed(args.seed.unwrap_or(42));

    let result = model.fit(&returns.values, &config);

    // Print model summary
    println!("{}", "═══════════════════════════════════════════════════════════════".magenta());
    println!("{}", "  Stochastic Volatility Model Results".magenta().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════".magenta());

    result.summary();

    // Interpret parameters
    println!("\n{}", "Parameter Interpretation:".yellow().bold());
    println!("{}", "─".repeat(50));

    let phi = result.phi_mean();
    let sigma_eta = result.sigma_eta_mean();
    let mu = result.mu_mean();

    // Half-life of volatility shocks
    let half_life = -1.0 / phi.ln();

    println!("  Persistence (phi = {:.4}):", phi);
    if phi > 0.95 {
        println!("    Volatility is highly persistent");
        println!("    Half-life of shocks: {:.1} periods", half_life);
    } else if phi > 0.8 {
        println!("    Volatility is moderately persistent");
        println!("    Half-life of shocks: {:.1} periods", half_life);
    } else {
        println!("    Volatility mean-reverts relatively quickly");
        println!("    Half-life of shocks: {:.1} periods", half_life);
    }

    println!("\n  Volatility of volatility (sigma_eta = {:.4}):", sigma_eta);
    if sigma_eta > 0.3 {
        println!("    Volatility varies substantially over time");
    } else if sigma_eta > 0.15 {
        println!("    Moderate variation in volatility");
    } else {
        println!("    Relatively stable volatility");
    }

    // Average volatility level
    let avg_vol = (mu / 2.0).exp();
    println!("\n  Average volatility level: {:.2}%", avg_vol * 100.0);

    // Show volatility time series
    println!("\n{}", "Volatility Time Series (sampled):".yellow().bold());

    let n_points = result.mean_volatility.len();
    let sample_step = (n_points / 10).max(1);

    let mut rows = Vec::new();
    for i in (0..n_points).step_by(sample_step) {
        if i < returns.timestamps.len() {
            let ts = returns.timestamps[i];
            let datetime = chrono::DateTime::from_timestamp_millis(ts)
                .map(|dt| dt.format("%m-%d %H:%M").to_string())
                .unwrap_or_else(|| format!("{}", i));

            let ret = returns.values[i];
            let vol = result.mean_volatility[i];
            let vol_low = result.volatility_ci_low[i];
            let vol_high = result.volatility_ci_high[i];

            // Determine regime
            let avg_vol = result.mean_volatility.iter().sum::<f64>() / result.mean_volatility.len() as f64;
            let regime = if vol > avg_vol * 1.5 {
                "HIGH".red().to_string()
            } else if vol < avg_vol * 0.7 {
                "LOW".green().to_string()
            } else {
                "NORMAL".to_string()
            };

            rows.push(VolatilityRow {
                period: datetime,
                return_pct: format!("{:+.2}%", ret * 100.0),
                volatility: format!("{:.2}%", vol * 100.0),
                ci: format!("[{:.2}%, {:.2}%]", vol_low * 100.0, vol_high * 100.0),
                regime,
            });
        }
    }

    let table = Table::new(rows).to_string();
    println!("{}", table);

    // Current volatility analysis
    println!("\n{}", "Current Market Conditions:".yellow().bold());
    println!("{}", "─".repeat(50));

    let current_vol = result.mean_volatility.last().unwrap_or(&0.0);
    let avg_vol: f64 = result.mean_volatility.iter().sum::<f64>() / result.mean_volatility.len() as f64;
    let vol_percentile = {
        let mut sorted = result.mean_volatility.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let position = sorted.iter().filter(|&&v| v <= *current_vol).count();
        position as f64 / sorted.len() as f64 * 100.0
    };

    println!("  Current volatility:     {:>10.2}%", current_vol * 100.0);
    println!("  Average volatility:     {:>10.2}%", avg_vol * 100.0);
    println!("  Volatility percentile:  {:>10.0}%", vol_percentile);

    // Risk implications
    println!("\n{}", "Risk Implications:".green().bold());
    println!("{}", "─".repeat(50));

    if vol_percentile > 80.0 {
        println!(
            "  {} Current volatility is in the top 20%!",
            "High Risk:".red().bold()
        );
        println!("  Consider reducing position sizes");
        println!("  Widen stop-loss orders");
    } else if vol_percentile < 20.0 {
        println!(
            "  {} Current volatility is in the bottom 20%",
            "Low Volatility:".green().bold()
        );
        println!("  Potentially quieter market conditions");
        println!("  May be opportunity for range-bound strategies");
    } else {
        println!(
            "  {} Current volatility is normal",
            "Normal:".yellow().bold()
        );
        println!("  Standard risk management applies");
    }

    // VaR adjustment based on current volatility
    let vol_ratio = current_vol / avg_vol;
    println!("\n  Volatility-adjusted position sizing:");
    println!("    If normal position = 100%:");
    println!("    Suggested position = {:.0}%", 100.0 / vol_ratio);

    // Expected range
    let expected_range = current_vol * 1.96; // 95% of moves
    let last_price = klines.last().map(|k| k.close).unwrap_or(0.0);

    println!("\n  Expected 95% price range (next period):");
    println!("    Current price: ${:.2}", last_price);
    println!("    Lower bound:   ${:.2} ({:+.2}%)",
        last_price * (1.0 - expected_range),
        -expected_range * 100.0
    );
    println!("    Upper bound:   ${:.2} ({:+.2}%)",
        last_price * (1.0 + expected_range),
        expected_range * 100.0
    );

    // Volatility forecast
    println!("\n{}", "Volatility Forecast:".yellow().bold());
    println!("{}", "─".repeat(50));

    // Simple AR(1) forecast
    let forecast_1 = avg_vol + phi * (current_vol - avg_vol);
    let forecast_5 = avg_vol + phi.powi(5) * (current_vol - avg_vol);
    let forecast_10 = avg_vol + phi.powi(10) * (current_vol - avg_vol);

    println!("  1-period ahead:  {:>10.2}%", forecast_1 * 100.0);
    println!("  5-period ahead:  {:>10.2}%", forecast_5 * 100.0);
    println!("  10-period ahead: {:>10.2}%", forecast_10 * 100.0);
    println!("  Long-run mean:   {:>10.2}%", avg_vol * 100.0);

    println!("\n{}", "═══════════════════════════════════════════════════════════════\n".magenta());

    Ok(())
}
