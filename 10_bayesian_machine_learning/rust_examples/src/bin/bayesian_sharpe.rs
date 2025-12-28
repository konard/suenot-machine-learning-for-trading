//! Bayesian Sharpe Ratio Example
//!
//! This example demonstrates how to compute and compare Sharpe ratios
//! using Bayesian inference with cryptocurrency data from Bybit.
//!
//! The Bayesian approach provides:
//! - Full posterior distribution of the Sharpe ratio
//! - Credible intervals (not just point estimates)
//! - Probability statements about performance

use anyhow::Result;
use bayesian_crypto::bayesian::inference::MCMCConfig;
use bayesian_crypto::bayesian::sharpe::BayesianSharpe;
use bayesian_crypto::data::{BybitClient, Returns, Symbol};
use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(name = "bayesian_sharpe")]
#[command(about = "Bayesian Sharpe ratio estimation and comparison for crypto")]
struct Args {
    /// First trading symbol
    #[arg(short = '1', long, default_value = "BTCUSDT")]
    symbol1: String,

    /// Second trading symbol (for comparison)
    #[arg(short = '2', long, default_value = "ETHUSDT")]
    symbol2: String,

    /// Time interval (1, 5, 15, 30, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "500")]
    limit: u32,

    /// Number of MCMC samples
    #[arg(long, default_value = "5000")]
    samples: usize,

    /// Random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "\n═══════════════════════════════════════════════════════════════".cyan());
    println!("{}", "  Bayesian Sharpe Ratio Analysis".cyan().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════\n".cyan());

    // Parse symbols
    let symbol1 = Symbol::from_str(&args.symbol1)
        .ok_or_else(|| anyhow::anyhow!("Unknown symbol: {}", args.symbol1))?;
    let symbol2 = Symbol::from_str(&args.symbol2)
        .ok_or_else(|| anyhow::anyhow!("Unknown symbol: {}", args.symbol2))?;

    // Fetch data
    let client = BybitClient::new();

    println!("Fetching data from Bybit...\n");

    let pb = ProgressBar::new(2);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap(),
    );

    pb.set_message(format!("Fetching {}", symbol1));
    let klines1 = client.get_klines(symbol1, &args.interval, args.limit).await?;
    pb.inc(1);

    pb.set_message(format!("Fetching {}", symbol2));
    let klines2 = client.get_klines(symbol2, &args.interval, args.limit).await?;
    pb.inc(1);
    pb.finish_with_message("Done!");

    println!();

    // Calculate returns
    let returns1 = Returns::from_klines(&klines1);
    let returns2 = Returns::from_klines(&klines2);

    // Print basic statistics
    println!("{}", "Return Statistics:".yellow().bold());
    println!("{}", "─".repeat(60));
    println!(
        "{:>20} {:>15} {:>15}",
        "", symbol1.to_string().green(), symbol2.to_string().blue()
    );
    println!(
        "{:>20} {:>15.4}% {:>15.4}%",
        "Mean (daily):",
        returns1.mean() * 100.0,
        returns2.mean() * 100.0
    );
    println!(
        "{:>20} {:>15.4}% {:>15.4}%",
        "Std Dev:",
        returns1.std() * 100.0,
        returns2.std() * 100.0
    );
    println!(
        "{:>20} {:>15.4} {:>15.4}",
        "Skewness:",
        returns1.skewness(),
        returns2.skewness()
    );
    println!(
        "{:>20} {:>15.4} {:>15.4}",
        "Kurtosis:",
        returns1.kurtosis(),
        returns2.kurtosis()
    );
    println!(
        "{:>20} {:>15.4}% {:>15.4}%",
        "Max Drawdown:",
        returns1.max_drawdown() * 100.0,
        returns2.max_drawdown() * 100.0
    );
    println!(
        "{:>20} {:>14.1}% {:>14.1}%",
        "Win Rate:",
        returns1.win_rate() * 100.0,
        returns2.win_rate() * 100.0
    );
    println!();

    // Frequentist Sharpe ratios (for comparison)
    // Assuming hourly data, annualization = 24 * 365 = 8760
    let annualization = match args.interval.as_str() {
        "1" => 525600.0,      // Minutes per year
        "5" => 105120.0,      // 5-min periods per year
        "15" => 35040.0,      // 15-min periods per year
        "30" => 17520.0,      // 30-min periods per year
        "60" => 8760.0,       // Hours per year
        "240" => 2190.0,      // 4-hour periods per year
        "D" => 365.0,         // Days per year
        _ => 8760.0,
    };

    let freq_sr1 = returns1.sharpe_ratio(annualization);
    let freq_sr2 = returns2.sharpe_ratio(annualization);

    println!("{}", "Frequentist Sharpe Ratios (point estimates):".yellow().bold());
    println!("  {}: {:.4}", symbol1, freq_sr1);
    println!("  {}: {:.4}", symbol2, freq_sr2);
    println!();

    // Bayesian Sharpe ratio estimation
    println!("{}", "Running Bayesian MCMC Estimation...".yellow().bold());
    println!(
        "  Samples: {}, Warmup: {}\n",
        args.samples,
        args.samples / 4
    );

    let estimator = BayesianSharpe::new(annualization);
    let config = MCMCConfig::new(args.samples)
        .with_warmup(args.samples / 4)
        .with_seed(args.seed.unwrap_or(42));

    // Estimate for symbol 1
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(format!("Estimating Sharpe ratio for {}...", symbol1));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let result1 = estimator.estimate(&returns1.values, &config);
    pb.finish_with_message(format!("{} estimation complete!", symbol1));

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    pb.set_message(format!("Estimating Sharpe ratio for {}...", symbol2));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let result2 = estimator.estimate(&returns2.values, &config);
    pb.finish_with_message(format!("{} estimation complete!", symbol2));

    println!();

    // Print Bayesian results
    println!("{}", "═══════════════════════════════════════════════════════════════".cyan());
    println!("{}", format!("  Bayesian Sharpe Ratio: {}", symbol1).cyan().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════".cyan());
    result1.summary();

    println!("{}", "═══════════════════════════════════════════════════════════════".cyan());
    println!("{}", format!("  Bayesian Sharpe Ratio: {}", symbol2).cyan().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════".cyan());
    result2.summary();

    // Comparison
    println!("{}", "═══════════════════════════════════════════════════════════════".magenta());
    println!("{}", "  Strategy Comparison".magenta().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════".magenta());

    // Calculate difference
    let n = result1.sharpe_samples.len().min(result2.sharpe_samples.len());
    let sharpe_diff: Vec<f64> = result1.sharpe_samples[..n]
        .iter()
        .zip(result2.sharpe_samples[..n].iter())
        .map(|(&s1, &s2)| s1 - s2)
        .collect();

    let prob_1_better = sharpe_diff.iter().filter(|&&d| d > 0.0).count() as f64 / n as f64;
    let mean_diff: f64 = sharpe_diff.iter().sum::<f64>() / n as f64;

    let mut sorted_diff = sharpe_diff.clone();
    sorted_diff.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let diff_ci_low = sorted_diff[(0.025 * n as f64) as usize];
    let diff_ci_high = sorted_diff[(0.975 * n as f64) as usize];

    println!("\nSharpe Ratio Difference ({} - {}):", symbol1, symbol2);
    println!("  Mean difference:     {:>10.4}", mean_diff);
    println!("  95% Credible Int:    [{:.4}, {:.4}]", diff_ci_low, diff_ci_high);
    println!();
    println!(
        "  P({} > {}):  {:>10.1}%",
        symbol1,
        symbol2,
        prob_1_better * 100.0
    );
    println!(
        "  P({} > {}):  {:>10.1}%",
        symbol2,
        symbol1,
        (1.0 - prob_1_better) * 100.0
    );

    // Interpretation
    println!("\n{}", "Interpretation:".green().bold());
    println!("{}", "─".repeat(50));

    if diff_ci_low > 0.0 {
        println!(
            "  {} appears to have a significantly higher Sharpe ratio.",
            symbol1.to_string().green()
        );
        println!(
            "  The 95% credible interval for the difference is entirely positive."
        );
    } else if diff_ci_high < 0.0 {
        println!(
            "  {} appears to have a significantly higher Sharpe ratio.",
            symbol2.to_string().blue()
        );
        println!(
            "  The 95% credible interval for the difference is entirely negative."
        );
    } else {
        println!("  The difference between the two assets is not conclusive.");
        println!("  The 95% credible interval includes zero.");
        println!("  More data may be needed to distinguish performance.");
    }

    // Practical significance
    if mean_diff.abs() < 0.1 {
        println!("\n  The difference is likely not practically significant (< 0.1 SR).");
    } else if mean_diff.abs() < 0.3 {
        println!("\n  The difference is moderate but worth considering.");
    } else {
        println!("\n  The difference is substantial and practically meaningful.");
    }

    // Risk thresholds
    println!("\n{}", "Risk-Adjusted Performance Thresholds:".yellow().bold());
    println!(
        "  P({} SR > 0):    {:>8.1}%",
        symbol1,
        result1.prob_positive() * 100.0
    );
    println!(
        "  P({} SR > 0):    {:>8.1}%",
        symbol2,
        result2.prob_positive() * 100.0
    );
    println!(
        "  P({} SR > 0.5):  {:>8.1}%",
        symbol1,
        result1.prob_exceeds(0.5) * 100.0
    );
    println!(
        "  P({} SR > 0.5):  {:>8.1}%",
        symbol2,
        result2.prob_exceeds(0.5) * 100.0
    );
    println!(
        "  P({} SR > 1.0):  {:>8.1}%",
        symbol1,
        result1.prob_exceeds(1.0) * 100.0
    );
    println!(
        "  P({} SR > 1.0):  {:>8.1}%",
        symbol2,
        result2.prob_exceeds(1.0) * 100.0
    );

    println!("\n{}", "═══════════════════════════════════════════════════════════════\n".cyan());

    Ok(())
}
