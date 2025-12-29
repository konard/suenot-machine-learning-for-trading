//! Conjugate Priors Example: Dynamically estimating crypto price movement probabilities
//!
//! This example demonstrates how to use Beta-Binomial conjugate priors to
//! estimate the probability of price increases in cryptocurrency markets.
//!
//! We update our beliefs about the "up probability" as we observe new candles.

use anyhow::Result;
use bayesian_crypto::bayesian::distributions::{Beta, Distribution};
use bayesian_crypto::data::{BybitClient, Kline, Returns, Symbol};
use clap::Parser;
use colored::Colorize;
use tabled::{Table, Tabled};

#[derive(Parser, Debug)]
#[command(name = "conjugate_priors")]
#[command(about = "Bayesian estimation of crypto price movement probabilities")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Time interval (1, 5, 15, 30, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value = "200")]
    limit: u32,

    /// Prior alpha (initial successes + 1)
    #[arg(long, default_value = "1.0")]
    prior_alpha: f64,

    /// Prior beta (initial failures + 1)
    #[arg(long, default_value = "1.0")]
    prior_beta: f64,
}

#[derive(Tabled)]
struct UpdateRow {
    #[tabled(rename = "Candle #")]
    candle: usize,
    #[tabled(rename = "Direction")]
    direction: String,
    #[tabled(rename = "Ups")]
    ups: u64,
    #[tabled(rename = "Downs")]
    downs: u64,
    #[tabled(rename = "Prior Mean")]
    prior_mean: String,
    #[tabled(rename = "Posterior Mean")]
    posterior_mean: String,
    #[tabled(rename = "95% CI")]
    ci: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "\n═══════════════════════════════════════════════════════════════".blue());
    println!("{}", "  Bayesian Conjugate Prior: Price Movement Probability".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════\n".blue());

    // Parse symbol
    let symbol = Symbol::from_str(&args.symbol)
        .ok_or_else(|| anyhow::anyhow!("Unknown symbol: {}", args.symbol))?;

    println!("Fetching {} {} candles for {}...", args.limit, args.interval, symbol);

    // Fetch data from Bybit
    let client = BybitClient::new();
    let klines = client.get_klines(symbol, &args.interval, args.limit).await?;

    println!("Fetched {} candles\n", klines.len());

    // Calculate returns
    let returns = Returns::from_klines(&klines);

    println!("{}", "Return Statistics:".yellow().bold());
    println!("  Mean return:   {:>10.4}%", returns.mean() * 100.0);
    println!("  Std deviation: {:>10.4}%", returns.std() * 100.0);
    println!("  Win rate:      {:>10.1}%", returns.win_rate() * 100.0);
    println!("  Max return:    {:>10.4}%", returns.max() * 100.0);
    println!("  Min return:    {:>10.4}%", returns.min() * 100.0);
    println!();

    // Start with prior
    let prior = Beta::new(args.prior_alpha, args.prior_beta);

    println!("{}", "Prior Distribution:".yellow().bold());
    println!(
        "  Beta({:.1}, {:.1}) - Prior mean: {:.4}",
        prior.alpha,
        prior.beta,
        prior.mean()
    );
    println!();

    // Sequential Bayesian updating
    println!("{}", "Sequential Bayesian Updates:".yellow().bold());
    println!("{}", "(Showing updates at regular intervals)\n".dimmed());

    let mut current = prior;
    let mut ups = 0u64;
    let mut downs = 0u64;
    let mut rows = Vec::new();

    // Show updates at certain milestones
    let milestones: Vec<usize> = vec![1, 5, 10, 20, 50, 100, returns.len()];

    for (i, &ret) in returns.values.iter().enumerate() {
        let is_up = ret > 0.0;

        if is_up {
            ups += 1;
        } else {
            downs += 1;
        }

        // Update posterior
        let prior_mean = current.mean();
        current = current.update(if is_up { 1 } else { 0 }, if is_up { 0 } else { 1 });
        let ci = current.credible_interval(0.95);

        if milestones.contains(&(i + 1)) {
            rows.push(UpdateRow {
                candle: i + 1,
                direction: if is_up {
                    "UP".green().to_string()
                } else {
                    "DOWN".red().to_string()
                },
                ups,
                downs,
                prior_mean: format!("{:.4}", prior_mean),
                posterior_mean: format!("{:.4}", current.mean()),
                ci: format!("[{:.3}, {:.3}]", ci.0, ci.1),
            });
        }
    }

    let table = Table::new(rows).to_string();
    println!("{}", table);

    // Final posterior analysis
    println!("\n{}", "Final Posterior Analysis:".yellow().bold());
    println!("{}", "─".repeat(50));

    let final_ci = current.credible_interval(0.95);
    let final_ci_99 = current.credible_interval(0.99);

    println!(
        "  Posterior: Beta({:.1}, {:.1})",
        current.alpha, current.beta
    );
    println!("  Mean probability of UP: {:.4}", current.mean());
    println!("  Mode (most likely):     {:.4}", current.mode());
    println!("  Standard deviation:     {:.4}", current.std());
    println!();
    println!("  95% Credible Interval:  [{:.4}, {:.4}]", final_ci.0, final_ci.1);
    println!("  99% Credible Interval:  [{:.4}, {:.4}]", final_ci_99.0, final_ci_99.1);

    // Probability statements
    println!("\n{}", "Probability Statements:".yellow().bold());

    let prob_above_50 = 1.0 - Beta::new(current.alpha, current.beta).quantile(0.5);
    let prob_above_55 = 1.0 - Beta::new(current.alpha, current.beta).quantile(0.55);
    let prob_below_45 = Beta::new(current.alpha, current.beta).quantile(0.45);

    println!("  P(up_prob > 0.50) = {:.1}%", prob_above_50 * 100.0);
    println!("  P(up_prob > 0.55) = {:.1}%", prob_above_55 * 100.0);
    println!("  P(up_prob < 0.45) = {:.1}%", prob_below_45 * 100.0);

    // Comparison with frequentist estimate
    println!("\n{}", "Comparison with Frequentist Estimate:".yellow().bold());
    let freq_estimate = ups as f64 / (ups + downs) as f64;
    let freq_std = (freq_estimate * (1.0 - freq_estimate) / (ups + downs) as f64).sqrt();

    println!("  Frequentist point estimate: {:.4}", freq_estimate);
    println!(
        "  Frequentist 95% CI:         [{:.4}, {:.4}]",
        freq_estimate - 1.96 * freq_std,
        freq_estimate + 1.96 * freq_std
    );
    println!("  Bayesian posterior mean:    {:.4}", current.mean());
    println!(
        "  Bayesian 95% CI:            [{:.4}, {:.4}]",
        final_ci.0, final_ci.1
    );

    // Interpretation
    println!("\n{}", "Interpretation:".green().bold());
    if current.mean() > 0.52 {
        println!(
            "  The data suggests a slight bullish bias ({:.1}% probability of up moves)",
            current.mean() * 100.0
        );
    } else if current.mean() < 0.48 {
        println!(
            "  The data suggests a slight bearish bias ({:.1}% probability of up moves)",
            current.mean() * 100.0
        );
    } else {
        println!(
            "  The data is consistent with a roughly fair coin ({:.1}% up probability)",
            current.mean() * 100.0
        );
    }

    let ci_width = final_ci.1 - final_ci.0;
    if ci_width > 0.1 {
        println!(
            "  Uncertainty is still relatively high (CI width: {:.1}%)",
            ci_width * 100.0
        );
        println!("  More data would help narrow down the estimate.");
    } else {
        println!(
            "  The estimate is fairly precise (CI width: {:.1}%)",
            ci_width * 100.0
        );
    }

    println!("\n{}", "═══════════════════════════════════════════════════════════════\n".blue());

    Ok(())
}
