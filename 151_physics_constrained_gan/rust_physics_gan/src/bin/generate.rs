//! Generate synthetic financial paths and analyze their properties
//!
//! Generates paths using the Heston model (or simple GAN) and evaluates
//! how well they satisfy financial physics constraints.
//!
//! Usage:
//!   cargo run --bin generate -- --n-paths 1000 --seq-len 256 --regime bull
//!   cargo run --bin generate -- --n-paths 500 --model heston --output synthetic.csv

use anyhow::Result;
use clap::Parser;
use colored::Colorize;

use physics_constrained_gan::{
    compute_log_returns,
    detect_regime,
    gan::SimpleGenerator,
    heston::{self, HestonParams},
    physics::{self, PhysicsConfig},
    returns_to_prices,
    statistics,
    trading,
    MarketRegime,
    VolatilityLevel,
};

#[derive(Parser)]
#[command(name = "generate", about = "Generate and analyze synthetic financial paths")]
struct Args {
    /// Number of paths to generate
    #[arg(short, long, default_value = "500")]
    n_paths: usize,

    /// Sequence length per path
    #[arg(short, long, default_value = "256")]
    seq_len: usize,

    /// Generation model: "heston" or "gan"
    #[arg(short, long, default_value = "heston")]
    model: String,

    /// Target regime: bull, bear, sideways, crisis
    #[arg(short, long, default_value = "bull")]
    regime: String,

    /// Output CSV path for generated data
    #[arg(short, long)]
    output: Option<String>,

    /// Run backtest on generated paths
    #[arg(long)]
    backtest: bool,

    /// Initial price for conversion to prices
    #[arg(long, default_value = "50000.0")]
    initial_price: f64,
}

fn get_heston_params_for_regime(regime: &str) -> HestonParams {
    match regime {
        "bull" => HestonParams {
            mu: 0.15,
            kappa: 2.0,
            theta: 0.03,
            xi: 0.2,
            rho: -0.5,
            v0: 0.03,
            dt: 1.0 / 252.0,
        },
        "bear" => HestonParams {
            mu: -0.10,
            kappa: 1.5,
            theta: 0.06,
            xi: 0.4,
            rho: -0.8,
            v0: 0.06,
            dt: 1.0 / 252.0,
        },
        "sideways" => HestonParams {
            mu: 0.02,
            kappa: 3.0,
            theta: 0.02,
            xi: 0.15,
            rho: -0.3,
            v0: 0.02,
            dt: 1.0 / 252.0,
        },
        "crisis" => HestonParams {
            mu: -0.30,
            kappa: 1.0,
            theta: 0.10,
            xi: 0.6,
            rho: -0.9,
            v0: 0.12,
            dt: 1.0 / 252.0,
        },
        _ => HestonParams::default(),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "═══════════════════════════════════════════════════════".blue().bold());
    println!("{}", " Physics-Constrained GAN - Path Generator & Analyzer".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════".blue().bold());

    println!("\nGenerating {} paths of length {} using {} model",
             args.n_paths, args.seq_len, args.model.green());
    println!("Target regime: {}", args.regime.yellow());

    // ---- Generate paths ----
    let paths: Vec<Vec<f64>> = match args.model.as_str() {
        "heston" => {
            let params = get_heston_params_for_regime(&args.regime);
            println!("Heston params: mu={:.3}, kappa={:.3}, theta={:.4}, xi={:.3}, rho={:.3}",
                     params.mu, params.kappa, params.theta, params.xi, params.rho);
            heston::generate_multiple_paths(&params, args.n_paths, args.seq_len)
        }
        "gan" => {
            let gen = SimpleGenerator::new(64, 128, args.seq_len);
            gen.generate_batch(args.n_paths)
        }
        _ => {
            println!("{}", "Unknown model type. Using Heston.".yellow());
            let params = HestonParams::default();
            heston::generate_multiple_paths(&params, args.n_paths, args.seq_len)
        }
    };

    println!("Generated {} paths", paths.len());

    // ---- Analyze paths ----
    let all_returns: Vec<f64> = paths.iter().flatten().cloned().collect();
    let stats = statistics::compute_statistics(&all_returns);

    println!("\n{}", "Generated Data Statistics:".yellow().bold());
    println!("{}", stats);

    // Physics constraint evaluation
    let config = PhysicsConfig::default();
    let report = physics::evaluate_physics(&all_returns, &config);

    println!("\n{}", "Physics Constraints:".yellow().bold());
    println!("  Martingale loss:      {:.6} {}",
             report.martingale_loss,
             if report.martingale_loss < 0.001 { "(GOOD)".green() } else { "(NEEDS WORK)".yellow() });
    println!("  Vol clustering loss:  {:.6} {}",
             report.volatility_clustering_loss,
             if report.volatility_clustering_loss < 0.01 { "(GOOD)".green() } else { "(NEEDS WORK)".yellow() });
    println!("  Kurtosis loss:        {:.6} {}",
             report.kurtosis_loss,
             if report.kurtosis_loss < 10.0 { "(GOOD)".green() } else { "(NEEDS WORK)".yellow() });
    println!("  Leverage effect loss: {:.6} {}",
             report.leverage_effect_loss,
             if report.leverage_effect_loss < 0.05 { "(GOOD)".green() } else { "(NEEDS WORK)".yellow() });
    println!("  Autocorrelation loss: {:.6} {}",
             report.autocorrelation_loss,
             if report.autocorrelation_loss < 0.01 { "(GOOD)".green() } else { "(NEEDS WORK)".yellow() });
    println!("  Total physics loss:   {:.6}", report.total_physics_loss);

    // ---- Distribution analysis ----
    println!("\n{}", "Distribution Analysis:".yellow().bold());

    let percentiles = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0];
    let mut sorted = all_returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for p in &percentiles {
        let idx = ((p / 100.0) * sorted.len() as f64) as usize;
        let idx = idx.min(sorted.len() - 1);
        println!("  {:>5.1}th percentile: {:>10.6}", p, sorted[idx]);
    }

    // ---- Regime classification of generated paths ----
    println!("\n{}", "Regime Classification of Generated Paths:".yellow().bold());
    let mut regime_counts = [0usize; 4];
    for path in &paths {
        let (regime, _) = detect_regime(path);
        regime_counts[regime as usize] += 1;
    }
    println!("  Bull:     {:>5} ({:.1}%)",
             regime_counts[0], 100.0 * regime_counts[0] as f64 / paths.len() as f64);
    println!("  Bear:     {:>5} ({:.1}%)",
             regime_counts[1], 100.0 * regime_counts[1] as f64 / paths.len() as f64);
    println!("  Sideways: {:>5} ({:.1}%)",
             regime_counts[2], 100.0 * regime_counts[2] as f64 / paths.len() as f64);
    println!("  Crisis:   {:>5} ({:.1}%)",
             regime_counts[3], 100.0 * regime_counts[3] as f64 / paths.len() as f64);

    // ---- VaR and CVaR ----
    println!("\n{}", "Risk Metrics:".yellow().bold());
    let cumulative_returns: Vec<f64> = paths.iter()
        .map(|path| path.iter().sum::<f64>())
        .collect();
    let mut cum_sorted = cumulative_returns.clone();
    cum_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let var_95_idx = (0.05 * cum_sorted.len() as f64) as usize;
    let var_99_idx = (0.01 * cum_sorted.len() as f64) as usize;
    let var_95 = cum_sorted[var_95_idx];
    let var_99 = cum_sorted[var_99_idx];

    let cvar_95: f64 = cum_sorted[..var_95_idx + 1].iter().sum::<f64>()
        / (var_95_idx + 1) as f64;
    let cvar_99: f64 = cum_sorted[..var_99_idx + 1].iter().sum::<f64>()
        / (var_99_idx + 1).max(1) as f64;

    println!("  VaR (95%):   {:.4} ({:.2}%)", var_95, var_95 * 100.0);
    println!("  VaR (99%):   {:.4} ({:.2}%)", var_99, var_99 * 100.0);
    println!("  CVaR (95%):  {:.4} ({:.2}%)", cvar_95, cvar_95 * 100.0);
    println!("  CVaR (99%):  {:.4} ({:.2}%)", cvar_99, cvar_99 * 100.0);

    // ---- Save to CSV ----
    if let Some(output_path) = &args.output {
        let mut wtr = csv::Writer::from_path(output_path)?;

        // Header
        let header: Vec<String> = (0..args.seq_len).map(|i| format!("t_{}", i)).collect();
        wtr.write_record(&header)?;

        // Paths
        for path in &paths {
            let row: Vec<String> = path.iter().map(|r| format!("{:.8}", r)).collect();
            wtr.write_record(&row)?;
        }
        wtr.flush()?;
        println!("\nPaths saved to {}", output_path.green());
    }

    // ---- Optional backtest ----
    if args.backtest {
        println!("\n{}", "Running Backtest on Generated Scenarios:".yellow().bold());

        // Simple momentum strategy on first path
        let test_path = &paths[0];
        let signals: Vec<f64> = test_path.windows(10)
            .map(|w| {
                let avg: f64 = w.iter().sum::<f64>() / w.len() as f64;
                if avg > 0.0 { 1.0 } else if avg < 0.0 { -1.0 } else { 0.0 }
            })
            .collect();

        let result = trading::run_backtest(
            &test_path[9..],  // Align with signal
            &signals,
            0.001,
            100000.0,
        );

        println!("{}", result);

        // Monte Carlo aggregate
        println!("\n{}", "Monte Carlo Aggregate (all paths):".yellow().bold());
        let final_returns: Vec<f64> = paths.iter()
            .map(|p| p.iter().sum::<f64>())
            .collect();
        println!("  Mean cumulative return:  {:.4} ({:.2}%)",
                 statistics::mean(&final_returns),
                 statistics::mean(&final_returns) * 100.0);
        println!("  Std of cum. return:      {:.4}", statistics::std_dev(&final_returns));
        println!("  P(profit):               {:.2}%",
                 100.0 * final_returns.iter().filter(|&&r| r > 0.0).count() as f64
                 / final_returns.len() as f64);

        // Signal from Monte Carlo paths
        let signal = trading::generate_signal_from_paths(&paths, 0.3);
        println!("\n  Monte Carlo Signal: {} (confidence: {:.2}, size: {:.3})",
                 signal.direction,
                 signal.confidence,
                 signal.position_size);
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
