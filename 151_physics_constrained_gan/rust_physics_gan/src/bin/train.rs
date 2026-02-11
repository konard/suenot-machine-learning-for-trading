//! Simplified GAN training demonstration
//!
//! This binary demonstrates the training loop concept for a physics-constrained
//! GAN using a simplified linear generator. For production GPU-accelerated
//! training, use the Python implementation with PyTorch.
//!
//! Usage:
//!   cargo run --bin train -- --n-paths 500 --seq-len 256 --epochs 100

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use physics_constrained_gan::{
    bybit::BybitClient,
    compute_log_returns,
    create_windows,
    gan::SimpleGenerator,
    heston::{self, HestonParams},
    physics::{self, PhysicsConfig},
    statistics,
};

#[derive(Parser)]
#[command(name = "train", about = "Train (simplified) Physics-Constrained GAN")]
struct Args {
    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Sequence length for generated paths
    #[arg(short, long, default_value = "256")]
    seq_len: usize,

    /// Latent dimension
    #[arg(short = 'z', long, default_value = "64")]
    latent_dim: usize,

    /// Number of synthetic training paths
    #[arg(short, long, default_value = "500")]
    n_paths: usize,

    /// Use Bybit data instead of Heston synthetic
    #[arg(long)]
    use_bybit: bool,

    /// Bybit symbol (if --use-bybit)
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,

    /// Bybit interval (if --use-bybit)
    #[arg(long, default_value = "60")]
    interval: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "═══════════════════════════════════════════════════════".blue().bold());
    println!("{}", " Physics-Constrained GAN - Training (Simplified)".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════".blue().bold());

    // ---- Load or generate training data ----
    let training_windows: Vec<Vec<f64>>;

    if args.use_bybit {
        println!("\nFetching training data from Bybit ({})...", args.symbol.green());
        let client = BybitClient::new();
        let candles = client.fetch_extended(&args.symbol, &args.interval, 5000)?;
        let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let returns = compute_log_returns(&prices);
        training_windows = create_windows(&returns, args.seq_len, args.seq_len / 4);
        println!("Created {} training windows from {} candles",
                 training_windows.len(), candles.len());
    } else {
        println!("\nGenerating Heston model training data...");
        let params = HestonParams {
            mu: 0.05,
            kappa: 2.0,
            theta: 0.04,
            xi: 0.3,
            rho: -0.7,
            v0: 0.04,
            dt: 1.0 / 252.0,
        };
        training_windows = heston::generate_multiple_paths(&params, args.n_paths, args.seq_len);
        println!("Generated {} Heston paths of length {}", args.n_paths, args.seq_len);
    }

    if training_windows.is_empty() {
        println!("{}", "No training data available!".red());
        return Ok(());
    }

    // Compute target statistics from real data
    let all_returns: Vec<f64> = training_windows.iter().flatten().cloned().collect();
    let mut physics_config = PhysicsConfig::default();
    physics_config.set_targets_from_data(&all_returns);

    let real_stats = statistics::compute_statistics(&all_returns);
    println!("\n{}", "Real Data Statistics:".yellow().bold());
    println!("{}", real_stats);

    // ---- Initialize generator ----
    let hidden_dim = 128;
    let generator = SimpleGenerator::new(args.latent_dim, hidden_dim, args.seq_len);

    println!("\n{}", "Generator Architecture:".yellow().bold());
    println!("  Latent dim:  {}", args.latent_dim);
    println!("  Hidden dim:  {}", hidden_dim);
    println!("  Output dim:  {} (sequence length)", args.seq_len);

    // ---- Training loop ----
    println!("\n{}", "Training...".yellow().bold());
    println!("(Note: This is a simplified demonstration. For real training,");
    println!(" use the Python implementation with PyTorch GPU acceleration.)\n");

    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut best_physics_loss = f64::INFINITY;

    for epoch in 0..args.epochs {
        // Generate a batch of synthetic paths
        let batch_size = 50;
        let synthetic_batch = generator.generate_batch(batch_size);

        // Evaluate physics constraints on generated data
        let gen_all: Vec<f64> = synthetic_batch.iter().flatten().cloned().collect();
        let report = physics::evaluate_physics(&gen_all, &physics_config);

        if report.total_physics_loss < best_physics_loss {
            best_physics_loss = report.total_physics_loss;
        }

        // Log progress
        if epoch % 10 == 0 || epoch == args.epochs - 1 {
            pb.println(format!(
                "Epoch {:4} | Physics: {:.4} (best: {:.4}) | Kurt: {:.2} | Lev: {:.4} | ACF|r|(1): {:.4}",
                epoch,
                report.total_physics_loss,
                best_physics_loss,
                report.statistics.kurtosis,
                report.statistics.leverage_corr,
                report.statistics.acf_abs_lag1,
            ));
        }

        pb.inc(1);
    }

    pb.finish_with_message("Training complete");

    // ---- Final evaluation ----
    println!("\n{}", "Final Evaluation:".yellow().bold());

    let final_paths = generator.generate_batch(200);
    let final_returns: Vec<f64> = final_paths.iter().flatten().cloned().collect();
    let final_report = physics::evaluate_physics(&final_returns, &physics_config);

    println!("\n{}", "Generated Data Physics Report:".green().bold());
    println!("{}", final_report);

    // Compare with real data
    println!("\n{}", "Comparison:".yellow().bold());
    println!("{:<25} {:>12} {:>12}",
             "Metric", "Real", "Generated");
    println!("{}", "-".repeat(50));
    println!("{:<25} {:>12.6} {:>12.6}",
             "Mean", real_stats.mean, final_report.statistics.mean);
    println!("{:<25} {:>12.6} {:>12.6}",
             "Std", real_stats.std, final_report.statistics.std);
    println!("{:<25} {:>12.4} {:>12.4}",
             "Skewness", real_stats.skewness, final_report.statistics.skewness);
    println!("{:<25} {:>12.4} {:>12.4}",
             "Excess Kurtosis", real_stats.kurtosis, final_report.statistics.kurtosis);
    println!("{:<25} {:>12.4} {:>12.4}",
             "ACF(|r|, lag=1)", real_stats.acf_abs_lag1, final_report.statistics.acf_abs_lag1);
    println!("{:<25} {:>12.4} {:>12.4}",
             "ACF(r^2, lag=1)", real_stats.acf_sq_lag1, final_report.statistics.acf_sq_lag1);
    println!("{:<25} {:>12.4} {:>12.4}",
             "Leverage Corr", real_stats.leverage_corr, final_report.statistics.leverage_corr);

    println!("\n{}", "Done!".green().bold());
    println!("For production-quality training, use:");
    println!("  python train.py --symbol BTCUSDT --epochs 500");

    Ok(())
}
