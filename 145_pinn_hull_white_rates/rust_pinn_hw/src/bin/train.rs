//! Train the Hull-White PINN model.
//!
//! Usage:
//!     cargo run --bin train -- --epochs 100 --lr 0.001 --collocation 500

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use pinn_hull_white::{
    HullWhiteParams,
    model::{HullWhitePINN, PINNConfig},
    analytical::HullWhiteAnalytical,
    data::sample_treasury_curve,
};

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train Hull-White PINN model")]
struct Args {
    /// Mean reversion speed
    #[arg(short, long, default_value = "0.1")]
    mean_reversion: f64,

    /// Short rate volatility
    #[arg(short, long, default_value = "0.01")]
    sigma: f64,

    /// Current short rate
    #[arg(short, long, default_value = "0.03")]
    rate: f64,

    /// Maximum maturity (years)
    #[arg(short = 'T', long, default_value = "10.0")]
    t_max: f64,

    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    lr: f64,

    /// Number of collocation points per epoch
    #[arg(short, long, default_value = "200")]
    collocation: usize,

    /// Hidden layer sizes (comma-separated)
    #[arg(long, default_value = "32,32,16")]
    hidden: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("{}", "=" .repeat(60).blue());
    println!("{}", "  Hull-White PINN Training".bold().blue());
    println!("{}", "=" .repeat(60).blue());

    // Parse hidden layers
    let hidden_layers: Vec<usize> = args.hidden
        .split(',')
        .map(|s| s.trim().parse::<usize>().unwrap_or(32))
        .collect();

    let params = HullWhiteParams {
        a: args.mean_reversion,
        sigma: args.sigma,
        r0: args.rate,
    };

    println!("\n{}", "Model Parameters:".bold());
    println!("  Mean reversion (a):  {:.4}", params.a);
    println!("  Volatility (sigma):  {:.6}", params.sigma);
    println!("  Current rate (r0):   {:.4}", params.r0);
    println!("  Max maturity (T):    {:.1}", args.t_max);
    println!("  Hidden layers:       {:?}", hidden_layers);
    println!("  Epochs:              {}", args.epochs);
    println!("  Learning rate:       {:.1e}", args.lr);
    println!("  Collocation points:  {}", args.collocation);

    // Create PINN
    let config = PINNConfig {
        hidden_layers,
        activation: "tanh".to_string(),
        learning_rate: args.lr,
    };

    let mut pinn = HullWhitePINN::new(params.a, params.sigma, config);

    let total_params: usize = pinn.layers.iter().map(|l| {
        l.weights.len() + l.biases.len()
    }).sum();
    println!("  Total parameters:    {}", total_params);

    // Create analytical model for validation
    let hw = HullWhiteAnalytical::new_flat(params.a, params.sigma, params.r0);

    // Training
    println!("\n{}", "Training...".bold().yellow());

    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) loss={msg}"
        )?
        .progress_chars("=> "),
    );

    let loss_history = pinn.train(
        &params,
        args.t_max,
        args.epochs,
        args.collocation,
        Some(&|epoch, loss| {
            pb.set_position(epoch as u64 + 1);
            pb.set_message(format!("{:.2e}", loss));
        }),
    );

    pb.finish_with_message("done");

    // Validation
    println!("\n{}", "Validation: PINN vs Analytical".bold().green());
    println!("{:-<60}", "");
    println!(
        "{:>10} {:>12} {:>12} {:>12}",
        "Maturity", "PINN", "Analytical", "Error"
    );
    println!("{:-<60}", "");

    let test_maturities = vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
    for &t in &test_maturities {
        let p_pinn = pinn.predict(params.r0, 0.0, t);
        let p_anal = hw.bond_price(params.r0, 0.0, t);
        let error = (p_pinn - p_anal).abs();

        let error_str = if error < 1e-4 {
            format!("{:.2e}", error).green()
        } else if error < 1e-2 {
            format!("{:.2e}", error).yellow()
        } else {
            format!("{:.2e}", error).red()
        };

        println!(
            "{:>10.1} {:>12.6} {:>12.6} {:>12}",
            t, p_pinn, p_anal, error_str
        );
    }

    // Yield curve comparison
    println!("\n{}", "Yield Curve:".bold().green());
    let pinn_yields = pinn.yield_curve(params.r0, 0.0, &test_maturities);
    let anal_yields = hw.yield_curve(params.r0, 0.0, &test_maturities);

    for (i, &t) in test_maturities.iter().enumerate() {
        println!(
            "  T={:.1}  PINN={:.4}%  Analytical={:.4}%",
            t,
            pinn_yields[i] * 100.0,
            anal_yields[i] * 100.0,
        );
    }

    // Greeks
    println!("\n{}", "Greeks at T=5:".bold().green());
    let greeks = pinn.greeks(params.r0, 0.0, 5.0);
    println!("  {}", greeks);

    // Training summary
    println!("\n{}", "Training Summary:".bold());
    if let Some(first_loss) = loss_history.first() {
        println!("  Initial loss: {:.2e}", first_loss);
    }
    if let Some(final_loss) = loss_history.last() {
        println!("  Final loss:   {:.2e}", final_loss);
    }

    println!("\n{}", "Training complete.".bold().green());

    Ok(())
}
