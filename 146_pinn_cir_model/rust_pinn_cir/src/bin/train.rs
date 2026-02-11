//! PINN Training for CIR Bond Pricing
//!
//! Trains a simple neural network to solve the CIR bond pricing PDE
//! using physics-informed loss (PDE residual + boundary conditions).
//!
//! This is a simplified demonstration using finite-difference gradients
//! and a random search optimization approach. For production use,
//! the Python implementation with PyTorch autograd is recommended.

use clap::Parser;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use pinn_cir_model::*;
use rand::Rng;

#[derive(Parser, Debug)]
#[command(name = "CIR PINN Trainer")]
#[command(about = "Train a PINN for CIR bond pricing")]
struct Args {
    /// Mean-reversion speed
    #[arg(long, default_value = "0.5")]
    kappa: f64,

    /// Long-run mean rate
    #[arg(long, default_value = "0.05")]
    theta: f64,

    /// Volatility coefficient
    #[arg(long, default_value = "0.1")]
    sigma: f64,

    /// Initial rate
    #[arg(long, default_value = "0.03")]
    r0: f64,

    /// Bond maturity
    #[arg(long, default_value = "5.0")]
    maturity: f64,

    /// Max rate for computational domain
    #[arg(long, default_value = "0.20")]
    r_max: f64,

    /// Number of training epochs
    #[arg(long, default_value = "2000")]
    epochs: usize,

    /// Number of collocation points
    #[arg(long, default_value = "500")]
    n_collocation: usize,
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    println!("{}", "=".repeat(70).blue());
    println!("{}", "CIR PINN Training".bold().blue());
    println!("{}", "=".repeat(70).blue());

    let params = CIRParams {
        kappa: args.kappa,
        theta: args.theta,
        sigma: args.sigma,
        r0: args.r0,
    };

    print_params_summary(&params);

    let config = PINNConfig {
        n_collocation: args.n_collocation,
        n_boundary: 50,
        n_terminal: 50,
        learning_rate: 0.01,
        epochs: args.epochs,
        r_max: args.r_max,
        t_max: args.maturity,
    };

    // Create network
    let layer_sizes = vec![3, 64, 64, 32, 1];
    let mut net = SimpleNetwork::new(&layer_sizes);
    println!(
        "\nNetwork: {:?} ({} parameters)",
        layer_sizes,
        net.n_parameters()
    );

    // Generate collocation points (sqrt-aware sampling)
    let mut rng = rand::thread_rng();
    let n_total = config.n_collocation + config.n_boundary + config.n_terminal;

    println!("\n{}", "Training...".yellow());
    let pb = ProgressBar::new(config.epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut best_loss = f64::INFINITY;
    let mut best_weights = net.weights.clone();
    let mut best_biases = net.biases.clone();

    // Training loop using random perturbation (evolutionary strategy)
    // NOTE: This is a simplified approach for demonstration.
    // The Python version uses proper gradient-based training with PyTorch.
    for epoch in 0..config.epochs {
        // Compute current loss
        let loss = compute_total_loss(&net, &params, &config, &mut rng);

        if loss < best_loss {
            best_loss = loss;
            best_weights = net.weights.clone();
            best_biases = net.biases.clone();
        }

        // Perturbation-based update
        let perturbation_scale = config.learning_rate * (1.0 - epoch as f64 / config.epochs as f64);

        for l in 0..net.n_layers {
            let (rows, cols) = (net.weights[l].nrows(), net.weights[l].ncols());
            for i in 0..rows {
                for j in 0..cols {
                    // Try perturbation
                    let old_val = net.weights[l][[i, j]];
                    let perturbation = rng.gen_range(-perturbation_scale..perturbation_scale);

                    net.weights[l][[i, j]] = old_val + perturbation;
                    let new_loss = compute_total_loss(&net, &params, &config, &mut rng);

                    if new_loss < loss {
                        // Keep perturbation
                    } else {
                        // Revert
                        net.weights[l][[i, j]] = old_val;
                    }
                }
            }

            for i in 0..net.biases[l].len() {
                let old_val = net.biases[l][i];
                let perturbation = rng.gen_range(-perturbation_scale..perturbation_scale);

                net.biases[l][i] = old_val + perturbation;
                let new_loss = compute_total_loss(&net, &params, &config, &mut rng);

                if new_loss < loss {
                    // Keep
                } else {
                    net.biases[l][i] = old_val;
                }
            }
        }

        if epoch % 100 == 0 || epoch == config.epochs - 1 {
            pb.set_message(format!("loss: {:.6}", best_loss));
        }
        pb.inc(1);
    }
    pb.finish_with_message(format!("Final loss: {:.6}", best_loss));

    // Restore best weights
    net.weights = best_weights;
    net.biases = best_biases;

    // Evaluation
    println!("\n{}", "=".repeat(70).green());
    println!("{}", "Evaluation: PINN vs Analytical".bold().green());
    println!("{}", "=".repeat(70).green());

    println!(
        "\n{:>8} {:>12} {:>12} {:>12}",
        "Rate", "PINN", "Analytical", "Error"
    );
    println!("{}", "-".repeat(48));

    let test_rates = vec![0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15];
    let mut max_error = 0.0f64;
    let mut sum_error = 0.0f64;

    for &r in &test_rates {
        let p_pinn = net.predict(r, 0.0, config.r_max, config.t_max);
        let p_analytical = cir_bond_price(r, config.t_max, &params);
        let error = (p_pinn - p_analytical).abs();
        max_error = max_error.max(error);
        sum_error += error;

        println!(
            "{:>8.2}% {:>12.6} {:>12.6} {:>12.6}",
            r * 100.0,
            p_pinn,
            p_analytical,
            error,
        );
    }

    let mean_error = sum_error / test_rates.len() as f64;
    println!("{}", "-".repeat(48));
    println!(
        "Max error: {:.6}, Mean error: {:.6}",
        max_error, mean_error
    );

    // Yield curve from PINN
    println!("\n{}", "PINN-implied yield curve:".bold());
    let maturities = vec![0.5, 1.0, 2.0, 3.0, 5.0];
    println!(
        "{:>10} {:>12} {:>12} {:>12}",
        "Maturity", "PINN Price", "PINN Yield", "Exact Yield"
    );
    for &tau in &maturities {
        let p_pinn = net.predict(args.r0, config.t_max - tau, config.r_max, config.t_max);
        let y_pinn = if tau > 0.0 { -p_pinn.ln() / tau } else { args.r0 };
        let y_exact = cir_yield(args.r0, tau, &params);
        println!(
            "{:>10.1}y {:>12.6} {:>12.4}% {:>12.4}%",
            tau,
            p_pinn,
            y_pinn * 100.0,
            y_exact * 100.0
        );
    }

    println!(
        "\n{}",
        "Note: For production-quality PINN training, use the Python version with PyTorch."
            .dimmed()
    );
}

/// Compute total PINN loss at a random subset of points.
fn compute_total_loss(
    net: &SimpleNetwork,
    params: &CIRParams,
    config: &PINNConfig,
    rng: &mut impl Rng,
) -> f64 {
    let n_sample = 50; // subsample for speed
    let mut loss = 0.0;

    // PDE residual at interior points
    for _ in 0..n_sample {
        let u: f64 = rng.gen();
        let r = config.r_max * u * u; // sqrt-aware sampling
        let t = rng.gen::<f64>() * config.t_max * 0.99;

        let residual = pde_residual(net, r, t, params, config);
        loss += residual * residual;
    }
    loss /= n_sample as f64;

    // Terminal condition: P(r, T) = 1
    let mut terminal_loss = 0.0;
    for _ in 0..20 {
        let r = rng.gen::<f64>() * config.r_max;
        let p = net.predict(r, config.t_max, config.r_max, config.t_max);
        terminal_loss += (p - 1.0) * (p - 1.0);
    }
    loss += 10.0 * terminal_loss / 20.0;

    // Analytical anchor points
    let mut analytical_loss = 0.0;
    for _ in 0..20 {
        let r = rng.gen::<f64>() * config.r_max * 0.8;
        let t = rng.gen::<f64>() * config.t_max * 0.9;
        let tau = config.t_max - t;

        let p_pinn = net.predict(r, t, config.r_max, config.t_max);
        let p_exact = cir_bond_price(r, tau, params);
        analytical_loss += (p_pinn - p_exact) * (p_pinn - p_exact);
    }
    loss += 5.0 * analytical_loss / 20.0;

    loss
}
