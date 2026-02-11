//! Train a Lagrangian Neural Network on market data.
//!
//! Usage:
//!   cargo run --release --bin train -- --hidden-dim 32 --epochs 200 --lr 0.001

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use lagrangian_nn_trading::nn::LagrangianNN;
use lagrangian_nn_trading::data::{BybitClient, construct_config_space, normalize_config_space};
use lagrangian_nn_trading::utils::{SGDOptimizer, compute_loss_and_gradients, energy_along_trajectory};
use lagrangian_nn_trading::integrator::integrate_trajectory;
use rand::seq::SliceRandom;

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train Lagrangian NN for trading")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval
    #[arg(short, long, default_value = "5")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value_t = 5000)]
    limit: usize,

    /// MA window for config space
    #[arg(long, default_value_t = 20)]
    ma_window: usize,

    /// Hidden layer dimension
    #[arg(long, default_value_t = 32)]
    hidden_dim: usize,

    /// Number of hidden layers
    #[arg(long, default_value_t = 2)]
    num_layers: usize,

    /// Mass matrix regularization
    #[arg(long, default_value_t = 1e-4)]
    mass_reg: f64,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f64,

    /// Momentum for SGD
    #[arg(long, default_value_t = 0.9)]
    momentum: f64,

    /// Number of training epochs
    #[arg(long, default_value_t = 200)]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Output directory
    #[arg(short, long, default_value = "output")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Lagrangian NN Trading: Training ===");
    println!("Symbol:      {}", args.symbol);
    println!("Hidden dim:  {}", args.hidden_dim);
    println!("Num layers:  {}", args.num_layers);
    println!("Mass reg:    {}", args.mass_reg);
    println!("LR:          {}", args.lr);
    println!("Epochs:      {}", args.epochs);
    println!("Batch size:  {}", args.batch_size);
    println!();

    std::fs::create_dir_all(&args.output)?;

    // Fetch data
    let client = BybitClient::new();
    println!("Fetching data...");
    let candles = client.fetch_extended(&args.symbol, &args.interval, args.limit).await?;
    println!("Fetched {} candles", candles.len());

    if candles.len() < 100 {
        anyhow::bail!("Not enough data");
    }

    // Construct config space
    let config_data = construct_config_space(&candles, args.ma_window);
    let (normalized, _stats) = normalize_config_space(&config_data);
    let n = normalized.q.len();
    println!("Config space: {} samples, dim={}", n, normalized.q[0].len());

    // Train/test split
    let split = (n as f64 * 0.8) as usize;
    let q_train = &normalized.q[..split];
    let qdot_train = &normalized.qdot[..split];
    let qddot_train = &normalized.qddot[..split];
    let q_test = &normalized.q[split..];
    let qdot_test = &normalized.qdot[split..];
    let qddot_test = &normalized.qddot[split..];

    println!("Train: {}, Test: {}", split, n - split);

    // Create model
    let coord_dim = normalized.q[0].len();
    let mut model = LagrangianNN::new(coord_dim, args.hidden_dim, args.num_layers, args.mass_reg);
    println!("Model parameters: {}", model.num_parameters());

    // Create optimizer
    let mut optimizer = SGDOptimizer::new(model.num_parameters(), args.lr, args.momentum);

    // Training loop
    println!("\nTraining...");
    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    let mut best_val_loss = f64::INFINITY;
    let mut best_params = model.parameters();
    let mut rng = rand::thread_rng();

    for epoch in 0..args.epochs {
        // Shuffle training indices
        let mut indices: Vec<usize> = (0..split).collect();
        indices.shuffle(&mut rng);

        // Mini-batch training
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for chunk in indices.chunks(args.batch_size) {
            let q_batch: Vec<Vec<f64>> = chunk.iter().map(|&i| q_train[i].clone()).collect();
            let qdot_batch: Vec<Vec<f64>> = chunk.iter().map(|&i| qdot_train[i].clone()).collect();
            let qddot_batch: Vec<Vec<f64>> = chunk.iter().map(|&i| qddot_train[i].clone()).collect();

            let (loss, gradients) = compute_loss_and_gradients(
                &model, &q_batch, &qdot_batch, &qddot_batch,
            );

            let mut params = model.parameters();
            optimizer.step(&mut params, &gradients);
            model.set_parameters(&params);

            epoch_loss += loss;
            n_batches += 1;
        }

        epoch_loss /= n_batches as f64;

        // Validation (use smaller batch for speed)
        let val_size = q_test.len().min(100);
        let val_q: Vec<Vec<f64>> = q_test[..val_size].to_vec();
        let val_qdot: Vec<Vec<f64>> = qdot_test[..val_size].to_vec();
        let val_qddot: Vec<Vec<f64>> = qddot_test[..val_size].to_vec();

        let (val_loss, _) = compute_loss_and_gradients(&model, &val_q, &val_qdot, &val_qddot);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            best_params = model.parameters();
        }

        if epoch % 10 == 0 || epoch == args.epochs - 1 {
            println!(
                "Epoch {:4}/{}: train_loss={:.6}, val_loss={:.6}",
                epoch + 1, args.epochs, epoch_loss, val_loss
            );
        }

        pb.inc(1);
    }
    pb.finish_with_message("Training complete");

    // Restore best model
    model.set_parameters(&best_params);

    // Energy conservation check
    println!("\nEnergy conservation evaluation:");
    let q0 = &q_test[0];
    let qdot0 = &qdot_test[0];
    let (traj_q, traj_v) = integrate_trajectory(&model, q0, qdot0, 0.01, 100);
    let energies = energy_along_trajectory(&model, &traj_q, &traj_v);

    if let (Some(first), Some(last)) = (energies.first(), energies.last()) {
        let drift = (last - first) / (first.abs() + 1e-10);
        println!("  Initial energy: {:.6}", first);
        println!("  Final energy:   {:.6}", last);
        println!("  Relative drift: {:.6}%", drift * 100.0);
    }

    // Save model
    let model_json = serde_json::to_string_pretty(&model)?;
    let model_path = format!("{}/lnn_model.json", args.output);
    std::fs::write(&model_path, model_json)?;
    println!("\nModel saved to {}", model_path);

    println!("Best validation loss: {:.6}", best_val_loss);
    println!("\nTraining complete!");
    Ok(())
}
