//! # Neural ODE Portfolio CLI
//!
//! Command-line interface for Neural ODE portfolio optimization.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use neural_ode_crypto::prelude::*;
use neural_ode_crypto::data::{BybitClient, CandleData, Timeframe, TechnicalIndicators};
use neural_ode_crypto::strategy::{BacktestConfig, Backtester};

/// Neural ODE Portfolio Optimizer
#[derive(Parser)]
#[command(name = "neural_ode_cli")]
#[command(author = "ML4Trading")]
#[command(version = "0.1.0")]
#[command(about = "Neural ODE for cryptocurrency portfolio optimization")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch historical data from Bybit
    Fetch {
        /// Trading pairs (comma-separated)
        #[arg(short, long, default_value = "BTCUSDT,ETHUSDT,SOLUSDT")]
        symbols: String,

        /// Timeframe
        #[arg(short, long, default_value = "60")]
        interval: String,

        /// Number of candles
        #[arg(short, long, default_value = "1000")]
        limit: usize,

        /// Output file (CSV)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Train Neural ODE model
    Train {
        /// Input data file
        #[arg(short, long)]
        data: String,

        /// Number of assets
        #[arg(short = 'n', long, default_value = "3")]
        n_assets: usize,

        /// Hidden dimension
        #[arg(short = 'd', long, default_value = "16")]
        hidden_dim: usize,

        /// Number of epochs
        #[arg(short, long, default_value = "50")]
        epochs: usize,

        /// Output model file
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Run backtest
    Backtest {
        /// Input data file
        #[arg(short, long)]
        data: String,

        /// Model file (optional)
        #[arg(short, long)]
        model: Option<String>,

        /// Initial portfolio value
        #[arg(short = 'v', long, default_value = "100000")]
        initial_value: f64,

        /// Rebalance threshold
        #[arg(short, long, default_value = "0.02")]
        threshold: f64,
    },

    /// Demonstrate ODE solvers
    Demo {
        /// Solver to demonstrate
        #[arg(short, long, default_value = "all")]
        solver: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            symbols,
            interval,
            limit,
            output,
        } => {
            fetch_data(&symbols, &interval, limit, output.as_deref()).await?;
        }
        Commands::Train {
            data,
            n_assets,
            hidden_dim,
            epochs,
            output,
        } => {
            train_model(&data, n_assets, hidden_dim, epochs, output.as_deref())?;
        }
        Commands::Backtest {
            data,
            model,
            initial_value,
            threshold,
        } => {
            run_backtest(&data, model.as_deref(), initial_value, threshold)?;
        }
        Commands::Demo { solver } => {
            demo_solvers(&solver);
        }
    }

    Ok(())
}

async fn fetch_data(
    symbols: &str,
    interval: &str,
    limit: usize,
    output: Option<&str>,
) -> Result<()> {
    let client = BybitClient::new();
    let symbol_list: Vec<&str> = symbols.split(',').collect();

    info!("Fetching data for {} symbols", symbol_list.len());

    for symbol in &symbol_list {
        let candles = client.get_klines(symbol, interval, limit).await?;
        info!("  {} - {} candles", symbol, candles.len());

        if let Some(output_base) = output {
            let filename = format!("{}_{}.csv", output_base, symbol.to_lowercase());
            save_candles_csv(&candles, &filename)?;
            info!("  Saved to {}", filename);
        }
    }

    info!("Done!");
    Ok(())
}

fn save_candles_csv(candles: &[neural_ode_crypto::data::Candle], filename: &str) -> Result<()> {
    let mut wtr = csv::Writer::from_path(filename)?;

    wtr.write_record(&["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

    for candle in candles {
        wtr.write_record(&[
            candle.open_time.to_string(),
            candle.open.to_string(),
            candle.high.to_string(),
            candle.low.to_string(),
            candle.close.to_string(),
            candle.volume.to_string(),
            candle.turnover.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

fn train_model(
    _data: &str,
    n_assets: usize,
    hidden_dim: usize,
    epochs: usize,
    _output: Option<&str>,
) -> Result<()> {
    info!("Creating Neural ODE model with {} assets, {} hidden dim", n_assets, hidden_dim);

    let model = NeuralODEPortfolio::new(n_assets, 12, hidden_dim);
    info!("Model has {} parameters", model.num_params());

    info!("Training for {} epochs (demo mode)...", epochs);

    // In a real implementation, we would:
    // 1. Load data from file
    // 2. Create training samples
    // 3. Train the model
    // 4. Save the trained model

    info!("Training complete (demo)");
    Ok(())
}

fn run_backtest(
    _data: &str,
    _model: Option<&str>,
    initial_value: f64,
    threshold: f64,
) -> Result<()> {
    info!("Running backtest with initial value ${:.2}", initial_value);

    // Create model and rebalancer
    let model = NeuralODEPortfolio::new(3, 12, 16);
    let rebalancer = ContinuousRebalancer::new(model, threshold);

    // Create demo data
    let demo_data = create_demo_data();

    // Configure backtest
    let config = BacktestConfig {
        initial_value,
        transaction_cost: 0.001,
        rebalance_threshold: threshold,
        min_rebalance_interval: 24,
        benchmark: "equal_weight".to_string(),
    };

    let backtester = Backtester::new(config);
    let result = backtester.run(&rebalancer, &demo_data);

    // Print results
    info!("=== Backtest Results ===");
    info!("Total Return: {:.2}%", result.total_return * 100.0);
    info!("Benchmark Return: {:.2}%", result.benchmark_return * 100.0);
    info!("Annualized Return: {:.2}%", result.annualized_return * 100.0);
    info!("Annualized Volatility: {:.2}%", result.annualized_volatility * 100.0);
    info!("Sharpe Ratio: {:.3}", result.sharpe_ratio);
    info!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    info!("Number of Rebalances: {}", result.num_rebalances);
    info!("Total Costs: ${:.2}", result.total_costs);

    Ok(())
}

fn create_demo_data() -> Vec<CandleData> {
    use neural_ode_crypto::data::Candle;

    let n = 500;
    let mut rng = rand::thread_rng();

    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let base_prices = [45000.0, 2500.0, 100.0];

    symbols
        .iter()
        .zip(base_prices.iter())
        .map(|(symbol, &base)| {
            let mut price = base;
            let candles: Vec<Candle> = (0..n)
                .map(|i| {
                    // Random walk with drift
                    let change = (rand::random::<f64>() - 0.5) * 0.02;
                    price *= 1.0 + change;

                    Candle::new(
                        i as i64 * 3600000,
                        price,
                        price * (1.0 + rand::random::<f64>() * 0.01),
                        price * (1.0 - rand::random::<f64>() * 0.01),
                        price * (1.0 + (rand::random::<f64>() - 0.5) * 0.005),
                        1000.0 + rand::random::<f64>() * 500.0,
                        price * 1000.0,
                    )
                })
                .collect();

            CandleData::new(symbol.to_string(), Timeframe::Hour1, candles)
        })
        .collect()
}

fn demo_solvers(solver: &str) {
    use ndarray::Array1;
    use neural_ode_crypto::ode::{ClosureODE, ODESolver, EulerSolver, RK4Solver, Dopri5Solver};

    info!("=== ODE Solver Demonstration ===");
    info!("Solving: dz/dt = -z, z(0) = 1");
    info!("Exact solution: z(t) = e^(-t)");
    info!("");

    // Exponential decay ODE
    let ode = ClosureODE::new(
        |z: &Array1<f64>, _t: f64| -z.clone(),
        1,
    );

    let z0 = Array1::from_vec(vec![1.0]);
    let t_span = (0.0, 2.0);
    let n_steps = 21;

    let exact_final = (-2.0_f64).exp();

    let solvers: Vec<(&str, Box<dyn ODESolver>)> = match solver {
        "euler" => vec![("Euler", Box::new(EulerSolver::new(0.01)))],
        "rk4" => vec![("RK4", Box::new(RK4Solver::new(0.01)))],
        "dopri5" => vec![("Dopri5", Box::new(Dopri5Solver::default()))],
        _ => vec![
            ("Euler", Box::new(EulerSolver::new(0.01))),
            ("RK4", Box::new(RK4Solver::new(0.01))),
            ("Dopri5", Box::new(Dopri5Solver::default())),
        ],
    };

    for (name, solver) in solvers {
        let start = std::time::Instant::now();
        let (times, states) = solver.solve(&ode, z0.clone(), t_span, n_steps);
        let duration = start.elapsed();

        let final_value = states.last().unwrap()[0];
        let error = (final_value - exact_final).abs();

        info!("{} Solver:", name);
        info!("  Final value: {:.10}", final_value);
        info!("  Exact value: {:.10}", exact_final);
        info!("  Error:       {:.2e}", error);
        info!("  Time:        {:?}", duration);
        info!("");
    }
}
