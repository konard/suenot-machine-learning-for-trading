//! Predict Binary
//!
//! Uses a trained Neural ODE model to generate trading predictions
//! and run a backtest on cryptocurrency data.
//!
//! Usage:
//!   cargo run --bin predict -- --symbol BTCUSDT --window-size 10
//!   cargo run --bin predict -- --symbol ETHUSDT --strategy mean_reversion

use clap::Parser;
use colored::Colorize;
use neural_ode_trading::backtest::Backtester;
use neural_ode_trading::data::{generate_synthetic_klines, BybitClient};
use neural_ode_trading::model::NeuralODE;
use neural_ode_trading::ode_solver::{DormandPrinceSolver, EulerSolver, ODESolver, RK4Solver};
use ndarray::Array1;

#[derive(Parser, Debug)]
#[command(name = "predict")]
#[command(about = "Generate predictions and backtest Neural ODE trading strategy")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Lookback window size
    #[arg(long, default_value_t = 10)]
    window_size: usize,

    /// Hidden dimension
    #[arg(long, default_value_t = 32)]
    hidden_dim: usize,

    /// Trading strategy
    #[arg(long, default_value = "momentum")]
    strategy: String,

    /// Initial capital (USD)
    #[arg(long, default_value_t = 100_000.0)]
    capital: f64,

    /// Transaction cost (fraction)
    #[arg(long, default_value_t = 0.001)]
    cost: f64,

    /// Number of data points
    #[arg(long, default_value_t = 500)]
    n_data: usize,

    /// Use synthetic data
    #[arg(long)]
    synthetic: bool,

    /// Compare ODE solvers
    #[arg(long)]
    compare_solvers: bool,
}

fn compare_ode_solvers() {
    println!("\n{}", "ODE Solver Comparison".bold().cyan());
    println!("{}", "=".repeat(60));

    // Test problem: dy/dt = -y, y(0) = 1 => y(1) = exp(-1) = 0.367879...
    let y0 = Array1::from_vec(vec![1.0]);
    let expected = (-1.0_f64).exp();
    let f = |_t: f64, y: &Array1<f64>| -y.clone();

    // Euler
    let euler = EulerSolver;
    let start = std::time::Instant::now();
    let result_euler = euler.solve(&f, &y0, 0.0, 1.0, 0.001);
    let time_euler = start.elapsed();

    // RK4
    let rk4 = RK4Solver;
    let start = std::time::Instant::now();
    let result_rk4 = rk4.solve(&f, &y0, 0.0, 1.0, 0.01);
    let time_rk4 = start.elapsed();

    // Dormand-Prince
    let dopri = DormandPrinceSolver::new(1e-8, 1e-10);
    let start = std::time::Instant::now();
    let result_dopri = dopri.solve(&f, &y0, 0.0, 1.0, 0.1);
    let time_dopri = start.elapsed();

    println!(
        "\n{:<20} {:>12} {:>12} {:>15}",
        "Solver", "Result", "Error", "Time"
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:<20} {:>12.8} {:>12.2e} {:>12}us",
        "Euler (h=0.001)",
        result_euler[0],
        (result_euler[0] - expected).abs(),
        time_euler.as_micros()
    );
    println!(
        "{:<20} {:>12.8} {:>12.2e} {:>12}us",
        "RK4 (h=0.01)",
        result_rk4[0],
        (result_rk4[0] - expected).abs(),
        time_rk4.as_micros()
    );
    println!(
        "{:<20} {:>12.8} {:>12.2e} {:>12}us",
        "Dormand-Prince",
        result_dopri[0],
        (result_dopri[0] - expected).abs(),
        time_dopri.as_micros()
    );
    println!(
        "\n{:<20} {:>12.8}",
        "Exact (exp(-1))", expected
    );

    // Multi-dimensional test: Lotka-Volterra (predator-prey)
    println!("\n\n{}", "Lotka-Volterra System (2D ODE)".bold().cyan());
    println!("{}", "=".repeat(60));
    println!("  dx/dt = alpha*x - beta*x*y   (prey)");
    println!("  dy/dt = delta*x*y - gamma*y   (predator)");

    let lv = |_t: f64, y: &Array1<f64>| {
        let alpha = 1.1;
        let beta = 0.4;
        let delta = 0.1;
        let gamma = 0.4;

        let dx = alpha * y[0] - beta * y[0] * y[1];
        let dy = delta * y[0] * y[1] - gamma * y[1];
        Array1::from_vec(vec![dx, dy])
    };

    let y0_lv = Array1::from_vec(vec![10.0, 5.0]);
    let t_points: Vec<f64> = (0..=50).map(|i| i as f64 * 0.2).collect();

    let trajectory = rk4.solve_trajectory(&lv, &y0_lv, &t_points, 0.01);

    println!("\n{:<8} {:>12} {:>12}", "Time", "Prey", "Predator");
    println!("{}", "-".repeat(35));
    for (i, (t, y)) in trajectory.iter().enumerate() {
        if i % 10 == 0 {
            println!("{:<8.1} {:>12.4} {:>12.4}", t, y[0], y[1]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("{}", "================================================".cyan());
    println!("  {}", "Neural ODE Trading - Prediction & Backtest".bold());
    println!("{}", "================================================".cyan());
    println!("  Symbol:   {}", args.symbol);
    println!("  Strategy: {}", args.strategy);
    println!("  Capital:  ${:.0}", args.capital);
    println!("{}", "================================================".cyan());

    // Optionally compare solvers
    if args.compare_solvers {
        compare_ode_solvers();
        println!();
    }

    // Load data
    println!("\nLoading data...");
    let data = if args.synthetic {
        generate_synthetic_klines(&args.symbol, args.n_data)
    } else {
        let client = BybitClient::new();
        match client.fetch_klines(&args.symbol, "1", args.n_data) {
            Ok(data) => {
                println!("Fetched {} candles from Bybit", data.len());
                data
            }
            Err(e) => {
                println!("API error: {}. Using synthetic data.", e);
                generate_synthetic_klines(&args.symbol, args.n_data)
            }
        }
    };

    // Create model
    let input_dim = args.window_size * 3;
    let model = NeuralODE::new(input_dim, args.hidden_dim, 1, 0.05);
    println!("Model parameters: {}", model.num_parameters());

    // Generate signals
    println!("\nGenerating trading signals...");
    let backtester = Backtester::new(args.capital, args.cost);
    let signals = backtester.generate_signals(&model, &data, args.window_size);

    let buy_count = signals.iter().filter(|s| **s == neural_ode_trading::Signal::Buy).count();
    let sell_count = signals.iter().filter(|s| **s == neural_ode_trading::Signal::Sell).count();
    let hold_count = signals.iter().filter(|s| **s == neural_ode_trading::Signal::Hold).count();

    println!("  Total signals: {}", signals.len());
    println!("  Buy:  {} ({:.1}%)", buy_count, buy_count as f64 / signals.len().max(1) as f64 * 100.0);
    println!("  Sell: {} ({:.1}%)", sell_count, sell_count as f64 / signals.len().max(1) as f64 * 100.0);
    println!("  Hold: {} ({:.1}%)", hold_count, hold_count as f64 / signals.len().max(1) as f64 * 100.0);

    // Run backtest
    println!("\nRunning backtest...");
    let result = backtester.run(&data, &signals, args.window_size);
    result.print_summary();

    // Buy-and-hold comparison
    let first_price = data.first().map(|d| d.close).unwrap_or(1.0);
    let last_price = data.last().map(|d| d.close).unwrap_or(1.0);
    let bh_return = (last_price / first_price - 1.0) * 100.0;

    println!("  Benchmark (Buy & Hold): {:.2}%", bh_return);
    let excess = result.total_return * 100.0 - bh_return;
    if excess > 0.0 {
        println!(
            "  Excess Return: {}",
            format!("+{:.2}%", excess).green()
        );
    } else {
        println!(
            "  Excess Return: {}",
            format!("{:.2}%", excess).red()
        );
    }

    Ok(())
}
