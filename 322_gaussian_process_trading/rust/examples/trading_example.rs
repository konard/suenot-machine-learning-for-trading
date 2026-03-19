//! Gaussian Process Trading Example
//!
//! Fetches BTCUSDT daily candles from Bybit, fits a GP to recent prices,
//! and predicts the next several days with confidence intervals and trading signals.

use anyhow::Result;
use gaussian_process_trading::{
    fetch_bybit_candles, optimize_hyperparameters, trading_signal, GaussianProcess, KernelType,
};

fn main() -> Result<()> {
    println!("=== Gaussian Process BTCUSDT Trading ===\n");

    // ── Step 1: Fetch data from Bybit ────────────────────────────────────
    println!("Fetching BTCUSDT daily candles from Bybit...");
    let candles = match fetch_bybit_candles("BTCUSDT", "D", 60) {
        Ok(c) => {
            println!("Fetched {} candles.\n", c.len());
            c
        }
        Err(e) => {
            println!("Could not fetch live data ({}). Using synthetic data.\n", e);
            generate_synthetic_data()
        }
    };

    if candles.is_empty() {
        println!("No candle data available. Exiting.");
        return Ok(());
    }

    // ── Step 2: Prepare GP inputs ────────────────────────────────────────
    let n = candles.len();
    let train_size = n.saturating_sub(5).max(10);
    let train_candles = &candles[..train_size];

    let close_prices: Vec<f64> = train_candles.iter().map(|c| c.close).collect();
    let current_price = close_prices.last().copied().unwrap_or(0.0);

    // Normalize inputs to [0, 1] for better kernel conditioning
    let x_train: Vec<f64> = (0..train_size).map(|i| i as f64 / train_size as f64).collect();

    // Normalize prices (subtract mean, divide by std)
    let mean_price: f64 = close_prices.iter().sum::<f64>() / close_prices.len() as f64;
    let std_price: f64 = {
        let var: f64 = close_prices
            .iter()
            .map(|p| (p - mean_price).powi(2))
            .sum::<f64>()
            / close_prices.len() as f64;
        var.sqrt().max(1.0)
    };
    let y_train: Vec<f64> = close_prices.iter().map(|p| (p - mean_price) / std_price).collect();

    println!(
        "Training on {} data points. Price range: ${:.2} - ${:.2}",
        train_size,
        close_prices.iter().cloned().fold(f64::INFINITY, f64::min),
        close_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    println!("Current price: ${:.2}\n", current_price);

    // ── Step 3: Optimize hyperparameters ─────────────────────────────────
    println!("Optimizing GP hyperparameters (random search, 200 candidates)...");
    let (best_l, best_sf, best_sn, best_lml) =
        optimize_hyperparameters(&x_train, &y_train, "matern52", 200)?;
    println!(
        "Best hyperparameters: l={:.4}, sigma_f={:.4}, sigma_n={:.4}, LML={:.4}\n",
        best_l, best_sf, best_sn, best_lml
    );

    // ── Step 4: Fit GP with best hyperparameters ─────────────────────────
    let gp = GaussianProcess::fit(
        x_train.clone(),
        y_train.clone(),
        KernelType::Matern52 {
            length_scale: best_l,
            signal_variance: best_sf,
        },
        best_sn,
    )?;

    let lml = gp.log_marginal_likelihood();
    println!("GP fitted. Log marginal likelihood: {:.4}\n", lml);

    // ── Step 5: Predict future days ──────────────────────────────────────
    let predict_days = [1, 2, 3, 5, 7, 10];
    let step = 1.0 / train_size as f64;

    println!("--- Predictions ---");
    println!(
        "{:<8} {:<14} {:<12} {:<30} {:<12}",
        "Day", "Predicted", "+/- Std", "95% Confidence Interval", "Signal"
    );
    println!("{}", "-".repeat(80));

    for &day in &predict_days {
        let x_test = vec![1.0 + day as f64 * step];
        let (means_norm, vars_norm) = gp.predict(&x_test);

        // Denormalize
        let predicted_price = means_norm[0] * std_price + mean_price;
        let predicted_std = vars_norm[0].sqrt() * std_price;

        let lower = predicted_price - 1.96 * predicted_std;
        let upper = predicted_price + 1.96 * predicted_std;

        let signal = trading_signal(predicted_price, predicted_std, current_price);

        println!(
            "+{:<7} ${:<13.2} ${:<11.2} [${:.2} - ${:.2}]   {}",
            day, predicted_price, predicted_std, lower, upper, signal
        );
    }

    // ── Step 6: Regime detection via variance ────────────────────────────
    println!("\n--- Regime Detection (In-Sample Variance) ---");
    let recent_n = 5.min(x_train.len());
    let recent_xs: Vec<f64> = x_train[x_train.len() - recent_n..].to_vec();
    let (_, recent_vars) = gp.predict(&recent_xs);
    let avg_recent_var: f64 = recent_vars.iter().sum::<f64>() / recent_vars.len() as f64;
    let avg_recent_std = avg_recent_var.sqrt() * std_price;

    println!(
        "Average in-sample std (last {} points): ${:.2}",
        recent_n, avg_recent_std
    );
    if avg_recent_std > std_price * 0.5 {
        println!("WARNING: High in-sample uncertainty detected -- possible regime change!");
    } else {
        println!("Market regime appears stable.");
    }

    println!("\n=== Done ===");
    Ok(())
}

/// Generate synthetic BTCUSDT-like data for offline testing.
fn generate_synthetic_data() -> Vec<gaussian_process_trading::Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 67000.0_f64;
    let n = 60;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let change: f64 = rng.gen_range(-0.02..0.02);
        let open = price;
        price *= 1.0 + change;
        let close = price;
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(1000.0..5000.0);

        candles.push(gaussian_process_trading::Candle {
            timestamp: 1700000000000 + i as u64 * 86400000,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    candles
}
