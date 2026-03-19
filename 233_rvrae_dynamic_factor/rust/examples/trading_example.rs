use anyhow::Result;
use ndarray::Array1;
use rvrae_dynamic_factor::*;

fn main() -> Result<()> {
    println!("=== RVRAE Dynamic Factor Model - Trading Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Fetch market data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching market data from Bybit...");
    let symbols = ["BTCUSDT", "ETHUSDT"];
    let interval = "60"; // 1-hour candles
    let limit = 200;

    let mut all_returns: Vec<Vec<f64>> = Vec::new();

    for symbol in &symbols {
        match fetch_bybit_klines(symbol, interval, limit) {
            Ok(prices) => {
                println!(
                    "  {} : {} candles fetched, latest price = {:.2}",
                    symbol,
                    prices.len(),
                    prices.last().unwrap_or(&0.0)
                );
                let rets = log_returns(&prices);
                let (normed, mean, std) = normalize(&rets);
                println!(
                    "  {} : {} returns, mean={:.6}, std={:.6}",
                    symbol,
                    normed.len(),
                    mean,
                    std
                );
                all_returns.push(normed);
            }
            Err(e) => {
                println!("  {} : API error ({}), using synthetic data", symbol, e);
                let synthetic = generate_synthetic_returns(limit - 1);
                all_returns.push(synthetic);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 2. Combine into multivariate time-series
    // -----------------------------------------------------------------------
    let multivariate = combine_assets(&all_returns);
    println!(
        "\nMultivariate series: {} timesteps x {} assets",
        multivariate.len(),
        if multivariate.is_empty() {
            0
        } else {
            multivariate[0].len()
        }
    );

    // -----------------------------------------------------------------------
    // 3. Train RVRAE
    // -----------------------------------------------------------------------
    let input_size = symbols.len();
    let hidden_size = 16;
    let latent_size = 2;
    let epochs = 100;
    let learning_rate = 0.001;
    let annealing_epochs = 50;

    println!(
        "\nTraining RVRAE (input={}, hidden={}, latent={}, epochs={})...\n",
        input_size, hidden_size, latent_size, epochs
    );

    let mut model = RVRAE::new(input_size, hidden_size, latent_size);
    let losses = model.train(&multivariate, epochs, learning_rate, annealing_epochs);

    println!(
        "\nTraining complete. Final loss: {:.6}",
        losses.last().unwrap_or(&0.0)
    );

    // -----------------------------------------------------------------------
    // 4. Extract time-varying latent factors
    // -----------------------------------------------------------------------
    println!("\n--- Dynamic Factor Extraction ---");
    let factors = model.extract_factors(&multivariate);
    println!("Extracted {} factor vectors (latent dim = {})", factors.len(), latent_size);

    // Print first and last few factors
    println!("\nFirst 5 factor values:");
    for (t, z) in factors.iter().enumerate().take(5) {
        println!("  t={}: z = {:?}", t, z.to_vec());
    }
    println!("\nLast 5 factor values:");
    let start = factors.len().saturating_sub(5);
    for (t, z) in factors.iter().enumerate().skip(start) {
        println!("  t={}: z = {:?}", t, z.to_vec());
    }

    // -----------------------------------------------------------------------
    // 5. Analyze factor dynamics
    // -----------------------------------------------------------------------
    println!("\n--- Factor Dynamics Analysis ---");

    // Factor velocities
    let velocities = FactorDynamics::factor_velocity(&factors);
    let mean_vel: f64 = velocities.iter().sum::<f64>() / velocities.len().max(1) as f64;
    let max_vel = velocities.iter().cloned().fold(0.0_f64, f64::max);
    println!(
        "Factor velocity: mean={:.6}, max={:.6}",
        mean_vel, max_vel
    );

    // Autocorrelation for each factor dimension
    for dim in 0..latent_size {
        let factor_series: Vec<f64> = factors.iter().map(|z| z[dim]).collect();
        let ac1 = FactorDynamics::autocorrelation(&factor_series, 1);
        let ac5 = FactorDynamics::autocorrelation(&factor_series, 5);
        println!(
            "Factor {}: autocorr(lag=1)={:.4}, autocorr(lag=5)={:.4}",
            dim, ac1, ac5
        );
    }

    // Cross-factor correlation
    if latent_size >= 2 {
        let f0: Vec<f64> = factors.iter().map(|z| z[0]).collect();
        let f1: Vec<f64> = factors.iter().map(|z| z[1]).collect();
        let corr = FactorDynamics::factor_correlation(&f0, &f1);
        println!("Factor 0-1 correlation: {:.4}", corr);

        // Rolling correlation
        let window = 20;
        let rolling = FactorDynamics::rolling_correlation(&f0, &f1, window);
        if !rolling.is_empty() {
            let mean_roll: f64 = rolling.iter().sum::<f64>() / rolling.len() as f64;
            let min_roll = rolling.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_roll = rolling.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "Rolling correlation (w={}): mean={:.4}, min={:.4}, max={:.4}",
                window, mean_roll, min_roll, max_roll
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. Detect regime transitions
    // -----------------------------------------------------------------------
    println!("\n--- Regime Transition Detection ---");
    let regime_shifts = FactorDynamics::detect_regime_shifts(&factors, 2.0);
    if regime_shifts.is_empty() {
        println!("No significant regime shifts detected (threshold = 2 sigma).");
    } else {
        println!(
            "Detected {} regime shift(s) at timesteps: {:?}",
            regime_shifts.len(),
            regime_shifts
        );
    }

    // Lower threshold
    let mild_shifts = FactorDynamics::detect_regime_shifts(&factors, 1.5);
    println!(
        "Mild regime changes (1.5 sigma): {} timesteps",
        mild_shifts.len()
    );

    // -----------------------------------------------------------------------
    // 7. Factor-based trading signals
    // -----------------------------------------------------------------------
    println!("\n--- Factor-Based Trading Signals ---");

    if factors.len() >= 10 {
        // Factor momentum: direction of recent factor movement
        let recent = &factors[factors.len() - 5..];
        let oldest = &factors[factors.len() - 10..factors.len() - 5];

        let recent_mean: Array1<f64> = recent
            .iter()
            .fold(Array1::zeros(latent_size), |acc, z| acc + z)
            / 5.0;
        let oldest_mean: Array1<f64> = oldest
            .iter()
            .fold(Array1::zeros(latent_size), |acc, z| acc + z)
            / 5.0;

        let momentum = &recent_mean - &oldest_mean;
        println!("Factor momentum (last 5 vs prev 5): {:?}", momentum.to_vec());

        // Factor regime indicator
        let current_vel = if let Some(&v) = velocities.last() {
            v
        } else {
            0.0
        };
        let regime = if current_vel > mean_vel + mean_vel {
            "HIGH VOLATILITY - consider reducing exposure"
        } else if current_vel < mean_vel * 0.5 {
            "LOW VOLATILITY - stable regime"
        } else {
            "NORMAL - factors evolving steadily"
        };
        println!(
            "Current factor velocity: {:.6} -> Regime: {}",
            current_vel, regime
        );
    }

    println!("\n=== RVRAE Dynamic Factor Analysis Complete ===");
    Ok(())
}

/// Generate synthetic return series for fallback when API is unavailable.
fn generate_synthetic_returns(n: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut returns = Vec::with_capacity(n);

    // Regime-switching synthetic data
    let mut regime = 0; // 0 = low vol, 1 = high vol
    for _i in 0..n {
        // Switch regime occasionally
        if rng.gen::<f64>() < 0.02 {
            regime = 1 - regime;
        }
        let vol = if regime == 0 { 0.01 } else { 0.03 };
        let drift = if regime == 0 { 0.0001 } else { -0.0005 };
        let r = drift + vol * rng.gen_range(-1.0..1.0);
        returns.push(r);
    }

    let (normed, _, _) = rvrae_dynamic_factor::normalize(&returns);
    normed
}
