use anyhow::Result;
use data_augmentation_trading::*;
use ndarray::Array1;

fn print_separator() {
    println!("{}", "=".repeat(70));
}

fn print_series_summary(name: &str, data: &Array1<f64>) {
    let (mean, std_dev, min, max) = compute_stats(data);
    println!(
        "  {:<25} | len={:<4} | mean={:<10.2} | std={:<8.2} | min={:<10.2} | max={:<10.2}",
        name,
        data.len(),
        mean,
        std_dev,
        min,
        max
    );
}

fn print_returns_summary(name: &str, prices: &Array1<f64>) {
    let returns = compute_returns(prices);
    if returns.is_empty() {
        return;
    }
    let (mean, std_dev, min, max) = compute_stats(&returns);
    println!(
        "  {:<25} | mean={:<12.6} | std={:<10.6} | min={:<12.6} | max={:<12.6}",
        name, mean, std_dev, min, max
    );
}

/// Simple moving average model: predict next return as average of last `window` returns.
fn sma_predict(returns: &[f64], window: usize) -> f64 {
    if returns.len() < window {
        return 0.0;
    }
    let start = returns.len() - window;
    let sum: f64 = returns[start..].iter().sum();
    sum / window as f64
}

/// Evaluate a simple directional prediction model.
/// Returns accuracy: fraction of times predicted direction matches actual.
fn evaluate_directional_accuracy(
    train_prices: &Array1<f64>,
    test_prices: &Array1<f64>,
    window: usize,
) -> f64 {
    let train_returns = compute_returns(train_prices);
    let test_returns = compute_returns(test_prices);

    if test_returns.is_empty() || train_returns.is_empty() {
        return 0.0;
    }

    // Build all returns for prediction context
    let mut all_returns: Vec<f64> = train_returns.to_vec();
    let mut correct = 0;
    let total = test_returns.len();

    for i in 0..total {
        let pred = sma_predict(&all_returns, window);
        let actual = test_returns[i];

        if (pred >= 0.0 && actual >= 0.0) || (pred < 0.0 && actual < 0.0) {
            correct += 1;
        }

        all_returns.push(actual);
    }

    correct as f64 / total as f64
}

fn main() -> Result<()> {
    println!();
    print_separator();
    println!("  Chapter 288: Data Augmentation for Trading");
    println!("  Fetching BTCUSDT data from Bybit and applying augmentations");
    print_separator();
    println!();

    // ── Step 1: Fetch data from Bybit ──────────────────────────────────────
    println!("[1] Fetching BTCUSDT hourly data from Bybit...");
    let bars = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(bars) => {
            println!("    Fetched {} candles successfully.", bars.len());
            bars
        }
        Err(e) => {
            println!("    Warning: Could not fetch from Bybit: {}", e);
            println!("    Using synthetic sample data instead.");
            generate_synthetic_data()
        }
    };

    let prices = extract_close_prices(&bars);
    println!();

    // ── Step 2: Show original data stats ───────────────────────────────────
    println!("[2] Original data statistics:");
    print_series_summary("Original Prices", &prices);
    print_returns_summary("Original Returns", &prices);
    println!();

    // ── Step 3: Apply augmentations ────────────────────────────────────────
    println!("[3] Applying augmentation techniques:");
    println!();

    let mut augmenter = DataAugmenter::with_seed(42);

    // Time Warping
    let warped = augmenter.time_warp(&prices);
    print_series_summary("Time Warped", &warped);
    print_returns_summary("Time Warped Returns", &warped);
    println!();

    // Magnitude Scaling
    let scaled = augmenter.magnitude_scale(&prices);
    print_series_summary("Magnitude Scaled", &scaled);
    print_returns_summary("Magnitude Scaled Returns", &scaled);
    println!();

    // Jittering
    let jittered = augmenter.jitter(&prices);
    print_series_summary("Jittered", &jittered);
    print_returns_summary("Jittered Returns", &jittered);
    println!();

    // Window Slicing
    let sliced = augmenter.window_slice(&prices);
    print_series_summary("Window Sliced", &sliced);
    print_returns_summary("Window Sliced Returns", &sliced);
    println!();

    // Volatility Scaling
    let vol_scaled = augmenter.volatility_scale(&prices);
    print_series_summary("Volatility Scaled (1.5x)", &vol_scaled);
    print_returns_summary("Vol Scaled Returns", &vol_scaled);
    println!();

    // Combined augmentation
    let combined = augmenter.augment_all(&prices);
    print_series_summary("Combined (all techniques)", &combined);
    print_returns_summary("Combined Returns", &combined);
    println!();

    // ── Step 4: Show sample values ─────────────────────────────────────────
    print_separator();
    println!("  Sample values (first 10 data points):");
    print_separator();
    println!(
        "  {:>5} | {:>12} | {:>12} | {:>12} | {:>12}",
        "Index", "Original", "Warped", "Jittered", "VolScaled"
    );
    println!("  {}", "-".repeat(63));
    let show_n = 10.min(prices.len());
    for i in 0..show_n {
        println!(
            "  {:>5} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.2}",
            i, prices[i], warped[i], jittered[i], vol_scaled[i]
        );
    }
    println!();

    // ── Step 5: Train model with/without augmentation ──────────────────────
    print_separator();
    println!("  Model Comparison: With vs Without Augmentation");
    print_separator();
    println!();

    let split = (prices.len() as f64 * 0.7) as usize;
    let train_prices = prices.slice(ndarray::s![..split]).to_owned();
    let test_prices = prices.slice(ndarray::s![split..]).to_owned();

    println!(
        "  Train set: {} bars, Test set: {} bars",
        train_prices.len(),
        test_prices.len()
    );
    println!();

    // Without augmentation
    let sma_window = 10;
    let accuracy_no_aug = evaluate_directional_accuracy(&train_prices, &test_prices, sma_window);
    println!(
        "  [No Augmentation]   SMA({}) directional accuracy: {:.1}%",
        sma_window,
        accuracy_no_aug * 100.0
    );

    // With augmentation: create augmented training sets and average predictions
    let num_augmentations = 5;
    let mut total_accuracy = 0.0;

    for i in 0..num_augmentations {
        let mut aug = DataAugmenter::with_seed(100 + i as u64);
        let aug_train = aug.augment_all(&train_prices);
        let acc = evaluate_directional_accuracy(&aug_train, &test_prices, sma_window);
        total_accuracy += acc;
    }
    let avg_accuracy_aug = total_accuracy / num_augmentations as f64;

    println!(
        "  [With Augmentation] SMA({}) avg directional accuracy: {:.1}% (over {} augmented sets)",
        sma_window,
        avg_accuracy_aug * 100.0,
        num_augmentations
    );

    // Ensemble: combine original + augmented
    let ensemble_accuracy = (accuracy_no_aug + avg_accuracy_aug) / 2.0;
    println!(
        "  [Ensemble]          Combined accuracy estimate:      {:.1}%",
        ensemble_accuracy * 100.0
    );

    println!();
    print_separator();
    println!("  Key Insight: Data augmentation provides diverse training");
    println!("  perspectives, helping models generalize better to unseen");
    println!("  market conditions. The ensemble of original + augmented");
    println!("  data typically yields more robust predictions.");
    print_separator();
    println!();

    Ok(())
}

/// Generate synthetic OHLCV data when Bybit API is unavailable.
fn generate_synthetic_data() -> Vec<OhlcvBar> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let mut bars = Vec::new();
    let mut price = 50000.0_f64;
    let base_timestamp = 1700000000000_u64;

    for i in 0..200 {
        let return_pct = rng.gen_range(-0.02..0.02);
        let close = price * (1.0 + return_pct);
        let high = close * (1.0 + rng.gen_range(0.0..0.01));
        let low = close * (1.0 - rng.gen_range(0.0..0.01));
        let open = price;
        let volume = rng.gen_range(100.0..10000.0);

        bars.push(OhlcvBar {
            timestamp: base_timestamp + i * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    bars
}
