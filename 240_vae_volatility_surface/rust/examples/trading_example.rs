//! Trading example: VAE Volatility Surface
//!
//! Demonstrates the full pipeline:
//! 1. Fetch live BTC options data from Bybit
//! 2. Generate synthetic training surfaces via SABR
//! 3. Train a VAE on the surface distribution
//! 4. Encode/decode surfaces and check arbitrage conditions
//! 5. Detect relative value trading opportunities

use vae_volatility_surface::*;
use ndarray::Array1;

#[tokio::main]
async fn main() {
    println!("=== Chapter 240: VAE Volatility Surface ===\n");

    let moneyness = default_moneyness_grid();
    let maturity_days = default_maturity_days();
    let maturities_years: Vec<f64> = maturity_days.iter().map(|d| d / 365.0).collect();
    let grid_size = moneyness.len() * maturities_years.len();

    // -----------------------------------------------------------------------
    // Step 1: Fetch live BTC options data from Bybit
    // -----------------------------------------------------------------------
    println!("--- Step 1: Fetching BTC options from Bybit ---");
    let client = BybitClient::new();
    let live_surface = match client.get_option_tickers().await {
        Ok(tickers) => {
            println!("  Fetched {} option tickers from Bybit", tickers.len());
            let parsed = BybitClient::parse_tickers(&tickers);
            println!("  Parsed {} valid options", parsed.len());

            if parsed.len() > 10 {
                let calls: Vec<_> = parsed.iter().filter(|o| o.is_call).collect();
                let puts: Vec<_> = parsed.iter().filter(|o| !o.is_call).collect();
                println!("  Calls: {}, Puts: {}", calls.len(), puts.len());

                if let Some(opt) = parsed.first() {
                    println!("  Underlying price: ${:.2}", opt.underlying);
                }

                let surface = BybitClient::build_surface(&parsed, &moneyness, &maturity_days);
                println!("  Built surface: {} maturities × {} strikes", surface.maturities.len(), surface.moneyness.len());
                Some(surface)
            } else {
                println!("  Not enough options data, will use synthetic data");
                None
            }
        }
        Err(e) => {
            println!("  Could not fetch from Bybit: {}", e);
            println!("  Falling back to synthetic data");
            None
        }
    };
    println!();

    // -----------------------------------------------------------------------
    // Step 2: Generate synthetic training surfaces via SABR
    // -----------------------------------------------------------------------
    println!("--- Step 2: Generating synthetic SABR surfaces ---");
    let mut generator = SurfaceGenerator::new();
    let training_surfaces = generator.generate_batch(200, &moneyness, &maturities_years);
    println!("  Generated {} training surfaces", training_surfaces.len());
    println!("  Grid: {} maturities × {} strikes = {} points",
        maturities_years.len(), moneyness.len(), grid_size);

    // Show statistics of the first surface
    let sample_surface = &training_surfaces[0];
    println!("  Sample surface ATM vols:");
    for (i, &tau) in maturities_years.iter().enumerate() {
        println!("    τ = {:.3}y: ATM vol = {:.4}", tau, sample_surface.atm_vol(i));
    }

    let skew = compute_skew(sample_surface, 0);
    let bfly = compute_butterfly(sample_surface, 0);
    println!("  Short-term skew: {:.4}", skew);
    println!("  Short-term butterfly: {:.4}", bfly);
    println!();

    // -----------------------------------------------------------------------
    // Step 3: Train VAE on surface distribution
    // -----------------------------------------------------------------------
    println!("--- Step 3: Training VAE (β=1.5) ---");
    let latent_dim = 4;
    let hidden_dim = 32;
    let beta = 1.5;
    let mut vae = VAEModel::new(grid_size, hidden_dim, latent_dim, beta);
    println!("  Architecture: {} → {} → {} → {} → {}",
        grid_size, hidden_dim, latent_dim, hidden_dim, grid_size);
    println!("  β = {}", beta);

    let training_data: Vec<Vec<f64>> = training_surfaces.iter().map(|s| s.flatten()).collect();

    // Measure loss before training
    let x0 = Array1::from(training_data[0].clone());
    let (recon0, mu0, lv0) = vae.forward(&x0);
    let (loss_before, recon_before, kl_before) = vae.loss(&x0, &recon0, &mu0, &lv0);
    println!("  Before training: total={:.4}, recon={:.4}, KL={:.4}", loss_before, recon_before, kl_before);

    let (loss_after, recon_after, kl_after) = vae.train(&training_data, 50, 0.01);
    println!("  After training:  total={:.4}, recon={:.4}, KL={:.4}", loss_after, recon_after, kl_after);
    println!("  Loss reduction: {:.1}%", (1.0 - loss_after / loss_before) * 100.0);
    println!();

    // -----------------------------------------------------------------------
    // Step 4: Encode/decode and check arbitrage conditions
    // -----------------------------------------------------------------------
    println!("--- Step 4: Encode/Decode and Arbitrage Check ---");

    // Encode a surface
    let test_surface = &training_surfaces[10];
    let test_flat = Array1::from(test_surface.flatten());
    let (mu, log_var) = vae.encode(&test_flat);
    println!("  Latent representation (μ):");
    for i in 0..latent_dim {
        println!("    z_{}: μ={:.4}, σ={:.4}", i, mu[i], (log_var[i] * 0.5).exp());
    }

    // Decode back
    let z = vae.reparameterize(&mu, &log_var);
    let recon_flat = vae.decode(&z);

    // Reconstruction error
    let mse: f64 = test_flat.iter()
        .zip(recon_flat.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>() / grid_size as f64;
    println!("  Reconstruction MSE: {:.6}", mse);

    // Build reconstructed surface and check arbitrage
    let recon_ivols = ndarray::Array2::from_shape_vec(
        (maturities_years.len(), moneyness.len()),
        recon_flat.to_vec(),
    ).unwrap();
    let recon_surface = VolSurface::new(moneyness.clone(), maturities_years.clone(), recon_ivols);

    let (butterfly_violations, _) = ArbitrageChecker::check_butterfly(&recon_surface);
    let (calendar_violations, _) = ArbitrageChecker::check_calendar(&recon_surface);
    println!("  Butterfly violations: {}", butterfly_violations);
    println!("  Calendar violations: {}", calendar_violations);

    // Check original surfaces too
    let (orig_butterfly, _) = ArbitrageChecker::check_butterfly(test_surface);
    let (orig_calendar, _) = ArbitrageChecker::check_calendar(test_surface);
    println!("  Original surface: butterfly={}, calendar={}", orig_butterfly, orig_calendar);
    println!();

    // -----------------------------------------------------------------------
    // Step 5: Generate new surfaces from prior and detect trading signals
    // -----------------------------------------------------------------------
    println!("--- Step 5: Surface Generation and Trading Signals ---");

    // Sample from the prior
    println!("  Sampling 5 surfaces from VAE prior:");
    for i in 0..5 {
        let sampled = vae.sample();
        let avg_vol: f64 = sampled.iter().sum::<f64>() / sampled.len() as f64;
        let min_vol = sampled.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_vol = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("    Surface {}: avg_vol={:.4}, range=[{:.4}, {:.4}]",
            i + 1, avg_vol, min_vol, max_vol);
    }
    println!();

    // Relative value: compare observed vs VAE-reconstructed surface
    println!("  Relative Value Analysis:");
    let target = if live_surface.is_some() {
        println!("  (Using live Bybit surface)");
        live_surface.as_ref().unwrap()
    } else {
        println!("  (Using synthetic surface as proxy for observed market)");
        &training_surfaces[50]
    };

    let target_flat = Array1::from(target.flatten());
    let (t_mu, t_lv) = vae.encode(&target_flat);
    let t_z = vae.reparameterize(&t_mu, &t_lv);
    let fair_flat = vae.decode(&t_z);

    // Compute mispricing: observed - fair value
    let mut max_mispricing = 0.0f64;
    let mut max_mispricing_idx = (0, 0);

    for t in 0..maturities_years.len() {
        for k in 0..moneyness.len() {
            let idx = t * moneyness.len() + k;
            let observed = target_flat[idx];
            let fair = fair_flat[idx];
            let diff = (observed - fair).abs();
            if diff > max_mispricing {
                max_mispricing = diff;
                max_mispricing_idx = (t, k);
            }
        }
    }

    let (t_idx, k_idx) = max_mispricing_idx;
    let flat_idx = t_idx * moneyness.len() + k_idx;
    let observed_vol = target_flat[flat_idx];
    let fair_vol = fair_flat[flat_idx];
    let direction = if observed_vol > fair_vol { "SELL VOL (overpriced)" } else { "BUY VOL (underpriced)" };

    println!("  Largest mispricing found:");
    println!("    Maturity: {:.3}y, Moneyness: {:.3}", maturities_years[t_idx], moneyness[k_idx]);
    println!("    Observed IV: {:.4}", observed_vol);
    println!("    Fair IV:     {:.4}", fair_vol);
    println!("    Diff:        {:.4}", max_mispricing);
    println!("    Signal:      {}", direction);
    println!();

    println!("=== Done ===");
}
