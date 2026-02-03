//! Event study example using DiD methodology.
//!
//! Simulates analyzing the impact of a cryptocurrency exchange listing
//! on token prices using Difference-in-Differences.

use chrono::{Duration, Utc};
use std::collections::HashMap;

use diff_in_diff_trading::data::generate_synthetic_candles;
use diff_in_diff_trading::model::{DiDEstimator, PanelData};
use diff_in_diff_trading::trading::SignalGenerator;

fn main() {
    println!("=== Event Study: Crypto Exchange Listing Impact ===\n");

    // Scenario: A token (TOKEN_A) gets listed on a major exchange.
    // We want to measure the listing premium using DiD.

    let event_date = Utc::now() - Duration::days(30);
    let pre_window = 20; // 20 days before event
    let post_window = 10; // 10 days after event

    println!("Event: TOKEN_A listed on major exchange");
    println!("Event date: {}", event_date.format("%Y-%m-%d"));
    println!("Analysis window: {} days pre, {} days post\n", pre_window, post_window);

    // Generate synthetic price data
    // TOKEN_A (treated): experiences listing effect
    // TOKEN_B (control): similar token, no listing
    let token_a_candles = generate_synthetic_candles(pre_window + post_window, 100.0, "TOKEN_A");
    let token_b_candles = generate_synthetic_candles(pre_window + post_window, 100.0, "TOKEN_B");

    println!("Price data generated:");
    println!("  - TOKEN_A (treated): {} candles", token_a_candles.len());
    println!("  - TOKEN_B (control): {} candles", token_b_candles.len());
    println!();

    // Compute returns and add treatment effect to TOKEN_A post-listing
    let listing_premium = 0.08; // 8% listing premium

    // Convert candle data to panel format
    let mut panel_data: Vec<PanelData> = Vec::new();

    // Add TOKEN_A data
    for (i, candle) in token_a_candles.iter().enumerate() {
        let is_post = i >= pre_window;
        let event_time = i as i32 - pre_window as i32;

        // Compute return (with listing effect added post-event)
        let base_return = if i > 0 {
            (candle.close - token_a_candles[i - 1].close) / token_a_candles[i - 1].close
        } else {
            0.0
        };

        let outcome = if is_post {
            base_return + listing_premium / post_window as f64 // Spread effect over post period
        } else {
            base_return
        };

        panel_data.push(PanelData {
            unit_id: "TOKEN_A".to_string(),
            timestamp: candle.timestamp,
            outcome,
            treated: true,
            post_treatment: is_post,
            event_time: Some(event_time),
            covariates: HashMap::new(),
        });
    }

    // Add TOKEN_B data (control)
    for (i, candle) in token_b_candles.iter().enumerate() {
        let is_post = i >= pre_window;
        let event_time = i as i32 - pre_window as i32;

        let outcome = if i > 0 {
            (candle.close - token_b_candles[i - 1].close) / token_b_candles[i - 1].close
        } else {
            0.0
        };

        panel_data.push(PanelData {
            unit_id: "TOKEN_B".to_string(),
            timestamp: candle.timestamp,
            outcome,
            treated: false,
            post_treatment: is_post,
            event_time: Some(event_time),
            covariates: HashMap::new(),
        });
    }

    println!("Panel data prepared: {} observations", panel_data.len());
    println!();

    // Run DiD estimation
    println!("Running DiD estimation...\n");

    let estimator = DiDEstimator::new()
        .with_bootstrap(1000)
        .with_confidence(0.95);

    let results = estimator.fit(&panel_data);

    println!("=== DiD Results ===");
    println!("{}\n", results);

    // Compare with true effect
    println!("Comparison with true effect:");
    println!("  - True listing premium: {:.2}%", listing_premium * 100.0);
    println!(
        "  - Estimated premium:    {:.2}%",
        results.did_estimate * 100.0
    );
    println!(
        "  - 95% CI: [{:.2}%, {:.2}%]",
        results.ci_lower * 100.0,
        results.ci_upper * 100.0
    );
    println!();

    // Event study: Show time-series of effects
    println!("=== Event Study Plot (Cumulative Effect) ===");
    println!("Period | Treatment | Control | Diff");
    println!("{:-<50}", "");

    let mut cum_treat_pre = 0.0;
    let mut cum_ctrl_pre = 0.0;
    let mut cum_treat_post = 0.0;
    let mut cum_ctrl_post = 0.0;

    for event_time in -5..=5 {
        let treat_obs: Vec<_> = panel_data
            .iter()
            .filter(|d| d.treated && d.event_time == Some(event_time))
            .collect();

        let ctrl_obs: Vec<_> = panel_data
            .iter()
            .filter(|d| !d.treated && d.event_time == Some(event_time))
            .collect();

        if treat_obs.is_empty() || ctrl_obs.is_empty() {
            continue;
        }

        let treat_return = treat_obs.iter().map(|d| d.outcome).sum::<f64>() / treat_obs.len() as f64;
        let ctrl_return = ctrl_obs.iter().map(|d| d.outcome).sum::<f64>() / ctrl_obs.len() as f64;

        if event_time < 0 {
            cum_treat_pre += treat_return;
            cum_ctrl_pre += ctrl_return;
        } else {
            cum_treat_post += treat_return;
            cum_ctrl_post += ctrl_return;
        }

        let marker = if event_time == 0 { " <-- EVENT" } else { "" };
        println!(
            "t={:+3}  | {:+7.2}% | {:+7.2}% | {:+7.2}%{}",
            event_time,
            treat_return * 100.0,
            ctrl_return * 100.0,
            (treat_return - ctrl_return) * 100.0,
            marker
        );
    }

    println!("{:-<50}", "");
    println!(
        "Pre-event cumulative:  Treatment={:+.2}%, Control={:+.2}%",
        cum_treat_pre * 100.0,
        cum_ctrl_pre * 100.0
    );
    println!(
        "Post-event cumulative: Treatment={:+.2}%, Control={:+.2}%",
        cum_treat_post * 100.0,
        cum_ctrl_post * 100.0
    );
    println!();

    // Trading signal
    println!("=== Trading Signal ===");

    let signal_gen = SignalGenerator::new()
        .with_threshold(0.02)
        .with_significance(0.05);

    let signal = signal_gen.generate(&results);

    println!("Signal: {:?}", signal);
    println!("Significance: p = {:.4}", results.p_value);

    match signal {
        diff_in_diff_trading::trading::Signal::Long => {
            println!("Recommendation: BUY TOKEN_A (positive listing effect detected)");
        }
        diff_in_diff_trading::trading::Signal::Short => {
            println!("Recommendation: SELL TOKEN_A (negative effect detected)");
        }
        diff_in_diff_trading::trading::Signal::Flat => {
            println!("Recommendation: NO TRADE (effect not significant or too small)");
        }
    }

    println!("\n=== Event Study Complete ===");
}
