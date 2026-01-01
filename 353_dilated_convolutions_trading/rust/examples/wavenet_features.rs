//! Example: WaveNet Feature Extraction for Trading
//!
//! This example demonstrates:
//! 1. Fetching real market data
//! 2. Calculating technical features
//! 3. Applying WaveNet-style feature extraction
//! 4. Analyzing multi-scale patterns

use dilated_conv_trading::api::Interval;
use dilated_conv_trading::conv::DilatedConvStack;
use dilated_conv_trading::features::{Normalizer, TechnicalFeatures};
use dilated_conv_trading::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== WaveNet Feature Extraction ===\n");

    // 1. Fetch market data
    println!("1. Fetching market data from Bybit...\n");

    let client = BybitClient::new();
    let symbol = "BTCUSDT";
    let klines = client
        .get_klines_with_interval(symbol, Interval::Min15, 500)
        .await?;

    println!("   Fetched {} 15-minute candles for {}", klines.len(), symbol);
    println!(
        "   Price range: {:.2} - {:.2}",
        klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max)
    );

    // 2. Calculate technical features
    println!("\n2. Calculating technical features...\n");

    let feature_calc = TechnicalFeatures::new();
    let raw_features = feature_calc.calculate(&klines);

    println!("   Feature matrix shape: {:?}", raw_features.dim());
    println!("   Features:");
    println!("     0: Price returns");
    println!("     1: Log volume");
    println!("     2: High-low range %");
    println!("     3: Close position (0-1)");
    println!("     4: Volume MA ratio");

    // 3. Normalize features
    println!("\n3. Normalizing features...\n");

    let mut normalizer = Normalizer::new();
    let normalized = normalizer.fit_transform(&raw_features);

    println!("   Normalization statistics:");
    for (i, (mean, std)) in normalizer.means().iter().zip(normalizer.stds().iter()).enumerate() {
        println!("     Feature {}: mean={:.6}, std={:.6}", i, mean, std);
    }

    // 4. Create WaveNet model
    println!("\n4. Creating WaveNet-style model...\n");

    let dilation_rates = [1, 2, 4, 8, 16, 32, 64];
    let model = DilatedConvStack::new(5, 32, &dilation_rates);

    println!("   Configuration:");
    println!("     Input channels: 5");
    println!("     Residual channels: 32");
    println!("     Dilation rates: {:?}", dilation_rates);
    println!("     Number of blocks: {}", model.num_blocks());
    println!("     Receptive field: {} timesteps", model.receptive_field());
    println!(
        "     Time coverage: {:.1} hours (at 15min intervals)",
        model.receptive_field() as f64 * 15.0 / 60.0
    );

    // 5. Extract multi-scale features
    println!("\n5. Extracting multi-scale features...\n");

    let features_by_scale = model.extract_features(&normalized);

    println!("   Extracted features from each scale:");
    for (i, features) in features_by_scale.iter().enumerate() {
        let dilation = dilation_rates[i];
        let temporal_scale = dilation * 15; // in minutes
        println!(
            "     Block {} (d={}): shape {:?}, ~{}min patterns",
            i + 1,
            dilation,
            features.dim(),
            temporal_scale
        );
    }

    // 6. Generate predictions
    println!("\n6. Generating predictions...\n");

    let output = model.forward(&normalized);
    let last_pred = model.predict_last(&normalized);

    println!("   Output shape: {:?}", output.dim());
    println!("\n   Last timestep prediction:");
    println!("     Direction score: {:.4}", last_pred[0]);
    println!("     Magnitude score: {:.4}", last_pred[1]);
    println!("     Volatility score: {:.4}", last_pred[2]);

    // Interpret prediction
    let direction = if last_pred[0] > 0.3 {
        "BULLISH"
    } else if last_pred[0] < -0.3 {
        "BEARISH"
    } else {
        "NEUTRAL"
    };

    let confidence = last_pred[0].abs();

    println!("\n   Interpretation:");
    println!("     Signal: {} (confidence: {:.1}%)", direction, confidence * 100.0);

    // 7. Analyze feature importance by scale
    println!("\n7. Analyzing feature importance by scale...\n");

    let mut scale_variances = Vec::new();
    for (i, features) in features_by_scale.iter().enumerate() {
        let variance: f64 = features.iter().map(|x| x.powi(2)).sum::<f64>()
            / features.len() as f64;
        scale_variances.push((i, dilation_rates[i], variance));
    }

    let total_variance: f64 = scale_variances.iter().map(|(_, _, v)| v).sum();

    println!("   Variance contribution by scale:");
    for (i, dilation, variance) in &scale_variances {
        let pct = variance / total_variance * 100.0;
        let bar_len = (pct / 5.0) as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "     d={:3}: {:6.2}% {}",
            dilation, pct, bar
        );
    }

    // 8. Time series of predictions
    println!("\n8. Prediction time series (last 10 points)...\n");

    let seq_len = output.dim().1;
    println!("   Time  | Direction | Magnitude | Volatility | Signal");
    println!("   ------|-----------|-----------|------------|--------");

    for i in (seq_len - 10)..seq_len {
        let dir = output[[0, i]];
        let mag = output[[1, i]];
        let vol = output[[2, i]];
        let signal = if dir > 0.3 { "BUY " } else if dir < -0.3 { "SELL" } else { "HOLD" };

        println!(
            "   t-{:3} |   {:+.4}  |   {:.4}   |   {:.4}    | {}",
            seq_len - i - 1,
            dir,
            mag,
            vol,
            signal
        );
    }

    println!("\n=== Feature Extraction Complete ===");

    Ok(())
}
