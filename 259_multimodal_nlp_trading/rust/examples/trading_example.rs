use multimodal_nlp_trading::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Multimodal NLP Trading - Trading Example ===\n");

    // ── Step 1: Fetch live kline data from Bybit ────────────────────
    println!("[1] Fetching BTCUSDT klines from Bybit V5 API...\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "1", 50).await {
        Ok(k) => {
            println!("  Fetched {} kline bars", k.len());
            if let Some(last) = k.last() {
                println!(
                    "  Latest bar: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
                    last.open, last.high, last.low, last.close, last.volume
                );
            }
            k
        }
        Err(e) => {
            println!("  Could not fetch klines: {}. Falling back to synthetic data.", e);
            Vec::new()
        }
    };

    let orderbook = match client.get_orderbook("BTCUSDT", 5).await {
        Ok((bids, asks)) => {
            println!(
                "  Order book: {} bid levels, {} ask levels",
                bids.len(),
                asks.len()
            );
            if let (Some(best_bid), Some(best_ask)) = (bids.first(), asks.first()) {
                println!(
                    "  Best bid: {:.2} (qty {:.4}), Best ask: {:.2} (qty {:.4})",
                    best_bid.0, best_bid.1, best_ask.0, best_ask.1
                );
            }
            Some((bids, asks))
        }
        Err(e) => {
            println!("  Could not fetch orderbook: {}. Falling back to synthetic data.", e);
            None
        }
    };

    // ── Step 2: Build candlestick data for the visual modality ──────
    println!("\n[2] Preparing candlestick data for visual feature extraction...\n");

    let candles: Vec<(f64, f64, f64, f64, f64)> = if !klines.is_empty() {
        klines
            .iter()
            .map(|k| (k.open, k.high, k.low, k.close, k.volume))
            .collect()
    } else {
        println!("  Using {} synthetic candles.", 50);
        generate_synthetic_candles(50, 50_000.0)
    };

    println!("  Total candles available: {}", candles.len());

    let extractor = VisualFeatureExtractor::new();
    let visual_features = extractor.extract_features(&candles);
    println!("  Visual features (7-dim):");
    println!("    trend_direction  : {:.6}", visual_features[0]);
    println!("    avg_body_ratio   : {:.6}", visual_features[1]);
    println!("    avg_upper_shadow : {:.6}", visual_features[2]);
    println!("    avg_lower_shadow : {:.6}", visual_features[3]);
    println!("    volume_trend     : {:.6}", visual_features[4]);
    println!("    bullish_ratio    : {:.6}", visual_features[5]);
    println!("    volatility       : {:.6}", visual_features[6]);

    // Detect engulfing patterns on the last 10 candles
    let recent = &candles[candles.len().saturating_sub(10)..];
    let engulfing = extractor.detect_engulfing(recent);
    let bullish_eng: usize = engulfing.iter().filter(|&&x| x == 1).count();
    let bearish_eng: usize = engulfing.iter().filter(|&&x| x == -1).count();
    println!(
        "\n  Engulfing patterns (last {} bars): {} bullish, {} bearish",
        recent.len(),
        bullish_eng,
        bearish_eng
    );

    // ── Step 3: Encode text modality ────────────────────────────────
    println!("\n[3] Encoding text modality...\n");

    let texts = generate_sample_texts();
    let text_encoder = TextEncoder::new();

    for (i, text) in texts.iter().enumerate() {
        let score = text_encoder.sentiment_score(text);
        println!(
            "  [{:02}] score={:+.4}  \"{}\"",
            i + 1,
            score,
            if text.len() > 60 { &text[..60] } else { text }
        );
    }

    // Use the first text as the primary news input for the signal
    let primary_text = texts[0];
    let text_features = text_encoder.encode(primary_text);
    println!("\n  Primary text encoding (5-dim):");
    println!("    positive_count   : {}", text_features[0]);
    println!("    negative_count   : {}", text_features[1]);
    println!("    sentiment_score  : {:.4}", text_features[2]);
    println!("    word_count       : {}", text_features[3]);
    println!("    avg_word_length  : {:.4}", text_features[4]);

    // ── Step 4: Encode numerical modality ───────────────────────────
    println!("\n[4] Encoding numerical market features...\n");

    // Derive numerical features from live data where available
    let (ret, vol_ratio, spread_val) = if !candles.is_empty() {
        let n = candles.len();
        let ret = if n >= 2 && candles[n - 2].3 != 0.0 {
            (candles[n - 1].3 - candles[n - 2].3) / candles[n - 2].3
        } else {
            0.0
        };
        let avg_vol: f64 = candles.iter().map(|c| c.4).sum::<f64>() / n as f64;
        let vol_ratio = if avg_vol > 0.0 {
            candles[n - 1].4 / avg_vol
        } else {
            1.0
        };
        let spread = if let Some((ref bids, ref asks)) = orderbook {
            if let (Some(bb), Some(ba)) = (bids.first(), asks.first()) {
                (ba.0 - bb.0) / ba.0
            } else {
                0.001
            }
        } else {
            0.001
        };
        (ret, vol_ratio, spread)
    } else {
        (0.005, 1.1, 0.001)
    };

    let market_data = vec![ret, vol_ratio, visual_features[6], visual_features[0], spread_val];
    println!("  Numerical features (5-dim):");
    println!("    return           : {:+.6}", market_data[0]);
    println!("    volume_ratio     : {:.6}", market_data[1]);
    println!("    volatility       : {:.6}", market_data[2]);
    println!("    momentum         : {:+.6}", market_data[3]);
    println!("    spread           : {:.6}", market_data[4]);

    let num_encoder = NumericalEncoder::new(5, 8);
    let num_embedding = num_encoder.encode(&market_data);
    println!("\n  Numerical embedding (8-dim, tanh):");
    for (i, v) in num_embedding.iter().enumerate() {
        println!("    [{}] {:+.6}", i, v);
    }

    // ── Step 5: Fuse modalities with attention ──────────────────────
    println!("\n[5] Fusing modalities with attention mechanism...\n");

    let fusion = MultimodalFusion::new(5, 8, 7);
    let (fused, attn_weights) =
        fusion.fuse_with_weights(&text_features, &num_embedding, &visual_features);

    println!("  Attention weights (softmax):");
    println!("    text      : {:.4}  ({:.1}%)", attn_weights[0], attn_weights[0] * 100.0);
    println!("    numerical : {:.4}  ({:.1}%)", attn_weights[1], attn_weights[1] * 100.0);
    println!("    visual    : {:.4}  ({:.1}%)", attn_weights[2], attn_weights[2] * 100.0);
    println!("\n  Fused vector ({}-dim):", fused.len());
    for (i, v) in fused.iter().enumerate() {
        println!("    [{}] {:+.6}", i, v);
    }

    // ── Step 6: Generate trading signal ─────────────────────────────
    println!("\n[6] Generating trading signal...\n");

    let generator = TradingSignalGenerator::new();
    let signal = generator.generate_signal(primary_text, &market_data, &candles);

    println!("  ┌─────────────────────────────────┐");
    println!(
        "  │  Direction  : {:>18}  │",
        if signal.direction { "UP ▲" } else { "DOWN ▼" }
    );
    println!(
        "  │  Confidence : {:>16.2}%  │",
        signal.confidence * 100.0
    );
    println!("  ├─────────────────────────────────┤");
    println!("  │  Modality weights:              │");
    println!(
        "  │    text      : {:>14.4}     │",
        signal.modality_weights[0]
    );
    println!(
        "  │    numerical : {:>14.4}     │",
        signal.modality_weights[1]
    );
    println!(
        "  │    visual    : {:>14.4}     │",
        signal.modality_weights[2]
    );
    println!("  └─────────────────────────────────┘");

    // ── Step 7: Batch signals across sample texts ────────────────────
    println!("\n[7] Batch signals for all sample texts...\n");

    let market_data_samples = generate_synthetic_market_data(texts.len());

    for (i, (text, mdata)) in texts.iter().zip(market_data_samples.iter()).enumerate() {
        let sig = generator.generate_signal(text, mdata, &candles);
        println!(
            "  [{:02}] {} ({:.1}%)  text_w={:.2} num_w={:.2} vis_w={:.2}  \"{}\"",
            i + 1,
            if sig.direction { "UP  " } else { "DOWN" },
            sig.confidence * 100.0,
            sig.modality_weights[0],
            sig.modality_weights[1],
            sig.modality_weights[2],
            if text.len() > 50 { &text[..50] } else { text }
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
