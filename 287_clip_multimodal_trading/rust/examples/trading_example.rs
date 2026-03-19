use clip_multimodal_trading::*;
use ndarray::Array2;

fn main() -> anyhow::Result<()> {
    println!("=== CLIP Multimodal Trading Example ===\n");

    // ── 1. Initialize CLIP model ──────────────────────────────────────
    let embedding_dim = 64;
    let model = CLIPModel::new(embedding_dim, 42);
    println!("Initialized CLIP model with embedding_dim={}", embedding_dim);
    println!("  Temperature: {}", model.temperature);

    // ── 2. Fetch BTCUSDT data from Bybit ──────────────────────────────
    println!("\n--- Fetching BTCUSDT kline data from Bybit ---");
    let client = BybitClient::new();
    let price_data = match client.fetch_klines("BTCUSDT", "15", 50) {
        Ok(klines) => {
            println!("Fetched {} klines from Bybit", klines.len());
            if let Some(first) = klines.first() {
                println!(
                    "  First candle: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
                    first.open, first.high, first.low, first.close, first.volume
                );
            }
            if let Some(last) = klines.last() {
                println!(
                    "  Last candle:  O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
                    last.open, last.high, last.low, last.close, last.volume
                );
            }
            BybitClient::klines_to_array(&klines)
        }
        Err(e) => {
            println!("Could not fetch from Bybit ({}), using synthetic data", e);
            generate_synthetic_ohlcv("bullish", 50, 100)
        }
    };

    // ── 3. Encode price sequence ──────────────────────────────────────
    println!("\n--- Encoding price sequence ---");
    let price_embedding = model.price_encoder.encode(&price_data);
    println!(
        "Price embedding (first 8 dims): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        price_embedding[0], price_embedding[1], price_embedding[2], price_embedding[3],
        price_embedding[4], price_embedding[5], price_embedding[6], price_embedding[7]
    );

    // ── 4. Encode market sentiment text snippets ──────────────────────
    println!("\n--- Encoding text snippets ---");
    let text_snippets = vec![
        "Bitcoin surges past resistance on strong institutional buying pressure",
        "Crypto market crashes as regulatory fears spike and traders panic sell",
        "BTC consolidates in tight range with declining volume awaiting catalyst",
        "Wild volatility as whales manipulate the order book in both directions",
        "Steady accumulation phase with Bitcoin slowly grinding higher each day",
        "Sharp correction after overextended rally triggers mass liquidations",
    ];

    for snippet in &text_snippets {
        let emb = model.text_encoder.encode(snippet);
        let sim = cosine_similarity(&price_embedding, &emb);
        println!("  sim={:.4}  \"{}\"", sim, &snippet[..60.min(snippet.len())]);
    }

    // ── 5. Zero-shot regime classification ────────────────────────────
    println!("\n--- Zero-shot regime classification ---");
    let regime_labels = default_regime_labels();
    let classification = model.zero_shot_classify(&price_data, &regime_labels);

    println!("Current market regime classification:");
    for (i, (label, score)) in classification.iter().enumerate() {
        let marker = if i == 0 { " <-- BEST MATCH" } else { "" };
        println!("  [{:.4}] {}{}", score, label, marker);
    }

    // ── 6. Cross-modal retrieval with synthetic regimes ───────────────
    println!("\n--- Cross-modal retrieval: text -> price sequences ---");
    let regime_names = ["bullish", "bearish", "consolidation", "volatile"];
    let synthetic_sequences: Vec<Array2<f64>> = regime_names
        .iter()
        .map(|&r| generate_synthetic_ohlcv(r, 30, 77))
        .collect();

    let query_text = "Bitcoin price rallies sharply with massive green candles";
    println!("Query: \"{}\"", query_text);

    let retrieval_results = model.retrieve_prices_by_text(query_text, &synthetic_sequences);
    for (idx, score) in &retrieval_results {
        println!("  [{:.4}] regime: {}", score, regime_names[*idx]);
    }

    // ── 7. Similarity matrix visualization ────────────────────────────
    println!("\n--- Similarity matrix (text x price regimes) ---");
    let short_labels = ["bullish text", "bearish text", "consolidation text", "volatile text"];
    let short_texts: Vec<&str> = vec![
        "strong bullish trend with increasing volume and momentum",
        "bearish breakdown with panic selling and high volatility",
        "range-bound consolidation with low volatility",
        "volatile whipsaw with no clear direction",
    ];

    let text_embeddings = model.text_encoder.encode_batch(&short_texts);
    let price_embeddings = model.price_encoder.encode_batch(&synthetic_sequences);
    let sim_matrix = CLIPModel::cosine_similarity_matrix(&text_embeddings, &price_embeddings);

    print!("{:>25}", "");
    for name in &regime_names {
        print!("{:>14}", name);
    }
    println!();

    for i in 0..short_labels.len() {
        print!("{:>25}", short_labels[i]);
        for j in 0..regime_names.len() {
            print!("{:>14.4}", sim_matrix[[i, j]]);
        }
        println!();
    }

    // ── 8. Contrastive loss computation ───────────────────────────────
    println!("\n--- Contrastive loss computation ---");
    let loss = model.info_nce_loss(&text_embeddings, &price_embeddings);
    println!("InfoNCE loss on aligned text-price pairs: {:.4}", loss);
    println!("(Lower loss = better alignment between text and price embeddings)");

    println!("\n=== Example complete ===");
    Ok(())
}
