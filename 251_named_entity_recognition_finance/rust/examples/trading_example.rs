use named_entity_recognition_finance::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Named Entity Recognition for Finance - Trading Example ===\n");

    // ── Step 1: Initialize NER and fetch Bybit tickers ─────────────
    println!("[1] Initializing Financial NER system...\n");

    let mut ner = FinancialNER::new();

    let client = BybitClient::new();
    match client.get_tickers().await {
        Ok(tickers) => {
            let symbols: Vec<String> = tickers.iter().map(|t| t.symbol.clone()).collect();
            println!("  Fetched {} trading pairs from Bybit", symbols.len());
            if let Some(btc) = tickers.iter().find(|t| t.symbol == "BTCUSDT") {
                println!("  BTCUSDT last price: ${}", btc.last_price);
            }
            ner.add_tickers(&symbols);
            println!("  Added Bybit symbols to ticker gazetteer");
        }
        Err(e) => {
            println!("  Could not fetch Bybit tickers: {}. Using default gazetteer.", e);
        }
    }

    // ── Step 2: Process sample financial texts ─────────────────────
    println!("\n[2] Processing financial texts with NER...\n");

    let texts = sample_financial_texts();
    for (i, text) in texts.iter().enumerate() {
        println!("  Text {}: \"{}\"", i + 1, text);
        let entities = ner.extract_entities(text);
        for entity in &entities {
            println!("    {} {}", entity.entity_type, entity);
        }
        println!();
    }

    // ── Step 3: Fetch live market data and generate text ──────────
    println!("[3] Generating text from live market data...\n");

    match client.get_klines("BTCUSDT", "1", 5).await {
        Ok(klines) => {
            if let Some(last) = klines.last() {
                let generated_text = format!(
                    "BTCUSDT traded at ${:.2} on Bybit with a high of ${:.2} and volume of {:.2} BTC in the last minute",
                    last.close, last.high, last.volume
                );
                println!("  Generated: \"{}\"", generated_text);
                let entities = ner.extract_entities(&generated_text);
                for entity in &entities {
                    println!("    {} {}", entity.entity_type, entity);
                }
            }
        }
        Err(e) => {
            println!("  Could not fetch klines: {}. Using synthetic text.", e);
            let synthetic = "BTCUSDT traded at $67500.00 on Bybit with volume of 1234.56 BTC";
            println!("  Synthetic: \"{}\"", synthetic);
            let entities = ner.extract_entities(synthetic);
            for entity in &entities {
                println!("    {} {}", entity.entity_type, entity);
            }
        }
    }

    // ── Step 4: Demonstrate BIO tagging ───────────────────────────
    println!("\n[4] BIO Tagging demonstration...\n");

    let demo_text = "Goldman Sachs reported $2.5 billion revenue in Q3 2024";
    println!("  Text: \"{}\"", demo_text);
    println!();

    let tokens = ner.tokenize(demo_text);
    let tags = ner.predict(&tokens);

    println!("  {:>15} {:>12}", "Token", "BIO Tag");
    println!("  {:>15} {:>12}", "-----", "-------");
    for (token, tag) in tokens.iter().zip(tags.iter()) {
        println!("  {:>15} {:>12}", token.text, tag);
    }

    // ── Step 5: Train NER classifier ──────────────────────────────
    println!("\n[5] Training NER Classifier...\n");

    let training_data = generate_training_data(2000);
    let (train, test) = training_data.split_at(1600);

    let mut classifier = NERClassifier::new(10, 8, 0.01);
    println!("  Accuracy before training: {:.2}%", classifier.accuracy(test) * 100.0);

    classifier.train(&train.to_vec(), 100);
    let acc = classifier.accuracy(test);
    println!("  Accuracy after training:  {:.2}%", acc * 100.0);

    // Show per-class predictions
    println!("\n  Per-class test predictions:");
    let mut class_counts = vec![0usize; 8];
    let mut class_correct = vec![0usize; 8];
    for (features, label) in test {
        class_counts[*label] += 1;
        let (pred, _) = classifier.predict(features);
        if pred == *label {
            class_correct[*label] += 1;
        }
    }
    for c in 0..8 {
        if class_counts[c] > 0 {
            let class_acc = class_correct[c] as f64 / class_counts[c] as f64;
            println!(
                "    {:>14}: {}/{} correct ({:.1}%)",
                NERClassifier::class_name(c),
                class_correct[c],
                class_counts[c],
                class_acc * 100.0
            );
        }
    }

    // ── Step 6: Live entity extraction with classifier ────────────
    println!("\n[6] Live Entity Extraction...\n");

    let live_text = "Tesla acquired SolarCity for $2.6 billion and TSLA rose 8.5% in November 2016";
    println!("  Text: \"{}\"", live_text);
    let entities = ner.extract_entities(live_text);
    println!("  Extracted {} entities:", entities.len());
    for entity in &entities {
        println!("    {}", entity);
    }

    println!("\n=== Done ===");
    Ok(())
}
