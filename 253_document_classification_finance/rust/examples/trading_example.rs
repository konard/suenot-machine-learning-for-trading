use document_classification_finance::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Document Classification in Finance - Trading Example ===\n");

    // ── Step 1: Generate training corpus ─────────────────────────────
    println!("[1] Generating synthetic financial document corpus...\n");

    let documents = generate_synthetic_documents(40);
    println!("  Generated {} documents across {} categories", documents.len(), CATEGORIES.len());
    for cat in CATEGORIES {
        let count = documents.iter().filter(|d| d.label.as_deref() == Some(cat)).count();
        println!("    - {}: {} documents", cat, count);
    }

    // ── Step 2: Build TF-IDF features ────────────────────────────────
    println!("\n[2] Building TF-IDF feature vectors...\n");

    let mut vectorizer = TfIdfVectorizer::new();
    let vectors = vectorizer.fit_transform(&documents);
    let labels: Vec<String> = documents.iter().filter_map(|d| d.label.clone()).collect();

    println!("  Vocabulary size: {} terms", vectorizer.vocab_size());
    println!("  Feature vector dimension: {}", vectors[0].len());

    let top_features = vectorizer.top_features(10);
    println!("  Top 10 most discriminative terms:");
    for (word, idf) in &top_features {
        println!("    - {:20} IDF = {:.4}", word, idf);
    }

    // ── Step 3: Train Naive Bayes classifier ─────────────────────────
    println!("\n[3] Training Naive Bayes classifier...\n");

    let split = vectors.len() * 4 / 5;
    let (train_feat, test_feat) = vectors.split_at(split);
    let (train_labels, test_labels) = labels.split_at(split);

    let mut nb = NaiveBayesClassifier::new(1.0);
    nb.fit(train_feat, train_labels);

    let nb_acc = nb.accuracy(test_feat, test_labels);
    println!("  Naive Bayes accuracy: {:.1}%", nb_acc * 100.0);

    // ── Step 4: Train Softmax classifier ─────────────────────────────
    println!("\n[4] Training Softmax (Logistic Regression) classifier...\n");

    let classes: Vec<String> = CATEGORIES.iter().map(|&s| s.to_string()).collect();
    let mut softmax_clf = DocumentClassifier::new(vectorizer.vocab_size(), classes, 0.01);

    let acc_before = softmax_clf.accuracy(test_feat, test_labels);
    println!("  Accuracy before training: {:.1}%", acc_before * 100.0);

    softmax_clf.train(train_feat, train_labels, 100);
    let acc_after = softmax_clf.accuracy(test_feat, test_labels);
    println!("  Accuracy after training:  {:.1}%", acc_after * 100.0);

    // ── Step 5: Classify new documents ───────────────────────────────
    println!("\n[5] Classifying new financial documents...\n");

    let test_documents = vec![
        Document::new(
            "Company announces record quarterly revenue beating EPS estimates with raised guidance",
            None,
        ),
        Document::new(
            "Firm completes acquisition of rival in major merger deal valued at ten billion",
            None,
        ),
        Document::new(
            "SEC launches investigation into compliance violations imposing significant penalty",
            None,
        ),
        Document::new(
            "Credit rating downgrade following bankruptcy warning and impairment loss recognition",
            None,
        ),
        Document::new(
            "Market sees mixed trading session with tech stocks rising and energy declining",
            None,
        ),
    ];

    for doc in &test_documents {
        let feat = vectorizer.transform_one(doc);
        let (nb_pred, _) = nb.predict(&feat);
        let (sm_pred, sm_conf) = softmax_clf.predict(&feat);

        let text_preview: String = doc.text.chars().take(60).collect();
        println!("  \"{}...\"", text_preview);
        println!(
            "    NB: {:12} | Softmax: {:12} (conf: {:.1}%)",
            nb_pred,
            sm_pred,
            sm_conf * 100.0
        );
    }

    // ── Step 6: Keyword analysis ─────────────────────────────────────
    println!("\n[6] Financial keyword density analysis...\n");

    let detector = FinancialKeywordDetector::new();

    for doc in &test_documents {
        let text_preview: String = doc.text.chars().take(50).collect();
        let densities = detector.detect(doc);
        println!("  \"{}...\"", text_preview);
        for (group, density) in &densities {
            if *density > 0.0 {
                println!("    {:12}: {:.1}% keyword density", group, density * 100.0);
            }
        }
    }

    // ── Step 7: Fetch live data from Bybit ───────────────────────────
    println!("\n[7] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "15", 20).await {
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
            println!("  Could not fetch klines: {}. Using synthetic data.", e);
            Vec::new()
        }
    };

    // ── Step 8: Simulate document-driven trading signals ─────────────
    println!("\n[8] Simulating document classification trading signals...\n");

    let impact_data = generate_price_impact_data();
    println!("  Category price impact analysis:");
    for (category, avg_change, count) in &impact_data {
        let direction = if *avg_change >= 0.0 { "+" } else { "" };
        println!(
            "    {:12}: {}{:.2}% avg impact ({} events)",
            category, direction, avg_change, count
        );
    }

    // Simulate classifying a stream of documents and generating signals
    println!("\n  Trading signal simulation:");
    let signal_docs = vec![
        "Strong earnings beat with revenue surge and raised guidance outlook",
        "Major acquisition announced in strategic merger deal",
        "SEC enforcement action with compliance penalty issued",
        "Company warns of default risk with impairment charges",
        "Markets close mixed in routine trading session",
    ];

    let mut total_pnl = 0.0;
    for text in &signal_docs {
        let doc = Document::new(text, None);
        let feat = vectorizer.transform_one(&doc);
        let (category, confidence) = softmax_clf.predict(&feat);

        let impact = impact_data
            .iter()
            .find(|(c, _, _)| c == &category)
            .map(|(_, change, _)| *change)
            .unwrap_or(0.0);

        let position_size = confidence; // scale position by confidence
        let signal_pnl = impact * position_size;
        total_pnl += signal_pnl;

        let direction = if impact >= 0.0 { "LONG" } else { "SHORT" };
        let text_preview: String = text.chars().take(45).collect();
        println!(
            "    \"{}...\" -> {} ({:.0}% conf) -> {} signal -> P&L: {:+.2}%",
            text_preview,
            category,
            confidence * 100.0,
            direction,
            signal_pnl
        );
    }

    println!("\n  Total simulated P&L: {:+.2}%", total_pnl);

    // Show price context if available
    if !klines.is_empty() {
        let first = &klines[0];
        let last = klines.last().unwrap();
        let price_change = (last.close - first.open) / first.open * 100.0;
        println!(
            "  BTCUSDT price change over period: {:+.2}%",
            price_change
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
