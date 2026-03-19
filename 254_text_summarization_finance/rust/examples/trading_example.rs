use text_summarization_finance::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Text Summarization Finance - Trading Example ===\n");

    // ── Step 1: Fetch live data from Bybit ──────────────────────────
    println!("[1] Fetching BTCUSDT data from Bybit V5 API...\n");

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
            println!("  Could not fetch klines: {}. Using synthetic data.", e);
            generate_synthetic_klines(50, 50000.0)
        }
    };

    // ── Step 2: Create sample financial documents ───────────────────
    println!("\n[2] Creating sample financial documents...\n");

    let documents = generate_synthetic_documents();
    println!("  Generated {} financial documents", documents.len());
    for (i, doc) in documents.iter().enumerate() {
        let word_count = doc.split_whitespace().count();
        println!("  Document {}: {} words", i + 1, word_count);
    }

    // ── Step 3: Compute TF-IDF scores ───────────────────────────────
    println!("\n[3] Computing TF-IDF scores...\n");

    let doc_refs: Vec<&str> = documents.iter().map(|d| d.as_str()).collect();
    let vectorizer = TfIdfVectorizer::fit(&doc_refs);

    println!("  Vocabulary size: {} words", vectorizer.vocab_size());
    println!("  Corpus size: {} documents", vectorizer.num_documents());

    // Show TF-IDF vectors for first document
    let tfidf_vec = vectorizer.transform(&documents[0]);
    let non_zero: usize = tfidf_vec.iter().filter(|&&v| v > 0.0).count();
    println!("  Document 1 TF-IDF vector: {} non-zero entries out of {}", non_zero, tfidf_vec.len());

    // ── Step 4: Score and rank sentences ─────────────────────────────
    println!("\n[4] Scoring and ranking sentences...\n");

    let scorer = SentenceScorer::new();
    let sentences = sentence_split(&documents[0]);
    println!("  Document 1 has {} sentences", sentences.len());

    let scores = scorer.score_sentences(&sentences, &vectorizer);
    let mut ranked = scores.clone();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Top 3 sentences by score:");
    for (i, (idx, score)) in ranked.iter().take(3).enumerate() {
        let preview: String = sentences[*idx].chars().take(60).collect();
        println!("    {}. [score={:.4}] {}...", i + 1, score, preview);
    }

    // Show feature breakdown for top sentence
    let (top_idx, _) = ranked[0];
    let top_sent = &sentences[top_idx];
    println!("\n  Feature breakdown for top sentence:");
    println!("    Position score:   {:.4}", scorer.position_score(top_idx, sentences.len()));
    println!("    Length score:     {:.4}", scorer.length_score(top_sent));
    println!("    Financial density:{:.4}", scorer.financial_keyword_density(top_sent));
    println!("    Numerical density:{:.4}", scorer.numerical_density(top_sent));
    println!("    TF-IDF score:    {:.4}", vectorizer.sentence_score(top_sent));

    // ── Step 5: Generate extractive summaries ────────────────────────
    println!("\n[5] Generating extractive summaries...\n");

    let summarizer = TextSummarizer::new(2);

    for (i, doc) in documents.iter().enumerate() {
        let summary = summarizer.summarize(doc, &vectorizer);
        let original_words = doc.split_whitespace().count();
        let summary_words = summary.split_whitespace().count();
        let compression = 1.0 - (summary_words as f64 / original_words as f64);

        println!("  Document {} summary ({} -> {} words, {:.0}% compression):",
            i + 1, original_words, summary_words, compression * 100.0);

        let preview: String = summary.chars().take(120).collect();
        println!("    {}...\n", preview);
    }

    // ── Step 6: Compute sentiment of summaries ──────────────────────
    println!("[6] Computing sentiment of summaries...\n");

    let sentiment_scorer = SentimentScorer::new();

    for (i, doc) in documents.iter().enumerate() {
        let summary = summarizer.summarize(doc, &vectorizer);
        let sentiment = sentiment_scorer.score(&summary);
        let classification = sentiment_scorer.classify(&summary);

        println!("  Document {}: sentiment={:+.4} ({})", i + 1, sentiment, classification);
    }

    // ── Step 7: Generate trading signals based on sentiment ─────────
    println!("\n[7] Generating trading signals...\n");

    let trader = SummaryTrader::default();
    let results = trader.process_documents(&doc_refs, &summarizer, &vectorizer, &sentiment_scorer);

    println!("  {:>4} | {:>10} | {:>6} | Summary Preview", "Doc", "Sentiment", "Signal");
    println!("  {}", "-".repeat(60));

    for (i, (summary, sentiment, signal)) in results.iter().enumerate() {
        let preview: String = summary.chars().take(35).collect();
        println!("  {:>4} | {:>+10.4} | {:>6} | {}...", i + 1, sentiment, signal, preview);
    }

    // ── Step 8: Show prediction results with price data ─────────────
    println!("\n[8] Combining signals with price data...\n");

    // Compute ROUGE scores between summaries
    let summary1 = summarizer.summarize(&documents[0], &vectorizer);
    let summary2 = summarizer.summarize(&documents[3], &vectorizer);
    println!("  ROUGE scores between Document 1 and Document 4 summaries:");
    println!("    ROUGE-1: {:.4}", rouge_1(&summary1, &summary2));
    println!("    ROUGE-2: {:.4}", rouge_2(&summary1, &summary2));
    println!("    ROUGE-L: {:.4}", rouge_l(&summary1, &summary2));

    // Aggregate signal from all documents
    let avg_sentiment: f64 = results.iter().map(|(_, s, _)| s).sum::<f64>() / results.len() as f64;
    let aggregate_signal = trader.signal(avg_sentiment);

    println!("\n  Aggregate sentiment: {:+.4}", avg_sentiment);
    println!("  Aggregate signal: {}", aggregate_signal);

    // Show recent price action
    println!("\n  Recent price action (last 5 bars):");
    let start = klines.len().saturating_sub(5);
    for kline in &klines[start..] {
        let direction = if kline.close >= kline.open { "UP  " } else { "DOWN" };
        let change_pct = (kline.close - kline.open) / kline.open * 100.0;
        println!(
            "    {} | O={:.2} C={:.2} ({:+.2}%) V={:.2}",
            direction, kline.open, kline.close, change_pct, kline.volume
        );
    }

    // Price trend
    if klines.len() >= 2 {
        let first_close = klines[0].close;
        let last_close = klines.last().unwrap().close;
        let trend = (last_close - first_close) / first_close * 100.0;
        println!("\n  Price trend over {} bars: {:+.2}%", klines.len(), trend);
    }

    // Summary statistics
    let buy_count = results.iter().filter(|(_, _, s)| *s == TradingSignal::Buy).count();
    let sell_count = results.iter().filter(|(_, _, s)| *s == TradingSignal::Sell).count();
    let hold_count = results.iter().filter(|(_, _, s)| *s == TradingSignal::Hold).count();

    println!("\n  Signal distribution:");
    println!("    BUY:  {} documents", buy_count);
    println!("    SELL: {} documents", sell_count);
    println!("    HOLD: {} documents", hold_count);

    // Cosine similarity between document TF-IDF vectors
    println!("\n  Document similarity matrix (cosine similarity):");
    let tfidf_vecs: Vec<Vec<f64>> = documents
        .iter()
        .map(|d| vectorizer.transform(d))
        .collect();

    print!("        ");
    for i in 0..documents.len().min(5) {
        print!("  Doc{}  ", i + 1);
    }
    println!();

    for i in 0..documents.len().min(5) {
        print!("  Doc{} ", i + 1);
        for j in 0..documents.len().min(5) {
            let sim = cosine_similarity(&tfidf_vecs[i], &tfidf_vecs[j]);
            print!("  {:.3}  ", sim);
        }
        println!();
    }

    println!("\n=== Done ===");
    Ok(())
}
