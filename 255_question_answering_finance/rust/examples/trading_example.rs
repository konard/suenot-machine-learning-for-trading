use question_answering_finance::*;

fn main() {
    println!("=== Chapter 255: Financial Question Answering ===\n");

    // ── Build the QA Pipeline ──────────────────────────────────────
    let mut pipeline = FinancialQAPipeline::new();

    // Load sample documents
    let samples = generate_sample_documents();
    for (title, content, source, doc_type, date) in &samples {
        let id = pipeline.add_document(title, content, source, doc_type.clone(), date);
        println!("Added document [{}]: {}", id, title);
    }
    println!("\nDocument store contains {} documents.\n", pipeline.store.len());

    // ── Question Classification Demo ───────────────────────────────
    println!("=== Question Classification ===\n");
    let questions = vec![
        "What was Apple's revenue in Q3 2024?",
        "What is the growth rate of Bitcoin dominance?",
        "Did Apple's revenue increase year over year?",
        "Which had higher revenue, iPhone or Services?",
        "When did Bitcoin reach its all-time high?",
        "What is the total value locked in DeFi?",
        "How much did the S&P 500 return in 2024?",
        "Is the Federal Reserve cutting rates?",
        "Compare Bitcoin and Ethereum market cap.",
        "What is the average lending rate on Aave?",
    ];

    for q in &questions {
        let q_type = pipeline.classifier.classify(q);
        println!("  Q: {}", q);
        println!("  Type: {:?}\n", q_type);
    }

    // ── Full QA Pipeline Demo ──────────────────────────────────────
    println!("=== Full QA Pipeline ===\n");

    let qa_questions = vec![
        "What was Apple's revenue in Q3 2024?",
        "What is the growth rate of Bitcoin dominance?",
        "Did Apple's revenue increase year over year?",
        "What is the total value locked in DeFi protocols?",
        "What was the S&P 500 earnings per share growth?",
        "What is the federal funds rate?",
    ];

    for q in &qa_questions {
        println!("Q: {}", q);
        let answers = pipeline.ask(q);
        for (i, ans) in answers.iter().take(2).enumerate() {
            println!(
                "  Answer {}: {} (confidence: {:.2}, type: {:?}, doc: {:?})",
                i + 1,
                ans.text,
                ans.confidence,
                ans.question_type,
                ans.source_doc_id,
            );
        }
        println!();
    }

    // ── Numerical Reasoning Demo ───────────────────────────────────
    println!("=== Numerical Reasoning ===\n");
    let reasoner = NumericalReasoner::new();

    let context = "Revenue was $100 million in 2023 and $120 million in 2024.";
    println!("Context: {}", context);

    let nums = reasoner.parse_numbers(context);
    println!("Parsed numbers: {:?}", nums);

    let ops = vec![
        ("Growth rate", NumericalOp::GrowthRate),
        ("Difference", NumericalOp::Difference),
        ("Ratio", NumericalOp::Ratio),
        ("Sum", NumericalOp::Sum),
        ("Average", NumericalOp::Average),
    ];

    for (name, op) in &ops {
        if let Some(result) = reasoner.compute(op, &nums) {
            println!("  {}: {:.4}", name, result);
        }
    }

    println!();

    // ── Document Retrieval Demo ────────────────────────────────────
    println!("=== Document Retrieval ===\n");

    let search_queries = vec![
        "Bitcoin ETF trading volume",
        "Apple earnings revenue",
        "Federal Reserve interest rate",
        "DeFi lending protocol",
    ];

    for query in &search_queries {
        println!("Search: \"{}\"", query);
        let results = pipeline.store.search(query, 3);
        for (doc_id, score) in &results {
            if let Some(doc) = pipeline.store.get_document(*doc_id) {
                println!("  [{:.4}] {}", score, doc.title);
            }
        }
        println!();
    }

    // ── Confidence Calibration Demo ────────────────────────────────
    println!("=== Confidence Calibration ===\n");

    let eval_data = generate_qa_evaluation_data(1000);
    let ece = expected_calibration_error(&eval_data, 10);
    println!("Expected Calibration Error (ECE) on synthetic data: {:.4}", ece);
    println!("(Lower is better; 0.0 = perfectly calibrated)\n");

    // ── Bybit Integration Note ─────────────────────────────────────
    println!("=== Bybit API Integration ===\n");
    println!("The BybitClient supports fetching live market data:");
    println!("  - Kline (candlestick) data via /v5/market/kline");
    println!("  - Order book snapshots via /v5/market/orderbook");
    println!();
    println!("Example usage (requires async runtime):");
    println!("  let client = BybitClient::new();");
    println!("  let klines = client.get_klines(\"BTCUSDT\", \"60\", 10).await?;");
    println!("  let (bids, asks) = client.get_orderbook(\"BTCUSDT\", 25).await?;");
    println!();
    println!("This data can be added to the DocumentStore for real-time QA.");

    println!("\n=== Done ===");
}
