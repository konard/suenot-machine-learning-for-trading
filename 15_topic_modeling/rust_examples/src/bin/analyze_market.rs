//! Market Analysis with Topic Modeling
//!
//! This example demonstrates a complete pipeline for:
//! - Fetching live market data and announcements from Bybit
//! - Preprocessing and analyzing text content
//! - Extracting market themes using topic modeling
//! - Correlating topics with price movements

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use topic_modeling::api::bybit::{BybitClient, Kline, MarketDocument};
use topic_modeling::models::lda::{LdaConfig, LDA};
use topic_modeling::models::lsi::LSI;
use topic_modeling::preprocessing::tokenizer::Tokenizer;
use topic_modeling::preprocessing::vectorizer::{CountVectorizer, TfIdfVectorizer};

fn main() -> Result<()> {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Cryptocurrency Market Analysis with Topic Modeling      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Initialize Bybit client
    let client = BybitClient::new();

    // Step 1: Fetch market data
    println!("📊 Fetching Market Data...\n");

    let market_overview = fetch_market_overview(&client)?;
    print_market_overview(&market_overview);

    // Step 2: Fetch and analyze announcements
    println!("\n📰 Fetching Recent Announcements...\n");

    let documents = fetch_or_generate_documents(&client)?;
    println!("Loaded {} documents for analysis\n", documents.len());

    // Step 3: Preprocess text
    println!("🔧 Preprocessing Documents...\n");

    let tokenizer = Tokenizer::for_crypto().min_length(3);
    let texts: Vec<String> = documents.iter().map(|d| d.full_text()).collect();
    let tokenized: Vec<Vec<String>> = texts.iter().map(|t| tokenizer.tokenize(t)).collect();

    // Build vocabulary statistics
    let vocab = tokenizer.build_vocabulary(&tokenized);
    let vocab_freq = tokenizer.vocabulary_with_frequencies(&tokenized);

    println!("Vocabulary Statistics:");
    println!("  Total unique terms: {}", vocab.len());
    println!("  Top terms by document frequency:");
    for (term, freq) in vocab_freq.iter().take(10) {
        println!("    {}: {} documents", term, freq);
    }

    // Step 4: Compare LSI and LDA
    println!("\n🔬 Topic Analysis Comparison\n");
    println!("{}", "=".repeat(60));

    // LSI Analysis
    println!("\n📐 LSI (Latent Semantic Indexing) Results:\n");

    let mut tfidf_vectorizer = TfIdfVectorizer::new()
        .min_df(1)
        .max_df_ratio(0.9)
        .max_features(200);

    let tfidf_matrix = tfidf_vectorizer.fit_transform(&tokenized);
    let tfidf_vocab = tfidf_vectorizer.get_vocabulary().clone();
    let tfidf_terms: Vec<String> = (0..tfidf_vectorizer.vocabulary_size())
        .filter_map(|i| tfidf_vectorizer.get_term(i).cloned())
        .collect();

    let n_topics = 4;
    let mut lsi = LSI::new(n_topics)?;
    lsi.fit(&tfidf_matrix, tfidf_vocab, tfidf_terms)?;

    let lsi_topics = lsi.get_topics(6)?;
    for topic in &lsi_topics {
        println!("  Topic {}: (variance: {:.1}%)", topic.index, topic.variance_ratio * 100.0);
        let words: Vec<String> = topic
            .top_words
            .iter()
            .take(5)
            .map(|(w, _)| w.clone())
            .collect();
        println!("    Keywords: {}\n", words.join(", "));
    }

    // LDA Analysis
    println!("📊 LDA (Latent Dirichlet Allocation) Results:\n");

    let mut count_vectorizer = CountVectorizer::new()
        .min_df(1)
        .max_df_ratio(0.9)
        .max_features(200);

    let count_matrix = count_vectorizer.fit_transform(&tokenized);
    let count_vocab = count_vectorizer.get_vocabulary().clone();
    let count_terms: Vec<String> = (0..count_vectorizer.vocabulary_size())
        .filter_map(|i| count_vectorizer.get_term(i).cloned())
        .collect();

    let lda_config = LdaConfig::new(n_topics)
        .alpha(0.1)
        .beta(0.01)
        .n_iterations(300)
        .burn_in(50)
        .random_seed(42);

    let mut lda = LDA::new(lda_config)?;
    lda.fit(&count_matrix, count_vocab, count_terms)?;

    let lda_topics = lda.get_topics(6)?;
    for topic in &lda_topics {
        println!("  Topic {}: (prevalence: {:.1}%)", topic.index, topic.prevalence * 100.0);
        let words: Vec<String> = topic
            .top_words
            .iter()
            .take(5)
            .map(|(w, _)| w.clone())
            .collect();
        println!("    Keywords: {}\n", words.join(", "));
    }

    // Step 5: Document categorization
    println!("{}", "=".repeat(60));
    println!("\n📑 Document Categorization\n");

    let dominant_topics = lda.dominant_topics()?;
    let doc_topics = lda.get_document_topics()?;

    // Group documents by topic
    let mut topic_docs: HashMap<usize, Vec<(usize, &MarketDocument, f64)>> = HashMap::new();

    for (i, doc) in documents.iter().enumerate() {
        let topic = dominant_topics[i];
        let prob = doc_topics[[i, topic]];
        topic_docs.entry(topic).or_default().push((i, doc, prob));
    }

    // Print documents grouped by topic
    for topic_idx in 0..n_topics {
        if let Some(docs) = topic_docs.get(&topic_idx) {
            let topic_info = &lda_topics[topic_idx];
            let keywords: Vec<String> = topic_info
                .top_words
                .iter()
                .take(3)
                .map(|(w, _)| w.clone())
                .collect();

            println!("Topic {} ({}):", topic_idx, keywords.join(", "));
            println!("{}", "-".repeat(40));

            for (_, doc, prob) in docs.iter().take(3) {
                let preview: String = doc.title.chars().take(50).collect();
                println!("  [{:.0}%] {}", prob * 100.0, preview);
            }
            println!();
        }
    }

    // Step 6: Symbol-topic correlation
    println!("{}", "=".repeat(60));
    println!("\n🔗 Symbol-Topic Correlation\n");

    let mut symbol_topics: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, doc) in documents.iter().enumerate() {
        for symbol in &doc.symbols {
            symbol_topics
                .entry(symbol.clone())
                .or_default()
                .push(dominant_topics[i]);
        }
    }

    for (symbol, topics) in &symbol_topics {
        if !topics.is_empty() {
            // Count topic frequencies
            let mut topic_counts: HashMap<usize, usize> = HashMap::new();
            for &topic in topics {
                *topic_counts.entry(topic).or_insert(0) += 1;
            }

            // Find dominant topic for this symbol
            let dominant = topic_counts.iter().max_by_key(|(_, &count)| count);

            if let Some((&topic, &count)) = dominant {
                let topic_info = &lda_topics[topic];
                let keywords: Vec<String> = topic_info
                    .top_words
                    .iter()
                    .take(3)
                    .map(|(w, _)| w.clone())
                    .collect();

                println!(
                    "  {} -> Topic {} ({}) - {} mentions",
                    symbol,
                    topic,
                    keywords.join(", "),
                    count
                );
            }
        }
    }

    // Step 7: Model quality metrics
    println!("\n{}", "=".repeat(60));
    println!("\n📈 Model Quality Metrics\n");

    let perplexity = lda.perplexity(&count_matrix)?;
    println!("LDA Perplexity: {:.2}", perplexity);

    if let Some(avg_coherence) = lda.average_coherence() {
        println!("LDA Average Coherence: {:.4}", avg_coherence);
    }

    let explained_var = lsi.explained_variance_ratio()?;
    println!("LSI Explained Variance: {:.2}%", explained_var * 100.0);

    // Step 8: Trading insights
    println!("\n{}", "=".repeat(60));
    println!("\n💡 Trading Insights\n");

    generate_trading_insights(&lda_topics, &market_overview);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Analysis Complete                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Market overview structure
struct MarketOverview {
    btc_price: f64,
    btc_change: f64,
    eth_price: f64,
    eth_change: f64,
    top_gainers: Vec<(String, f64)>,
    top_losers: Vec<(String, f64)>,
}

/// Fetch market overview from Bybit
fn fetch_market_overview(client: &BybitClient) -> Result<MarketOverview> {
    let tickers = client.get_tickers("spot", None).unwrap_or_default();

    let mut btc_price = 0.0;
    let mut btc_change = 0.0;
    let mut eth_price = 0.0;
    let mut eth_change = 0.0;
    let mut changes: Vec<(String, f64)> = Vec::new();

    for ticker in &tickers {
        let price: f64 = ticker.last_price.parse().unwrap_or(0.0);
        let change: f64 = ticker.price_24h_pcnt.parse().unwrap_or(0.0) * 100.0;

        if ticker.symbol == "BTCUSDT" {
            btc_price = price;
            btc_change = change;
        } else if ticker.symbol == "ETHUSDT" {
            eth_price = price;
            eth_change = change;
        }

        if ticker.symbol.ends_with("USDT") {
            changes.push((ticker.symbol.clone(), change));
        }
    }

    changes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_gainers: Vec<(String, f64)> = changes.iter().take(5).cloned().collect();
    let top_losers: Vec<(String, f64)> = changes.iter().rev().take(5).cloned().collect();

    Ok(MarketOverview {
        btc_price,
        btc_change,
        eth_price,
        eth_change,
        top_gainers,
        top_losers,
    })
}

/// Print market overview
fn print_market_overview(overview: &MarketOverview) {
    println!("Current Market State:");
    println!(
        "  BTC/USDT: ${:.2} ({:+.2}%)",
        overview.btc_price, overview.btc_change
    );
    println!(
        "  ETH/USDT: ${:.2} ({:+.2}%)",
        overview.eth_price, overview.eth_change
    );

    if !overview.top_gainers.is_empty() {
        println!("\n  Top Gainers:");
        for (symbol, change) in &overview.top_gainers {
            println!("    {}: {:+.2}%", symbol, change);
        }
    }

    if !overview.top_losers.is_empty() {
        println!("\n  Top Losers:");
        for (symbol, change) in &overview.top_losers {
            println!("    {}: {:+.2}%", symbol, change);
        }
    }
}

/// Fetch documents or generate sample data
fn fetch_or_generate_documents(client: &BybitClient) -> Result<Vec<MarketDocument>> {
    // Try to fetch real announcements
    if let Ok(announcements) = client.get_announcements("en-US", 30) {
        if !announcements.is_empty() {
            println!("Fetched {} real announcements from Bybit", announcements.len());
            return Ok(announcements
                .iter()
                .map(MarketDocument::from_announcement)
                .collect());
        }
    }

    // Generate sample data
    println!("Using sample data for demonstration");

    let sample_docs = vec![
        ("New BTC/USDT Trading Pair Features", "Enhanced trading features for Bitcoin including advanced order types and improved liquidity.", vec!["BTC"]),
        ("Ethereum Staking Rewards Update", "ETH staking rewards have been updated. Validators can now earn competitive APY.", vec!["ETH"]),
        ("System Maintenance Notice", "Scheduled maintenance for trading systems. All services will resume shortly.", vec![]),
        ("New Listing: SOLANA", "SOL is now available for spot trading. Start trading Solana today.", vec!["SOL"]),
        ("DeFi Integration Announcement", "New DeFi protocols integrated with our platform for enhanced yield opportunities.", vec!["ETH"]),
        ("Bitcoin ETF Discussion", "Market analysis of Bitcoin ETF implications for institutional investors.", vec!["BTC"]),
        ("NFT Marketplace Launch", "Explore unique digital collectibles on our new NFT marketplace.", vec![]),
        ("Trading Competition Results", "Congratulations to the winners of our BTC trading competition.", vec!["BTC"]),
        ("Regulatory Compliance Update", "Enhanced compliance measures implemented across all trading pairs.", vec![]),
        ("Layer 2 Scaling Solutions", "Improved transaction speeds with Layer 2 integration for Ethereum.", vec!["ETH"]),
        ("Crypto Market Weekly Report", "Bitcoin and Ethereum lead market recovery amid positive sentiment.", vec!["BTC", "ETH"]),
        ("Smart Contract Security", "New security audit completed for all DeFi smart contracts.", vec!["ETH"]),
        ("Institutional Trading Features", "Advanced trading tools for institutional Bitcoin investors.", vec!["BTC"]),
        ("Cross-Chain Bridge Update", "Improved cross-chain transfers between major blockchain networks.", vec![]),
        ("Stablecoin Trading Pairs", "New USDT and USDC trading pairs added for major cryptocurrencies.", vec!["USDT"]),
    ];

    Ok(sample_docs
        .into_iter()
        .enumerate()
        .map(|(i, (title, content, symbols))| MarketDocument {
            id: format!("sample_{}", i),
            title: title.to_string(),
            content: content.to_string(),
            timestamp: Utc::now().timestamp_millis() as u64 - (i as u64 * 3600000),
            symbols: symbols.into_iter().map(String::from).collect(),
            doc_type: "announcement".to_string(),
            market_context: None,
        })
        .collect())
}

/// Generate trading insights from topics
fn generate_trading_insights(
    topics: &[topic_modeling::models::lda::LdaTopic],
    market: &MarketOverview,
) {
    println!("Based on topic analysis and market conditions:\n");

    for topic in topics {
        let keywords: Vec<&str> = topic.top_words.iter().take(3).map(|(w, _)| w.as_str()).collect();

        let insight = if keywords.iter().any(|&w| w.contains("bitcoin") || w.contains("btc")) {
            if market.btc_change > 0.0 {
                format!("Bitcoin-related topics ({:.1}% prevalence) align with positive BTC momentum ({:+.2}%)",
                    topic.prevalence * 100.0, market.btc_change)
            } else {
                format!("Bitcoin topics remain active ({:.1}% prevalence) despite price decline ({:+.2}%)",
                    topic.prevalence * 100.0, market.btc_change)
            }
        } else if keywords.iter().any(|&w| w.contains("ethereum") || w.contains("eth") || w.contains("defi")) {
            format!("Ethereum/DeFi themes represent {:.1}% of discussions. ETH: {:+.2}%",
                topic.prevalence * 100.0, market.eth_change)
        } else if keywords.iter().any(|&w| w.contains("regulation") || w.contains("compliance")) {
            format!("Regulatory topics at {:.1}% prevalence - monitor for policy impacts",
                topic.prevalence * 100.0)
        } else {
            format!("Topic '{}' represents {:.1}% of market discussions",
                keywords.join(", "), topic.prevalence * 100.0)
        };

        println!("  • {}", insight);
    }

    println!("\n  ⚠️  Note: This analysis is for educational purposes only.");
    println!("      Always conduct thorough research before trading.");
}
