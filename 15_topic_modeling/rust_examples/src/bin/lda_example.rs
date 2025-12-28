//! LDA (Latent Dirichlet Allocation) Example
//!
//! This example demonstrates how to:
//! - Load documents from a dataset
//! - Preprocess and tokenize text
//! - Build a document-term matrix
//! - Apply LDA for topic extraction
//! - Evaluate model quality with coherence metrics

use anyhow::Result;
use std::path::PathBuf;
use topic_modeling::models::lda::{LdaConfig, LDA};
use topic_modeling::preprocessing::tokenizer::Tokenizer;
use topic_modeling::preprocessing::vectorizer::CountVectorizer;
use topic_modeling::utils::evaluation::{Evaluator, ModelSummary};
use topic_modeling::utils::io::DocumentDataset;

fn main() -> Result<()> {
    env_logger::init();

    println!("=== LDA Topic Modeling Example ===\n");

    // Load documents
    let documents = load_or_create_sample_data()?;
    println!("Loaded {} documents\n", documents.len());

    // Step 1: Tokenize documents
    println!("Step 1: Tokenizing documents...");
    let tokenizer = Tokenizer::for_crypto().min_length(3);

    let tokenized: Vec<Vec<String>> = documents
        .iter()
        .map(|doc| tokenizer.tokenize(doc))
        .collect();

    // Print sample tokenization
    if let Some(first_doc) = tokenized.first() {
        println!(
            "  Sample tokens: {:?}",
            &first_doc[..first_doc.len().min(10)]
        );
    }

    // Step 2: Build count matrix (bag of words)
    println!("\nStep 2: Building document-term matrix...");
    let mut vectorizer = CountVectorizer::new()
        .min_df(1)
        .max_df_ratio(0.9)
        .max_features(300);

    let dtm = vectorizer.fit_transform(&tokenized);
    println!(
        "  Matrix shape: {} documents x {} terms",
        dtm.nrows(),
        dtm.ncols()
    );

    // Get vocabulary for topic interpretation
    let vocabulary = vectorizer.get_vocabulary().clone();
    let terms: Vec<String> = (0..vectorizer.vocabulary_size())
        .filter_map(|i| vectorizer.get_term(i).cloned())
        .collect();

    println!("  Vocabulary size: {}", terms.len());

    // Step 3: Configure and train LDA
    let n_topics = 5;
    println!("\nStep 3: Training LDA with {} topics...", n_topics);
    println!("  This may take a moment...");

    let config = LdaConfig::new(n_topics)
        .alpha(0.1)          // Sparse document-topic distribution
        .beta(0.01)          // Sparse topic-word distribution
        .n_iterations(500)   // Training iterations
        .burn_in(50)         // Burn-in period
        .random_seed(42);    // For reproducibility

    let mut lda = LDA::new(config)?;
    lda.fit(&dtm, vocabulary, terms.clone())?;

    println!("  Training complete!");

    // Step 4: Display topics
    println!("\n=== Discovered Topics ===\n");
    let topics = lda.get_topics(8)?;

    for topic in &topics {
        println!("{}", topic);
        println!();
    }

    // Step 5: Model evaluation
    println!("=== Model Evaluation ===\n");

    // Calculate perplexity
    let perplexity = lda.perplexity(&dtm)?;
    println!("Perplexity: {:.2}", perplexity);
    println!("  (Lower is better, indicates how well model predicts held-out data)");

    // Topic coherence
    if let Some(coherences) = lda.get_coherence_scores() {
        println!("\nTopic Coherence Scores (PMI-based):");
        for (i, score) in coherences.iter().enumerate() {
            println!("  Topic {}: {:.4}", i, score);
        }

        if let Some(avg) = lda.average_coherence() {
            println!("\nAverage Coherence: {:.4}", avg);
            println!("  (Higher is better, measures semantic quality of topics)");
        }
    }

    // Compute model summary
    let topic_words: Vec<Vec<String>> = topics
        .iter()
        .map(|t| t.top_words.iter().map(|(w, _)| w.clone()).collect())
        .collect();

    let evaluator = Evaluator::new().with_dtm(dtm.clone(), terms);
    let summary = ModelSummary::from_topics(&topic_words, &evaluator, Some(perplexity));

    println!("\n--- Full Model Summary ---");
    summary.print();

    // Step 6: Document-topic analysis
    println!("\n=== Document-Topic Analysis ===\n");
    let doc_topics = lda.get_document_topics()?;
    let dominant = lda.dominant_topics()?;

    println!("Document assignments:");
    for (i, doc_text) in documents.iter().enumerate().take(10) {
        let preview: String = doc_text.chars().take(40).collect();
        let topic = dominant[i];
        let prob = doc_topics[[i, topic]];

        println!(
            "  Doc {:2}: Topic {} ({:.2}%) - {}...",
            i + 1,
            topic,
            prob * 100.0,
            preview
        );
    }

    // Step 7: Topic distribution for each document
    println!("\n=== Detailed Document Analysis ===\n");
    for (i, doc_text) in documents.iter().enumerate().take(3) {
        let preview: String = doc_text.chars().take(50).collect();
        println!("Document {}: {}...", i + 1, preview);
        println!("  Topic distribution:");

        let mut topic_probs: Vec<(usize, f64)> = (0..n_topics)
            .map(|t| (t, doc_topics[[i, t]]))
            .collect();
        topic_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (topic, prob) in topic_probs.iter().take(3) {
            println!("    Topic {}: {:.2}%", topic, prob * 100.0);
        }
        println!();
    }

    // Step 8: Transform new document
    println!("=== Transforming New Document ===\n");
    let new_doc = "Ethereum smart contracts enable decentralized applications and DeFi protocols";
    println!("New document: {}\n", new_doc);

    let new_tokens = vec![tokenizer.tokenize(new_doc)];
    let new_dtm = vectorizer.transform(&new_tokens);
    let new_topics = lda.transform(&new_dtm)?;

    println!("Topic distribution for new document:");
    let mut probs: Vec<(usize, f64)> = (0..n_topics)
        .map(|t| (t, new_topics[[0, t]]))
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (topic, prob) in probs {
        println!("  Topic {}: {:.2}%", topic, prob * 100.0);
    }

    // Step 9: Log-likelihood history
    println!("\n=== Training Convergence ===\n");
    let ll_history = lda.log_likelihood_history();
    if !ll_history.is_empty() {
        println!("Log-likelihood history (sampled):");
        let step = (ll_history.len() / 10).max(1);
        for (i, ll) in ll_history.iter().enumerate().step_by(step) {
            println!("  Iteration {}: {:.2}", i + 50, ll); // +50 for burn-in
        }
        println!(
            "\nFinal log-likelihood: {:.2}",
            ll_history.last().unwrap_or(&0.0)
        );
    }

    println!("\n=== LDA Example Complete ===");
    Ok(())
}

/// Load dataset from file or create sample data
fn load_or_create_sample_data() -> Result<Vec<String>> {
    // Try to load from file first
    let data_path = PathBuf::from("data/sample_crypto_news.json");

    if data_path.exists() {
        println!("Loading dataset from {:?}...", data_path);
        let dataset = DocumentDataset::load_json(&data_path)?;
        return Ok(dataset.get_texts());
    }

    // Create sample data with clearer topic structure
    println!("Creating sample data...");

    let sample_texts = vec![
        // Topic 1: Bitcoin trading and investment
        "Bitcoin price surges as institutional investors increase their holdings in BTC markets",
        "Trading volume for bitcoin reaches record levels on cryptocurrency exchanges worldwide",
        "BTC technical analysis shows bullish patterns as traders anticipate price breakout",
        "Bitcoin investment funds report massive inflows as market sentiment turns positive",
        "Cryptocurrency traders watch bitcoin support and resistance levels for entry points",
        "BTC market capitalization grows as retail and institutional demand increases",
        "Bitcoin trading strategies for volatile market conditions and risk management",

        // Topic 2: Ethereum and smart contracts
        "Ethereum network upgrades improve smart contract execution and reduce gas fees",
        "DeFi applications on Ethereum attract billions in total value locked",
        "Smart contract developers build decentralized applications on Ethereum platform",
        "Ethereum staking rewards attract validators to secure the proof of stake network",
        "Layer 2 solutions for Ethereum scale smart contract transactions efficiently",
        "Decentralized finance protocols leverage Ethereum smart contracts for lending",
        "Ethereum ecosystem growth drives innovation in blockchain technology",

        // Topic 3: NFTs and digital assets
        "NFT marketplace launches new features for digital art collectors and creators",
        "Digital collectibles market expands as NFT adoption reaches mainstream audiences",
        "Blockchain gaming integrates NFT technology for verifiable in-game assets",
        "Artists embrace NFT platforms for selling unique digital artwork and creations",
        "NFT collections attract celebrity endorsements and high-profile sales",
        "Metaverse projects incorporate NFT ownership for virtual land and items",
        "Digital asset marketplaces compete for NFT trading volume and user engagement",

        // Topic 4: Regulation and compliance
        "Cryptocurrency regulation frameworks evolve as governments address digital assets",
        "Exchange compliance requirements increase with new anti-money laundering rules",
        "Regulatory clarity for digital assets impacts institutional adoption decisions",
        "Government oversight of cryptocurrency markets aims to protect retail investors",
        "Stablecoin regulation proposals address systemic risk and reserve requirements",
        "Compliance costs rise as exchanges implement enhanced verification procedures",
        "Regulatory authorities worldwide coordinate on cryptocurrency policy standards",

        // Topic 5: Blockchain technology
        "Blockchain technology innovations enable new consensus mechanisms and security",
        "Zero knowledge proofs provide privacy solutions for blockchain transactions",
        "Cross-chain protocols connect different blockchain networks for interoperability",
        "Decentralized storage solutions leverage blockchain for data availability",
        "Cryptographic advances improve blockchain scalability and transaction throughput",
        "Distributed ledger technology transforms supply chain and identity management",
        "Blockchain infrastructure development attracts venture capital investment",
    ];

    Ok(sample_texts.into_iter().map(String::from).collect())
}
