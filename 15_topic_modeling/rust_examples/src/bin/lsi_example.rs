//! LSI (Latent Semantic Indexing) Example
//!
//! This example demonstrates how to:
//! - Load documents from a dataset
//! - Preprocess and tokenize text
//! - Build a TF-IDF matrix
//! - Apply LSI for topic extraction
//! - Analyze document similarity

use anyhow::Result;
use std::path::PathBuf;
use topic_modeling::models::lsi::LSI;
use topic_modeling::preprocessing::tokenizer::Tokenizer;
use topic_modeling::preprocessing::vectorizer::TfIdfVectorizer;
use topic_modeling::utils::io::DocumentDataset;

fn main() -> Result<()> {
    env_logger::init();

    println!("=== LSI Topic Modeling Example ===\n");

    // Try to load dataset, or create sample data
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
        println!("  Sample tokens: {:?}", &first_doc[..first_doc.len().min(10)]);
    }

    // Step 2: Build TF-IDF matrix
    println!("\nStep 2: Building TF-IDF matrix...");
    let mut vectorizer = TfIdfVectorizer::new()
        .min_df(1)
        .max_df_ratio(0.95)
        .max_features(500);

    let tfidf_matrix = vectorizer.fit_transform(&tokenized);
    println!(
        "  Matrix shape: {} documents x {} terms",
        tfidf_matrix.nrows(),
        tfidf_matrix.ncols()
    );

    // Get vocabulary for topic interpretation
    let vocabulary = vectorizer.get_vocabulary().clone();
    let terms: Vec<String> = (0..vectorizer.vocabulary_size())
        .filter_map(|i| vectorizer.get_term(i).cloned())
        .collect();

    println!("  Vocabulary size: {}", terms.len());

    // Show top TF-IDF terms
    println!("\n  Top discriminative terms (by IDF):");
    for (term, idf) in vectorizer.top_terms_by_idf(10) {
        println!("    {}: {:.3}", term, idf);
    }

    // Step 3: Apply LSI
    let n_topics = 5;
    println!("\nStep 3: Applying LSI with {} topics...", n_topics);

    let mut lsi = LSI::new(n_topics)?;
    lsi.fit(&tfidf_matrix, vocabulary, terms)?;

    // Step 4: Display topics
    println!("\n=== Discovered Topics ===\n");
    let topics = lsi.get_topics(8)?;

    for topic in &topics {
        println!("{}", topic);
        println!();
    }

    // Step 5: Show explained variance
    println!("=== Model Statistics ===\n");
    let total_variance = lsi.explained_variance_ratio()?;
    println!("Total explained variance: {:.2}%", total_variance * 100.0);

    // Step 6: Document-topic analysis
    println!("\n=== Document-Topic Analysis ===\n");
    let doc_topics = lsi.get_document_topics()?;

    for (i, doc_text) in documents.iter().enumerate().take(5) {
        let preview: String = doc_text.chars().take(50).collect();
        println!("Document {}: {}...", i + 1, preview);

        let topic_dist = lsi.get_document_topic_distribution(i)?;
        println!("  Top topics:");
        for (topic_idx, weight) in topic_dist.iter().take(3) {
            println!("    Topic {}: {:.3}", topic_idx, weight);
        }
        println!();
    }

    // Step 7: Document similarity
    println!("=== Document Similarity ===\n");
    if documents.len() >= 2 {
        // Find most similar documents to the first document
        println!("Documents most similar to Document 1:");
        let similar = lsi.most_similar_documents(0, 3)?;

        for (doc_idx, similarity) in similar {
            let preview: String = documents[doc_idx].chars().take(50).collect();
            println!("  Doc {}: {:.3} - {}...", doc_idx + 1, similarity, preview);
        }
    }

    // Step 8: Transform new document
    println!("\n=== Transforming New Document ===\n");
    let new_doc = "Bitcoin trading volume increases as institutional investors accumulate BTC";
    println!("New document: {}\n", new_doc);

    let new_tokens = vec![tokenizer.tokenize(new_doc)];
    let new_tfidf = vectorizer.transform(&new_tokens);
    let new_topics = lsi.transform(&new_tfidf)?;

    println!("Topic distribution for new document:");
    for topic_idx in 0..n_topics {
        println!("  Topic {}: {:.3}", topic_idx, new_topics[[0, topic_idx]]);
    }

    println!("\n=== LSI Example Complete ===");
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

    // Create sample data
    println!("Creating sample data...");

    let sample_texts = vec![
        // Bitcoin/trading theme
        "Bitcoin price analysis shows bullish momentum as trading volume increases significantly on major exchanges",
        "BTC technical analysis indicates strong support levels traders should watch for potential breakout",
        "Cryptocurrency trading strategies for bitcoin volatility management and risk assessment",
        "Bitcoin market cap reaches new highs as institutional trading activity continues to grow",
        "Technical indicators suggest bitcoin entering accumulation phase with increased trading interest",

        // Ethereum/DeFi theme
        "Ethereum smart contracts enable decentralized finance applications transforming traditional banking",
        "DeFi protocols on Ethereum network see increased total value locked as yield farming grows",
        "Ethereum gas fees optimization strategies for smart contract deployment and execution",
        "Layer 2 scaling solutions for Ethereum improve transaction throughput and reduce costs",
        "Decentralized exchanges on Ethereum provide liquidity for token swaps without intermediaries",

        // NFT/Gaming theme
        "NFT marketplace activity increases as digital art collections attract mainstream attention",
        "Blockchain gaming integrates NFT ownership for in-game assets and virtual real estate",
        "Digital collectibles market evolves with new NFT standards and interoperability features",
        "Gaming metaverse projects leverage blockchain for verifiable digital asset ownership",
        "NFT creators explore new revenue models through royalties and community engagement",

        // Regulation theme
        "Cryptocurrency regulation framework discussions continue in legislative sessions worldwide",
        "Compliance requirements for digital asset exchanges increase regulatory scrutiny",
        "Government oversight of stablecoin issuers addresses systemic risk concerns",
        "Regulatory clarity for cryptocurrency businesses impacts market participant behavior",
        "Anti-money laundering rules for crypto exchanges require enhanced verification procedures",

        // Technology theme
        "Zero knowledge proof technology enables privacy-preserving blockchain transactions",
        "Cross-chain interoperability protocols connect different blockchain networks seamlessly",
        "Consensus mechanism improvements enhance blockchain security and energy efficiency",
        "Cryptographic innovations in blockchain technology address scalability challenges",
        "Decentralized storage solutions provide censorship-resistant data availability",
    ];

    Ok(sample_texts.into_iter().map(String::from).collect())
}
