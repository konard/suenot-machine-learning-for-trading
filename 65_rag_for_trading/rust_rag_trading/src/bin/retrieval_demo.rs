//! Document retrieval demonstration.
//!
//! This example shows how to:
//! - Create and index financial documents
//! - Search for relevant information
//! - Filter by ticker symbol

use chrono::Utc;
use rag_trading::{Document, DocumentRetriever, DocumentType};

fn main() {
    println!("=== RAG Trading: Document Retrieval Demo ===\n");

    // Create a document retriever
    let mut retriever = DocumentRetriever::new();

    // Add sample financial documents
    let documents = vec![
        Document::new(
            "Tesla reported record quarterly deliveries of 500,000 vehicles, \
             beating analyst expectations. The company's China sales showed \
             particularly strong growth. Multiple analysts upgraded their \
             price targets following the announcement.",
            Some("TSLA".to_string()),
            "Reuters",
            Utc::now(),
        ),
        Document::new(
            "Apple's latest iPhone launch exceeded sales expectations in the \
             first weekend. The new AI features are driving strong demand. \
             Services revenue continues to grow at double-digit rates.",
            Some("AAPL".to_string()),
            "Bloomberg",
            Utc::now(),
        ),
        Document::new(
            "NVIDIA's data center revenue grew 200% year-over-year as AI \
             chip demand continues to surge. The company raised guidance \
             for the next quarter, citing strong enterprise adoption.",
            Some("NVDA".to_string()),
            "CNBC",
            Utc::now(),
        ),
        Document::new(
            "Tesla faces increasing competition in China from BYD and other \
             local EV makers. Margins are under pressure due to price cuts. \
             Some analysts expressed concerns about near-term profitability.",
            Some("TSLA".to_string()),
            "Financial Times",
            Utc::now(),
        ),
        Document::new(
            "Microsoft reported strong cloud growth with Azure revenue up 29%. \
             The company's AI investments are starting to show returns. \
             Enterprise customers are rapidly adopting Copilot.",
            Some("MSFT".to_string()),
            "Wall Street Journal",
            Utc::now(),
        ),
        Document::new(
            "Federal Reserve minutes suggest rates will stay higher for longer. \
             Markets are repricing expectations for rate cuts in 2024.",
            None,
            "Federal Reserve",
            Utc::now(),
        ).with_type(DocumentType::Other),
    ];

    // Add documents to the retriever
    retriever.add_documents(documents);
    println!("Indexed {} documents\n", retriever.document_count());

    // Example 1: Search for Tesla-related news
    println!("--- Search: 'Tesla delivery growth' ---");
    let results = retriever.search("Tesla delivery growth", 3, None);
    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. [Score: {:.3}] {} - {}...",
            i + 1,
            result.score,
            result.document.source,
            &result.document.text[..80.min(result.document.text.len())]
        );
    }
    println!();

    // Example 2: Search with ticker filter
    println!("--- Search: 'sales growth' filtered by TSLA ---");
    let results = retriever.search("sales growth", 3, Some("TSLA"));
    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. [Score: {:.3}] {} - {}...",
            i + 1,
            result.score,
            result.document.source,
            &result.document.text[..80.min(result.document.text.len())]
        );
    }
    println!();

    // Example 3: Search for AI-related news
    println!("--- Search: 'AI artificial intelligence chip' ---");
    let results = retriever.search("AI artificial intelligence chip", 3, None);
    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. [Score: {:.3}] [{}] {}",
            i + 1,
            result.score,
            result.document.ticker.as_deref().unwrap_or("N/A"),
            result.document.source
        );
    }
    println!();

    // Example 4: Economic data search
    println!("--- Search: 'interest rates monetary policy' ---");
    let results = retriever.search("interest rates monetary policy", 2, None);
    for result in &results {
        println!(
            "[{:?}] {} - {}",
            result.document.doc_type, result.document.source, result.document.text
        );
    }

    println!("\n=== Demo Complete ===");
}
