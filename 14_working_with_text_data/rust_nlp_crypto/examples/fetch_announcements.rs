//! Пример: Получение анонсов с Bybit
//!
//! Демонстрирует работу с API Bybit для получения
//! анонсов и новостей криптовалютной биржи.
//!
//! Запуск:
//! ```bash
//! cargo run --example fetch_announcements
//! ```

use anyhow::Result;
use rust_nlp_crypto::api::{AnnouncementFilter, BybitClient, assess_trading_relevance, TradingRelevance};
use rust_nlp_crypto::models::AnnouncementType;

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("   Bybit Announcements Fetcher");
    println!("═══════════════════════════════════════════════════════════\n");

    // Создаём клиент
    let client = BybitClient::new();

    // Получаем последние анонсы
    println!("📥 Fetching latest announcements...\n");
    let announcements = client.get_announcements(20).await?;

    println!("Found {} announcements\n", announcements.len());

    // Выводим все анонсы с оценкой важности
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ All Announcements                                           │");
    println!("├─────────────────────────────────────────────────────────────┤");

    for ann in &announcements {
        let relevance = assess_trading_relevance(ann);
        let relevance_emoji = match relevance {
            TradingRelevance::High => "🔥",
            TradingRelevance::Medium => "📊",
            TradingRelevance::Low => "📝",
        };

        println!("│");
        println!("│ {} [{}] {}", relevance_emoji, ann.publish_time.format("%m-%d %H:%M"), ann.title);
        println!("│   Type: {:?}", ann.announcement_type);

        if !ann.symbols.is_empty() {
            println!("│   Symbols: {}", ann.symbols.join(", "));
        }
    }

    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Фильтруем только листинги
    println!("🆕 New Listings:");
    println!("─────────────────");
    let listing_filter = AnnouncementFilter::new().with_type(AnnouncementType::NewListing);
    let listings = listing_filter.apply(&announcements);

    if listings.is_empty() {
        println!("   No new listings in recent announcements");
    } else {
        for listing in &listings {
            println!("   • {} ({})", listing.title, listing.publish_time.format("%Y-%m-%d"));
        }
    }

    // Фильтруем по ключевым словам
    println!("\n🔍 Announcements mentioning 'BTC' or 'ETH':");
    println!("────────────────────────────────────────────");
    let keyword_filter = AnnouncementFilter::new()
        .with_keywords(vec!["BTC".to_string(), "ETH".to_string(), "Bitcoin".to_string(), "Ethereum".to_string()]);
    let btc_eth = keyword_filter.apply(&announcements);

    if btc_eth.is_empty() {
        println!("   No announcements mentioning BTC or ETH");
    } else {
        for ann in &btc_eth {
            println!("   • {}", ann.title);
        }
    }

    // Высокоприоритетные анонсы
    println!("\n🔥 High Priority for Trading:");
    println!("──────────────────────────────");
    let high_priority: Vec<_> = announcements
        .iter()
        .filter(|a| assess_trading_relevance(a) == TradingRelevance::High)
        .collect();

    if high_priority.is_empty() {
        println!("   No high-priority announcements");
    } else {
        for ann in high_priority {
            println!("   🚨 {}", ann.title);
            println!("      {:?}", ann.announcement_type);
        }
    }

    println!("\n✅ Done!\n");

    Ok(())
}
