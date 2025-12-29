//! Пример: Генерация торговых сигналов
//!
//! Демонстрирует комбинирование анализа настроений
//! с рыночными данными для генерации торговых сигналов.
//!
//! Запуск:
//! ```bash
//! cargo run --example trading_signals
//! ```

use anyhow::Result;
use rust_nlp_crypto::api::{BybitClient, Interval, TechnicalIndicators};
use rust_nlp_crypto::sentiment::SentimentAnalyzer;
use rust_nlp_crypto::signals::{SignalConfig, SignalGenerator};

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("   Crypto Trading Signals Generator");
    println!("═══════════════════════════════════════════════════════════\n");

    // Создаём клиент Bybit
    let client = BybitClient::new();

    // Символы для анализа
    let symbols = vec!["BTC", "ETH", "SOL"];

    // ═══════════════════════════════════════════════════════════
    // 1. Получение рыночных данных
    // ═══════════════════════════════════════════════════════════
    println!("1️⃣  MARKET DATA");
    println!("──────────────────────────────────────────────\n");

    for symbol in &symbols {
        let ticker_symbol = format!("{}USDT", symbol);

        match client.get_ticker(&ticker_symbol).await {
            Ok(ticker) => {
                let trend_emoji = if ticker.price_24h_pcnt > 0.0 {
                    "📈"
                } else {
                    "📉"
                };
                println!(
                    "  {} {} ${:.2} ({:+.2}%)",
                    trend_emoji, symbol, ticker.last_price, ticker.price_24h_pcnt * 100.0
                );
            }
            Err(e) => {
                println!("  ⚠️  {} - Failed to fetch: {}", symbol, e);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 2. Технический анализ
    // ═══════════════════════════════════════════════════════════
    println!("\n2️⃣  TECHNICAL ANALYSIS");
    println!("──────────────────────────────────────────────\n");

    for symbol in &symbols {
        let ticker_symbol = format!("{}USDT", symbol);

        match client
            .get_klines(&ticker_symbol, Interval::H1.as_str(), 100)
            .await
        {
            Ok(klines) => {
                if klines.len() >= 20 {
                    // Рассчитываем индикаторы
                    let sma_short = TechnicalIndicators::sma(&klines, 7);
                    let sma_long = TechnicalIndicators::sma(&klines, 20);
                    let rsi = TechnicalIndicators::rsi(&klines, 14);
                    let trend = TechnicalIndicators::trend(&klines, 7, 20);

                    println!("  {} {}:", symbol, ticker_symbol);

                    if let (Some(short), Some(long)) = (sma_short.last(), sma_long.last()) {
                        println!("    SMA(7):  ${:.2}", short);
                        println!("    SMA(20): ${:.2}", long);
                    }

                    if let Some(rsi_val) = rsi.last() {
                        let rsi_status = if *rsi_val > 70.0 {
                            "Overbought ⚠️"
                        } else if *rsi_val < 30.0 {
                            "Oversold ⚠️"
                        } else {
                            "Normal"
                        };
                        println!("    RSI(14): {:.1} ({})", rsi_val, rsi_status);
                    }

                    if let Some(t) = trend {
                        println!("    Trend:   {}", t);
                    }

                    println!();
                }
            }
            Err(e) => {
                println!("  ⚠️  {} - Failed to fetch klines: {}", symbol, e);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 3. Анализ новостей
    // ═══════════════════════════════════════════════════════════
    println!("3️⃣  NEWS SENTIMENT ANALYSIS");
    println!("──────────────────────────────────────────────\n");

    println!("📥 Fetching announcements...");
    let announcements = client.get_announcements(30).await?;
    println!("   Found {} announcements\n", announcements.len());

    let analyzer = SentimentAnalyzer::new();

    // Анализируем последние 10 анонсов
    let recent: Vec<_> = announcements.iter().take(10).collect();

    println!("Recent announcements sentiment:\n");
    for ann in &recent {
        let text = format!("{} {}", ann.title, ann.description);
        let result = analyzer.analyze(&text);

        let emoji = match result.polarity {
            rust_nlp_crypto::models::Polarity::Positive => "🟢",
            rust_nlp_crypto::models::Polarity::Negative => "🔴",
            rust_nlp_crypto::models::Polarity::Neutral => "⚪",
        };

        println!(
            "  {} [{:+.2}] {}",
            emoji,
            result.score,
            truncate(&ann.title, 50)
        );
    }

    // ═══════════════════════════════════════════════════════════
    // 4. Генерация сигналов
    // ═══════════════════════════════════════════════════════════
    println!("\n4️⃣  TRADING SIGNALS");
    println!("──────────────────────────────────────────────\n");

    // Настраиваем генератор сигналов
    let config = SignalConfig {
        min_confidence: 0.2,
        min_texts: 2,
        sentiment_weight: 0.6,
        technical_weight: 0.4,
        strong_signal_threshold: 0.7,
    };

    let generator = SignalGenerator::new().with_config(config);

    // Генерируем сигналы для каждого символа
    for symbol in &symbols {
        println!("Generating signal for {}...", symbol);

        // Фильтруем анонсы по символу
        let relevant: Vec<_> = announcements
            .iter()
            .filter(|a| {
                a.symbols
                    .iter()
                    .any(|s| s.to_uppercase() == symbol.to_uppercase())
                    || a.title.to_uppercase().contains(symbol)
            })
            .collect();

        if relevant.len() < 2 {
            println!("  ⚠️  Not enough relevant news for {}\n", symbol);
            continue;
        }

        // Получаем рыночные данные для технического анализа
        let ticker_symbol = format!("{}USDT", symbol);
        let klines = client
            .get_klines(&ticker_symbol, Interval::H1.as_str(), 100)
            .await
            .ok();

        // Готовим тексты
        let texts: Vec<String> = relevant
            .iter()
            .map(|a| format!("{} {}", a.title, a.description))
            .collect();

        // Генерируем сигнал
        let signal = if let Some(ref klines) = klines {
            generator.generate_with_technicals(symbol, &texts, klines)
        } else {
            generator.generate_from_texts(symbol, &texts)
        };

        match signal {
            Some(signal) => {
                println!("{}", signal);
            }
            None => {
                println!("  ℹ️  Unable to generate confident signal for {}\n", symbol);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 5. Демо с симулированными данными
    // ═══════════════════════════════════════════════════════════
    println!("5️⃣  DEMO WITH SAMPLE TEXTS");
    println!("──────────────────────────────────────────────\n");

    let demo_texts = vec![
        "PEPE showing incredible momentum, up 50% in 24h! Very bullish!".to_string(),
        "New listing on Bybit: PEPE perpetual contracts now available!".to_string(),
        "PEPE breaking all resistance levels, to the moon! 🚀".to_string(),
        "Community is extremely bullish on PEPE right now".to_string(),
        "Major whale accumulation detected for PEPE".to_string(),
    ];

    let demo_signal = generator.generate_from_texts("PEPE", &demo_texts);

    println!("Demo: Analyzing positive PEPE sentiment...\n");

    if let Some(signal) = demo_signal {
        println!("{}", signal);
    }

    // Негативный сценарий
    let negative_texts = vec![
        "Market crash incoming! Sell everything now!".to_string(),
        "This looks like a massive rug pull, be careful!".to_string(),
        "Terrible performance, investors losing confidence".to_string(),
        "Project team dumping tokens, very bearish".to_string(),
    ];

    let negative_signal = generator.generate_from_texts("SCAM", &negative_texts);

    println!("Demo: Analyzing negative sentiment...\n");

    if let Some(signal) = negative_signal {
        println!("{}", signal);
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("   Analysis Complete! 🎉");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("⚠️  DISCLAIMER:");
    println!("   This is for educational purposes only.");
    println!("   Do NOT use these signals for real trading");
    println!("   without proper backtesting and risk management.\n");

    Ok(())
}

/// Обрезать текст до указанной длины
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
