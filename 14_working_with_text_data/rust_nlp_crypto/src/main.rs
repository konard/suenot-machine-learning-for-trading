//! CLI для NLP анализа криптовалютных данных
//!
//! Использование:
//! ```bash
//! cargo run -- --help
//! cargo run -- analyze --symbol BTCUSDT
//! cargo run -- signals --symbol ETHUSDT
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use rust_nlp_crypto::{
    api::BybitClient,
    nlp::{Preprocessor, Tokenizer, Vectorizer, BagOfWords, TfIdf},
    sentiment::SentimentAnalyzer,
    signals::SignalGenerator,
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "nlp_crypto")]
#[command(author = "ML for Trading")]
#[command(version = "0.1.0")]
#[command(about = "NLP analysis for cryptocurrency trading", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Уровень логирования
    #[arg(short, long, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Анализ настроений текста
    Analyze {
        /// Текст для анализа (или --symbol для анализа новостей)
        #[arg(short, long)]
        text: Option<String>,

        /// Символ криптовалюты для анализа новостей
        #[arg(short, long)]
        symbol: Option<String>,

        /// Количество новостей для анализа
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Генерация торговых сигналов
    Signals {
        /// Символ криптовалюты
        #[arg(short, long)]
        symbol: String,

        /// Количество новостей для анализа
        #[arg(short, long, default_value = "20")]
        limit: usize,

        /// Включить технический анализ
        #[arg(short, long)]
        technical: bool,
    },

    /// Демонстрация NLP pipeline
    Demo {
        /// Текст для демонстрации
        #[arg(short, long, default_value = "Bitcoin is showing strong bullish momentum today!")]
        text: String,
    },

    /// Получить анонсы Bybit
    Announcements {
        /// Количество анонсов
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Фильтр по символу
        #[arg(short, long)]
        symbol: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Настройка логирования
    let log_level = match cli.log_level.as_str() {
        "debug" => Level::DEBUG,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::Analyze { text, symbol, limit } => {
            run_analyze(text, symbol, limit).await?;
        }
        Commands::Signals { symbol, limit, technical } => {
            run_signals(&symbol, limit, technical).await?;
        }
        Commands::Demo { text } => {
            run_demo(&text)?;
        }
        Commands::Announcements { limit, symbol } => {
            run_announcements(limit, symbol).await?;
        }
    }

    Ok(())
}

async fn run_analyze(text: Option<String>, symbol: Option<String>, limit: usize) -> Result<()> {
    let analyzer = SentimentAnalyzer::new();

    if let Some(text) = text {
        // Анализ одного текста
        println!("\n📝 Analyzing text...\n");
        let result = analyzer.analyze(&text);

        println!("Text: {}", result.text);
        println!("Polarity: {:?}", result.polarity);
        println!("Score: {:.3}", result.score);
        println!("Confidence: {:.1}%", result.confidence * 100.0);

        if !result.key_words.is_empty() {
            println!("\nKey words:");
            for word in &result.key_words {
                let sign = if word.score > 0.0 { "+" } else { "" };
                println!("  • {} ({}{:.2})", word.word, sign, word.score);
            }
        }
    } else if let Some(symbol) = symbol {
        // Анализ новостей для символа
        println!("\n📊 Fetching announcements for {}...\n", symbol);

        let client = BybitClient::new();
        let announcements = client.get_announcements(limit).await?;

        // Фильтруем по символу
        let relevant: Vec<_> = announcements
            .iter()
            .filter(|a| {
                a.symbols.iter().any(|s| s.to_uppercase() == symbol.to_uppercase())
                    || a.title.to_uppercase().contains(&symbol.to_uppercase())
            })
            .collect();

        println!("Found {} relevant announcements\n", relevant.len());

        let mut results = Vec::new();
        for ann in &relevant {
            let text = format!("{} {}", ann.title, ann.description);
            let result = analyzer.analyze(&text);
            println!(
                "{:?} [{:.2}] {}",
                result.polarity, result.score, ann.title
            );
            results.push(result);
        }

        if !results.is_empty() {
            let aggregated = analyzer.aggregate(&results);
            println!("\n{}", aggregated);
        }
    } else {
        println!("Please provide --text or --symbol");
    }

    Ok(())
}

async fn run_signals(symbol: &str, limit: usize, technical: bool) -> Result<()> {
    println!("\n🎯 Generating trading signal for {}...\n", symbol);

    let client = BybitClient::new();
    let generator = SignalGenerator::new();

    // Получаем анонсы
    info!("Fetching announcements...");
    let announcements = client.get_announcements(limit).await?;

    // Генерируем сигнал
    let signal = if technical {
        // Получаем рыночные данные
        info!("Fetching market data...");
        let klines = client.get_klines(&format!("{}USDT", symbol), "60", 100).await?;

        let texts: Vec<String> = announcements
            .iter()
            .map(|a| format!("{} {}", a.title, a.description))
            .collect();

        generator.generate_with_technicals(symbol, &texts, &klines)
    } else {
        generator.generate_from_announcements(symbol, &announcements)
    };

    match signal {
        Some(signal) => {
            println!("{}", signal);
        }
        None => {
            println!("❌ Unable to generate signal. Not enough data or low confidence.");
            println!("\nTips:");
            println!("  • Try increasing --limit to analyze more announcements");
            println!("  • Check if the symbol is mentioned in recent news");
        }
    }

    Ok(())
}

fn run_demo(text: &str) -> Result<()> {
    println!("\n🔬 NLP Pipeline Demo\n");
    println!("Input text: \"{}\"\n", text);

    // 1. Tokenization
    println!("1️⃣  TOKENIZATION");
    println!("─────────────────");
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize(text);
    for token in &tokens {
        println!(
            "   {:15} -> {:15} [{:?}]",
            token.original, token.normalized, token.token_type
        );
    }

    // 2. Preprocessing
    println!("\n2️⃣  PREPROCESSING");
    println!("─────────────────");
    let preprocessor = Preprocessor::new();
    let token_strings: Vec<String> = tokens.iter().map(|t| t.normalized.clone()).collect();
    let processed = preprocessor.process(&token_strings);
    println!("   Before: {:?}", token_strings);
    println!("   After:  {:?}", processed);

    // 3. Bag of Words
    println!("\n3️⃣  BAG OF WORDS");
    println!("────────────────");
    let mut bow = BagOfWords::new();
    let docs = vec![processed.clone()];
    let dtm = bow.fit_transform(&docs);
    println!("   Vocabulary size: {}", dtm.n_terms());
    println!("   Terms: {:?}", dtm.terms);
    if let Some(vec) = dtm.get_document_vector(0) {
        println!("   Vector: {:?}", vec);
    }

    // 4. TF-IDF
    println!("\n4️⃣  TF-IDF");
    println!("──────────");
    let mut tfidf = TfIdf::new();
    let docs2 = vec![
        processed.clone(),
        vec!["bitcoin".to_string(), "great".to_string()],
        vec!["ethereum".to_string(), "strong".to_string()],
    ];
    let tfidf_dtm = tfidf.fit_transform(&docs2);
    println!("   Documents: {}", tfidf_dtm.n_documents());
    println!("   Terms: {}", tfidf_dtm.n_terms());
    let top = tfidf.top_terms(5);
    println!("   Top terms by IDF:");
    for (term, idf) in top {
        println!("     • {} (IDF: {:.3})", term, idf);
    }

    // 5. Sentiment Analysis
    println!("\n5️⃣  SENTIMENT ANALYSIS");
    println!("───────────────────────");
    let analyzer = SentimentAnalyzer::new();
    let result = analyzer.analyze(text);
    println!("   Polarity: {:?}", result.polarity);
    println!("   Score: {:.3}", result.score);
    println!("   Confidence: {:.1}%", result.confidence * 100.0);
    println!("   Key words:");
    for word in &result.key_words {
        let sign = if word.score > 0.0 { "+" } else { "" };
        println!("     • {} ({}{:.2})", word.word, sign, word.score);
    }

    println!("\n✅ Demo complete!\n");

    Ok(())
}

async fn run_announcements(limit: usize, symbol: Option<String>) -> Result<()> {
    println!("\n📢 Fetching Bybit Announcements...\n");

    let client = BybitClient::new();
    let announcements = client.get_announcements(limit).await?;

    let filtered: Vec<_> = if let Some(ref sym) = symbol {
        announcements
            .iter()
            .filter(|a| {
                a.symbols.iter().any(|s| s.to_uppercase() == sym.to_uppercase())
                    || a.title.to_uppercase().contains(&sym.to_uppercase())
            })
            .collect()
    } else {
        announcements.iter().collect()
    };

    println!("Found {} announcements\n", filtered.len());

    for ann in filtered {
        println!("───────────────────────────────────────");
        println!("📌 {}", ann.title);
        println!("   Type: {:?}", ann.announcement_type);
        println!("   Date: {}", ann.publish_time.format("%Y-%m-%d %H:%M"));
        if !ann.symbols.is_empty() {
            println!("   Symbols: {}", ann.symbols.join(", "));
        }
        if let Some(ref url) = ann.url {
            println!("   URL: {}", url);
        }
    }

    Ok(())
}
