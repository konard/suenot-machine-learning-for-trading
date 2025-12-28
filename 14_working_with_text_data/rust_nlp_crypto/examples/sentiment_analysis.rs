//! Пример: Анализ настроений крипто-текстов
//!
//! Демонстрирует полный NLP pipeline:
//! 1. Токенизация
//! 2. Предобработка
//! 3. Анализ настроений
//! 4. Наивный Байес
//!
//! Запуск:
//! ```bash
//! cargo run --example sentiment_analysis
//! ```

use rust_nlp_crypto::nlp::{BagOfWords, Preprocessor, TfIdf, Tokenizer, Vectorizer};
use rust_nlp_crypto::sentiment::{CryptoLexicon, NaiveBayesClassifier, SentimentAnalyzer, SentimentLexicon};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("   Crypto Sentiment Analysis Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    // Примеры текстов для анализа
    let sample_texts = vec![
        "Bitcoin is mooning! Very bullish on BTC right now 🚀",
        "Market is crashing, this looks like a scam. Selling everything.",
        "ETH showing stable growth, could be a good entry point",
        "Just got liquidated on my leveraged position... terrible day",
        "New partnership announced! This is huge for the ecosystem",
        "Not sure about this project, seems risky",
        "HODL! Diamond hands will be rewarded 💎",
        "The market seems uncertain, waiting for clearer signals",
    ];

    // ═══════════════════════════════════════════════════════════
    // 1. Демонстрация лексикона
    // ═══════════════════════════════════════════════════════════
    println!("1️⃣  CRYPTO LEXICON");
    println!("──────────────────────────────────────────────\n");

    let lexicon = CryptoLexicon::new();
    let stats = lexicon.stats();
    println!("Lexicon statistics:");
    println!("  • Positive words: {}", stats.positive_count);
    println!("  • Negative words: {}", stats.negative_count);
    println!("  • Modifiers: {}", stats.modifier_count);
    println!("  • Negations: {}", stats.negation_count);

    println!("\nSample word scores:");
    let test_words = ["moon", "crash", "bullish", "scam", "hodl", "dip"];
    for word in test_words {
        if let Some(score) = lexicon.get_score(word) {
            let bar = create_score_bar(score);
            println!("  {:10} {:+.2} {}", word, score, bar);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 2. Токенизация
    // ═══════════════════════════════════════════════════════════
    println!("\n2️⃣  TOKENIZATION");
    println!("──────────────────────────────────────────────\n");

    let tokenizer = Tokenizer::new();
    let sample = "Bitcoin $BTC is up 10%! Check @CryptoNews #bullish https://example.com";

    println!("Input: \"{}\"\n", sample);
    println!("Tokens:");

    let tokens = tokenizer.tokenize(sample);
    for token in &tokens {
        println!(
            "  {:20} -> {:15} [{:?}]",
            token.original, token.normalized, token.token_type
        );
    }

    // ═══════════════════════════════════════════════════════════
    // 3. Предобработка
    // ═══════════════════════════════════════════════════════════
    println!("\n3️⃣  PREPROCESSING");
    println!("──────────────────────────────────────────────\n");

    let preprocessor = Preprocessor::new().with_crypto_stopwords();

    let raw_tokens: Vec<String> = vec![
        "the", "bitcoin", "is", "running", "very", "bullish", "today", "crypto",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    println!("Before: {:?}", raw_tokens);
    let processed = preprocessor.process(&raw_tokens);
    println!("After:  {:?}", processed);

    println!("\nStemming examples:");
    let stem_examples = ["running", "bullish", "trading", "mooning", "crashed"];
    for word in stem_examples {
        println!("  {} -> {}", word, preprocessor.stem(word));
    }

    // ═══════════════════════════════════════════════════════════
    // 4. Bag of Words & TF-IDF
    // ═══════════════════════════════════════════════════════════
    println!("\n4️⃣  VECTORIZATION (Bag of Words & TF-IDF)");
    println!("──────────────────────────────────────────────\n");

    // Подготавливаем документы
    let docs: Vec<Vec<String>> = sample_texts
        .iter()
        .map(|text| {
            let tokens = tokenizer.tokenize_to_strings(text);
            preprocessor.process(&tokens)
        })
        .collect();

    // Bag of Words
    let mut bow = BagOfWords::new().with_min_df(1);
    let bow_matrix = bow.fit_transform(&docs);

    println!("Bag of Words:");
    println!("  Documents: {}", bow_matrix.n_documents());
    println!("  Vocabulary size: {}", bow_matrix.n_terms());
    println!("  Sample terms: {:?}", &bow_matrix.terms[..bow_matrix.terms.len().min(10)]);

    // TF-IDF
    let mut tfidf = TfIdf::new().with_min_df(1);
    let tfidf_matrix = tfidf.fit_transform(&docs);

    println!("\nTF-IDF:");
    println!("  Top terms by importance:");
    let top_terms = tfidf.top_terms(10);
    for (term, idf) in top_terms {
        println!("    • {:15} (IDF: {:.3})", term, idf);
    }

    // ═══════════════════════════════════════════════════════════
    // 5. Анализ настроений
    // ═══════════════════════════════════════════════════════════
    println!("\n5️⃣  SENTIMENT ANALYSIS");
    println!("──────────────────────────────────────────────\n");

    let analyzer = SentimentAnalyzer::new();

    println!("Analyzing sample texts:\n");

    let mut results = Vec::new();
    for text in &sample_texts {
        let result = analyzer.analyze(text);
        results.push(result.clone());

        let emoji = match result.polarity {
            rust_nlp_crypto::models::Polarity::Positive => "😊",
            rust_nlp_crypto::models::Polarity::Negative => "😟",
            rust_nlp_crypto::models::Polarity::Neutral => "😐",
        };

        let bar = create_score_bar(result.score);

        println!("{} {:+.2} {} \"{}\"", emoji, result.score, bar, truncate(text, 50));

        if !result.key_words.is_empty() {
            let keywords: Vec<String> = result
                .key_words
                .iter()
                .take(3)
                .map(|w| format!("{}({:+.1})", w.word, w.score))
                .collect();
            println!("         Keywords: {}", keywords.join(", "));
        }
        println!();
    }

    // Агрегированные результаты
    println!("📊 Aggregated Results:");
    println!("─────────────────────");
    let aggregated = analyzer.aggregate(&results);
    println!("{}", aggregated);

    // ═══════════════════════════════════════════════════════════
    // 6. Наивный Байес
    // ═══════════════════════════════════════════════════════════
    println!("\n6️⃣  NAIVE BAYES CLASSIFIER");
    println!("──────────────────────────────────────────────\n");

    // Создаём обучающие данные
    let train_docs = vec![
        vec!["moon", "bullish", "pump", "gains"],
        vec!["great", "excellent", "profit", "winning"],
        vec!["crash", "dump", "scam", "rekt"],
        vec!["terrible", "horrible", "lost", "liquidated"],
        vec!["okay", "stable", "normal", "waiting"],
        vec!["uncertain", "maybe", "possibly", "neutral"],
    ]
    .iter()
    .map(|words| words.iter().map(|s| s.to_string()).collect())
    .collect::<Vec<Vec<String>>>();

    let train_labels = vec![
        "positive", "positive", "negative", "negative", "neutral", "neutral",
    ];

    // Обучаем классификатор
    let mut classifier = NaiveBayesClassifier::new();
    classifier.fit(&train_docs, &train_labels.iter().map(|s| s.to_string()).collect::<Vec<_>>());

    println!("Classifier trained!");
    println!("  Classes: {:?}", classifier.classes());
    println!("  Vocabulary size: {}", classifier.vocab_size());

    // Тестируем
    println!("\nClassifying new texts:");

    let test_samples = vec![
        vec!["bullish", "moon", "gains"],
        vec!["crash", "dump", "terrible"],
        vec!["stable", "waiting", "normal"],
    ];

    for sample in &test_samples {
        let prediction = classifier.predict(sample);
        let probs = classifier.predict_proba(sample);

        println!("\n  Input: {:?}", sample);
        println!("  Prediction: {:?}", prediction);
        println!("  Probabilities:");
        for (class, prob) in &probs {
            let bar_len = (prob * 20.0) as usize;
            let bar = "█".repeat(bar_len);
            println!("    {:10} {:.1}% {}", class, prob * 100.0, bar);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("   Demo Complete! 🎉");
    println!("═══════════════════════════════════════════════════════════\n");
}

/// Создать визуальную полоску для оценки
fn create_score_bar(score: f64) -> String {
    let normalized = ((score + 1.0) / 2.0 * 20.0) as usize;
    let bar: String = (0..20)
        .map(|i| {
            if i < normalized {
                if score > 0.0 {
                    '█'
                } else {
                    '▓'
                }
            } else {
                '░'
            }
        })
        .collect();

    format!("[{}]", bar)
}

/// Обрезать текст до указанной длины
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
