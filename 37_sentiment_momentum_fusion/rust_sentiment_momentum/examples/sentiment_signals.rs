//! Пример: Sentiment сигналы
//!
//! Демонстрирует анализ sentiment и генерацию сигналов
//! на основе настроений.
//!
//! Запуск: cargo run --example sentiment_signals

use anyhow::Result;
use chrono::{Duration, Utc};
use rust_sentiment_momentum::sentiment::{
    MockSentimentGenerator, SentimentAggregator, SentimentAnalyzer, SentimentMessage,
};

fn main() -> Result<()> {
    println!("=== Анализ Sentiment ===\n");

    // 1. Анализатор настроений на основе словаря
    println!("--- Анализатор настроений ---\n");

    let analyzer = SentimentAnalyzer::new();

    let test_texts = [
        "Bitcoin to the moon! Very bullish!",
        "Market crash incoming, sell everything!",
        "BTC price is at 50000 today",
        "Extremely bullish on ETH, hodl strong!",
        "This looks like a scam, stay away",
        "Not sure about this, wait and see",
        "Крипта на луну! Рост продолжается!",
        "Ужас, паника на рынке!",
    ];

    for text in &test_texts {
        let score = analyzer.analyze(text);
        println!("Текст: \"{}\"", text);
        println!(
            "  Score: {:+.2} | Label: {} | Confidence: {:.0}%\n",
            score.score,
            score.label(),
            score.confidence * 100.0
        );
    }

    // 2. Генерация тестовых данных
    println!("--- Генерация тестовых сообщений ---\n");

    let mut generator = MockSentimentGenerator::new(42, 0.2, 0.3, 50);
    let now = Utc::now();
    let start = now - Duration::hours(24);

    let messages = generator.generate_messages("BTCUSDT", start, now);
    println!("Сгенерировано {} сообщений за 24 часа\n", messages.len());

    // Статистика по источникам
    let twitter_count = messages.iter().filter(|m| m.source == "twitter").count();
    let reddit_count = messages.iter().filter(|m| m.source == "reddit").count();
    let telegram_count = messages.iter().filter(|m| m.source == "telegram").count();

    println!("По источникам:");
    println!("  Twitter: {}", twitter_count);
    println!("  Reddit: {}", reddit_count);
    println!("  Telegram: {}", telegram_count);
    println!();

    // Средний sentiment
    let avg_sentiment: f64 =
        messages.iter().map(|m| m.sentiment).sum::<f64>() / messages.len() as f64;
    let bullish = messages.iter().filter(|m| m.sentiment > 0.1).count();
    let bearish = messages.iter().filter(|m| m.sentiment < -0.1).count();
    let neutral = messages.len() - bullish - bearish;

    println!("Распределение настроений:");
    println!("  Средний sentiment: {:.2}", avg_sentiment);
    println!(
        "  Бычьих: {} ({:.0}%)",
        bullish,
        bullish as f64 / messages.len() as f64 * 100.0
    );
    println!(
        "  Медвежьих: {} ({:.0}%)",
        bearish,
        bearish as f64 / messages.len() as f64 * 100.0
    );
    println!(
        "  Нейтральных: {} ({:.0}%)",
        neutral,
        neutral as f64 / messages.len() as f64 * 100.0
    );
    println!();

    // 3. Агрегация настроений
    println!("--- Агрегация настроений ---\n");

    let aggregator = SentimentAggregator::new(12.0, 10); // 12 часов halflife

    // Агрегация по часам
    let hourly = aggregator.aggregate_by_windows("BTCUSDT", &messages, 1, start, now);

    println!("Почасовая агрегация ({} точек):\n", hourly.len());

    // Последние 5 часов
    println!("Последние 5 часов:");
    for agg in hourly.iter().rev().take(5).rev() {
        println!(
            "  {}: sentiment={:+.2}, volume={}, confidence={:.0}%",
            agg.timestamp.format("%H:%M"),
            agg.sentiment,
            agg.volume,
            agg.confidence * 100.0
        );
    }
    println!();

    // Тренд sentiment
    if hourly.len() >= 2 {
        let first_half: f64 = hourly[..hourly.len() / 2]
            .iter()
            .map(|a| a.sentiment)
            .sum::<f64>()
            / (hourly.len() / 2) as f64;

        let second_half: f64 = hourly[hourly.len() / 2..]
            .iter()
            .map(|a| a.sentiment)
            .sum::<f64>()
            / (hourly.len() - hourly.len() / 2) as f64;

        let trend = second_half - first_half;

        println!("Тренд sentiment:");
        println!("  Первая половина: {:.2}", first_half);
        println!("  Вторая половина: {:.2}", second_half);
        println!(
            "  Тренд: {:+.2} ({})",
            trend,
            if trend > 0.05 {
                "растущий"
            } else if trend < -0.05 {
                "падающий"
            } else {
                "стабильный"
            }
        );
    }
    println!();

    // 4. Скользящий sentiment
    println!("--- Скользящий средний sentiment ---\n");

    let rolling = aggregator.rolling_sentiment(&hourly, 3);

    if rolling.len() >= 5 {
        println!("Последние 5 значений MA(3):");
        for (i, &val) in rolling.iter().rev().take(5).rev().enumerate() {
            println!(
                "  {}: {:.2}",
                hourly.len() - 5 + i,
                val
            );
        }
    }
    println!();

    // 5. Сигналы на основе sentiment
    println!("=== Генерация сигналов ===\n");

    if let Some(last) = hourly.last() {
        let signal_strength = last.sentiment * last.confidence;

        println!("Текущее состояние:");
        println!("  Sentiment: {:+.2}", last.sentiment);
        println!("  Уверенность: {:.0}%", last.confidence * 100.0);
        println!("  Сила сигнала: {:+.2}", signal_strength);
        println!();

        let signal = if signal_strength > 0.3 {
            "СИЛЬНЫЙ LONG"
        } else if signal_strength > 0.1 {
            "LONG"
        } else if signal_strength < -0.3 {
            "СИЛЬНЫЙ SHORT"
        } else if signal_strength < -0.1 {
            "SHORT"
        } else {
            "НЕЙТРАЛЬНЫЙ"
        };

        println!("Сигнал: {}", signal);

        // Дополнительные условия
        if last.bullish_ratio > 0.7 {
            println!("  + Высокая доля бычьих сообщений ({:.0}%)", last.bullish_ratio * 100.0);
        } else if last.bullish_ratio < 0.3 {
            println!("  + Высокая доля медвежьих сообщений ({:.0}%)", (1.0 - last.bullish_ratio) * 100.0);
        }

        if last.volume > 50 {
            println!("  + Высокий объём сообщений ({})", last.volume);
        } else if last.volume < 10 {
            println!("  - Низкий объём сообщений ({}) - сигнал менее надёжен", last.volume);
        }
    }

    Ok(())
}
