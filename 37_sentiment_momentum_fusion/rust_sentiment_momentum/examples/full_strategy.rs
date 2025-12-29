//! Пример: Полная стратегия Sentiment-Momentum Fusion
//!
//! Демонстрирует полный цикл:
//! 1. Получение данных с Bybit
//! 2. Генерация sentiment данных
//! 3. Расчёт моментумов
//! 4. Fusion сигналы
//! 5. Обнаружение дивергенций
//!
//! Запуск: cargo run --example full_strategy

use anyhow::Result;
use chrono::{Duration, Utc};
use rust_sentiment_momentum::{
    api::BybitClient,
    momentum::{calculate_price_momentum, calculate_sentiment_momentum},
    sentiment::MockSentimentGenerator,
    signals::{DivergenceDetector, FusionStrategy},
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     SENTIMENT-MOMENTUM FUSION STRATEGY DEMONSTRATION       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Параметры
    let symbol = "BTCUSDT";
    let interval = "60"; // 1 час
    let limit = 100u32;
    let momentum_period = 20;
    let price_weight = 0.6;
    let sentiment_weight = 0.4;

    println!("Параметры:");
    println!("  Символ: {}", symbol);
    println!("  Интервал: {} минут", interval);
    println!("  Количество свечей: {}", limit);
    println!("  Период моментума: {}", momentum_period);
    println!(
        "  Веса: Price={:.0}%, Sentiment={:.0}%",
        price_weight * 100.0,
        sentiment_weight * 100.0
    );
    println!();

    // ========================================
    // Шаг 1: Получение данных с Bybit
    // ========================================
    println!("═══ Шаг 1: Получение данных с Bybit ═══\n");

    let client = BybitClient::new();
    let klines = client
        .fetch_klines(symbol, interval, None, None, Some(limit))
        .await?;

    println!("Получено {} свечей", klines.len());

    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();

    if let (Some(first), Some(last)) = (prices.first(), prices.last()) {
        let total_change = (last - first) / first * 100.0;
        println!("Начальная цена: ${:.2}", first);
        println!("Конечная цена: ${:.2}", last);
        println!("Общее изменение: {:+.2}%", total_change);
    }
    println!();

    // ========================================
    // Шаг 2: Генерация sentiment данных
    // ========================================
    println!("═══ Шаг 2: Генерация sentiment данных ═══\n");

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let start_time = Utc::now() - Duration::hours(limit as i64);
    let mut generator = MockSentimentGenerator::new(42, 0.0, 0.25, 80);

    let sentiment_data = generator.generate_with_price_correlation(
        symbol,
        &returns,
        start_time,
        1, // 1 час
        0.5, // 50% корреляция
    );

    let sentiments: Vec<f64> = sentiment_data.iter().map(|s| s.sentiment).collect();

    let avg_sentiment: f64 = sentiments.iter().sum::<f64>() / sentiments.len() as f64;
    let bullish_ratio = sentiments.iter().filter(|&&s| s > 0.1).count() as f64 / sentiments.len() as f64;

    println!("Сгенерировано {} sentiment точек", sentiments.len());
    println!("Средний sentiment: {:+.2}", avg_sentiment);
    println!("Доля бычьих: {:.0}%", bullish_ratio * 100.0);
    println!();

    // ========================================
    // Шаг 3: Расчёт моментумов
    // ========================================
    println!("═══ Шаг 3: Расчёт моментумов ═══\n");

    let price_momentum = calculate_price_momentum(&prices, momentum_period);
    let sentiment_momentum = calculate_sentiment_momentum(&sentiments, momentum_period);

    println!("Price momentum: {} точек", price_momentum.len());
    println!("Sentiment momentum: {} точек", sentiment_momentum.len());

    if let Some(pm) = price_momentum.last() {
        println!(
            "\nПоследний Price momentum: {:+.4} (z={:+.2})",
            pm.value, pm.z_score
        );
    }

    if let Some(sm) = sentiment_momentum.last() {
        println!(
            "Последний Sentiment momentum: {:+.4} (z={:+.2})",
            sm.value, sm.z_score
        );
    }
    println!();

    // ========================================
    // Шаг 4: Fusion стратегия
    // ========================================
    println!("═══ Шаг 4: Fusion стратегия ═══\n");

    let strategy = FusionStrategy::new(price_weight, sentiment_weight);

    let price_mom_values: Vec<f64> = price_momentum.iter().map(|m| m.z_score).collect();
    let sent_mom_values: Vec<f64> = sentiment_momentum.iter().map(|m| m.value).collect();

    let signals = strategy.calculate_signals(&price_mom_values, &sent_mom_values);

    println!("Сгенерировано {} сигналов\n", signals.len());

    // Статистика сигналов
    let strong_long = signals.iter().filter(|s| s.signal.value > 0.5).count();
    let long = signals
        .iter()
        .filter(|s| s.signal.value > 0.2 && s.signal.value <= 0.5)
        .count();
    let neutral = signals
        .iter()
        .filter(|s| s.signal.value.abs() <= 0.2)
        .count();
    let short = signals
        .iter()
        .filter(|s| s.signal.value < -0.2 && s.signal.value >= -0.5)
        .count();
    let strong_short = signals.iter().filter(|s| s.signal.value < -0.5).count();

    println!("Распределение сигналов:");
    println!(
        "  Strong Long: {} ({:.0}%)",
        strong_long,
        strong_long as f64 / signals.len() as f64 * 100.0
    );
    println!(
        "  Long: {} ({:.0}%)",
        long,
        long as f64 / signals.len() as f64 * 100.0
    );
    println!(
        "  Neutral: {} ({:.0}%)",
        neutral,
        neutral as f64 / signals.len() as f64 * 100.0
    );
    println!(
        "  Short: {} ({:.0}%)",
        short,
        short as f64 / signals.len() as f64 * 100.0
    );
    println!(
        "  Strong Short: {} ({:.0}%)",
        strong_short,
        strong_short as f64 / signals.len() as f64 * 100.0
    );
    println!();

    // ========================================
    // Шаг 5: Обнаружение дивергенций
    // ========================================
    println!("═══ Шаг 5: Обнаружение дивергенций ═══\n");

    let detector = DivergenceDetector::default();
    let stats = detector.count_divergences(&price_mom_values, &sent_mom_values);

    println!("Статистика дивергенций:");
    println!("  Бычьих: {}", stats.bullish_count);
    println!("  Медвежьих: {}", stats.bearish_count);
    println!("  Сильных: {}", stats.strong_count);
    println!(
        "  Доля дивергенций: {:.1}%",
        stats.divergence_ratio() * 100.0
    );
    println!();

    // Последняя сильная дивергенция
    if let Some((idx, div)) = detector.find_last_strong_divergence(&price_mom_values, &sent_mom_values) {
        println!(
            "Последняя сильная дивергенция на позиции {}:",
            idx
        );
        println!("  Тип: {:?}", div.divergence_type);
        println!("  Сила: {:.0}%", div.strength * 100.0);
        println!(
            "  Price momentum: {:+.3}",
            div.price_momentum
        );
        println!(
            "  Sentiment momentum: {:+.3}",
            div.sentiment_momentum
        );
    }
    println!();

    // ========================================
    // Итоговый сигнал
    // ========================================
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                    ТЕКУЩИЙ СИГНАЛ                          ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    if let Some(last) = signals.last() {
        println!("Тип сигнала: {:?}", last.signal.signal_type);
        println!("Значение: {:+.2}", last.signal.value);
        println!("Уверенность: {:.0}%", last.confidence * 100.0);
        println!();
        println!("Компоненты:");
        println!(
            "  Price momentum (z): {:+.2}",
            last.price_component
        );
        println!(
            "  Sentiment momentum: {:+.2}",
            last.sentiment_component
        );

        if let Some(ref div) = last.divergence {
            println!();
            println!("⚠️  ВНИМАНИЕ: Обнаружена дивергенция!");
            println!("   Тип: {:?}", div.divergence_type);
            println!("   Сила: {:.0}%", div.strength * 100.0);

            if div.is_strong {
                println!();
                println!("   ⚠️  СИЛЬНАЯ ДИВЕРГЕНЦИЯ");
                println!("   Рекомендация: Осторожность, возможен разворот тренда");
            }
        } else {
            println!();
            println!("✓ Дивергенция отсутствует - сигнал подтверждён");
        }

        println!();

        // Рекомендация
        let recommendation = if last.divergence.as_ref().map(|d| d.is_strong).unwrap_or(false) {
            "ОСТОРОЖНОСТЬ - подождите подтверждения"
        } else if last.signal.value > 0.5 && last.confidence > 0.6 {
            "СИЛЬНАЯ ПОКУПКА"
        } else if last.signal.value > 0.2 {
            "ПОКУПКА"
        } else if last.signal.value < -0.5 && last.confidence > 0.6 {
            "СИЛЬНАЯ ПРОДАЖА"
        } else if last.signal.value < -0.2 {
            "ПРОДАЖА"
        } else {
            "ОЖИДАНИЕ - нет чёткого сигнала"
        };

        println!("═══════════════════════════════════════");
        println!("  РЕКОМЕНДАЦИЯ: {}", recommendation);
        println!("═══════════════════════════════════════");
    }

    println!("\n✓ Анализ завершён\n");

    Ok(())
}
