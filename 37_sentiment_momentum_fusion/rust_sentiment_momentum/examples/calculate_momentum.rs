//! Пример: Расчёт моментума
//!
//! Демонстрирует расчёт ценового и sentiment моментума
//! на данных Bybit.
//!
//! Запуск: cargo run --example calculate_momentum

use anyhow::Result;
use rust_sentiment_momentum::{
    api::BybitClient,
    momentum::{
        calculate_price_momentum, calculate_sentiment_momentum, cumulative_return, ema,
        volatility, MomentumCalculator, PriceMomentum,
    },
    sentiment::MockSentimentGenerator,
};
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Расчёт моментума ===\n");

    // Получаем данные
    let client = BybitClient::new();
    let klines = client
        .fetch_klines("BTCUSDT", "60", None, None, Some(100))
        .await?;

    println!("Получено {} свечей для BTCUSDT\n", klines.len());

    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();

    // Расчёт различных типов моментума
    println!("--- Ценовой моментум ---\n");

    // PriceMomentum с различными периодами
    for period in [5, 10, 20] {
        if let Some(mom) = PriceMomentum::calculate(&prices, period) {
            println!("Период {}-свечей:", period);
            println!("  ROC: {:+.2}%", mom.roc * 100.0);
            println!("  Отклонение от SMA: {:+.2}%", mom.deviation_from_sma * 100.0);
            println!("  RSI: {:.1}", mom.rsi);
            println!(
                "  Направление: {}",
                if mom.is_bullish() {
                    "Бычий"
                } else if mom.is_bearish() {
                    "Медвежий"
                } else {
                    "Нейтральный"
                }
            );
            println!("  Сила: {:.0}%", mom.strength() * 100.0);
            println!();
        }
    }

    // Кумулятивная доходность
    println!("--- Кумулятивная доходность ---\n");
    for period in [10, 20, 50] {
        if let Some(ret) = cumulative_return(&prices, period) {
            println!("  {}-свечей: {:+.2}%", period, ret * 100.0);
        }
    }
    println!();

    // Волатильность
    println!("--- Волатильность ---\n");
    for period in [10, 20, 50] {
        if let Some(vol) = volatility(&prices, period) {
            println!("  {}-свечей: {:.2}%", period, vol * 100.0);
        }
    }
    println!();

    // EMA
    println!("--- Экспоненциальное скользящее среднее ---\n");
    let ema_20 = ema(&prices, 20);
    if let Some(&last_ema) = ema_20.last() {
        let last_price = *prices.last().unwrap();
        let deviation = (last_price - last_ema) / last_ema * 100.0;
        println!("  EMA(20): ${:.2}", last_ema);
        println!("  Текущая цена: ${:.2}", last_price);
        println!("  Отклонение: {:+.2}%", deviation);
    }
    println!();

    // Множественные периоды с калькулятором
    println!("--- MomentumCalculator (множественные периоды) ---\n");

    let calculator = MomentumCalculator::new(&[5, 10, 20]);
    let all_momentums = calculator.calculate_all_price(&prices);

    for (i, period_results) in all_momentums.iter().enumerate() {
        let period = [5, 10, 20][i];
        if let Some(last) = period_results.last() {
            println!(
                "  Период {}: ROC={:+.4}, Z-score={:+.2}",
                period, last.value, last.z_score
            );
        }
    }
    println!();

    // Sentiment моментум
    println!("=== Sentiment моментум ===\n");

    // Генерируем sentiment данные
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let start_time = Utc::now() - Duration::hours(100);
    let mut generator = MockSentimentGenerator::new(42, 0.0, 0.3, 100);
    let sentiment_data =
        generator.generate_with_price_correlation("BTCUSDT", &returns, start_time, 1, 0.6);

    let sentiments: Vec<f64> = sentiment_data.iter().map(|s| s.sentiment).collect();

    println!("Сгенерировано {} sentiment точек\n", sentiments.len());

    // Sentiment статистика
    let avg_sentiment: f64 = sentiments.iter().sum::<f64>() / sentiments.len() as f64;
    let bullish_ratio =
        sentiments.iter().filter(|&&s| s > 0.1).count() as f64 / sentiments.len() as f64;

    println!("Средний sentiment: {:.2}", avg_sentiment);
    println!("Доля бычьих: {:.0}%", bullish_ratio * 100.0);
    println!();

    // Sentiment моментум
    let sent_momentum = calculate_sentiment_momentum(&sentiments, 10);
    if let Some(last) = sent_momentum.last() {
        println!("Sentiment моментум (10 периодов):");
        println!("  Изменение: {:+.3}", last.value);
        println!("  Z-score: {:+.2}", last.z_score);
        println!(
            "  Направление: {}",
            match last.direction {
                1 => "Растущий",
                -1 => "Падающий",
                _ => "Нейтральный",
            }
        );
    }
    println!();

    // Сравнение моментумов
    println!("=== Сравнение Price vs Sentiment моментума ===\n");

    let price_mom = calculate_price_momentum(&prices, 20);
    let sent_mom = calculate_sentiment_momentum(&sentiments, 20);

    if let (Some(p), Some(s)) = (price_mom.last(), sent_mom.last()) {
        println!("Период 20:");
        println!("  Price momentum: {:+.4} (z={:+.2})", p.value, p.z_score);
        println!("  Sentiment momentum: {:+.4} (z={:+.2})", s.value, s.z_score);

        let aligned = (p.direction == 1 && s.direction == 1)
            || (p.direction == -1 && s.direction == -1);

        if aligned {
            println!("\n  ✓ Моментумы согласованы - сильный сигнал");
        } else if p.direction != 0 && s.direction != 0 && p.direction != s.direction {
            println!("\n  ⚠ Дивергенция между ценой и sentiment!");
        } else {
            println!("\n  → Смешанные сигналы");
        }
    }

    Ok(())
}
