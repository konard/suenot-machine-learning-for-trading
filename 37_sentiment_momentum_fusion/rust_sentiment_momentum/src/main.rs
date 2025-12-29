//! Sentiment-Momentum Fusion CLI
//!
//! Командный интерфейс для анализа криптовалютных данных
//! с использованием стратегии Sentiment-Momentum Fusion.

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use rust_sentiment_momentum::{
    api::BybitClient,
    momentum::{calculate_price_momentum, calculate_sentiment_momentum},
    sentiment::MockSentimentGenerator,
    signals::{DivergenceDetector, FusionStrategy},
};

/// Sentiment-Momentum Fusion Strategy CLI
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Символ торговой пары (например, BTCUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Интервал свечей (1, 5, 15, 60, 240, D)
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Количество свечей для анализа
    #[arg(short, long, default_value_t = 100)]
    limit: u32,

    /// Период для расчёта моментума
    #[arg(short, long, default_value_t = 20)]
    momentum_period: usize,

    /// Вес ценового моментума (0.0 - 1.0)
    #[arg(long, default_value_t = 0.6)]
    price_weight: f64,

    /// Вес sentiment-моментума (0.0 - 1.0)
    #[arg(long, default_value_t = 0.4)]
    sentiment_weight: f64,

    /// Использовать тестовую сеть Bybit
    #[arg(long)]
    testnet: bool,

    /// Уровень логирования (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Настройка логирования
    let log_level = match args.log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder().with_max_level(log_level).finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Запуск Sentiment-Momentum Fusion Strategy");
    info!("Символ: {}", args.symbol);
    info!("Интервал: {}", args.interval);

    // Создаём клиент Bybit
    let client = if args.testnet {
        info!("Использование тестовой сети");
        BybitClient::testnet()
    } else {
        BybitClient::new()
    };

    // Получаем данные с биржи
    info!("Получение данных с Bybit...");
    let klines = client
        .fetch_klines(&args.symbol, &args.interval, None, None, Some(args.limit))
        .await?;

    if klines.is_empty() {
        println!("Нет данных для анализа");
        return Ok(());
    }

    info!("Получено {} свечей", klines.len());

    // Извлекаем цены закрытия
    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

    // Рассчитываем доходности для генерации sentiment
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| {
            if w[0] != 0.0 {
                (w[1] - w[0]) / w[0]
            } else {
                0.0
            }
        })
        .collect();

    // Генерируем симулированные sentiment-данные
    // (в реальном приложении здесь был бы API социальных сетей)
    info!("Генерация sentiment-данных...");
    let start_time = klines.first().map(|k| k.datetime()).unwrap_or_else(Utc::now);
    let mut sentiment_generator = MockSentimentGenerator::new(42, 0.0, 0.25, 100);
    let sentiment_data = sentiment_generator.generate_with_price_correlation(
        &args.symbol,
        &returns,
        start_time,
        1, // 1 час на интервал
        0.5, // 50% корреляция с ценой
    );

    let sentiments: Vec<f64> = sentiment_data.iter().map(|s| s.sentiment).collect();

    // Рассчитываем моментум
    info!("Расчёт моментума...");
    let price_momentum = calculate_price_momentum(&prices, args.momentum_period);
    let sentiment_momentum = calculate_sentiment_momentum(&sentiments, args.momentum_period);

    // Создаём стратегию
    let strategy = FusionStrategy::new(args.price_weight, args.sentiment_weight);
    let divergence_detector = DivergenceDetector::default();

    // Получаем значения моментума
    let price_mom_values: Vec<f64> = price_momentum.iter().map(|m| m.z_score).collect();
    let sent_mom_values: Vec<f64> = sentiment_momentum.iter().map(|m| m.value).collect();

    // Генерируем сигналы
    info!("Генерация сигналов...");
    let signals = strategy.calculate_signals(&price_mom_values, &sent_mom_values);

    // Выводим результаты
    println!("\n========================================");
    println!("  SENTIMENT-MOMENTUM FUSION ANALYSIS");
    println!("========================================\n");

    println!("Символ: {}", args.symbol);
    println!("Период анализа: {} свечей", klines.len());
    println!(
        "Веса: Price={:.0}%, Sentiment={:.0}%",
        args.price_weight * 100.0,
        args.sentiment_weight * 100.0
    );
    println!();

    // Текущее состояние
    if let Some(last_kline) = klines.last() {
        println!("Текущая цена: ${:.2}", last_kline.close);
    }

    if let Some(last_sentiment) = sentiments.last() {
        println!(
            "Текущий sentiment: {:.2} ({})",
            last_sentiment,
            if *last_sentiment > 0.1 {
                "Позитивный"
            } else if *last_sentiment < -0.1 {
                "Негативный"
            } else {
                "Нейтральный"
            }
        );
    }

    println!();

    // Последний сигнал
    if let Some(last_signal) = signals.last() {
        println!("----------------------------------------");
        println!("ТЕКУЩИЙ СИГНАЛ:");
        println!("----------------------------------------");
        println!("  Тип: {:?}", last_signal.signal.signal_type);
        println!("  Сила: {:.2}", last_signal.signal.value);
        println!("  Уверенность: {:.0}%", last_signal.confidence * 100.0);
        println!(
            "  Price momentum: {:.2}",
            last_signal.price_component
        );
        println!(
            "  Sentiment momentum: {:.2}",
            last_signal.sentiment_component
        );

        if let Some(ref div) = last_signal.divergence {
            println!();
            println!("  ⚠️  ДИВЕРГЕНЦИЯ: {:?}", div.divergence_type);
            println!("      Сила: {:.0}%", div.strength * 100.0);
            if div.is_strong {
                println!("      ⚠️  СИЛЬНАЯ ДИВЕРГЕНЦИЯ - будьте осторожны!");
            }
        }
    }

    println!();

    // Статистика дивергенций
    let div_stats = divergence_detector.count_divergences(&price_mom_values, &sent_mom_values);
    println!("----------------------------------------");
    println!("СТАТИСТИКА ДИВЕРГЕНЦИЙ:");
    println!("----------------------------------------");
    println!("  Бычьих дивергенций: {}", div_stats.bullish_count);
    println!("  Медвежьих дивергенций: {}", div_stats.bearish_count);
    println!("  Сильных дивергенций: {}", div_stats.strong_count);
    println!(
        "  Доля дивергенций: {:.1}%",
        div_stats.divergence_ratio() * 100.0
    );

    println!();

    // Распределение сигналов
    let long_signals = signals.iter().filter(|s| s.signal.value > 0.2).count();
    let short_signals = signals.iter().filter(|s| s.signal.value < -0.2).count();
    let neutral_signals = signals.len() - long_signals - short_signals;

    println!("----------------------------------------");
    println!("РАСПРЕДЕЛЕНИЕ СИГНАЛОВ:");
    println!("----------------------------------------");
    println!("  Long сигналов: {} ({:.1}%)", long_signals, long_signals as f64 / signals.len() as f64 * 100.0);
    println!("  Short сигналов: {} ({:.1}%)", short_signals, short_signals as f64 / signals.len() as f64 * 100.0);
    println!("  Нейтральных: {} ({:.1}%)", neutral_signals, neutral_signals as f64 / signals.len() as f64 * 100.0);

    println!("\n========================================\n");

    info!("Анализ завершён");

    Ok(())
}
