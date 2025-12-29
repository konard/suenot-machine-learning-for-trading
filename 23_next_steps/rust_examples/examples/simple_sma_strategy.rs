//! Пример простой SMA Crossover стратегии
//!
//! Запуск: cargo run --example simple_sma_strategy

use anyhow::Result;
use ml4t_bybit::client::BybitClient;
use ml4t_bybit::data::Interval;
use ml4t_bybit::indicators::{Indicator, SMA};
use ml4t_bybit::strategies::{Signal, SmaCrossStrategy, Strategy};

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════╗");
    println!("║       SMA Crossover Strategy Demo          ║");
    println!("╚════════════════════════════════════════════╝");
    println!();

    // Создаём клиент
    let client = BybitClient::new();

    // Символ для анализа
    let symbol = "BTCUSDT";

    println!("📊 Анализ {} с использованием SMA Crossover", symbol);
    println!();

    // Получаем исторические данные
    let klines = client
        .get_klines(symbol, Interval::H1, Some(200))
        .await?;

    println!("Получено {} свечей", klines.len());
    println!();

    // Извлекаем цены закрытия
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

    // Создаём индикаторы
    let fast_sma = SMA::new(10);
    let slow_sma = SMA::new(20);

    // Рассчитываем SMA
    let fast_values = fast_sma.calculate(&closes);
    let slow_values = slow_sma.calculate(&closes);

    // Выводим последние значения
    println!("═══════════════════════════════════════════════");
    println!("               Текущие значения                ");
    println!("═══════════════════════════════════════════════");

    if let Some(last_price) = closes.last() {
        println!("  Текущая цена:  ${:.2}", last_price);
    }

    if let Some(fast) = fast_values.last() {
        println!("  SMA(10):       ${:.2}", fast);
    }

    if let Some(slow) = slow_values.last() {
        println!("  SMA(20):       ${:.2}", slow);
    }

    // Определяем текущее положение
    if let (Some(&fast), Some(&slow)) = (fast_values.last(), slow_values.last()) {
        let position = if fast > slow {
            "🟢 Быстрая SMA ВЫШЕ медленной (бычий тренд)"
        } else if fast < slow {
            "🔴 Быстрая SMA НИЖЕ медленной (медвежий тренд)"
        } else {
            "🟡 SMA пересекаются"
        };
        println!();
        println!("  {}", position);
    }

    println!();

    // Используем стратегию для генерации сигнала
    println!("═══════════════════════════════════════════════");
    println!("                  Сигналы                      ");
    println!("═══════════════════════════════════════════════");

    let strategy = SmaCrossStrategy::new(10, 20);
    println!("Стратегия: {}", strategy.name());
    println!();

    // Генерируем сигналы для всех точек
    let signals = strategy.generate_signals(&klines);

    // Считаем статистику сигналов
    let buy_signals = signals.iter().filter(|s| **s == Signal::Buy).count();
    let sell_signals = signals.iter().filter(|s| **s == Signal::Sell).count();

    println!("За последние {} свечей:", klines.len());
    println!("  • Сигналов на покупку:  {}", buy_signals);
    println!("  • Сигналов на продажу:  {}", sell_signals);
    println!();

    // Текущий сигнал
    let current_signal = strategy.generate_signal(&klines);
    let signal_str = match current_signal {
        Signal::Buy => "🟢 ПОКУПКА",
        Signal::Sell => "🔴 ПРОДАЖА",
        Signal::Hold => "🟡 УДЕРЖАНИЕ",
    };

    println!("Текущий сигнал: {}", signal_str);
    println!();

    // Показываем последние 5 пересечений
    println!("═══════════════════════════════════════════════");
    println!("           Последние пересечения               ");
    println!("═══════════════════════════════════════════════");

    let mut crossovers: Vec<(usize, Signal)> = Vec::new();
    for (i, signal) in signals.iter().enumerate() {
        if *signal != Signal::Hold {
            crossovers.push((i + strategy.min_bars(), *signal));
        }
    }

    for (idx, signal) in crossovers.iter().rev().take(5) {
        let kline = &klines[*idx];
        let datetime = chrono::DateTime::from_timestamp_millis(kline.timestamp as i64)
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let signal_str = match signal {
            Signal::Buy => "🟢 BUY ",
            Signal::Sell => "🔴 SELL",
            _ => "      ",
        };

        println!("  {} | {} | ${:.2}", datetime, signal_str, kline.close);
    }

    println!();

    // Визуализация последних значений
    println!("═══════════════════════════════════════════════");
    println!("          Последние 10 значений SMA            ");
    println!("═══════════════════════════════════════════════");
    println!();
    println!("  {:>12} │ {:>12} │ {:>12} │ {:>8}", "Цена", "SMA(10)", "SMA(20)", "Разница");
    println!("  ─────────────┼──────────────┼──────────────┼─────────");

    let offset = fast_values.len() - slow_values.len();
    let n = slow_values.len().min(10);

    for i in (slow_values.len() - n)..slow_values.len() {
        let price = closes[closes.len() - slow_values.len() + i];
        let fast = fast_values[i + offset];
        let slow = slow_values[i];
        let diff = fast - slow;

        let indicator = if diff > 0.0 { "+" } else { "" };

        println!(
            "  {:>12.2} │ {:>12.2} │ {:>12.2} │ {:>7}{:.2}",
            price, fast, slow, indicator, diff.abs()
        );
    }

    println!();
    println!("✅ Анализ завершён!");

    Ok(())
}
