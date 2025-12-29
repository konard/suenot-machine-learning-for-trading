//! Пример стратегии на основе RSI
//!
//! Запуск: cargo run --example rsi_strategy

use anyhow::Result;
use ml4t_bybit::client::BybitClient;
use ml4t_bybit::data::Interval;
use ml4t_bybit::indicators::{Indicator, RSI, MACD, BollingerBands};
use ml4t_bybit::strategies::{RsiStrategy, Signal, Strategy};

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════╗");
    println!("║          RSI Strategy Analysis             ║");
    println!("╚════════════════════════════════════════════╝");
    println!();

    let client = BybitClient::new();
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"];

    println!("Анализ криптовалют с использованием RSI, MACD и Bollinger Bands");
    println!();

    // Заголовок таблицы
    println!("┌────────────┬───────────┬────────┬────────────────┬──────────────┬──────────┐");
    println!("│   Symbol   │   Price   │  RSI   │     MACD       │   BB Band    │  Signal  │");
    println!("├────────────┼───────────┼────────┼────────────────┼──────────────┼──────────┤");

    for symbol in &symbols {
        match analyze_symbol(&client, symbol).await {
            Ok(analysis) => {
                println!(
                    "│ {:>10} │ {:>9} │ {:>6} │ {:>14} │ {:>12} │ {:>8} │",
                    analysis.symbol,
                    analysis.price,
                    analysis.rsi,
                    analysis.macd,
                    analysis.bb_position,
                    analysis.signal
                );
            }
            Err(e) => {
                println!("│ {:>10} │ Error: {:40} │", symbol, e.to_string());
            }
        }
    }

    println!("└────────────┴───────────┴────────┴────────────────┴──────────────┴──────────┘");
    println!();

    // Детальный анализ BTC
    println!("═══════════════════════════════════════════════");
    println!("         Детальный анализ BTCUSDT              ");
    println!("═══════════════════════════════════════════════");
    println!();

    let klines = client
        .get_klines("BTCUSDT", Interval::H1, Some(200))
        .await?;

    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

    // RSI
    let rsi = RSI::new(14);
    let rsi_values = rsi.calculate(&closes);

    if let Some(&current_rsi) = rsi_values.last() {
        println!("📊 RSI(14): {:.2}", current_rsi);
        println!();

        // Визуализация RSI
        let rsi_bar = create_rsi_bar(current_rsi);
        println!("   Oversold                      Overbought");
        println!("   0       30        50        70       100");
        println!("   ├────────┼─────────┼─────────┼────────┤");
        println!("   {}", rsi_bar);
        println!();

        // Интерпретация
        let interpretation = if current_rsi > 70.0 {
            "⚠️  ПЕРЕКУПЛЕННОСТЬ - возможен откат вниз"
        } else if current_rsi < 30.0 {
            "⚠️  ПЕРЕПРОДАННОСТЬ - возможен отскок вверх"
        } else if current_rsi > 50.0 {
            "📈 Бычий импульс"
        } else {
            "📉 Медвежий импульс"
        };
        println!("   {}", interpretation);
    }

    println!();

    // MACD
    let macd = MACD::standard();
    let macd_result = macd.calculate(&closes);

    if !macd_result.macd_line.is_empty() {
        let macd_value = *macd_result.macd_line.last().unwrap();
        let signal_value = *macd_result.signal_line.last().unwrap();
        let histogram = *macd_result.histogram.last().unwrap();

        println!("📊 MACD(12,26,9):");
        println!("   MACD Line:   {:.2}", macd_value);
        println!("   Signal Line: {:.2}", signal_value);
        println!("   Histogram:   {:.2} {}", histogram, if histogram > 0.0 { "🟢" } else { "🔴" });
        println!();

        let macd_interpretation = if histogram > 0.0 && macd_value > signal_value {
            "📈 Бычий импульс усиливается"
        } else if histogram < 0.0 && macd_value < signal_value {
            "📉 Медвежий импульс усиливается"
        } else if histogram > 0.0 {
            "⚠️  Бычий импульс ослабевает"
        } else {
            "⚠️  Медвежий импульс ослабевает"
        };
        println!("   {}", macd_interpretation);
    }

    println!();

    // Bollinger Bands
    let bb = BollingerBands::standard();
    let bb_result = bb.calculate(&closes);

    if !bb_result.upper.is_empty() {
        let upper = *bb_result.upper.last().unwrap();
        let middle = *bb_result.middle.last().unwrap();
        let lower = *bb_result.lower.last().unwrap();
        let current_price = *closes.last().unwrap();
        let percent_b = *bb_result.percent_b.last().unwrap();

        println!("📊 Bollinger Bands(20,2):");
        println!("   Upper Band:  ${:.2}", upper);
        println!("   Middle:      ${:.2}", middle);
        println!("   Lower Band:  ${:.2}", lower);
        println!("   Current:     ${:.2}", current_price);
        println!("   %B:          {:.2}%", percent_b * 100.0);
        println!();

        let bb_interpretation = if current_price > upper {
            "⚠️  Цена ВЫШЕ верхней полосы - возможен откат"
        } else if current_price < lower {
            "⚠️  Цена НИЖЕ нижней полосы - возможен отскок"
        } else if percent_b > 0.8 {
            "📈 Цена у верхней границы"
        } else if percent_b < 0.2 {
            "📉 Цена у нижней границы"
        } else {
            "⏸️  Цена в середине диапазона"
        };
        println!("   {}", bb_interpretation);
    }

    println!();

    // Общий сигнал стратегии
    let rsi_strategy = RsiStrategy::standard();
    let signal = rsi_strategy.generate_signal(&klines);

    println!("═══════════════════════════════════════════════");
    println!("              ИТОГОВЫЙ СИГНАЛ                  ");
    println!("═══════════════════════════════════════════════");

    let signal_emoji = match signal {
        Signal::Buy => "🟢 ПОКУПКА",
        Signal::Sell => "🔴 ПРОДАЖА",
        Signal::Hold => "🟡 УДЕРЖАНИЕ",
    };

    println!();
    println!("                 {}", signal_emoji);
    println!();

    println!("⚠️  ВАЖНО: Это не финансовая рекомендация!");
    println!("   Всегда проводите собственный анализ перед торговлей.");
    println!();

    Ok(())
}

struct SymbolAnalysis {
    symbol: String,
    price: String,
    rsi: String,
    macd: String,
    bb_position: String,
    signal: String,
}

async fn analyze_symbol(client: &BybitClient, symbol: &str) -> Result<SymbolAnalysis> {
    let klines = client
        .get_klines(symbol, Interval::H1, Some(100))
        .await?;

    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let current_price = *closes.last().unwrap_or(&0.0);

    // RSI
    let rsi = RSI::new(14);
    let rsi_values = rsi.calculate(&closes);
    let current_rsi = rsi_values.last().copied().unwrap_or(50.0);

    let rsi_str = format!("{:.1}", current_rsi);
    let rsi_with_indicator = if current_rsi > 70.0 {
        format!("{}⬆", rsi_str)
    } else if current_rsi < 30.0 {
        format!("{}⬇", rsi_str)
    } else {
        rsi_str
    };

    // MACD
    let macd = MACD::standard();
    let macd_result = macd.calculate(&closes);

    let macd_str = if !macd_result.histogram.is_empty() {
        let hist = *macd_result.histogram.last().unwrap();
        if hist > 0.0 {
            format!("{:.2} 🟢", hist)
        } else {
            format!("{:.2} 🔴", hist)
        }
    } else {
        "N/A".to_string()
    };

    // Bollinger Bands position
    let bb = BollingerBands::standard();
    let bb_result = bb.calculate(&closes);

    let bb_str = if !bb_result.percent_b.is_empty() {
        let percent_b = *bb_result.percent_b.last().unwrap();
        if percent_b > 0.8 {
            "Upper ⬆".to_string()
        } else if percent_b < 0.2 {
            "Lower ⬇".to_string()
        } else {
            "Middle".to_string()
        }
    } else {
        "N/A".to_string()
    };

    // Signal
    let rsi_strategy = RsiStrategy::standard();
    let signal = rsi_strategy.generate_signal(&klines);
    let signal_str = match signal {
        Signal::Buy => "BUY 🟢",
        Signal::Sell => "SELL 🔴",
        Signal::Hold => "HOLD 🟡",
    };

    Ok(SymbolAnalysis {
        symbol: symbol.to_string(),
        price: format!("${:.2}", current_price),
        rsi: rsi_with_indicator,
        macd: macd_str,
        bb_position: bb_str,
        signal: signal_str.to_string(),
    })
}

fn create_rsi_bar(rsi: f64) -> String {
    let position = ((rsi / 100.0) * 40.0) as usize;
    let mut bar = String::new();

    for i in 0..40 {
        if i == position {
            bar.push('█');
        } else if i < 12 {
            // 0-30 zone
            bar.push('░');
        } else if i > 28 {
            // 70-100 zone
            bar.push('░');
        } else {
            bar.push('─');
        }
    }

    bar
}
