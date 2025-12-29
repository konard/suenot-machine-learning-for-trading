//! # Analyze Factors Example
//!
//! Пример анализа риск-факторов, извлеченных автоэнкодером.
//!
//! ```bash
//! # Сначала загрузите данные и обучите модель
//! cargo run --example fetch_data
//! cargo run --example train_model
//!
//! # Затем проанализируйте факторы
//! cargo run --example analyze_factors
//! ```

use anyhow::Result;
use clap::Parser;
use crypto_autoencoders::{
    utils, Autoencoder, DataProcessor, NormalizationMethod, RiskFactorAnalyzer,
};
use std::path::PathBuf;

/// Анализ риск-факторов
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Путь к файлу с признаками
    #[arg(short, long, default_value = "data/btcusdt_1h_features.csv")]
    input: PathBuf,

    /// Размер скрытого представления
    #[arg(short, long, default_value_t = 8)]
    latent_size: usize,

    /// Количество эпох обучения
    #[arg(short, long, default_value_t = 50)]
    epochs: usize,

    /// Количество кластеров для анализа режимов
    #[arg(short, long, default_value_t = 3)]
    n_clusters: usize,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║       Анализ риск-факторов криптовалют               ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // Проверяем наличие файла
    if !args.input.exists() {
        println!("Файл {:?} не найден!", args.input);
        println!("\nСначала загрузите данные:");
        println!("  cargo run --example fetch_data");
        return Ok(());
    }

    // Загружаем данные
    println!("Загрузка данных из {:?}...", args.input);
    let features = utils::load_features(&args.input)?;
    println!(
        "Загружено {} образцов x {} признаков\n",
        features.nrows(),
        features.ncols()
    );

    let data = features.to_array();

    // Обучаем автоэнкодер
    println!("Обучение автоэнкодера...");
    let hidden_sizes = vec![32, 16];
    let mut ae = Autoencoder::deep(features.ncols(), &hidden_sizes, args.latent_size);
    ae.fit(&data, args.epochs, 0.001);

    let train_error = ae.reconstruction_error(&data);
    println!("  Ошибка реконструкции: {:.6}\n", train_error);

    // Анализируем риск-факторы
    println!("═══════════════════════════════════════════════════════");
    println!("                   РИСК-ФАКТОРЫ");
    println!("═══════════════════════════════════════════════════════\n");

    let analyzer = RiskFactorAnalyzer::new();
    let risk_factors = analyzer.analyze(&mut ae, &features);

    for factor in &risk_factors {
        println!("┌─────────────────────────────────────────────────────┐");
        println!("│ {}  ", factor.name);
        println!("├─────────────────────────────────────────────────────┤");
        println!(
            "│ Объясненная дисперсия: {:.2}%",
            factor.explained_variance * 100.0
        );
        println!("│ Среднее: {:.4}, Std: {:.4}", factor.statistics.mean, factor.statistics.std);
        println!(
            "│ Min: {:.4}, Max: {:.4}",
            factor.statistics.min, factor.statistics.max
        );
        println!(
            "│ Асимметрия: {:.4}, Эксцесс: {:.4}",
            factor.statistics.skewness, factor.statistics.kurtosis
        );

        // Топ-5 коррелирующих признаков
        let mut correlations: Vec<_> = factor.feature_correlations.iter().collect();
        correlations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        println!("│");
        println!("│ Топ корреляции с признаками:");
        for (name, corr) in correlations.iter().take(5) {
            let bar_len = (corr.abs() * 20.0) as usize;
            let bar = "█".repeat(bar_len);
            let sign = if **corr >= 0.0 { "+" } else { "-" };
            println!("│   {:>20}: {:>+.3} {}{}", name, corr, sign, bar);
        }
        println!("└─────────────────────────────────────────────────────┘\n");
    }

    // Важность признаков
    println!("═══════════════════════════════════════════════════════");
    println!("                 ВАЖНОСТЬ ПРИЗНАКОВ");
    println!("═══════════════════════════════════════════════════════\n");

    let importance = analyzer.feature_importance(&mut ae, &features);
    let mut importance_sorted: Vec<_> = importance.iter().collect();
    importance_sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("(Насколько хорошо признак восстанавливается автоэнкодером)\n");
    for (name, score) in &importance_sorted {
        let bar_len = (*score * 30.0) as usize;
        let bar = "▓".repeat(bar_len) + &"░".repeat(30 - bar_len);
        println!("  {:>20}: {:.2}% {}", name, score * 100.0, bar);
    }

    // Анализ временной динамики
    println!("\n═══════════════════════════════════════════════════════");
    println!("              ВРЕМЕННАЯ ДИНАМИКА ФАКТОРОВ");
    println!("═══════════════════════════════════════════════════════\n");

    let latent = ae.transform(&data);
    let temporal_stats = analyzer.temporal_analysis(&latent, 20);

    for stats in &temporal_stats {
        let trend_dir = if stats.trend > 0.01 {
            "↗ Восходящий"
        } else if stats.trend < -0.01 {
            "↘ Нисходящий"
        } else {
            "→ Боковой"
        };

        let persistence = if stats.autocorrelation > 0.5 {
            "Высокая персистентность"
        } else if stats.autocorrelation > 0.2 {
            "Умеренная персистентность"
        } else {
            "Низкая персистентность"
        };

        println!(
            "  Фактор {}: тренд={:.4} ({}), автокорр={:.4} ({})",
            stats.factor_index + 1,
            stats.trend,
            trend_dir,
            stats.autocorrelation,
            persistence
        );
    }

    // Кластеризация рыночных режимов
    println!("\n═══════════════════════════════════════════════════════");
    println!("              РЫНОЧНЫЕ РЕЖИМЫ (кластеры)");
    println!("═══════════════════════════════════════════════════════\n");

    let cluster_labels = analyzer.cluster_regimes(&latent, args.n_clusters);

    // Подсчитываем размеры кластеров
    let mut cluster_counts = vec![0usize; args.n_clusters];
    for &label in &cluster_labels {
        cluster_counts[label] += 1;
    }

    for (i, &count) in cluster_counts.iter().enumerate() {
        let pct = count as f64 / cluster_labels.len() as f64 * 100.0;

        // Вычисляем средние значения факторов для кластера
        let cluster_means: Vec<f64> = (0..latent.ncols())
            .map(|j| {
                let sum: f64 = latent
                    .outer_iter()
                    .enumerate()
                    .filter(|(idx, _)| cluster_labels[*idx] == i)
                    .map(|(_, row)| row[j])
                    .sum();
                sum / count as f64
            })
            .collect();

        let regime_name = classify_regime(&cluster_means);

        println!("  Режим {} ({}):", i + 1, regime_name);
        println!("    Частота: {} образцов ({:.1}%)", count, pct);
        print!("    Средние факторы: [");
        for (j, &mean) in cluster_means.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{:.3}", mean);
        }
        println!("]");
        println!();
    }

    // Последовательность режимов
    println!("  Последовательность последних 50 точек:");
    print!("  ");
    for &label in cluster_labels.iter().rev().take(50).rev() {
        print!("{}", label + 1);
    }
    println!("\n");

    // Интерпретация
    println!("═══════════════════════════════════════════════════════");
    println!("                    ИНТЕРПРЕТАЦИЯ");
    println!("═══════════════════════════════════════════════════════\n");

    println!("  Автоэнкодер сжал {} признаков в {} латентных факторов.", features.ncols(), args.latent_size);
    println!("  Эти факторы представляют скрытые риск-факторы рынка.\n");

    println!("  Рекомендации по использованию:");
    println!("  1. Используйте факторы как входы для торговых моделей");
    println!("  2. Отслеживайте смену режимов для управления рисками");
    println!("  3. Высокие значения волатильного фактора → уменьшить позиции");
    println!("  4. Персистентные факторы хороши для трендовых стратегий\n");

    Ok(())
}

/// Классифицирует режим по средним значениям факторов
fn classify_regime(means: &[f64]) -> &'static str {
    // Простая эвристика на основе средних значений
    let total: f64 = means.iter().sum();
    let variance: f64 = means.iter().map(|x| x.powi(2)).sum::<f64>() / means.len() as f64;

    if variance > 0.5 {
        "Высокая волатильность"
    } else if total > 0.5 {
        "Бычий тренд"
    } else if total < -0.5 {
        "Медвежий тренд"
    } else {
        "Консолидация"
    }
}
