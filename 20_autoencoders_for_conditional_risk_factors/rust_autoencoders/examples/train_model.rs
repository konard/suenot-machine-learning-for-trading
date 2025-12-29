//! # Train Model Example
//!
//! Пример обучения различных автоэнкодеров на криптовалютных данных.
//!
//! ```bash
//! # Сначала загрузите данные
//! cargo run --example fetch_data
//!
//! # Затем обучите модель
//! cargo run --example train_model -- --model deep --epochs 100
//! ```

use anyhow::Result;
use clap::{Parser, ValueEnum};
use crypto_autoencoders::{
    utils, Autoencoder, ConditionalAutoencoder, DataProcessor, DenoisingAutoencoder,
    NormalizationMethod, VariationalAutoencoder,
};
use ndarray::Array2;
use std::path::PathBuf;

/// Тип модели
#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    /// Простой автоэнкодер
    Simple,
    /// Глубокий автоэнкодер
    Deep,
    /// Шумоподавляющий автоэнкодер
    Denoising,
    /// Вариационный автоэнкодер
    Vae,
    /// Условный автоэнкодер
    Conditional,
}

/// Обучение автоэнкодера на криптовалютных данных
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Путь к файлу с признаками
    #[arg(short, long, default_value = "data/btcusdt_1h_features.csv")]
    input: PathBuf,

    /// Тип модели
    #[arg(short, long, value_enum, default_value = "deep")]
    model: ModelType,

    /// Размер скрытого представления
    #[arg(short, long, default_value_t = 8)]
    latent_size: usize,

    /// Количество эпох обучения
    #[arg(short, long, default_value_t = 100)]
    epochs: usize,

    /// Скорость обучения
    #[arg(long, default_value_t = 0.001)]
    lr: f64,

    /// Стандартное отклонение шума (для denoising)
    #[arg(long, default_value_t = 0.1)]
    noise_std: f64,

    /// Папка для сохранения результатов
    #[arg(short, long, default_value = "data")]
    output_dir: PathBuf,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("=== Обучение автоэнкодера для криптовалют ===\n");

    // Проверяем наличие файла
    if !args.input.exists() {
        println!("Файл {:?} не найден!", args.input);
        println!("Сначала загрузите данные:");
        println!("  cargo run --example fetch_data");
        return Ok(());
    }

    // Загружаем данные
    println!("Загрузка данных из {:?}...", args.input);
    let features = utils::load_features(&args.input)?;
    println!(
        "Загружено {} образцов x {} признаков",
        features.nrows(),
        features.ncols()
    );

    // Преобразуем в ndarray
    let data = features.to_array();
    let input_size = features.ncols();

    println!("\nПризнаки:");
    for (i, name) in features.names.iter().enumerate() {
        let col: Vec<f64> = data.column(i).to_vec();
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        let min = col.iter().copied().fold(f64::INFINITY, f64::min);
        let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!("  {}: mean={:.4}, min={:.4}, max={:.4}", name, mean, min, max);
    }

    // Разделяем на train/test (80/20)
    let split_idx = (data.nrows() as f64 * 0.8) as usize;
    let train_data = data.slice(ndarray::s![..split_idx, ..]).to_owned();
    let test_data = data.slice(ndarray::s![split_idx.., ..]).to_owned();

    println!(
        "\nРазделение данных: {} train, {} test",
        train_data.nrows(),
        test_data.nrows()
    );

    // Обучаем модель в зависимости от типа
    match args.model {
        ModelType::Simple => train_simple_autoencoder(&train_data, &test_data, &args)?,
        ModelType::Deep => train_deep_autoencoder(&train_data, &test_data, &args)?,
        ModelType::Denoising => train_denoising_autoencoder(&train_data, &test_data, &args)?,
        ModelType::Vae => train_vae(&train_data, &test_data, &args)?,
        ModelType::Conditional => train_conditional_autoencoder(&train_data, &test_data, &args)?,
    }

    Ok(())
}

fn train_simple_autoencoder(train: &Array2<f64>, test: &Array2<f64>, args: &Args) -> Result<()> {
    println!("\n=== Простой автоэнкодер ===");
    println!("Архитектура: {} -> {} -> {}", train.ncols(), args.latent_size, train.ncols());

    let mut ae = Autoencoder::new(train.ncols(), args.latent_size);

    println!("\nОбучение ({} эпох, lr={})...", args.epochs, args.lr);
    ae.fit(train, args.epochs, args.lr);

    // Оцениваем
    let train_error = ae.reconstruction_error(train);
    let test_error = ae.reconstruction_error(test);

    println!("\nРезультаты:");
    println!("  Train MSE: {:.6}", train_error);
    println!("  Test MSE:  {:.6}", test_error);

    // Сохраняем латентные представления
    let latent = ae.transform(test);
    save_latent(&latent, &args.output_dir, "simple")?;

    // Визуализируем историю потерь
    println!("\nИстория потерь:");
    print_loss_chart(ae.loss_history());

    Ok(())
}

fn train_deep_autoencoder(train: &Array2<f64>, test: &Array2<f64>, args: &Args) -> Result<()> {
    println!("\n=== Глубокий автоэнкодер ===");

    let hidden_sizes = vec![64, 32];
    println!(
        "Архитектура: {} -> {:?} -> {} -> {:?} -> {}",
        train.ncols(),
        hidden_sizes,
        args.latent_size,
        hidden_sizes.iter().rev().collect::<Vec<_>>(),
        train.ncols()
    );

    let mut ae = Autoencoder::deep(train.ncols(), &hidden_sizes, args.latent_size);

    println!("\nОбучение ({} эпох, lr={})...", args.epochs, args.lr);
    ae.fit(train, args.epochs, args.lr);

    let train_error = ae.reconstruction_error(train);
    let test_error = ae.reconstruction_error(test);

    println!("\nРезультаты:");
    println!("  Train MSE: {:.6}", train_error);
    println!("  Test MSE:  {:.6}", test_error);

    let latent = ae.transform(test);
    save_latent(&latent, &args.output_dir, "deep")?;

    println!("\nИстория потерь:");
    print_loss_chart(ae.loss_history());

    Ok(())
}

fn train_denoising_autoencoder(train: &Array2<f64>, test: &Array2<f64>, args: &Args) -> Result<()> {
    println!("\n=== Шумоподавляющий автоэнкодер ===");
    println!("Уровень шума: {}", args.noise_std);

    let hidden_sizes = vec![32];
    let mut dae = DenoisingAutoencoder::deep(
        train.ncols(),
        &hidden_sizes,
        args.latent_size,
        args.noise_std,
    );

    println!("\nОбучение ({} эпох, lr={})...", args.epochs, args.lr);
    dae.fit(train, args.epochs, args.lr);

    // Тестируем очистку шума
    let noisy_test = add_noise(test, args.noise_std);
    let cleaned = dae.denoise(&noisy_test);

    let noisy_error = mse(&noisy_test, test);
    let cleaned_error = mse(&cleaned, test);

    println!("\nРезультаты:");
    println!("  MSE зашумленных данных: {:.6}", noisy_error);
    println!("  MSE после очистки:      {:.6}", cleaned_error);
    println!("  Улучшение: {:.2}%", (1.0 - cleaned_error / noisy_error) * 100.0);

    Ok(())
}

fn train_vae(train: &Array2<f64>, test: &Array2<f64>, args: &Args) -> Result<()> {
    println!("\n=== Вариационный автоэнкодер (VAE) ===");

    let hidden_size = 64;
    println!(
        "Архитектура: {} -> {} -> {} (mu, log_var) -> {} -> {}",
        train.ncols(),
        hidden_size,
        args.latent_size,
        hidden_size,
        train.ncols()
    );

    let mut vae = VariationalAutoencoder::new(train.ncols(), hidden_size, args.latent_size);

    println!("\nОбучение ({} эпох, lr={})...", args.epochs, args.lr);
    vae.fit(train, args.epochs, args.lr);

    // Генерируем новые данные
    println!("\nГенерация 5 новых образцов из латентного пространства:");
    let generated = vae.generate(5);
    for (i, sample) in generated.outer_iter().enumerate() {
        let vals: Vec<String> = sample.iter().map(|v| format!("{:.3}", v)).collect();
        println!("  Sample {}: [{}]", i + 1, vals.join(", "));
    }

    Ok(())
}

fn train_conditional_autoencoder(
    train: &Array2<f64>,
    test: &Array2<f64>,
    args: &Args,
) -> Result<()> {
    println!("\n=== Условный автоэнкодер ===");

    // Используем первые 3 признака как условия
    let condition_size = 3;
    let input_size = train.ncols() - condition_size;

    println!(
        "Вход: {} признаков, условия: {} признака",
        input_size, condition_size
    );

    // Разделяем данные на вход и условия
    let train_input = train.slice(ndarray::s![.., condition_size..]).to_owned();
    let train_cond = train.slice(ndarray::s![.., ..condition_size]).to_owned();
    let test_input = test.slice(ndarray::s![.., condition_size..]).to_owned();
    let test_cond = test.slice(ndarray::s![.., ..condition_size]).to_owned();

    let hidden_size = 32;
    let mut cae = ConditionalAutoencoder::new(
        input_size,
        condition_size,
        hidden_size,
        args.latent_size,
    );

    println!("\nОбучение ({} эпох, lr={})...", args.epochs, args.lr);
    cae.fit(&train_input, &train_cond, args.epochs, args.lr);

    // Тестируем
    let latent = cae.transform(&test_input, &test_cond);

    println!("\nРезультаты:");
    println!("  Размер латентного представления: {} x {}", latent.nrows(), latent.ncols());

    // Статистика латентных переменных
    println!("\nСтатистика латентных переменных:");
    for i in 0..latent.ncols() {
        let col: Vec<f64> = latent.column(i).to_vec();
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        let std = (col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64).sqrt();
        println!("  Фактор {}: mean={:.4}, std={:.4}", i + 1, mean, std);
    }

    save_latent(&latent, &args.output_dir, "conditional")?;

    Ok(())
}

// ============ Вспомогательные функции ============

fn add_noise(data: &Array2<f64>, std: f64) -> Array2<f64> {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0, std).unwrap();
    let mut rng = rand::thread_rng();
    data.mapv(|x| x + normal.sample(&mut rng))
}

fn mse(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).mapv(|x| x.powi(2)).sum() / (a.len() as f64)
}

fn save_latent(latent: &Array2<f64>, output_dir: &PathBuf, model_name: &str) -> Result<()> {
    let path = output_dir.join(format!("{}_latent.csv", model_name));

    let file = std::fs::File::create(&path)?;
    let mut writer = csv::Writer::from_writer(file);

    // Заголовок
    let header: Vec<String> = (0..latent.ncols())
        .map(|i| format!("factor_{}", i + 1))
        .collect();
    writer.write_record(&header)?;

    // Данные
    for row in latent.outer_iter() {
        let record: Vec<String> = row.iter().map(|v| format!("{:.6}", v)).collect();
        writer.write_record(&record)?;
    }

    writer.flush()?;
    println!("\nЛатентные представления сохранены в {:?}", path);

    Ok(())
}

fn print_loss_chart(history: &[f64]) {
    if history.is_empty() {
        return;
    }

    let min = history.iter().copied().fold(f64::INFINITY, f64::min);
    let max = history.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    println!("  Начальная потеря: {:.6}", history.first().unwrap());
    println!("  Финальная потеря: {:.6}", history.last().unwrap());
    println!("  Минимум: {:.6}", min);

    // Простой ASCII график
    let width = 50;
    let height = 10;
    let range = max - min;

    if range > 1e-10 {
        let step = history.len() / width.min(history.len());
        let sampled: Vec<f64> = history.iter().step_by(step.max(1)).copied().collect();

        println!("\n  {:>8.4} ┤", max);
        for h in (0..height).rev() {
            let threshold = min + (h as f64 / height as f64) * range;
            let line: String = sampled
                .iter()
                .map(|&v| if v >= threshold { '*' } else { ' ' })
                .collect();
            if h == height / 2 {
                println!("  {:>8.4} ┤{}", (max + min) / 2.0, line);
            } else {
                println!("           │{}", line);
            }
        }
        println!("  {:>8.4} ┴{}", min, "─".repeat(sampled.len()));
        println!("           Epoch 1{}Epoch {}", " ".repeat(sampled.len() - 15), history.len());
    }
}
