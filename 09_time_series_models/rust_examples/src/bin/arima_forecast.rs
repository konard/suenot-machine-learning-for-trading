//! Пример: Прогнозирование с помощью ARIMA
//!
//! Строит ARIMA модель и делает прогноз цен.
//!
//! Использование:
//! ```
//! cargo run --bin arima_forecast -- --file data/BTCUSDT_1h.csv --horizon 24
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use crypto_time_series::api::load_candles;
use crypto_time_series::models::{ArimaModel, ArimaParams, auto_arima};
use crypto_time_series::TimeSeries;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "ARIMA forecasting for cryptocurrency")]
struct Args {
    /// Input CSV file with candle data
    #[arg(short, long)]
    file: PathBuf,

    /// Forecast horizon (number of periods)
    #[arg(short = 'H', long, default_value = "24")]
    horizon: usize,

    /// AR order (p)
    #[arg(short, long)]
    p: Option<usize>,

    /// Differencing order (d)
    #[arg(short, long)]
    d: Option<usize>,

    /// MA order (q)
    #[arg(short, long)]
    q: Option<usize>,

    /// Use automatic model selection
    #[arg(long)]
    auto: bool,

    /// Use log returns for modeling
    #[arg(long)]
    log_returns: bool,

    /// Confidence level for intervals
    #[arg(long, default_value = "0.95")]
    confidence: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "ARIMA Forecasting".bold().blue());
    println!("{}", "=".repeat(60).blue());

    // Загружаем данные
    let candles = load_candles(&args.file)?;
    println!(
        "\n{} {} candles from {}",
        "Loaded:".green(),
        candles.len(),
        args.file.display()
    );

    let ts = TimeSeries::from_candles("data", &candles);

    // Подготавливаем данные
    let (data, transform_name) = if args.log_returns {
        (ts.log_returns(), "log returns")
    } else {
        (ts.data.clone(), "prices")
    };

    println!("{} {} ({} observations)\n", "Modeling:".cyan(), transform_name, data.len());

    // Строим модель
    let model = if args.auto {
        println!("{}", "Auto-selecting ARIMA order...".yellow());
        auto_arima(&data, 5, 2, 5)
    } else {
        let p = args.p.unwrap_or(1);
        let d = args.d.unwrap_or(1);
        let q = args.q.unwrap_or(1);

        println!(
            "{} ARIMA({},{},{})",
            "Fitting:".yellow(),
            p, d, q
        );

        let params = ArimaParams::new(p, d, q);
        ArimaModel::fit(&data, params)
    };

    let model = model.ok_or_else(|| anyhow::anyhow!("Failed to fit ARIMA model"))?;

    // Выводим информацию о модели
    println!("\n{}", model.summary());

    // Делаем прогноз
    println!("\n{} (horizon = {})", "Forecasting".bold(), args.horizon);
    println!("{}", "-".repeat(40));

    let forecast = model.forecast_interval(&data, args.horizon, args.confidence);

    // Выводим прогнозы
    println!(
        "\n{:>4} {:>12} {:>12} {:>12}",
        "Step", "Lower", "Point", "Upper"
    );
    println!("{}", "-".repeat(48));

    for i in 0..args.horizon {
        let point = forecast.point[i];
        let lower = forecast.lower[i];
        let upper = forecast.upper[i];

        // Преобразуем обратно в цены если нужно
        let (point_display, lower_display, upper_display) = if args.log_returns && !ts.data.is_empty() {
            let last_price = *ts.data.last().unwrap();
            let cumulative: f64 = forecast.point[..=i].iter().sum();
            let price_forecast = last_price * cumulative.exp();
            let price_lower = last_price * (cumulative + forecast.lower[i] - forecast.point[i]).exp();
            let price_upper = last_price * (cumulative + forecast.upper[i] - forecast.point[i]).exp();
            (price_forecast, price_lower, price_upper)
        } else {
            (point, lower, upper)
        };

        println!(
            "{:>4} {:>12.4} {:>12.4} {:>12.4}",
            i + 1,
            lower_display,
            point_display,
            upper_display
        );
    }

    // Статистика прогноза
    println!("\n{}", "Forecast Summary".bold());
    println!("{}", "-".repeat(40));

    let last_value = *data.last().unwrap_or(&0.0);
    let final_forecast = forecast.point.last().unwrap_or(&0.0);
    let direction = if *final_forecast > last_value {
        "UP".green()
    } else {
        "DOWN".red()
    };

    println!("Current value: {:.4}", last_value);
    println!("Forecast ({} steps): {:.4}", args.horizon, final_forecast);
    println!("Expected direction: {}", direction);
    println!(
        "Confidence interval ({:.0}%): [{:.4}, {:.4}]",
        args.confidence * 100.0,
        forecast.lower.last().unwrap_or(&0.0),
        forecast.upper.last().unwrap_or(&0.0)
    );

    // Проверяем качество модели на последних данных (backtesting)
    if data.len() > args.horizon + 50 {
        println!("\n{}", "Backtesting".bold());
        println!("{}", "-".repeat(40));

        let train_size = data.len() - args.horizon;
        let train_data = &data[..train_size];
        let test_data = &data[train_size..];

        if let Some(backtest_model) = ArimaModel::fit(train_data, model.params.clone()) {
            let backtest_forecast = backtest_model.forecast(train_data, args.horizon);

            // Вычисляем ошибки
            let mse: f64 = test_data
                .iter()
                .zip(backtest_forecast.iter())
                .map(|(a, f)| (a - f).powi(2))
                .sum::<f64>()
                / args.horizon as f64;

            let mae: f64 = test_data
                .iter()
                .zip(backtest_forecast.iter())
                .map(|(a, f)| (a - f).abs())
                .sum::<f64>()
                / args.horizon as f64;

            let rmse = mse.sqrt();

            println!("RMSE: {:.6}", rmse);
            println!("MAE: {:.6}", mae);

            // Directional accuracy
            let correct_direction: usize = test_data
                .windows(2)
                .zip(backtest_forecast.windows(2))
                .filter(|(actual, forecast)| {
                    (actual[1] > actual[0]) == (forecast[1] > forecast[0])
                })
                .count();

            let total_directions = (args.horizon - 1).max(1);
            let accuracy = correct_direction as f64 / total_directions as f64 * 100.0;

            println!("Directional accuracy: {:.1}%", accuracy);
        }
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
