//! # Пример: Предсказание волатильности
//!
//! Демонстрация расчёта и предсказания волатильности.
//!
//! ## Запуск
//!
//! ```bash
//! cargo run --example volatility_forecast
//! ```

use options_greeks_ml::volatility::{
    GarchPredictor, RealizedVolatility, VolatilityFeatures, VolatilityPredictor,
    VolatilityRiskPremium,
};
use rand::Rng;

fn main() {
    println!("=== Volatility Forecasting Demo ===\n");

    // Генерируем тестовые данные цен
    let prices = generate_price_series(100, 40000.0, 0.02);

    println!("Generated {} days of price data", prices.len());
    println!("  Start price: ${:.2}", prices.first().unwrap());
    println!("  End price:   ${:.2}", prices.last().unwrap());
    println!();

    // 1. Расчёт реализованной волатильности
    println!("=== Realized Volatility ===\n");

    let rv_calc = RealizedVolatility::crypto();

    let rv_5d = rv_calc.calculate(&prices, Some(5)).unwrap_or(0.0);
    let rv_10d = rv_calc.calculate(&prices, Some(10)).unwrap_or(0.0);
    let rv_20d = rv_calc.calculate(&prices, Some(20)).unwrap_or(0.0);
    let rv_60d = rv_calc.calculate(&prices, Some(60)).unwrap_or(0.0);

    println!("Historical Realized Volatility:");
    println!("  5-day:  {:.1}%", rv_5d * 100.0);
    println!("  10-day: {:.1}%", rv_10d * 100.0);
    println!("  20-day: {:.1}%", rv_20d * 100.0);
    println!("  60-day: {:.1}%", rv_60d * 100.0);
    println!();

    // Vol of vol
    if let Some(vov) = rv_calc.vol_of_vol(&prices, 20, 20) {
        println!("Volatility of Volatility (20d): {:.2}%", vov * 100.0);
    }
    println!();

    // 2. Предсказание с использованием признаков
    println!("=== ML-based Prediction ===\n");

    let current_iv = 0.55; // Текущая IV на рынке

    let features = VolatilityFeatures::from_prices(&prices, None, Some(current_iv));

    println!("Features:");
    println!("  rv_5d:     {:.2}%", features.rv_5d.unwrap_or(0.0) * 100.0);
    println!("  rv_10d:    {:.2}%", features.rv_10d.unwrap_or(0.0) * 100.0);
    println!("  rv_20d:    {:.2}%", features.rv_20d.unwrap_or(0.0) * 100.0);
    println!("  rv_60d:    {:.2}%", features.rv_60d.unwrap_or(0.0) * 100.0);
    println!("  vol_of_vol: {:.2}%", features.vol_of_vol.unwrap_or(0.0) * 100.0);
    println!("  return_5d: {:.2}%", features.return_5d.unwrap_or(0.0) * 100.0);
    println!("  current_iv: {:.2}%", features.current_iv.unwrap_or(0.0) * 100.0);
    println!();

    let predictor = VolatilityPredictor::default_weights(7);
    let predicted_rv = predictor.predict(&features);

    println!("7-day RV Prediction: {:.1}%", predicted_rv * 100.0);
    println!();

    // 3. GARCH модель
    println!("=== GARCH(1,1) Model ===\n");

    let mut garch = GarchPredictor::new();

    // Подаём доходности в модель
    let returns = RealizedVolatility::log_returns(&prices);
    for r in returns.iter().take(50) {
        garch.update(*r);
    }

    println!("GARCH Parameters:");
    println!("  Current volatility: {:.1}%", garch.current_volatility() * 100.0);
    println!("  Long-run volatility: {:.1}%", garch.long_run_volatility() * 100.0);
    println!();

    println!("GARCH Forecasts:");
    for days in [1, 7, 14, 30] {
        let forecast = garch.forecast(days);
        println!("  {}-day forecast: {:.1}%", days, forecast * 100.0);
    }
    println!();

    // 4. Volatility Risk Premium
    println!("=== Volatility Risk Premium ===\n");

    let mut vrp = VolatilityRiskPremium::default_crypto();

    // Добавляем историю VRP
    for i in 0..50 {
        // Симулируем: IV обычно выше RV
        let hist_iv = 0.50 + (i as f64 * 0.001);
        let hist_rv = 0.45 + (i as f64 * 0.0008);
        vrp.add_observation(hist_iv - hist_rv);
    }

    println!("Current Market:");
    println!("  Implied Volatility:  {:.1}%", current_iv * 100.0);
    println!("  Predicted RV:        {:.1}%", predicted_rv * 100.0);
    println!("  VRP (IV - RV):       {:.1}%", (current_iv - predicted_rv) * 100.0);
    println!();

    // Статистика VRP
    if let Some(stats) = vrp.statistics() {
        println!("VRP Statistics:");
        println!("  Mean VRP:       {:.2}%", stats.mean * 100.0);
        println!("  Std Dev:        {:.2}%", stats.std * 100.0);
        println!("  Current Z-score: {:.2}", stats.current_zscore);
        println!("  % Positive:     {:.1}%", stats.pct_positive * 100.0);
        println!();
    }

    // Торговый сигнал
    let signal = vrp.trading_signal(current_iv, predicted_rv, vrp.statistics().as_ref());

    println!("Trading Signal:");
    println!("  Action:     {}", signal.action);
    println!("  Edge:       {:.2}%", signal.edge * 100.0);
    println!("  Confidence: {:.1}%", signal.confidence * 100.0);
    println!("  Reason:     {}", signal.reason);
    println!();

    // 5. Анализ разных сценариев
    println!("=== Scenario Analysis ===\n");

    println!("Trading signals at different IV levels:");
    for iv in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70] {
        let sig = vrp.trading_signal(iv, predicted_rv, None);
        let arrow = match sig.action {
            options_greeks_ml::volatility::VrpAction::SellVolatility => "SELL",
            options_greeks_ml::volatility::VrpAction::BuyVolatility => "BUY ",
            options_greeks_ml::volatility::VrpAction::NoTrade => "HOLD",
        };
        println!(
            "  IV {:.0}% vs RV {:.0}%: {} (edge: {:.1}%)",
            iv * 100.0,
            predicted_rv * 100.0,
            arrow,
            sig.edge * 100.0
        );
    }
}

/// Генерация случайных цен с заданной волатильностью
fn generate_price_series(n: usize, start_price: f64, daily_vol: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price = start_price;

    for _ in 0..n {
        prices.push(price);
        let daily_return: f64 = rng.gen_range(-daily_vol..daily_vol);
        price *= 1.0 + daily_return;
    }

    prices
}
