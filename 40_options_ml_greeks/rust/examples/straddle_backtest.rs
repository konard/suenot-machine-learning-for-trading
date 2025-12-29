//! # Пример: Бэктест стратегии страддлов
//!
//! Симуляция торговли страддлами с дельта-хеджированием.
//!
//! ## Запуск
//!
//! ```bash
//! cargo run --example straddle_backtest
//! ```

use chrono::{Duration, Utc};
use options_greeks_ml::{
    greeks::{BlackScholes, Greeks, OptionType},
    models::{OptionContract, OptionPosition, Portfolio},
    strategy::{DeltaHedger, GammaScalper, PnlAttribution},
};
use rand::Rng;

fn main() {
    println!("=== Straddle Strategy Backtest ===\n");

    // Параметры симуляции
    let initial_spot = 42000.0;
    let strike = 42000.0;
    let days_to_expiry = 7;
    let initial_iv = 0.55;
    let realized_vol = 0.45; // Реальная волатильность ниже IV

    println!("Strategy Parameters:");
    println!("  Initial spot:    ${:.2}", initial_spot);
    println!("  Strike:          ${:.2}", strike);
    println!("  Days to expiry:  {}", days_to_expiry);
    println!("  Initial IV:      {:.1}%", initial_iv * 100.0);
    println!("  Realized Vol:    {:.1}% (actual)", realized_vol * 100.0);
    println!("  Strategy:        Short Straddle (sell vol)");
    println!();

    // Генерируем путь цены
    let price_path = generate_price_path(initial_spot, realized_vol, days_to_expiry);

    println!("Price Path:");
    for (i, &price) in price_path.iter().enumerate() {
        let change = (price / initial_spot - 1.0) * 100.0;
        println!("  Day {}: ${:.2} ({:+.2}%)", i, price, change);
    }
    println!();

    // Создаём портфель и открываем позицию
    let mut portfolio = Portfolio::new(100000.0);
    let expiry = Utc::now() + Duration::days(days_to_expiry as i64);

    // Начальные цены опционов
    let bs_initial = BlackScholes::crypto(initial_spot, strike, days_to_expiry as f64, initial_iv);

    let call_price = bs_initial.call_price();
    let put_price = bs_initial.put_price();
    let straddle_price = call_price + put_price;

    println!("Opening Position:");
    println!("  Call price:     ${:.2}", call_price);
    println!("  Put price:      ${:.2}", put_price);
    println!("  Straddle price: ${:.2}", straddle_price);

    // Создаём позиции (short straddle = -1 контракт каждого)
    let call = OptionContract::new("BTC", strike, expiry, OptionType::Call, call_price, initial_iv);
    let put = OptionContract::new("BTC", strike, expiry, OptionType::Put, put_price, initial_iv);

    let call_pos = OptionPosition::new(call, -1.0, call_price)
        .with_greeks(bs_initial.call_greeks());
    let put_pos = OptionPosition::new(put, -1.0, put_price)
        .with_greeks(bs_initial.put_greeks());

    portfolio.add_option(call_pos);
    portfolio.add_option(put_pos);
    portfolio.underlying_spot = initial_spot;

    // Получаем премию
    portfolio.cash += straddle_price;

    let initial_greeks = portfolio.total_greeks();
    println!("  Initial Greeks:");
    println!("    Delta: {:.4}", initial_greeks.delta);
    println!("    Gamma: {:.6}", initial_greeks.gamma);
    println!("    Theta: ${:.2}/day", initial_greeks.theta);
    println!("    Vega:  ${:.2}/1%", initial_greeks.vega);
    println!();

    // Симуляция с дельта-хеджированием
    println!("=== Daily Simulation ===\n");

    let mut hedger = DeltaHedger::default_crypto();
    let mut total_theta = 0.0;
    let mut total_gamma_pnl = 0.0;
    let mut daily_moves: Vec<f64> = Vec::new();

    for day in 0..days_to_expiry {
        let current_price = price_path[day];
        let days_left = (days_to_expiry - day) as f64;

        // Обновляем модель
        let bs_current = BlackScholes::crypto(current_price, strike, days_left, initial_iv);

        // Обновляем греки в портфеле
        let call_greeks = bs_current.call_greeks();
        let put_greeks = bs_current.put_greeks();

        // Обновляем позиции (упрощённо через общие греки)
        portfolio.underlying_spot = current_price;

        // Расчёт дневного P&L
        let prev_price = if day > 0 { price_path[day - 1] } else { initial_spot };
        let daily_move = (current_price - prev_price) / prev_price;
        daily_moves.push(daily_move);

        // Theta P&L (положительный для short)
        let theta_pnl = -(call_greeks.theta + put_greeks.theta);
        total_theta += theta_pnl;

        // Gamma P&L (отрицательный для short при движениях)
        let gamma = call_greeks.gamma + put_greeks.gamma;
        let gamma_pnl = -0.5 * gamma * current_price.powi(2) * daily_move.powi(2);
        total_gamma_pnl += gamma_pnl;

        println!(
            "Day {}: Spot ${:.2} | Move {:+.2}% | Theta ${:+.2} | Gamma ${:+.2}",
            day,
            current_price,
            daily_move * 100.0,
            theta_pnl,
            gamma_pnl
        );

        // Дельта-хедж
        let current_delta = -(call_greeks.delta + put_greeks.delta);
        if let Some(trade) = hedger.execute_hedge(&mut portfolio, current_price) {
            println!(
                "         Hedge: {} {:.4} BTC @ ${:.2} (cost: ${:.2})",
                if trade.quantity > 0.0 { "BUY" } else { "SELL" },
                trade.quantity.abs(),
                trade.price,
                trade.transaction_cost
            );
        }
    }

    println!();

    // Финальные расчёты
    println!("=== Final Results ===\n");

    let final_spot = *price_path.last().unwrap();
    let spot_change = final_spot - initial_spot;

    // P&L на экспирации
    let call_payoff = (final_spot - strike).max(0.0);
    let put_payoff = (strike - final_spot).max(0.0);
    let straddle_payoff = call_payoff + put_payoff;

    // Short straddle P&L = премия - payoff
    let options_pnl = straddle_price - straddle_payoff;

    println!("Market:");
    println!("  Final spot:    ${:.2}", final_spot);
    println!("  Spot change:   {:+.2} ({:+.2}%)", spot_change, (spot_change / initial_spot) * 100.0);
    println!();

    println!("Options P&L:");
    println!("  Premium collected: ${:.2}", straddle_price);
    println!("  Call payoff:       ${:.2}", call_payoff);
    println!("  Put payoff:        ${:.2}", put_payoff);
    println!("  Straddle payoff:   ${:.2}", straddle_payoff);
    println!("  Options P&L:       ${:+.2}", options_pnl);
    println!();

    println!("Greeks Attribution:");
    println!("  Theta P&L:   ${:+.2} ({:.0}% of premium)",
        total_theta,
        (total_theta / straddle_price) * 100.0
    );
    println!("  Gamma P&L:   ${:+.2}",total_gamma_pnl);
    println!();

    println!("Hedging:");
    println!("  Number of hedges:   {}", hedger.num_hedges());
    println!("  Transaction costs:  ${:.2}", hedger.total_costs());
    println!("  Hedge P&L:          ${:+.2}", hedger.hedging_pnl(final_spot));
    println!();

    let total_pnl = options_pnl - hedger.total_costs() + hedger.hedging_pnl(final_spot);
    println!("Total Strategy P&L: ${:+.2}", total_pnl);
    println!();

    // Анализ
    let vol_edge = initial_iv - realized_vol;
    let theoretical_pnl = vol_edge * initial_greeks.vega.abs() * 100.0;

    println!("Analysis:");
    println!("  Volatility edge:     {:.1}%", vol_edge * 100.0);
    println!("  Theoretical P&L:     ${:.2} (vega × vol edge)", theoretical_pnl);
    println!("  Actual P&L:          ${:.2}", total_pnl);
    println!("  Implementation cost: ${:.2}", theoretical_pnl - total_pnl);

    // Статус
    if total_pnl > 0.0 {
        println!("\n  Strategy PROFITABLE - IV was correctly identified as overpriced!");
    } else {
        println!("\n  Strategy UNPROFITABLE - Large moves exceeded theta decay");
    }
}

/// Генерация случайного пути цены
fn generate_price_path(start: f64, daily_vol: f64, days: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(days + 1);
    let mut price = start;

    prices.push(price);

    for _ in 0..days {
        let daily_return: f64 = rng.gen_range(-daily_vol * 2.0..daily_vol * 2.0);
        price *= 1.0 + daily_return / 365.0_f64.sqrt() * 10.0;
        prices.push(price);
    }

    prices
}
