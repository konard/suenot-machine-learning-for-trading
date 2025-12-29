//! # Пример: Расчёт греков опционов
//!
//! Демонстрация расчёта цен опционов и греков по модели Блэка-Шоулза.
//!
//! ## Запуск
//!
//! ```bash
//! cargo run --example calculate_greeks
//! ```

use options_greeks_ml::greeks::{BlackScholes, OptionType};

fn main() {
    println!("=== Options Greeks Calculator ===\n");

    // Параметры опциона
    let spot = 42000.0;      // Текущая цена BTC
    let strike = 42000.0;    // ATM страйк
    let days = 7.0;          // 7 дней до экспирации
    let iv = 0.55;           // 55% подразумеваемая волатильность

    println!("Input Parameters:");
    println!("  Spot Price:     ${:.2}", spot);
    println!("  Strike Price:   ${:.2}", strike);
    println!("  Days to Expiry: {:.0}", days);
    println!("  Volatility:     {:.1}%", iv * 100.0);
    println!();

    // Создаём модель Black-Scholes для крипты (ставка = 0)
    let bs = BlackScholes::crypto(spot, strike, days, iv);

    // Расчёт цен
    let call_price = bs.call_price();
    let put_price = bs.put_price();
    let straddle_price = bs.straddle_price();

    println!("Option Prices:");
    println!("  Call:     ${:.2}", call_price);
    println!("  Put:      ${:.2}", put_price);
    println!("  Straddle: ${:.2}", straddle_price);
    println!();

    // Расчёт греков для Call
    let call_greeks = bs.call_greeks();
    println!("Call Greeks:");
    println!("  Delta (Δ): {:.4}", call_greeks.delta);
    println!("    -> Если BTC вырастет на $100, call вырастет на ${:.2}", call_greeks.delta * 100.0);
    println!("  Gamma (Γ): {:.6}", call_greeks.gamma);
    println!("    -> Дельта изменится на {:.4} при движении на $100", call_greeks.gamma * 100.0);
    println!("  Theta (Θ): ${:.2}/день", call_greeks.theta);
    println!("    -> Опцион теряет ${:.2} в день из-за времени", -call_greeks.theta);
    println!("  Vega (ν):  ${:.2}/1%", call_greeks.vega);
    println!("    -> При росте IV на 1% опцион вырастет на ${:.2}", call_greeks.vega);
    println!();

    // Расчёт греков для Put
    let put_greeks = bs.put_greeks();
    println!("Put Greeks:");
    println!("  Delta (Δ): {:.4}", put_greeks.delta);
    println!("  Gamma (Γ): {:.6}", put_greeks.gamma);
    println!("  Theta (Θ): ${:.2}/день", put_greeks.theta);
    println!("  Vega (ν):  ${:.2}/1%", put_greeks.vega);
    println!();

    // Греки страддла
    let straddle_greeks = bs.straddle_greeks();
    println!("Straddle Greeks (Call + Put):");
    println!("  Delta (Δ): {:.4} (nearly zero for ATM)", straddle_greeks.delta);
    println!("  Gamma (Γ): {:.6} (double the gamma)", straddle_greeks.gamma);
    println!("  Theta (Θ): ${:.2}/день (double the decay)", straddle_greeks.theta);
    println!("  Vega (ν):  ${:.2}/1% (double the vega)", straddle_greeks.vega);
    println!();

    // Демонстрация расчёта IV
    println!("=== Implied Volatility Calculation ===\n");

    let market_price = 850.0; // Рыночная цена call опциона
    println!("Market call price: ${:.2}", market_price);

    match BlackScholes::implied_volatility(
        spot,
        strike,
        days / 365.0,
        0.0,
        market_price,
        OptionType::Call,
    ) {
        Ok(calculated_iv) => {
            println!("Calculated IV: {:.2}%", calculated_iv * 100.0);

            // Проверка
            let bs_check = BlackScholes::crypto(spot, strike, days, calculated_iv);
            println!("Verification - recalculated price: ${:.2}", bs_check.call_price());
        }
        Err(e) => {
            println!("Failed to calculate IV: {}", e);
        }
    }
    println!();

    // Демонстрация влияния параметров
    println!("=== Sensitivity Analysis ===\n");

    println!("Call price at different spot prices:");
    for delta_pct in [-5.0, -2.0, 0.0, 2.0, 5.0] {
        let new_spot = spot * (1.0 + delta_pct / 100.0);
        let bs_new = BlackScholes::crypto(new_spot, strike, days, iv);
        let new_price = bs_new.call_price();
        let pnl = new_price - call_price;
        println!(
            "  Spot ${:.0} ({:+.0}%): Call ${:.2} (P&L: {:+.2})",
            new_spot, delta_pct, new_price, pnl
        );
    }
    println!();

    println!("Call price at different IVs:");
    for delta_iv in [-10.0, -5.0, 0.0, 5.0, 10.0] {
        let new_iv = iv + delta_iv / 100.0;
        if new_iv > 0.0 {
            let bs_new = BlackScholes::crypto(spot, strike, days, new_iv);
            let new_price = bs_new.call_price();
            let pnl = new_price - call_price;
            println!(
                "  IV {:.0}% ({:+.0}%): Call ${:.2} (P&L: {:+.2})",
                new_iv * 100.0, delta_iv, new_price, pnl
            );
        }
    }
    println!();

    println!("Call price as time passes:");
    for days_left in [7.0, 5.0, 3.0, 1.0, 0.1] {
        let bs_new = BlackScholes::crypto(spot, strike, days_left, iv);
        let new_price = bs_new.call_price();
        let decay = call_price - new_price;
        println!(
            "  {} days left: Call ${:.2} (Decay: ${:.2})",
            days_left, new_price, decay
        );
    }
}
