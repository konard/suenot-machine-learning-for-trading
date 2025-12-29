//! # Модель Блэка-Шоулза
//!
//! Реализация модели ценообразования европейских опционов.
//!
//! ## Формулы
//!
//! Call = S × N(d₁) - K × e^(-rT) × N(d₂)
//! Put  = K × e^(-rT) × N(-d₂) - S × N(-d₁)
//!
//! где:
//! - d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
//! - d₂ = d₁ - σ√T

use super::{Greeks, OptionType};
use statrs::distribution::{ContinuousCDF, Normal};

/// Модель Блэка-Шоулза для ценообразования опционов
#[derive(Debug, Clone)]
pub struct BlackScholes {
    /// Текущая цена базового актива
    pub spot: f64,
    /// Страйк-цена опциона
    pub strike: f64,
    /// Время до экспирации (в годах)
    pub time_to_expiry: f64,
    /// Безрисковая процентная ставка
    pub risk_free_rate: f64,
    /// Волатильность (годовая)
    pub volatility: f64,
    /// Кэш для N(0,1)
    normal: Normal,
}

impl BlackScholes {
    /// Создать новый экземпляр модели
    ///
    /// # Arguments
    ///
    /// * `spot` - Текущая цена актива
    /// * `strike` - Страйк-цена опциона
    /// * `time_to_expiry` - Время до экспирации в годах (7 дней = 7/365)
    /// * `risk_free_rate` - Безрисковая ставка (0.05 = 5%)
    /// * `volatility` - Волатильность (0.55 = 55%)
    ///
    /// # Пример
    ///
    /// ```rust
    /// use options_greeks_ml::greeks::BlackScholes;
    ///
    /// let bs = BlackScholes::new(42000.0, 42000.0, 7.0/365.0, 0.05, 0.55);
    /// let price = bs.call_price();
    /// ```
    pub fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
    ) -> Self {
        Self {
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            normal: Normal::new(0.0, 1.0).unwrap(),
        }
    }

    /// Создать модель для криптовалюты (ставка = 0)
    pub fn crypto(spot: f64, strike: f64, days_to_expiry: f64, volatility: f64) -> Self {
        Self::new(spot, strike, days_to_expiry / 365.0, 0.0, volatility)
    }

    /// Расчёт d1
    fn d1(&self) -> f64 {
        let numerator = (self.spot / self.strike).ln()
            + (self.risk_free_rate + self.volatility.powi(2) / 2.0) * self.time_to_expiry;
        let denominator = self.volatility * self.time_to_expiry.sqrt();

        if denominator.abs() < 1e-10 {
            if self.spot > self.strike {
                f64::INFINITY
            } else if self.spot < self.strike {
                f64::NEG_INFINITY
            } else {
                0.0
            }
        } else {
            numerator / denominator
        }
    }

    /// Расчёт d2
    fn d2(&self) -> f64 {
        self.d1() - self.volatility * self.time_to_expiry.sqrt()
    }

    /// N(x) - функция нормального распределения
    fn n(&self, x: f64) -> f64 {
        if x.is_infinite() {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            self.normal.cdf(x)
        }
    }

    /// n(x) - плотность нормального распределения
    fn pdf(&self, x: f64) -> f64 {
        if x.is_infinite() {
            0.0
        } else {
            (-x.powi(2) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
        }
    }

    /// Цена call опциона
    pub fn call_price(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return (self.spot - self.strike).max(0.0);
        }

        let d1 = self.d1();
        let d2 = self.d2();

        self.spot * self.n(d1)
            - self.strike * (-self.risk_free_rate * self.time_to_expiry).exp() * self.n(d2)
    }

    /// Цена put опциона
    pub fn put_price(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return (self.strike - self.spot).max(0.0);
        }

        let d1 = self.d1();
        let d2 = self.d2();

        self.strike * (-self.risk_free_rate * self.time_to_expiry).exp() * self.n(-d2)
            - self.spot * self.n(-d1)
    }

    /// Цена опциона по типу
    pub fn price(&self, option_type: OptionType) -> f64 {
        match option_type {
            OptionType::Call => self.call_price(),
            OptionType::Put => self.put_price(),
        }
    }

    /// Цена страддла (Call + Put с одним страйком)
    pub fn straddle_price(&self) -> f64 {
        self.call_price() + self.put_price()
    }

    /// Расчёт греков для call опциона
    pub fn call_greeks(&self) -> Greeks {
        self.greeks_for_type(OptionType::Call)
    }

    /// Расчёт греков для put опциона
    pub fn put_greeks(&self) -> Greeks {
        self.greeks_for_type(OptionType::Put)
    }

    /// Расчёт греков по типу опциона
    pub fn greeks_for_type(&self, option_type: OptionType) -> Greeks {
        if self.time_to_expiry <= 0.0 {
            return Greeks::zero();
        }

        let d1 = self.d1();
        let d2 = self.d2();
        let sqrt_t = self.time_to_expiry.sqrt();
        let exp_rt = (-self.risk_free_rate * self.time_to_expiry).exp();

        // Delta
        let delta = match option_type {
            OptionType::Call => self.n(d1),
            OptionType::Put => self.n(d1) - 1.0,
        };

        // Gamma (одинаковая для call и put)
        let gamma = self.pdf(d1) / (self.spot * self.volatility * sqrt_t);

        // Theta (в день)
        let theta = match option_type {
            OptionType::Call => {
                -self.spot * self.pdf(d1) * self.volatility / (2.0 * sqrt_t)
                    - self.risk_free_rate * self.strike * exp_rt * self.n(d2)
            }
            OptionType::Put => {
                -self.spot * self.pdf(d1) * self.volatility / (2.0 * sqrt_t)
                    + self.risk_free_rate * self.strike * exp_rt * self.n(-d2)
            }
        } / 365.0;

        // Vega (на 1% изменения IV)
        let vega = self.spot * self.pdf(d1) * sqrt_t / 100.0;

        // Rho (на 1% изменения ставки)
        let rho = match option_type {
            OptionType::Call => self.strike * self.time_to_expiry * exp_rt * self.n(d2) / 100.0,
            OptionType::Put => -self.strike * self.time_to_expiry * exp_rt * self.n(-d2) / 100.0,
        };

        Greeks::new(delta, gamma, theta, vega, rho)
    }

    /// Греки страддла (Call + Put)
    pub fn straddle_greeks(&self) -> Greeks {
        let call = self.call_greeks();
        let put = self.put_greeks();
        call.add(&put, 1.0)
    }

    /// Расчёт подразумеваемой волатильности из цены опциона
    /// методом Ньютона-Рафсона
    pub fn implied_volatility(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        market_price: f64,
        option_type: OptionType,
    ) -> Result<f64, &'static str> {
        const MAX_ITERATIONS: usize = 100;
        const TOLERANCE: f64 = 1e-6;

        // Начальное приближение
        let mut sigma = 0.3;

        for _ in 0..MAX_ITERATIONS {
            let bs = BlackScholes::new(spot, strike, time_to_expiry, risk_free_rate, sigma);
            let price = bs.price(option_type);
            let vega = bs.greeks_for_type(option_type).vega * 100.0; // Восстанавливаем масштаб

            let diff = price - market_price;

            if diff.abs() < TOLERANCE {
                return Ok(sigma);
            }

            if vega.abs() < 1e-10 {
                return Err("Vega too small for Newton-Raphson");
            }

            sigma -= diff / vega;

            // Ограничения на sigma
            if sigma <= 0.001 {
                sigma = 0.001;
            } else if sigma > 5.0 {
                sigma = 5.0;
            }
        }

        Err("Implied volatility did not converge")
    }

    /// Обновить цену спота
    pub fn with_spot(&self, new_spot: f64) -> Self {
        Self::new(
            new_spot,
            self.strike,
            self.time_to_expiry,
            self.risk_free_rate,
            self.volatility,
        )
    }

    /// Обновить время до экспирации
    pub fn with_time(&self, new_time: f64) -> Self {
        Self::new(
            self.spot,
            self.strike,
            new_time,
            self.risk_free_rate,
            self.volatility,
        )
    }

    /// Обновить волатильность
    pub fn with_volatility(&self, new_vol: f64) -> Self {
        Self::new(
            self.spot,
            self.strike,
            self.time_to_expiry,
            self.risk_free_rate,
            new_vol,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atm_call_price() {
        // ATM опцион, 30 дней, 20% волатильность
        let bs = BlackScholes::new(100.0, 100.0, 30.0 / 365.0, 0.05, 0.20);
        let price = bs.call_price();

        // Цена должна быть примерно 2.5-3.5
        assert!(price > 2.0 && price < 4.0, "Call price: {}", price);
    }

    #[test]
    fn test_put_call_parity() {
        // Put-Call Parity: C - P = S - K*e^(-rT)
        let bs = BlackScholes::new(100.0, 100.0, 30.0 / 365.0, 0.05, 0.20);

        let call = bs.call_price();
        let put = bs.put_price();
        let parity = bs.spot
            - bs.strike * (-bs.risk_free_rate * bs.time_to_expiry).exp();

        assert!(
            (call - put - parity).abs() < 0.01,
            "Put-Call parity violated: {} vs {}",
            call - put,
            parity
        );
    }

    #[test]
    fn test_delta_range() {
        let bs = BlackScholes::new(100.0, 100.0, 30.0 / 365.0, 0.05, 0.20);

        let call_delta = bs.call_greeks().delta;
        let put_delta = bs.put_greeks().delta;

        // Call delta: 0 to 1
        assert!(
            call_delta >= 0.0 && call_delta <= 1.0,
            "Call delta: {}",
            call_delta
        );

        // Put delta: -1 to 0
        assert!(
            put_delta >= -1.0 && put_delta <= 0.0,
            "Put delta: {}",
            put_delta
        );

        // Call delta + |Put delta| = 1 (approximately for ATM)
        assert!(
            (call_delta + put_delta.abs() - 1.0).abs() < 0.01,
            "Delta sum: {}",
            call_delta + put_delta.abs()
        );
    }

    #[test]
    fn test_straddle_delta_near_zero() {
        // ATM страддл должен иметь дельту близкую к нулю
        let bs = BlackScholes::new(100.0, 100.0, 30.0 / 365.0, 0.05, 0.20);
        let straddle = bs.straddle_greeks();

        assert!(
            straddle.delta.abs() < 0.1,
            "Straddle delta: {}",
            straddle.delta
        );
    }

    #[test]
    fn test_implied_volatility() {
        let bs = BlackScholes::new(100.0, 100.0, 30.0 / 365.0, 0.05, 0.25);
        let price = bs.call_price();

        let iv = BlackScholes::implied_volatility(100.0, 100.0, 30.0 / 365.0, 0.05, price, OptionType::Call)
            .unwrap();

        assert!((iv - 0.25).abs() < 0.001, "IV: {} vs expected 0.25", iv);
    }

    #[test]
    fn test_crypto_model() {
        // Криптовалютный опцион (ставка = 0)
        let bs = BlackScholes::crypto(42000.0, 42000.0, 7.0, 0.55);

        let price = bs.straddle_price();
        assert!(price > 0.0, "Straddle price should be positive");

        let greeks = bs.straddle_greeks();
        // Gamma страддла = 2 * gamma одного опциона
        assert!(greeks.gamma > 0.0, "Gamma should be positive");
    }
}
