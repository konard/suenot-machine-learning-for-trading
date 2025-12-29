//! # Расчёт и анализ подразумеваемой волатильности

use crate::greeks::{BlackScholes, OptionType};

/// Калькулятор подразумеваемой волатильности
#[derive(Debug, Clone)]
pub struct ImpliedVolatility;

/// Данные поверхности волатильности
#[derive(Debug, Clone)]
pub struct VolatilitySurface {
    /// Страйки
    pub strikes: Vec<f64>,
    /// Экспирации (в днях)
    pub expiries: Vec<f64>,
    /// IV матрица [strike_idx][expiry_idx]
    pub ivs: Vec<Vec<f64>>,
    /// Спот цена
    pub spot: f64,
}

/// Точка на кривой улыбки волатильности
#[derive(Debug, Clone)]
pub struct SmilePoint {
    pub strike: f64,
    pub moneyness: f64, // K/S
    pub delta: f64,
    pub iv: f64,
}

impl ImpliedVolatility {
    /// Расчёт IV из цены опциона
    pub fn from_price(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        market_price: f64,
        option_type: OptionType,
    ) -> Result<f64, &'static str> {
        BlackScholes::implied_volatility(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            market_price,
            option_type,
        )
    }

    /// Расчёт ATM IV (среднее call и put)
    pub fn atm_iv(
        spot: f64,
        time_to_expiry: f64,
        call_price: f64,
        put_price: f64,
    ) -> Result<f64, &'static str> {
        let call_iv =
            Self::from_price(spot, spot, time_to_expiry, 0.0, call_price, OptionType::Call)?;
        let put_iv =
            Self::from_price(spot, spot, time_to_expiry, 0.0, put_price, OptionType::Put)?;

        Ok((call_iv + put_iv) / 2.0)
    }

    /// Расчёт IV skew (разница между OTM put и OTM call)
    pub fn skew(
        spot: f64,
        time_to_expiry: f64,
        put_strike: f64,
        put_price: f64,
        call_strike: f64,
        call_price: f64,
    ) -> Result<f64, &'static str> {
        let put_iv =
            Self::from_price(spot, put_strike, time_to_expiry, 0.0, put_price, OptionType::Put)?;
        let call_iv = Self::from_price(
            spot,
            call_strike,
            time_to_expiry,
            0.0,
            call_price,
            OptionType::Call,
        )?;

        Ok(put_iv - call_iv)
    }

    /// Расчёт term structure slope (разница IV разных экспираций)
    pub fn term_slope(short_term_iv: f64, long_term_iv: f64) -> f64 {
        long_term_iv - short_term_iv
    }

    /// Построение улыбки волатильности
    pub fn build_smile(
        spot: f64,
        time_to_expiry: f64,
        strikes: &[f64],
        prices: &[f64],
        option_types: &[OptionType],
    ) -> Vec<SmilePoint> {
        strikes
            .iter()
            .zip(prices.iter())
            .zip(option_types.iter())
            .filter_map(|((strike, price), opt_type)| {
                let iv = Self::from_price(spot, *strike, time_to_expiry, 0.0, *price, *opt_type)
                    .ok()?;

                let bs = BlackScholes::crypto(spot, *strike, time_to_expiry * 365.0, iv);
                let delta = bs.greeks_for_type(*opt_type).delta;

                Some(SmilePoint {
                    strike: *strike,
                    moneyness: strike / spot,
                    delta,
                    iv,
                })
            })
            .collect()
    }
}

impl VolatilitySurface {
    /// Создать новую поверхность волатильности
    pub fn new(spot: f64, strikes: Vec<f64>, expiries: Vec<f64>, ivs: Vec<Vec<f64>>) -> Self {
        Self {
            spot,
            strikes,
            expiries,
            ivs,
        }
    }

    /// Получить IV для конкретного страйка и экспирации
    pub fn get_iv(&self, strike: f64, expiry: f64) -> Option<f64> {
        // Находим ближайший страйк
        let strike_idx = self
            .strikes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - strike).abs())
                    .partial_cmp(&((*b - strike).abs()))
                    .unwrap()
            })?
            .0;

        // Находим ближайшую экспирацию
        let expiry_idx = self
            .expiries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((*a - expiry).abs())
                    .partial_cmp(&((*b - expiry).abs()))
                    .unwrap()
            })?
            .0;

        self.ivs.get(strike_idx)?.get(expiry_idx).copied()
    }

    /// Интерполяция IV (линейная)
    pub fn interpolate_iv(&self, strike: f64, expiry: f64) -> Option<f64> {
        // Упрощённая линейная интерполяция
        // В продакшене использовать более сложные методы (spline, SABR)
        self.get_iv(strike, expiry)
    }

    /// ATM IV для данной экспирации
    pub fn atm_iv(&self, expiry: f64) -> Option<f64> {
        self.get_iv(self.spot, expiry)
    }

    /// Skew для данной экспирации
    pub fn skew(&self, expiry: f64, delta_put: f64, delta_call: f64) -> Option<f64> {
        // Находим страйки по дельте
        let put_strike = self.spot * (1.0 - delta_put.abs() * 0.1); // Упрощение
        let call_strike = self.spot * (1.0 + delta_call * 0.1);

        let put_iv = self.get_iv(put_strike, expiry)?;
        let call_iv = self.get_iv(call_strike, expiry)?;

        Some(put_iv - call_iv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implied_volatility() {
        // Создаём опцион с известной IV
        let bs = BlackScholes::crypto(42000.0, 42000.0, 7.0, 0.55);
        let price = bs.call_price();

        // Восстанавливаем IV
        let iv = ImpliedVolatility::from_price(
            42000.0,
            42000.0,
            7.0 / 365.0,
            0.0,
            price,
            OptionType::Call,
        )
        .unwrap();

        assert!(
            (iv - 0.55).abs() < 0.01,
            "IV should be ~0.55, got {}",
            iv
        );
    }

    #[test]
    fn test_atm_iv() {
        let bs = BlackScholes::crypto(100.0, 100.0, 30.0, 0.30);

        let atm_iv = ImpliedVolatility::atm_iv(100.0, 30.0 / 365.0, bs.call_price(), bs.put_price())
            .unwrap();

        assert!(
            (atm_iv - 0.30).abs() < 0.01,
            "ATM IV should be ~0.30, got {}",
            atm_iv
        );
    }
}
