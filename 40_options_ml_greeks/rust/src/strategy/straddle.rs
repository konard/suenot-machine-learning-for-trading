//! # Straddle Strategy
//!
//! Торговля волатильностью через ATM страддлы.
//! Straddle = Call + Put с одинаковым страйком.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::greeks::{BlackScholes, Greeks, OptionType};
use crate::models::OptionContract;
use crate::volatility::{VolatilityPredictor, VolatilityRiskPremium, VrpAction};

/// Параметры страддла
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StraddleParams {
    /// Страйк (обычно ATM)
    pub strike: f64,
    /// Дата экспирации
    pub expiry: DateTime<Utc>,
    /// IV страддла (среднее call и put)
    pub iv: f64,
    /// Предсказанная RV
    pub predicted_rv: f64,
    /// Edge (IV - predicted RV)
    pub edge: f64,
    /// Направление (long/short)
    pub direction: StraddleDirection,
    /// Количество контрактов
    pub contracts: f64,
    /// Цена call
    pub call_price: f64,
    /// Цена put
    pub put_price: f64,
    /// Общая цена страддла
    pub straddle_price: f64,
    /// Общая вега
    pub total_vega: f64,
    /// Общая гамма
    pub total_gamma: f64,
    /// Общая тета
    pub total_theta: f64,
}

/// Направление страддла
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StraddleDirection {
    /// Long straddle (покупка vol)
    Long,
    /// Short straddle (продажа vol)
    Short,
}

impl std::fmt::Display for StraddleDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StraddleDirection::Long => write!(f, "Long"),
            StraddleDirection::Short => write!(f, "Short"),
        }
    }
}

/// Конфигурация стратегии
#[derive(Debug, Clone)]
pub struct StraddleConfig {
    /// Минимальный edge для входа
    pub min_edge: f64,
    /// Максимальная вега экспозиция
    pub max_vega: f64,
    /// Максимальное количество контрактов
    pub max_contracts: f64,
    /// Предпочтительные экспирации (в днях)
    pub preferred_expiries: Vec<f64>,
    /// Минимальное время до экспирации (дни)
    pub min_dte: f64,
}

impl Default for StraddleConfig {
    fn default() -> Self {
        Self {
            min_edge: 0.02,               // 2% минимальный edge
            max_vega: 10000.0,            // Максимальная vega $10,000
            max_contracts: 10.0,          // Максимум 10 контрактов
            preferred_expiries: vec![7.0, 14.0, 30.0], // Недельные, двухнедельные, месячные
            min_dte: 2.0,                 // Минимум 2 дня до экспирации
        }
    }
}

/// Стратегия торговли страддлами
#[derive(Debug)]
pub struct StraddleStrategy {
    /// Конфигурация
    config: StraddleConfig,
    /// Предсказатель волатильности
    predictor: VolatilityPredictor,
    /// Анализатор VRP
    vrp: VolatilityRiskPremium,
    /// Активные страддлы
    active_straddles: Vec<StraddleParams>,
}

impl StraddleStrategy {
    /// Создать новую стратегию
    pub fn new(
        config: StraddleConfig,
        predictor: VolatilityPredictor,
        vrp: VolatilityRiskPremium,
    ) -> Self {
        Self {
            config,
            predictor,
            vrp,
            active_straddles: Vec::new(),
        }
    }

    /// Создать со стандартными настройками
    pub fn default_crypto() -> Self {
        Self::new(
            StraddleConfig::default(),
            VolatilityPredictor::default_weights(7),
            VolatilityRiskPremium::default_crypto(),
        )
    }

    /// Найти ATM страйк
    pub fn find_atm_strike(spot: f64, available_strikes: &[f64]) -> Option<f64> {
        available_strikes
            .iter()
            .min_by(|a, b| {
                ((*a - spot).abs())
                    .partial_cmp(&((*b - spot).abs()))
                    .unwrap()
            })
            .copied()
    }

    /// Оценить страддл
    pub fn evaluate_straddle(
        &self,
        spot: f64,
        strike: f64,
        expiry: DateTime<Utc>,
        call: &OptionContract,
        put: &OptionContract,
        predicted_rv: f64,
    ) -> Option<StraddleParams> {
        // Среднее IV
        let straddle_iv = (call.iv + put.iv) / 2.0;

        // Edge
        let edge = straddle_iv - predicted_rv;

        if edge.abs() < self.config.min_edge {
            return None;
        }

        let direction = if edge > 0.0 {
            StraddleDirection::Short // IV переоценена, продаём
        } else {
            StraddleDirection::Long // IV недооценена, покупаем
        };

        // Расчёт греков
        let days_to_expiry = call.days_to_expiry();
        let bs = BlackScholes::crypto(spot, strike, days_to_expiry, straddle_iv);

        let call_greeks = bs.call_greeks();
        let put_greeks = bs.put_greeks();

        let total_vega = call_greeks.vega + put_greeks.vega;
        let total_gamma = call_greeks.gamma + put_greeks.gamma;
        let total_theta = call_greeks.theta + put_greeks.theta;

        // Размер позиции на основе vega-лимита
        let max_contracts = if total_vega > 0.0 {
            (self.config.max_vega / total_vega).min(self.config.max_contracts)
        } else {
            self.config.max_contracts
        };

        Some(StraddleParams {
            strike,
            expiry,
            iv: straddle_iv,
            predicted_rv,
            edge,
            direction,
            contracts: max_contracts,
            call_price: call.price,
            put_price: put.price,
            straddle_price: call.price + put.price,
            total_vega,
            total_gamma,
            total_theta,
        })
    }

    /// Получить торговый сигнал
    pub fn get_signal(
        &self,
        current_iv: f64,
        predicted_rv: f64,
    ) -> (VrpAction, f64) {
        let signal = self.vrp.trading_signal(current_iv, predicted_rv, None);
        (signal.action, signal.edge)
    }

    /// Добавить активный страддл
    pub fn add_straddle(&mut self, straddle: StraddleParams) {
        self.active_straddles.push(straddle);
    }

    /// Удалить страддл
    pub fn remove_straddle(&mut self, strike: f64, expiry: DateTime<Utc>) {
        self.active_straddles.retain(|s| s.strike != strike || s.expiry != expiry);
    }

    /// Активные страддлы
    pub fn active_straddles(&self) -> &[StraddleParams] {
        &self.active_straddles
    }

    /// Общая вега позиции
    pub fn total_vega(&self) -> f64 {
        self.active_straddles
            .iter()
            .map(|s| {
                let multiplier = match s.direction {
                    StraddleDirection::Long => 1.0,
                    StraddleDirection::Short => -1.0,
                };
                s.total_vega * s.contracts * multiplier
            })
            .sum()
    }

    /// Общая гамма позиции
    pub fn total_gamma(&self) -> f64 {
        self.active_straddles
            .iter()
            .map(|s| {
                let multiplier = match s.direction {
                    StraddleDirection::Long => 1.0,
                    StraddleDirection::Short => -1.0,
                };
                s.total_gamma * s.contracts * multiplier
            })
            .sum()
    }

    /// Общая тета позиции
    pub fn total_theta(&self) -> f64 {
        self.active_straddles
            .iter()
            .map(|s| {
                let multiplier = match s.direction {
                    StraddleDirection::Long => 1.0,
                    StraddleDirection::Short => -1.0,
                };
                s.total_theta * s.contracts * multiplier
            })
            .sum()
    }

    /// Расчёт ожидаемого P&L при заданной реализованной волатильности
    pub fn expected_pnl(&self, realized_vol: f64) -> f64 {
        self.active_straddles
            .iter()
            .map(|s| {
                let vol_diff = match s.direction {
                    StraddleDirection::Short => s.iv - realized_vol,
                    StraddleDirection::Long => realized_vol - s.iv,
                };
                // Приблизительный P&L через vega
                vol_diff * s.total_vega * 100.0 * s.contracts
            })
            .sum()
    }
}

/// Результат бэктеста страддла
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StraddleBacktestResult {
    /// Количество сделок
    pub num_trades: usize,
    /// Процент прибыльных
    pub win_rate: f64,
    /// Общий P&L
    pub total_pnl: f64,
    /// Средний P&L на сделку
    pub avg_pnl: f64,
    /// Максимальный P&L
    pub max_pnl: f64,
    /// Минимальный P&L
    pub min_pnl: f64,
    /// Sharpe ratio (если применимо)
    pub sharpe: Option<f64>,
    /// P&L атрибуция
    pub theta_pnl: f64,
    pub gamma_pnl: f64,
    pub vega_pnl: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_find_atm_strike() {
        let strikes = vec![40000.0, 41000.0, 42000.0, 43000.0, 44000.0];

        let atm = StraddleStrategy::find_atm_strike(42500.0, &strikes);
        assert_eq!(atm, Some(42000.0).or(Some(43000.0))); // Ближайший

        let atm = StraddleStrategy::find_atm_strike(42000.0, &strikes);
        assert_eq!(atm, Some(42000.0));
    }

    #[test]
    fn test_straddle_evaluation() {
        let strategy = StraddleStrategy::default_crypto();

        let expiry = Utc::now() + Duration::days(7);

        let call = OptionContract::new("BTC", 42000.0, expiry, OptionType::Call, 800.0, 0.55);
        let put = OptionContract::new("BTC", 42000.0, expiry, OptionType::Put, 750.0, 0.53);

        // IV = 54%, predicted RV = 45% → должны продавать
        let result = strategy.evaluate_straddle(42000.0, 42000.0, expiry, &call, &put, 0.45);

        assert!(result.is_some());
        let straddle = result.unwrap();
        assert_eq!(straddle.direction, StraddleDirection::Short);
        assert!(straddle.edge > 0.05); // Edge ~9%
    }

    #[test]
    fn test_straddle_no_edge() {
        let strategy = StraddleStrategy::default_crypto();

        let expiry = Utc::now() + Duration::days(7);

        let call = OptionContract::new("BTC", 42000.0, expiry, OptionType::Call, 800.0, 0.50);
        let put = OptionContract::new("BTC", 42000.0, expiry, OptionType::Put, 750.0, 0.50);

        // IV = 50%, predicted RV = 49% → edge слишком маленький
        let result = strategy.evaluate_straddle(42000.0, 42000.0, expiry, &call, &put, 0.49);

        assert!(result.is_none()); // Нет сделки
    }

    #[test]
    fn test_total_greeks() {
        let mut strategy = StraddleStrategy::default_crypto();

        let expiry = Utc::now() + Duration::days(7);

        strategy.add_straddle(StraddleParams {
            strike: 42000.0,
            expiry,
            iv: 0.55,
            predicted_rv: 0.45,
            edge: 0.10,
            direction: StraddleDirection::Short,
            contracts: 2.0,
            call_price: 800.0,
            put_price: 750.0,
            straddle_price: 1550.0,
            total_vega: 50.0,
            total_gamma: 0.002,
            total_theta: -20.0,
        });

        // Short straddle → отрицательная vega, положительная theta
        assert!(strategy.total_vega() < 0.0);
        assert!(strategy.total_theta() > 0.0);
    }
}
