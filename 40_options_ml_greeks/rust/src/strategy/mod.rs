//! # Торговые стратегии
//!
//! Реализация стратегий торговли волатильностью:
//! - Straddle trading
//! - Delta hedging
//! - Gamma scalping

mod delta_hedger;
mod straddle;
mod gamma_scalper;

pub use delta_hedger::DeltaHedger;
pub use straddle::StraddleStrategy;
pub use gamma_scalper::GammaScalper;

use serde::{Deserialize, Serialize};

/// Результат торговой операции
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    /// Успешно ли выполнена операция
    pub success: bool,
    /// Сообщение
    pub message: String,
    /// Цена исполнения
    pub execution_price: Option<f64>,
    /// Количество
    pub quantity: Option<f64>,
    /// Комиссия
    pub fee: Option<f64>,
}

impl TradeResult {
    pub fn success(message: impl Into<String>, price: f64, qty: f64, fee: f64) -> Self {
        Self {
            success: true,
            message: message.into(),
            execution_price: Some(price),
            quantity: Some(qty),
            fee: Some(fee),
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            execution_price: None,
            quantity: None,
            fee: None,
        }
    }
}

/// P&L атрибуция по грекам
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnlAttribution {
    /// P&L от дельты
    pub delta_pnl: f64,
    /// P&L от гаммы (реализованные движения)
    pub gamma_pnl: f64,
    /// P&L от теты (временной распад)
    pub theta_pnl: f64,
    /// P&L от веги (изменение IV)
    pub vega_pnl: f64,
    /// Общий P&L
    pub total_pnl: f64,
    /// Необъяснённый P&L (ошибки модели, высшие греки)
    pub unexplained: f64,
}

impl PnlAttribution {
    /// Расчёт атрибуции
    pub fn calculate(
        delta: f64,
        gamma: f64,
        theta: f64,
        vega: f64,
        spot_change: f64,
        iv_change: f64,
        days: f64,
        actual_pnl: f64,
    ) -> Self {
        let delta_pnl = delta * spot_change;
        let gamma_pnl = 0.5 * gamma * spot_change.powi(2);
        let theta_pnl = theta * days;
        let vega_pnl = vega * iv_change * 100.0; // vega per 1%

        let explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl;
        let unexplained = actual_pnl - explained;

        Self {
            delta_pnl,
            gamma_pnl,
            theta_pnl,
            vega_pnl,
            total_pnl: actual_pnl,
            unexplained,
        }
    }

    /// Доля каждого грека в общем P&L
    pub fn attribution_pct(&self) -> (f64, f64, f64, f64) {
        let total_abs = self.delta_pnl.abs()
            + self.gamma_pnl.abs()
            + self.theta_pnl.abs()
            + self.vega_pnl.abs();

        if total_abs < 0.0001 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        (
            self.delta_pnl.abs() / total_abs,
            self.gamma_pnl.abs() / total_abs,
            self.theta_pnl.abs() / total_abs,
            self.vega_pnl.abs() / total_abs,
        )
    }
}

impl std::fmt::Display for PnlAttribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "P&L Attribution:\n  Delta: ${:.2}\n  Gamma: ${:.2}\n  Theta: ${:.2}\n  Vega:  ${:.2}\n  Total: ${:.2}\n  Unexplained: ${:.2}",
            self.delta_pnl, self.gamma_pnl, self.theta_pnl, self.vega_pnl, self.total_pnl, self.unexplained
        )
    }
}
