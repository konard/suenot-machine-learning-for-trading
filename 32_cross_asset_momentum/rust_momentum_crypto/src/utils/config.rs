//! Конфигурация
//!
//! Этот модуль содержит структуры конфигурации для стратегии.

use serde::{Deserialize, Serialize};

/// Главная конфигурация стратегии
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Название стратегии
    pub name: String,
    /// Описание
    pub description: String,
    /// Вселенная активов
    pub universe: Vec<String>,
    /// Конфигурация моментума
    pub momentum: MomentumSettings,
    /// Конфигурация портфеля
    pub portfolio: PortfolioSettings,
    /// Конфигурация торговли
    pub trading: TradingSettings,
}

/// Настройки моментума
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumSettings {
    /// Период time-series momentum (дни)
    pub ts_lookback: usize,
    /// Период cross-sectional momentum (дни)
    pub cs_lookback: usize,
    /// Количество топ активов
    pub top_n: usize,
    /// Пропуск периода
    pub skip_period: usize,
    /// Использовать множественные периоды
    pub multi_period: bool,
    /// Веса периодов (если multi_period = true)
    pub period_weights: Option<Vec<(usize, f64)>>,
}

/// Настройки портфеля
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSettings {
    /// Начальный капитал
    pub initial_capital: f64,
    /// Целевая волатильность
    pub target_volatility: f64,
    /// Максимальный вес одного актива
    pub max_weight: f64,
    /// Использовать risk parity
    pub use_risk_parity: bool,
    /// Максимальный leverage
    pub max_leverage: f64,
}

/// Настройки торговли
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSettings {
    /// Период ребалансировки (дни)
    pub rebalance_period: usize,
    /// Порог ребалансировки (изменение веса)
    pub rebalance_threshold: f64,
    /// Комиссия (процент)
    pub commission: f64,
    /// Slippage (процент)
    pub slippage: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            name: "Crypto Dual Momentum".to_string(),
            description: "Cross-asset momentum strategy for cryptocurrencies".to_string(),
            universe: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "SOLUSDT".to_string(),
                "BNBUSDT".to_string(),
                "XRPUSDT".to_string(),
                "ADAUSDT".to_string(),
                "AVAXUSDT".to_string(),
                "DOTUSDT".to_string(),
            ],
            momentum: MomentumSettings::default(),
            portfolio: PortfolioSettings::default(),
            trading: TradingSettings::default(),
        }
    }
}

impl Default for MomentumSettings {
    fn default() -> Self {
        Self {
            ts_lookback: 30,
            cs_lookback: 30,
            top_n: 3,
            skip_period: 1,
            multi_period: false,
            period_weights: None,
        }
    }
}

impl Default for PortfolioSettings {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            target_volatility: 0.30,
            max_weight: 0.40,
            use_risk_parity: true,
            max_leverage: 1.0,
        }
    }
}

impl Default for TradingSettings {
    fn default() -> Self {
        Self {
            rebalance_period: 7,
            rebalance_threshold: 0.05,
            commission: 0.001,
            slippage: 0.0005,
        }
    }
}

impl StrategyConfig {
    /// Загрузить конфигурацию из JSON файла
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Сохранить конфигурацию в JSON файл
    pub fn to_file(&self, path: &str) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Создать агрессивную конфигурацию (больше риска)
    pub fn aggressive() -> Self {
        Self {
            name: "Aggressive Crypto Momentum".to_string(),
            description: "High-risk momentum strategy".to_string(),
            universe: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "SOLUSDT".to_string(),
                "AVAXUSDT".to_string(),
            ],
            momentum: MomentumSettings {
                ts_lookback: 14,
                cs_lookback: 14,
                top_n: 2,
                skip_period: 0,
                multi_period: false,
                period_weights: None,
            },
            portfolio: PortfolioSettings {
                initial_capital: 10000.0,
                target_volatility: 0.50,
                max_weight: 0.50,
                use_risk_parity: false,
                max_leverage: 1.5,
            },
            trading: TradingSettings {
                rebalance_period: 3,
                rebalance_threshold: 0.03,
                commission: 0.001,
                slippage: 0.001,
            },
        }
    }

    /// Создать консервативную конфигурацию (меньше риска)
    pub fn conservative() -> Self {
        Self {
            name: "Conservative Crypto Momentum".to_string(),
            description: "Low-risk momentum strategy".to_string(),
            universe: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
            ],
            momentum: MomentumSettings {
                ts_lookback: 60,
                cs_lookback: 60,
                top_n: 2,
                skip_period: 7,
                multi_period: true,
                period_weights: Some(vec![
                    (14, 0.2),
                    (30, 0.3),
                    (60, 0.5),
                ]),
            },
            portfolio: PortfolioSettings {
                initial_capital: 10000.0,
                target_volatility: 0.15,
                max_weight: 0.30,
                use_risk_parity: true,
                max_leverage: 0.8,
            },
            trading: TradingSettings {
                rebalance_period: 14,
                rebalance_threshold: 0.10,
                commission: 0.001,
                slippage: 0.0005,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = StrategyConfig::default();
        assert!(!config.universe.is_empty());
        assert!(config.momentum.top_n > 0);
    }

    #[test]
    fn test_aggressive_config() {
        let config = StrategyConfig::aggressive();
        assert!(config.portfolio.target_volatility > 0.30);
        assert!(config.momentum.ts_lookback < 30);
    }

    #[test]
    fn test_conservative_config() {
        let config = StrategyConfig::conservative();
        assert!(config.portfolio.target_volatility < 0.30);
        assert!(config.momentum.ts_lookback > 30);
    }

    #[test]
    fn test_serialization() {
        let config = StrategyConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: StrategyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.name, parsed.name);
    }
}
