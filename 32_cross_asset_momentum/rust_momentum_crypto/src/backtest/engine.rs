//! Движок бэктестинга
//!
//! Этот модуль содержит движок для симуляции торговой стратегии
//! на исторических данных.

use crate::data::{Portfolio, PriceSeries};
use crate::momentum::{DualMomentum, DualMomentumConfig};
use crate::strategy::{WeightCalculator, WeightConfig};
use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

/// Конфигурация бэктеста
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Начальный капитал
    pub initial_capital: f64,
    /// Комиссия за сделку (в процентах)
    pub commission: f64,
    /// Slippage (в процентах)
    pub slippage: f64,
    /// Период ребалансировки (в днях)
    pub rebalance_period: usize,
    /// Минимальное изменение веса для ребалансировки
    pub rebalance_threshold: f64,
    /// Использовать фракционные позиции
    pub allow_fractional: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission: 0.001, // 0.1% на Bybit
            slippage: 0.0005,  // 0.05%
            rebalance_period: 7, // Еженедельно
            rebalance_threshold: 0.05,
            allow_fractional: true,
        }
    }
}

/// Сделка
#[derive(Debug, Clone)]
pub struct Trade {
    /// Временная метка
    pub timestamp: DateTime<Utc>,
    /// Символ
    pub symbol: String,
    /// Направление (true = покупка, false = продажа)
    pub is_buy: bool,
    /// Количество
    pub quantity: f64,
    /// Цена
    pub price: f64,
    /// Комиссия
    pub commission: f64,
    /// Общая стоимость
    pub total_cost: f64,
}

/// Снимок портфеля
#[derive(Debug, Clone)]
pub struct PortfolioSnapshot {
    /// Временная метка
    pub timestamp: DateTime<Utc>,
    /// Стоимость портфеля
    pub value: f64,
    /// Доля в кеше
    pub cash_ratio: f64,
    /// Позиции (symbol -> value)
    pub positions: HashMap<String, f64>,
    /// Веса (symbol -> weight)
    pub weights: HashMap<String, f64>,
}

/// Результат бэктеста
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Начальный капитал
    pub initial_capital: f64,
    /// Конечный капитал
    pub final_capital: f64,
    /// Общая доходность
    pub total_return: f64,
    /// Годовая доходность (CAGR)
    pub cagr: f64,
    /// Волатильность (годовая)
    pub volatility: f64,
    /// Sharpe Ratio
    pub sharpe_ratio: f64,
    /// Sortino Ratio
    pub sortino_ratio: f64,
    /// Максимальная просадка
    pub max_drawdown: f64,
    /// Calmar Ratio
    pub calmar_ratio: f64,
    /// Количество сделок
    pub num_trades: usize,
    /// Общая комиссия
    pub total_commission: f64,
    /// Turnover
    pub turnover: f64,
    /// История портфеля
    pub portfolio_history: Vec<PortfolioSnapshot>,
    /// История сделок
    pub trades: Vec<Trade>,
}

/// Движок бэктеста
pub struct BacktestEngine {
    config: BacktestConfig,
    momentum_config: DualMomentumConfig,
    weight_config: WeightConfig,
}

impl BacktestEngine {
    /// Создать новый движок
    pub fn new(
        config: BacktestConfig,
        momentum_config: DualMomentumConfig,
        weight_config: WeightConfig,
    ) -> Self {
        Self {
            config,
            momentum_config,
            weight_config,
        }
    }

    /// Запустить бэктест
    pub fn run(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<BacktestResult> {
        // Инициализация
        let mut cash = self.config.initial_capital;
        let mut positions: HashMap<String, f64> = HashMap::new(); // symbol -> quantity
        let mut portfolio_history: Vec<PortfolioSnapshot> = Vec::new();
        let mut trades: Vec<Trade> = Vec::new();
        let mut total_commission = 0.0;

        let dual_momentum = DualMomentum::new(self.momentum_config.clone());
        let weight_calc = WeightCalculator::new(self.weight_config.clone());

        // Находим все уникальные даты
        let mut all_dates: Vec<DateTime<Utc>> = Vec::new();
        for series in price_data.values() {
            for candle in &series.candles {
                if candle.timestamp >= start_date && candle.timestamp <= end_date {
                    if !all_dates.contains(&candle.timestamp) {
                        all_dates.push(candle.timestamp);
                    }
                }
            }
        }
        all_dates.sort();

        let mut last_rebalance_idx: Option<usize> = None;

        for (idx, &date) in all_dates.iter().enumerate() {
            // Получаем цены на текущую дату
            let current_prices: HashMap<String, f64> = price_data
                .iter()
                .filter_map(|(symbol, series)| {
                    series
                        .candles
                        .iter()
                        .find(|c| c.timestamp <= date)
                        .map(|c| (symbol.clone(), c.close))
                })
                .collect();

            // Рассчитываем стоимость портфеля
            let portfolio_value = self.calculate_portfolio_value(cash, &positions, &current_prices);

            // Проверяем, нужно ли ребалансировать
            let should_rebalance = match last_rebalance_idx {
                None => idx >= self.momentum_config.ts_lookback,
                Some(last_idx) => idx - last_idx >= self.config.rebalance_period,
            };

            if should_rebalance {
                // Генерируем сигналы на основе данных до текущей даты
                let historical_data = self.get_historical_data(price_data, date);

                if let Ok(analysis) = dual_momentum.analyze(&historical_data) {
                    // Рассчитываем целевые веса
                    let target_weights: HashMap<String, f64> = analysis
                        .iter()
                        .filter(|a| a.selected)
                        .map(|a| (a.symbol.clone(), a.weight))
                        .collect();

                    // Выполняем ребалансировку
                    let (new_cash, new_positions, new_trades) = self.rebalance(
                        cash,
                        &positions,
                        &target_weights,
                        &current_prices,
                        date,
                    );

                    cash = new_cash;
                    positions = new_positions;
                    total_commission += new_trades.iter().map(|t| t.commission).sum::<f64>();
                    trades.extend(new_trades);
                    last_rebalance_idx = Some(idx);
                }
            }

            // Сохраняем снимок
            let weights = self.calculate_weights(&positions, &current_prices, portfolio_value);
            let snapshot = PortfolioSnapshot {
                timestamp: date,
                value: portfolio_value,
                cash_ratio: cash / portfolio_value,
                positions: positions
                    .iter()
                    .map(|(s, &q)| (s.clone(), q * current_prices.get(s).unwrap_or(&0.0)))
                    .collect(),
                weights,
            };
            portfolio_history.push(snapshot);
        }

        // Рассчитываем метрики
        let result = self.calculate_metrics(
            portfolio_history,
            trades,
            total_commission,
        );

        Ok(result)
    }

    /// Получить исторические данные до определённой даты
    fn get_historical_data(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        until_date: DateTime<Utc>,
    ) -> HashMap<String, PriceSeries> {
        price_data
            .iter()
            .map(|(symbol, series)| {
                let mut new_series = PriceSeries::new(symbol.clone(), series.interval.clone());
                for candle in &series.candles {
                    if candle.timestamp <= until_date {
                        new_series.push(candle.clone());
                    }
                }
                (symbol.clone(), new_series)
            })
            .collect()
    }

    /// Рассчитать стоимость портфеля
    fn calculate_portfolio_value(
        &self,
        cash: f64,
        positions: &HashMap<String, f64>,
        prices: &HashMap<String, f64>,
    ) -> f64 {
        let positions_value: f64 = positions
            .iter()
            .map(|(symbol, &quantity)| quantity * prices.get(symbol).unwrap_or(&0.0))
            .sum();

        cash + positions_value
    }

    /// Рассчитать веса позиций
    fn calculate_weights(
        &self,
        positions: &HashMap<String, f64>,
        prices: &HashMap<String, f64>,
        total_value: f64,
    ) -> HashMap<String, f64> {
        if total_value == 0.0 {
            return HashMap::new();
        }

        positions
            .iter()
            .map(|(symbol, &quantity)| {
                let value = quantity * prices.get(symbol).unwrap_or(&0.0);
                (symbol.clone(), value / total_value)
            })
            .collect()
    }

    /// Выполнить ребалансировку
    fn rebalance(
        &self,
        cash: f64,
        positions: &HashMap<String, f64>,
        target_weights: &HashMap<String, f64>,
        prices: &HashMap<String, f64>,
        timestamp: DateTime<Utc>,
    ) -> (f64, HashMap<String, f64>, Vec<Trade>) {
        let mut new_cash = cash;
        let mut new_positions = positions.clone();
        let mut trades = Vec::new();

        let total_value = self.calculate_portfolio_value(cash, positions, prices);
        let current_weights = self.calculate_weights(positions, prices, total_value);

        // Сначала продаём то, что нужно
        for (symbol, &current_weight) in &current_weights {
            let target_weight = target_weights.get(symbol).unwrap_or(&0.0);

            if current_weight > *target_weight + self.config.rebalance_threshold {
                let weight_diff = current_weight - target_weight;
                let value_to_sell = weight_diff * total_value;

                if let Some(&price) = prices.get(symbol) {
                    let quantity_to_sell = value_to_sell / price;
                    let effective_price = price * (1.0 - self.config.slippage);
                    let proceeds = quantity_to_sell * effective_price;
                    let commission = proceeds * self.config.commission;

                    new_cash += proceeds - commission;
                    *new_positions.entry(symbol.clone()).or_insert(0.0) -= quantity_to_sell;

                    trades.push(Trade {
                        timestamp,
                        symbol: symbol.clone(),
                        is_buy: false,
                        quantity: quantity_to_sell,
                        price: effective_price,
                        commission,
                        total_cost: proceeds,
                    });
                }
            }
        }

        // Удаляем нулевые позиции
        new_positions.retain(|_, &mut q| q > 0.0001);

        // Затем покупаем то, что нужно
        for (symbol, &target_weight) in target_weights {
            let current_weight = current_weights.get(symbol).unwrap_or(&0.0);

            if target_weight > *current_weight + self.config.rebalance_threshold {
                let weight_diff = target_weight - current_weight;
                let value_to_buy = weight_diff * total_value;

                if let Some(&price) = prices.get(symbol) {
                    if value_to_buy > 0.0 && new_cash >= value_to_buy {
                        let effective_price = price * (1.0 + self.config.slippage);
                        let commission = value_to_buy * self.config.commission;
                        let quantity_to_buy = (value_to_buy - commission) / effective_price;

                        new_cash -= value_to_buy;
                        *new_positions.entry(symbol.clone()).or_insert(0.0) += quantity_to_buy;

                        trades.push(Trade {
                            timestamp,
                            symbol: symbol.clone(),
                            is_buy: true,
                            quantity: quantity_to_buy,
                            price: effective_price,
                            commission,
                            total_cost: value_to_buy,
                        });
                    }
                }
            }
        }

        (new_cash, new_positions, trades)
    }

    /// Рассчитать метрики
    fn calculate_metrics(
        &self,
        portfolio_history: Vec<PortfolioSnapshot>,
        trades: Vec<Trade>,
        total_commission: f64,
    ) -> BacktestResult {
        if portfolio_history.is_empty() {
            return BacktestResult {
                initial_capital: self.config.initial_capital,
                final_capital: self.config.initial_capital,
                total_return: 0.0,
                cagr: 0.0,
                volatility: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                calmar_ratio: 0.0,
                num_trades: 0,
                total_commission: 0.0,
                turnover: 0.0,
                portfolio_history: Vec::new(),
                trades: Vec::new(),
            };
        }

        let values: Vec<f64> = portfolio_history.iter().map(|s| s.value).collect();
        let initial_capital = values[0];
        let final_capital = *values.last().unwrap();
        let total_return = (final_capital - initial_capital) / initial_capital;

        // Доходности
        let returns: Vec<f64> = values.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

        // Волатильность (аннуализированная)
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let daily_vol = variance.sqrt();
        let volatility = daily_vol * (365.0_f64).sqrt();

        // CAGR
        let first_date = portfolio_history.first().unwrap().timestamp;
        let last_date = portfolio_history.last().unwrap().timestamp;
        let years = (last_date - first_date).num_days() as f64 / 365.0;
        let cagr = if years > 0.0 {
            (final_capital / initial_capital).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Sharpe Ratio (assuming 0% risk-free rate for crypto)
        let sharpe_ratio = if volatility > 0.0 {
            cagr / volatility
        } else {
            0.0
        };

        // Sortino Ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_deviation = if !downside_returns.is_empty() {
            let sum_sq: f64 = downside_returns.iter().map(|r| r.powi(2)).sum();
            (sum_sq / downside_returns.len() as f64).sqrt() * (365.0_f64).sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_deviation > 0.0 {
            cagr / downside_deviation
        } else {
            0.0
        };

        // Maximum Drawdown
        let mut max_value = values[0];
        let mut max_drawdown = 0.0;
        for &value in &values {
            max_value = max_value.max(value);
            let drawdown = (max_value - value) / max_value;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Calmar Ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            cagr / max_drawdown
        } else {
            0.0
        };

        // Turnover
        let total_traded: f64 = trades.iter().map(|t| t.total_cost).sum();
        let avg_portfolio_value = values.iter().sum::<f64>() / values.len() as f64;
        let turnover = total_traded / avg_portfolio_value;

        BacktestResult {
            initial_capital,
            final_capital,
            total_return,
            cagr,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            num_trades: trades.len(),
            total_commission,
            turnover,
            portfolio_history,
            trades,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Candle;

    fn create_test_data() -> HashMap<String, PriceSeries> {
        let mut data = HashMap::new();

        let mut btc = PriceSeries::new("BTCUSDT".to_string(), "D".to_string());
        let mut eth = PriceSeries::new("ETHUSDT".to_string(), "D".to_string());

        let base_date = Utc::now() - Duration::days(100);

        for i in 0..100 {
            let date = base_date + Duration::days(i);

            // BTC: восходящий тренд
            let btc_price = 30000.0 + (i as f64) * 100.0 + ((i as f64 * 0.5).sin() * 500.0);
            btc.push(Candle::new(date, btc_price, btc_price * 1.01, btc_price * 0.99, btc_price, 1000.0));

            // ETH: более волатильный восходящий тренд
            let eth_price = 2000.0 + (i as f64) * 10.0 + ((i as f64 * 0.7).sin() * 100.0);
            eth.push(Candle::new(date, eth_price, eth_price * 1.02, eth_price * 0.98, eth_price, 500.0));
        }

        data.insert("BTCUSDT".to_string(), btc);
        data.insert("ETHUSDT".to_string(), eth);

        data
    }

    #[test]
    fn test_backtest_engine() {
        let data = create_test_data();

        let config = BacktestConfig::default();
        let momentum_config = DualMomentumConfig {
            ts_lookback: 14,
            cs_lookback: 14,
            top_n: 2,
            risk_free_rate: 0.0,
            skip_period: 0,
            equal_weight: true,
        };
        let weight_config = WeightConfig::default();

        let engine = BacktestEngine::new(config, momentum_config, weight_config);

        let start = Utc::now() - Duration::days(80);
        let end = Utc::now();

        let result = engine.run(&data, start, end).unwrap();

        // Проверяем базовые свойства результата
        assert!(result.final_capital > 0.0);
        assert!(!result.portfolio_history.is_empty());
    }
}
