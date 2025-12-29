//! Backtesting
//!
//! Бэктестинг торговых стратегий на исторических данных.

use super::{Position, Signal, SignalGenerator, TradingStrategy};
use crate::api::Kline;
use crate::model::QuantilePrediction;
use serde::{Deserialize, Serialize};

/// Конфигурация бэктеста
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Начальный капитал
    pub initial_capital: f64,

    /// Комиссия за сделку (процент)
    pub commission: f64,

    /// Slippage (процент)
    pub slippage: f64,

    /// Использовать ли маржу
    pub use_margin: bool,

    /// Максимальное плечо
    pub max_leverage: f64,

    /// Risk-free rate для Sharpe (годовой)
    pub risk_free_rate: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission: 0.001, // 0.1%
            slippage: 0.0005,  // 0.05%
            use_margin: false,
            max_leverage: 1.0,
            risk_free_rate: 0.02, // 2% годовых
        }
    }
}

/// Результат бэктеста
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Общая доходность (процент)
    pub total_return: f64,

    /// Годовая доходность
    pub annual_return: f64,

    /// Sharpe Ratio
    pub sharpe_ratio: f64,

    /// Sortino Ratio
    pub sortino_ratio: f64,

    /// Maximum Drawdown (процент)
    pub max_drawdown: f64,

    /// Calmar Ratio (annual return / max drawdown)
    pub calmar_ratio: f64,

    /// Количество сделок
    pub num_trades: usize,

    /// Win rate (процент выигрышных сделок)
    pub win_rate: f64,

    /// Profit factor (сумма прибылей / сумма убытков)
    pub profit_factor: f64,

    /// Средняя прибыль на сделку
    pub avg_trade_return: f64,

    /// Финальный капитал
    pub final_capital: f64,

    /// Equity curve (история капитала)
    pub equity_curve: Vec<f64>,

    /// Возвраты по периодам
    pub returns: Vec<f64>,
}

impl BacktestResult {
    /// Выводит результаты в читаемом формате
    pub fn print_summary(&self) {
        println!("\n=== Backtest Results ===");
        println!("Total Return: {:.2}%", self.total_return * 100.0);
        println!("Annual Return: {:.2}%", self.annual_return * 100.0);
        println!("Sharpe Ratio: {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio: {:.2}", self.sortino_ratio);
        println!("Max Drawdown: {:.2}%", self.max_drawdown * 100.0);
        println!("Calmar Ratio: {:.2}", self.calmar_ratio);
        println!("Number of Trades: {}", self.num_trades);
        println!("Win Rate: {:.2}%", self.win_rate * 100.0);
        println!("Profit Factor: {:.2}", self.profit_factor);
        println!("Avg Trade Return: {:.4}%", self.avg_trade_return * 100.0);
        println!(
            "Final Capital: ${:.2} (from ${:.2})",
            self.final_capital,
            self.equity_curve.first().unwrap_or(&0.0)
        );
    }
}

/// Бэктестер
pub struct Backtester {
    /// Конфигурация
    config: BacktestConfig,

    /// Торговая стратегия
    strategy: TradingStrategy,
}

impl Backtester {
    /// Создает новый бэктестер
    pub fn new(config: BacktestConfig, signal_generator: SignalGenerator) -> Self {
        Self {
            config,
            strategy: TradingStrategy::new(signal_generator),
        }
    }

    /// Запускает бэктест
    pub fn run(
        &mut self,
        predictions: &[QuantilePrediction],
        prices: &[f64],
        timestamps: &[i64],
    ) -> BacktestResult {
        assert_eq!(predictions.len(), prices.len());
        assert_eq!(predictions.len(), timestamps.len());

        let n = predictions.len();
        if n < 2 {
            return self.empty_result();
        }

        let mut capital = self.config.initial_capital;
        let mut equity_curve = vec![capital];
        let mut returns = Vec::new();
        let mut trade_returns = Vec::new();

        let mut current_position: Option<(Signal, f64, f64)> = None; // (signal, size, entry_price)

        for i in 0..n {
            let position = self.strategy.update(&predictions[i], timestamps[i], prices[i]);

            // Вычисляем P&L для текущей позиции
            if let Some((prev_signal, prev_size, entry_price)) = current_position {
                if i > 0 {
                    let price_change = (prices[i] - prices[i - 1]) / prices[i - 1];
                    let position_return = match prev_signal {
                        Signal::Long => price_change * prev_size,
                        Signal::Short => -price_change * prev_size,
                        _ => 0.0,
                    };

                    let net_return = position_return - self.config.commission * prev_size.abs();
                    capital *= 1.0 + net_return;
                    returns.push(net_return);
                }

                // Если позиция закрылась, записываем результат сделки
                if position.signal != prev_signal && prev_signal != Signal::Hold {
                    let exit_price = prices[i] * (1.0 - self.config.slippage);
                    let trade_return = match prev_signal {
                        Signal::Long => (exit_price - entry_price) / entry_price,
                        Signal::Short => (entry_price - exit_price) / entry_price,
                        _ => 0.0,
                    };
                    trade_returns.push(trade_return);
                }
            } else {
                returns.push(0.0);
            }

            // Обновляем текущую позицию
            if position.signal == Signal::Long || position.signal == Signal::Short {
                let entry_price = if current_position.map(|p| p.0) != Some(position.signal) {
                    prices[i] * (1.0 + self.config.slippage)
                } else {
                    current_position.map(|p| p.2).unwrap_or(prices[i])
                };
                current_position = Some((position.signal, position.size, entry_price));
            } else {
                current_position = None;
            }

            equity_curve.push(capital);
        }

        self.calculate_metrics(equity_curve, returns, trade_returns)
    }

    /// Вычисляет метрики
    fn calculate_metrics(
        &self,
        equity_curve: Vec<f64>,
        returns: Vec<f64>,
        trade_returns: Vec<f64>,
    ) -> BacktestResult {
        let initial = self.config.initial_capital;
        let final_capital = *equity_curve.last().unwrap_or(&initial);

        // Total return
        let total_return = (final_capital - initial) / initial;

        // Предполагаем часовые данные, 8760 часов в году
        let periods_per_year = 8760.0;
        let n_periods = returns.len() as f64;
        let annual_factor = periods_per_year / n_periods.max(1.0);

        // Annual return
        let annual_return = (1.0 + total_return).powf(annual_factor) - 1.0;

        // Sharpe Ratio
        let avg_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let std_return = Self::std(&returns);
        let excess_return = avg_return - self.config.risk_free_rate / periods_per_year;
        let sharpe_ratio = if std_return > 1e-10 {
            excess_return / std_return * (periods_per_year).sqrt()
        } else {
            0.0
        };

        // Sortino Ratio (только downside deviation)
        let downside_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_std = Self::std(&downside_returns);
        let sortino_ratio = if downside_std > 1e-10 {
            excess_return / downside_std * (periods_per_year).sqrt()
        } else {
            sharpe_ratio
        };

        // Max Drawdown
        let max_drawdown = Self::calculate_max_drawdown(&equity_curve);

        // Calmar Ratio
        let calmar_ratio = if max_drawdown > 1e-10 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let num_trades = trade_returns.len();
        let winning_trades = trade_returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = if num_trades > 0 {
            winning_trades as f64 / num_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = trade_returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = trade_returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
        let profit_factor = if gross_loss > 1e-10 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = if num_trades > 0 {
            trade_returns.iter().sum::<f64>() / num_trades as f64
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annual_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            num_trades,
            win_rate,
            profit_factor,
            avg_trade_return,
            final_capital,
            equity_curve,
            returns,
        }
    }

    /// Стандартное отклонение
    fn std(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Максимальная просадка
    fn calculate_max_drawdown(equity: &[f64]) -> f64 {
        let mut max_dd = 0.0;
        let mut peak = equity[0];

        for &value in equity {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    /// Пустой результат
    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            total_return: 0.0,
            annual_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            num_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_trade_return: 0.0,
            final_capital: self.config.initial_capital,
            equity_curve: vec![self.config.initial_capital],
            returns: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_predictions(returns: &[f64]) -> Vec<QuantilePrediction> {
        returns
            .iter()
            .map(|&r| {
                let values = Array2::from_shape_vec(
                    (1, 3),
                    vec![r - 0.01, r, r + 0.01],
                )
                .unwrap();
                QuantilePrediction::new(vec![0.1, 0.5, 0.9], values)
            })
            .collect()
    }

    #[test]
    fn test_backtester() {
        let config = BacktestConfig::default();
        let signal_gen = SignalGenerator::new();
        let mut backtester = Backtester::new(config, signal_gen);

        // Симулируем 10 периодов
        let predictions = make_predictions(&[0.02, 0.01, -0.01, 0.02, 0.01, -0.02, 0.01, 0.02, -0.01, 0.01]);
        let prices: Vec<f64> = (0..10).map(|i| 100.0 * (1.0 + 0.01 * i as f64)).collect();
        let timestamps: Vec<i64> = (0..10).map(|i| i as i64 * 3600000).collect();

        let result = backtester.run(&predictions, &prices, &timestamps);

        assert!(result.equity_curve.len() > 0);
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 95.0, 100.0, 90.0, 95.0];
        let max_dd = Backtester::calculate_max_drawdown(&equity);

        // Max DD: from 110 to 90 = (110-90)/110 ≈ 0.182
        assert!((max_dd - 0.182).abs() < 0.01);
    }

    #[test]
    fn test_std() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = Backtester::std(&values);

        // std of [1,2,3,4,5] ≈ 1.414
        assert!((std - 1.414).abs() < 0.01);
    }
}
