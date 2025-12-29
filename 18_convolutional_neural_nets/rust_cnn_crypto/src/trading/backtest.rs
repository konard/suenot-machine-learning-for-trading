//! Бэктестинг торговой стратегии

use super::{Signal, Strategy, StrategyConfig, StrategyStatistics};
use crate::bybit::Kline;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Результаты бэктеста
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Статистика стратегии
    pub statistics: StrategyStatistics,
    /// История equity
    pub equity_curve: Vec<(i64, f64)>,
    /// История drawdown
    pub drawdown_curve: Vec<(i64, f64)>,
    /// Максимальный drawdown
    pub max_drawdown: f64,
    /// Коэффициент Шарпа (annualized)
    pub sharpe_ratio: f64,
    /// Коэффициент Сортино
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Количество дней бэктеста
    pub trading_days: usize,
}

impl std::fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.statistics)?;
        writeln!(f, "=== Risk Metrics ===")?;
        writeln!(f, "Max Drawdown: {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Sharpe Ratio: {:.2}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio: {:.2}", self.sortino_ratio)?;
        writeln!(f, "Calmar Ratio: {:.2}", self.calmar_ratio)?;
        writeln!(f, "Trading Days: {}", self.trading_days)?;
        Ok(())
    }
}

/// Бэктестер
pub struct Backtest {
    config: StrategyConfig,
    initial_capital: f64,
    symbol: String,
}

impl Backtest {
    /// Создание бэктестера
    pub fn new(symbol: &str, initial_capital: f64, config: StrategyConfig) -> Self {
        Self {
            config,
            initial_capital,
            symbol: symbol.to_string(),
        }
    }

    /// Запуск бэктеста на исторических данных
    pub fn run(&self, klines: &[Kline], signals: &[Signal]) -> BacktestResult {
        info!(
            "Starting backtest with {} klines and {} signals",
            klines.len(),
            signals.len()
        );

        let mut strategy = Strategy::new(self.config.clone(), self.initial_capital);
        let mut equity_curve = Vec::new();
        let mut returns = Vec::new();

        // Создаём индекс сигналов по времени
        let mut signal_map: std::collections::HashMap<i64, &Signal> =
            signals.iter().map(|s| (s.timestamp, s)).collect();

        let mut prev_equity = self.initial_capital;

        for kline in klines {
            // Обновляем позиции (проверка стоп-лоссов/тейк-профитов)
            strategy.update(kline.close, kline.timestamp);

            // Проверяем сигналы
            if let Some(signal) = signal_map.remove(&kline.timestamp) {
                let _position = strategy.process_signal(signal, &self.symbol);
            }

            // Записываем equity
            let current_equity = strategy.equity(kline.close);
            equity_curve.push((kline.timestamp, current_equity));

            // Вычисляем доходность
            if prev_equity > 0.0 {
                let ret = (current_equity - prev_equity) / prev_equity;
                returns.push(ret);
            }
            prev_equity = current_equity;
        }

        // Закрываем все позиции в конце
        if let Some(last_kline) = klines.last() {
            strategy.close_all(last_kline.close, last_kline.timestamp);
        }

        // Вычисляем drawdown
        let (drawdown_curve, max_drawdown) = self.calculate_drawdown(&equity_curve);

        // Вычисляем метрики
        let sharpe_ratio = self.calculate_sharpe_ratio(&returns);
        let sortino_ratio = self.calculate_sortino_ratio(&returns);
        let calmar_ratio = if max_drawdown > 0.0 {
            let annual_return =
                (strategy.capital() / self.initial_capital).powf(365.0 / klines.len() as f64)
                    - 1.0;
            annual_return / max_drawdown
        } else {
            0.0
        };

        let trading_days = klines.len() / 96; // Примерно 96 15-минутных свечей в день

        let statistics = strategy.statistics(self.initial_capital);

        info!("Backtest completed. Final capital: ${:.2}", strategy.capital());

        BacktestResult {
            statistics,
            equity_curve,
            drawdown_curve,
            max_drawdown,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            trading_days,
        }
    }

    /// Вычисление drawdown
    fn calculate_drawdown(&self, equity_curve: &[(i64, f64)]) -> (Vec<(i64, f64)>, f64) {
        let mut drawdown_curve = Vec::new();
        let mut max_equity = 0.0;
        let mut max_drawdown = 0.0;

        for (timestamp, equity) in equity_curve {
            if *equity > max_equity {
                max_equity = *equity;
            }

            let drawdown = if max_equity > 0.0 {
                (max_equity - equity) / max_equity
            } else {
                0.0
            };

            drawdown_curve.push((*timestamp, drawdown));

            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        (drawdown_curve, max_drawdown)
    }

    /// Вычисление коэффициента Шарпа
    fn calculate_sharpe_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        if std > 0.0 {
            // Annualized (assuming 15-min intervals, ~35000 periods per year)
            let annualized_return = mean * 35040.0;
            let annualized_std = std * (35040.0_f64).sqrt();
            annualized_return / annualized_std
        } else {
            0.0
        }
    }

    /// Вычисление коэффициента Сортино
    fn calculate_sortino_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside.is_empty() {
            return if mean > 0.0 { f64::INFINITY } else { 0.0 };
        }

        let downside_variance: f64 =
            downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std > 0.0 {
            let annualized_return = mean * 35040.0;
            let annualized_downside_std = downside_std * (35040.0_f64).sqrt();
            annualized_return / annualized_downside_std
        } else {
            0.0
        }
    }

    /// Экспорт результатов в CSV
    pub fn export_equity_curve(&self, result: &BacktestResult, path: &str) -> std::io::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        writer.write_record(&["timestamp", "equity", "drawdown"])?;

        for ((ts, equity), (_, dd)) in result
            .equity_curve
            .iter()
            .zip(result.drawdown_curve.iter())
        {
            writer.write_record(&[ts.to_string(), equity.to_string(), dd.to_string()])?;
        }

        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize, base_price: f64) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let price = base_price + (i as f64).sin() * 100.0;
                Kline {
                    timestamp: i as i64 * 900000, // 15 min intervals
                    open: price,
                    high: price + 50.0,
                    low: price - 50.0,
                    close: price + 20.0,
                    volume: 100.0,
                    turnover: 5000000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest() {
        let klines = create_test_klines(100, 50000.0);
        let signals: Vec<Signal> = klines
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 10 == 0)
            .map(|(_, k)| Signal::new(k.timestamp, [0.1, 0.2, 0.7], k.close, 0.5))
            .collect();

        let backtest = Backtest::new("BTCUSDT", 10000.0, StrategyConfig::default());
        let result = backtest.run(&klines, &signals);

        assert!(result.equity_curve.len() > 0);
    }

    #[test]
    fn test_drawdown_calculation() {
        let backtest = Backtest::new("BTCUSDT", 10000.0, StrategyConfig::default());
        let equity = vec![
            (0, 10000.0),
            (1, 11000.0),
            (2, 10500.0),
            (3, 12000.0),
            (4, 10800.0),
        ];

        let (dd, max_dd) = backtest.calculate_drawdown(&equity);

        assert!(max_dd > 0.0);
        assert!(max_dd < 1.0);
    }
}
