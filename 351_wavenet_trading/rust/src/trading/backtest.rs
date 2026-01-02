//! Бэктестинг торговых стратегий

use crate::types::{Candle, Signal, PerformanceMetrics};
use super::strategy::calculate_metrics;

/// Результат бэктеста
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub equity_curve: Vec<f64>,
    pub returns: Vec<f64>,
    pub positions: Vec<f64>,
    pub signals: Vec<Signal>,
    pub trades: Vec<Trade>,
    pub metrics: PerformanceMetrics,
}

/// Информация о сделке
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub entry_price: f64,
    pub exit_price: f64,
    pub position: f64,  // 1.0 = long, -1.0 = short
    pub pnl: f64,
    pub pnl_pct: f64,
}

/// Конфигурация бэктеста
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission: f64,
    pub slippage: f64,
    pub position_size: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission: 0.001,  // 0.1%
            slippage: 0.0005,   // 0.05%
            position_size: 1.0, // 100% капитала
        }
    }
}

/// Движок бэктестинга
pub struct Backtester {
    pub config: BacktestConfig,
}

impl Backtester {
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Запустить бэктест на основе сигналов
    pub fn run(&self, candles: &[Candle], signals: &[Signal]) -> BacktestResult {
        let n = candles.len().min(signals.len());
        if n == 0 {
            return BacktestResult {
                equity_curve: vec![self.config.initial_capital],
                returns: Vec::new(),
                positions: Vec::new(),
                signals: Vec::new(),
                trades: Vec::new(),
                metrics: PerformanceMetrics::default(),
            };
        }

        let mut equity = self.config.initial_capital;
        let mut equity_curve = vec![equity];
        let mut returns = Vec::new();
        let mut positions = Vec::new();
        let mut trades = Vec::new();

        let mut current_position = 0.0;
        let mut entry_price = 0.0;
        let mut entry_idx = 0;

        for i in 0..n - 1 {
            let signal = signals[i];
            let current_price = candles[i].close;
            let next_price = candles[i + 1].close;

            let target_position = match signal {
                Signal::Buy => 1.0,
                Signal::Sell => -1.0,
                Signal::Hold => current_position,
            };

            // Если позиция меняется
            if target_position != current_position {
                // Закрываем текущую позицию
                if current_position != 0.0 {
                    let exit_price = current_price * (1.0 - self.config.slippage * current_position.signum());
                    let pnl_pct = (exit_price / entry_price - 1.0) * current_position;
                    let commission_cost = self.config.commission * 2.0; // entry + exit
                    let net_pnl_pct = pnl_pct - commission_cost;
                    let pnl = equity * net_pnl_pct;

                    equity += pnl;

                    trades.push(Trade {
                        entry_idx,
                        exit_idx: i,
                        entry_price,
                        exit_price,
                        position: current_position,
                        pnl,
                        pnl_pct: net_pnl_pct,
                    });
                }

                // Открываем новую позицию
                if target_position != 0.0 {
                    entry_price = current_price * (1.0 + self.config.slippage * target_position.signum());
                    entry_idx = i;
                }

                current_position = target_position;
            }

            // Рассчитываем дневную доходность
            let daily_return = if current_position != 0.0 {
                let price_change = (next_price - current_price) / current_price;
                price_change * current_position * self.config.position_size
            } else {
                0.0
            };

            returns.push(daily_return);
            positions.push(current_position);
            equity_curve.push(equity * (1.0 + daily_return));
            equity = equity_curve.last().copied().unwrap();
        }

        // Закрываем финальную позицию
        if current_position != 0.0 {
            let exit_price = candles[n - 1].close;
            let pnl_pct = (exit_price / entry_price - 1.0) * current_position;
            let net_pnl_pct = pnl_pct - self.config.commission * 2.0;

            trades.push(Trade {
                entry_idx,
                exit_idx: n - 1,
                entry_price,
                exit_price,
                position: current_position,
                pnl: equity * net_pnl_pct,
                pnl_pct: net_pnl_pct,
            });
        }

        let metrics = calculate_metrics(&returns);

        BacktestResult {
            equity_curve,
            returns,
            positions,
            signals: signals[..n].to_vec(),
            trades,
            metrics,
        }
    }

    /// Запустить бэктест на основе предсказанных доходностей
    pub fn run_with_predictions(
        &self,
        candles: &[Candle],
        predictions: &[f64],
        threshold: f64,
    ) -> BacktestResult {
        let signals: Vec<Signal> = predictions
            .iter()
            .map(|&p| {
                if p > threshold {
                    Signal::Buy
                } else if p < -threshold {
                    Signal::Sell
                } else {
                    Signal::Hold
                }
            })
            .collect();

        self.run(candles, &signals)
    }
}

impl BacktestResult {
    /// Вывести сводку результатов
    pub fn print_summary(&self) {
        println!("=== Backtest Results ===");
        println!();

        // Equity
        if let (Some(&first), Some(&last)) = (self.equity_curve.first(), self.equity_curve.last()) {
            println!("Initial Capital: ${:.2}", first);
            println!("Final Capital:   ${:.2}", last);
            println!("Total P&L:       ${:.2}", last - first);
        }

        println!();
        self.metrics.print_summary();

        println!();
        println!("=== Trade Statistics ===");
        println!("Total Trades:    {}", self.trades.len());

        if !self.trades.is_empty() {
            let winning: Vec<_> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
            let losing: Vec<_> = self.trades.iter().filter(|t| t.pnl <= 0.0).collect();

            println!("Winning Trades:  {}", winning.len());
            println!("Losing Trades:   {}", losing.len());

            let avg_win = if !winning.is_empty() {
                winning.iter().map(|t| t.pnl_pct).sum::<f64>() / winning.len() as f64
            } else {
                0.0
            };

            let avg_loss = if !losing.is_empty() {
                losing.iter().map(|t| t.pnl_pct).sum::<f64>() / losing.len() as f64
            } else {
                0.0
            };

            println!("Avg Win:         {:.2}%", avg_win * 100.0);
            println!("Avg Loss:        {:.2}%", avg_loss * 100.0);

            // Лучшая и худшая сделки
            if let Some(best) = self.trades.iter().max_by(|a, b| a.pnl_pct.partial_cmp(&b.pnl_pct).unwrap()) {
                println!("Best Trade:      {:.2}%", best.pnl_pct * 100.0);
            }
            if let Some(worst) = self.trades.iter().min_by(|a, b| a.pnl_pct.partial_cmp(&b.pnl_pct).unwrap()) {
                println!("Worst Trade:     {:.2}%", worst.pnl_pct * 100.0);
            }
        }
    }

    /// Экспортировать результаты в CSV формат
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("index,equity,return,position\n");

        for (i, ((eq, ret), pos)) in self.equity_curve.iter()
            .zip(self.returns.iter().chain(std::iter::once(&0.0)))
            .zip(self.positions.iter().chain(std::iter::once(&0.0)))
            .enumerate()
        {
            csv.push_str(&format!("{},{:.4},{:.6},{:.1}\n", i, eq, ret, pos));
        }

        csv
    }
}

/// Простой бэктест на основе позиций и доходностей
pub fn simple_backtest(positions: &[f64], returns: &[f64]) -> Vec<f64> {
    positions
        .iter()
        .zip(returns.iter())
        .map(|(&pos, &ret)| pos * ret)
        .collect()
}

/// Walk-forward оптимизация
pub struct WalkForwardOptimizer {
    pub train_size: usize,
    pub test_size: usize,
}

impl WalkForwardOptimizer {
    pub fn new(train_size: usize, test_size: usize) -> Self {
        Self {
            train_size,
            test_size,
        }
    }

    /// Разбить данные на train/test окна
    pub fn split_data<T: Clone>(&self, data: &[T]) -> Vec<(Vec<T>, Vec<T>)> {
        let mut splits = Vec::new();
        let window_size = self.train_size + self.test_size;

        let mut start = 0;
        while start + window_size <= data.len() {
            let train = data[start..start + self.train_size].to_vec();
            let test = data[start + self.train_size..start + window_size].to_vec();
            splits.push((train, test));
            start += self.test_size;
        }

        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candles(prices: &[f64]) -> Vec<Candle> {
        prices
            .iter()
            .map(|&p| Candle {
                timestamp: Utc::now(),
                open: p,
                high: p * 1.01,
                low: p * 0.99,
                close: p,
                volume: 1000.0,
            })
            .collect()
    }

    #[test]
    fn test_backtester() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let candles = create_test_candles(&prices);

        let signals: Vec<Signal> = prices
            .windows(2)
            .map(|w| {
                if w[1] > w[0] {
                    Signal::Buy
                } else {
                    Signal::Sell
                }
            })
            .collect();

        let backtester = Backtester::new(BacktestConfig::default());
        let result = backtester.run(&candles, &signals);

        assert!(!result.equity_curve.is_empty());
        assert!(!result.returns.is_empty());
    }

    #[test]
    fn test_walk_forward() {
        let data: Vec<i32> = (0..100).collect();
        let optimizer = WalkForwardOptimizer::new(60, 20);
        let splits = optimizer.split_data(&data);

        assert!(!splits.is_empty());
        for (train, test) in &splits {
            assert_eq!(train.len(), 60);
            assert_eq!(test.len(), 20);
        }
    }
}
