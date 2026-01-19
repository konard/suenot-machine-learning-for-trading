//! Бэктестинг торговой стратегии

use crate::api::Kline;
use crate::strategy::signals::{SignalGenerator, TradingSignal};

/// Результат бэктеста
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Общая доходность
    pub total_return: f64,
    /// Годовая доходность
    pub annualized_return: f64,
    /// Коэффициент Шарпа
    pub sharpe_ratio: f64,
    /// Коэффициент Сортино
    pub sortino_ratio: f64,
    /// Максимальная просадка
    pub max_drawdown: f64,
    /// Процент прибыльных сделок
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Количество сделок
    pub num_trades: usize,
    /// Средняя длительность сделки (в барах)
    pub avg_trade_duration: f64,
    /// Ежедневные returns
    pub daily_returns: Vec<f64>,
    /// Кривая equity
    pub equity_curve: Vec<f64>,
}

impl BacktestResult {
    /// Выводит сводку результатов
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results:\n\
             Total Return: {:.2}%\n\
             Annualized Return: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             Sortino Ratio: {:.2}\n\
             Max Drawdown: {:.2}%\n\
             Win Rate: {:.2}%\n\
             Profit Factor: {:.2}\n\
             Number of Trades: {}\n\
             Avg Trade Duration: {:.1} bars",
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.num_trades,
            self.avg_trade_duration,
        )
    }

    /// Проверяет, является ли стратегия прибыльной
    pub fn is_profitable(&self) -> bool {
        self.total_return > 0.0
    }
}

/// Торговая стратегия для бэктеста
pub struct TradingStrategy {
    /// Генератор сигналов
    signal_generator: SignalGenerator,
    /// Начальный капитал
    initial_capital: f64,
    /// Комиссия за сделку
    commission: f64,
    /// Slippage
    slippage: f64,
    /// Использовать стоп-лосс
    use_stop_loss: bool,
    /// Уровень стоп-лосса
    stop_loss_pct: f64,
    /// Использовать тейк-профит
    use_take_profit: bool,
    /// Уровень тейк-профита
    take_profit_pct: f64,
}

impl TradingStrategy {
    /// Создаёт стратегию с параметрами по умолчанию
    pub fn new(signal_generator: SignalGenerator) -> Self {
        Self {
            signal_generator,
            initial_capital: 10000.0,
            commission: 0.001, // 0.1%
            slippage: 0.0005,  // 0.05%
            use_stop_loss: false,
            stop_loss_pct: 0.02,
            use_take_profit: false,
            take_profit_pct: 0.03,
        }
    }

    /// Устанавливает начальный капитал
    pub fn with_capital(mut self, capital: f64) -> Self {
        self.initial_capital = capital;
        self
    }

    /// Устанавливает комиссию
    pub fn with_commission(mut self, commission: f64) -> Self {
        self.commission = commission;
        self
    }

    /// Включает стоп-лосс
    pub fn with_stop_loss(mut self, pct: f64) -> Self {
        self.use_stop_loss = true;
        self.stop_loss_pct = pct;
        self
    }

    /// Включает тейк-профит
    pub fn with_take_profit(mut self, pct: f64) -> Self {
        self.use_take_profit = true;
        self.take_profit_pct = pct;
        self
    }

    /// Запускает бэктест
    ///
    /// # Arguments
    ///
    /// * `klines` - Исторические свечи
    /// * `signals` - Сигналы для каждого бара (должны соответствовать klines)
    /// * `bars_per_year` - Количество баров в году (для annualized metrics)
    pub fn backtest(
        &self,
        klines: &[Kline],
        signals: &[TradingSignal],
        bars_per_year: usize,
    ) -> BacktestResult {
        assert_eq!(klines.len(), signals.len(), "Klines and signals must have same length");

        let n = klines.len();
        if n < 2 {
            return self.empty_result();
        }

        let mut capital = self.initial_capital;
        let mut position = 0.0; // -1, 0, 1
        let mut entry_price = 0.0;
        let mut equity_curve = vec![capital];
        let mut daily_returns = Vec::new();
        let mut trades: Vec<Trade> = Vec::new();
        let mut current_trade: Option<TradeBuilder> = None;

        for i in 1..n {
            let prev_close = klines[i - 1].close;
            let close = klines[i].close;

            // Рассчитываем return если есть позиция
            let position_return = if position != 0.0 {
                position * (close - prev_close) / prev_close
            } else {
                0.0
            };

            // Проверяем стоп-лосс и тейк-профит
            let should_exit = if position != 0.0 {
                let unrealized_pnl = position * (close - entry_price) / entry_price;

                (self.use_stop_loss && unrealized_pnl <= -self.stop_loss_pct) ||
                (self.use_take_profit && unrealized_pnl >= self.take_profit_pct)
            } else {
                false
            };

            // Получаем новый сигнал
            let new_signal = signals[i];
            let new_position = new_signal.position();

            // Проверяем, нужно ли менять позицию
            let position_changed = should_exit || (new_position != position);

            if position_changed {
                // Закрываем текущую позицию
                if position != 0.0 {
                    let trade_cost = capital * self.commission + capital * self.slippage;
                    capital -= trade_cost;

                    if let Some(builder) = current_trade.take() {
                        trades.push(builder.close(i, close));
                    }
                }

                // Открываем новую позицию (если не просто выход)
                if !should_exit && new_position != 0.0 {
                    let trade_cost = capital * self.commission + capital * self.slippage;
                    capital -= trade_cost;

                    position = new_position;
                    entry_price = close;

                    current_trade = Some(TradeBuilder {
                        entry_bar: i,
                        entry_price: close,
                        direction: position,
                    });
                } else if should_exit {
                    position = 0.0;
                } else {
                    position = new_position;
                    if new_position != 0.0 {
                        entry_price = close;
                        current_trade = Some(TradeBuilder {
                            entry_bar: i,
                            entry_price: close,
                            direction: position,
                        });
                    }
                }
            }

            // Обновляем капитал
            capital *= 1.0 + position_return;
            equity_curve.push(capital);
            daily_returns.push(position_return);
        }

        // Force-close any open position at the end so trade stats are accurate
        if position != 0.0 {
            let last_idx = n - 1;
            let last_close = klines[last_idx].close;

            let trade_cost = capital * self.commission + capital * self.slippage;
            capital -= trade_cost;

            if let Some(builder) = current_trade.take() {
                trades.push(builder.close(last_idx, last_close));
            }

            if let Some(last_equity) = equity_curve.last_mut() {
                *last_equity = capital;
            }
            if equity_curve.len() >= 2 {
                let prev_equity = equity_curve[equity_curve.len() - 2];
                if let Some(last_ret) = daily_returns.last_mut() {
                    *last_ret = capital / prev_equity - 1.0;
                }
            }
        }

        // Рассчитываем метрики
        self.calculate_metrics(
            &equity_curve,
            &daily_returns,
            &trades,
            bars_per_year,
        )
    }

    /// Рассчитывает метрики бэктеста
    fn calculate_metrics(
        &self,
        equity_curve: &[f64],
        daily_returns: &[f64],
        trades: &[Trade],
        bars_per_year: usize,
    ) -> BacktestResult {
        let n = daily_returns.len();

        if n == 0 {
            return self.empty_result();
        }

        // Total return
        let total_return = (equity_curve.last().unwrap_or(&self.initial_capital)
            / self.initial_capital) - 1.0;

        // Annualized return
        let years = n as f64 / bars_per_year as f64;
        let annualized_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Mean and std of returns
        let mean_return = daily_returns.iter().sum::<f64>() / n as f64;
        let variance = daily_returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / n as f64;
        let std_return = variance.sqrt();

        // Sharpe ratio (annualized)
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (bars_per_year as f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = daily_returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();

        let downside_var = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std = downside_var.sqrt();

        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (bars_per_year as f64).sqrt()
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = self.initial_capital;
        let mut max_drawdown = 0.0;

        for &equity in equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Trade statistics
        let winning_trades: Vec<&Trade> = trades.iter()
            .filter(|t| t.pnl > 0.0)
            .collect();

        let losing_trades: Vec<&Trade> = trades.iter()
            .filter(|t| t.pnl <= 0.0)
            .collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_duration = if !trades.is_empty() {
            trades.iter().map(|t| t.duration as f64).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades: trades.len(),
            avg_trade_duration,
            daily_returns: daily_returns.to_vec(),
            equity_curve: equity_curve.to_vec(),
        }
    }

    /// Возвращает пустой результат
    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            num_trades: 0,
            avg_trade_duration: 0.0,
            daily_returns: Vec::new(),
            equity_curve: vec![self.initial_capital],
        }
    }
}

impl Default for TradingStrategy {
    fn default() -> Self {
        Self::new(SignalGenerator::new())
    }
}

/// Информация о сделке
#[derive(Debug, Clone)]
struct Trade {
    entry_bar: usize,
    exit_bar: usize,
    entry_price: f64,
    exit_price: f64,
    direction: f64, // 1.0 for long, -1.0 for short
    pnl: f64,
    duration: usize,
}

/// Builder для создания сделки
struct TradeBuilder {
    entry_bar: usize,
    entry_price: f64,
    direction: f64,
}

impl TradeBuilder {
    fn close(self, exit_bar: usize, exit_price: f64) -> Trade {
        let pnl = self.direction * (exit_price - self.entry_price) / self.entry_price;

        Trade {
            entry_bar: self.entry_bar,
            exit_bar,
            entry_price: self.entry_price,
            exit_price,
            direction: self.direction,
            pnl,
            duration: exit_bar - self.entry_bar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize, trend: f64) -> Vec<Kline> {
        (0..n).map(|i| {
            let base = 100.0 + trend * i as f64 + (i as f64 * 0.1).sin() * 2.0;
            Kline {
                timestamp: i as u64 * 3600000,
                open: base,
                high: base + 1.0,
                low: base - 1.0,
                close: base + (i as f64 * 0.05).sin(),
                volume: 1000.0,
                turnover: 100000.0,
            }
        }).collect()
    }

    #[test]
    fn test_backtest_uptrend() {
        let klines = create_test_klines(100, 0.5); // Uptrend

        // Always long signals
        let signals = vec![TradingSignal::Long; 100];

        let strategy = TradingStrategy::default();
        let result = strategy.backtest(&klines, &signals, 8760);

        // In uptrend, long should be profitable
        assert!(result.total_return > 0.0, "Should be profitable in uptrend");
    }

    #[test]
    fn test_backtest_downtrend() {
        let klines = create_test_klines(100, -0.5); // Downtrend

        // Always short signals
        let signals = vec![TradingSignal::Short; 100];

        let strategy = TradingStrategy::default();
        let result = strategy.backtest(&klines, &signals, 8760);

        // In downtrend, short should be profitable
        assert!(result.total_return > 0.0, "Should be profitable in downtrend");
    }

    #[test]
    fn test_backtest_neutral() {
        let klines = create_test_klines(100, 0.0);

        // Always neutral signals
        let signals = vec![TradingSignal::Neutral; 100];

        let strategy = TradingStrategy::default();
        let result = strategy.backtest(&klines, &signals, 8760);

        // Neutral should have no trades
        assert_eq!(result.num_trades, 0);
        assert!((result.total_return).abs() < 0.01, "Should have near-zero return");
    }

    #[test]
    fn test_backtest_metrics() {
        let klines = create_test_klines(200, 0.3);
        let signals = vec![TradingSignal::Long; 200];

        let strategy = TradingStrategy::default()
            .with_capital(10000.0);

        let result = strategy.backtest(&klines, &signals, 8760);

        // Check metrics are reasonable
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
        assert_eq!(result.equity_curve.len(), 200);
    }

    #[test]
    fn test_backtest_with_stop_loss() {
        let klines = create_test_klines(100, -0.5); // Downtrend

        // Long signals in downtrend (bad trade)
        let signals = vec![TradingSignal::Long; 100];

        let strategy_no_sl = TradingStrategy::default();
        let strategy_with_sl = TradingStrategy::default()
            .with_stop_loss(0.02);

        let result_no_sl = strategy_no_sl.backtest(&klines, &signals, 8760);
        let result_with_sl = strategy_with_sl.backtest(&klines, &signals, 8760);

        // Stop loss should limit losses
        assert!(result_with_sl.max_drawdown <= result_no_sl.max_drawdown + 0.05);
    }

    #[test]
    fn test_result_summary() {
        let result = BacktestResult {
            total_return: 0.25,
            annualized_return: 0.15,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            max_drawdown: 0.1,
            win_rate: 0.6,
            profit_factor: 1.8,
            num_trades: 50,
            avg_trade_duration: 5.5,
            daily_returns: Vec::new(),
            equity_curve: Vec::new(),
        };

        let summary = result.summary();
        assert!(summary.contains("Total Return: 25.00%"));
        assert!(summary.contains("Sharpe Ratio: 1.50"));
    }
}
