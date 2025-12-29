//! Пример: Простая моментум-стратегия
//!
//! Демонстрирует:
//! - Построение торговой стратегии на основе индикаторов
//! - Генерация сигналов покупки/продажи
//! - Бэктестинг на исторических данных

use alpha_factors::{
    BybitClient,
    api::Interval,
    factors,
};
use alpha_factors::data::kline::KlineVec;

/// Торговый сигнал
#[derive(Debug, Clone, Copy, PartialEq)]
enum TradeSignal {
    Buy,
    Sell,
    Hold,
}

/// Позиция
#[derive(Debug, Clone)]
struct Position {
    entry_price: f64,
    entry_time: i64,
    size: f64,
}

/// Результат сделки
#[derive(Debug)]
struct Trade {
    entry_price: f64,
    exit_price: f64,
    pnl_percent: f64,
    holding_period: i64,
}

/// Моментум-стратегия
///
/// Правила:
/// - Покупаем когда RSI < 30 И цена ниже нижней полосы Боллинджера
/// - Продаём когда RSI > 70 ИЛИ цена выше верхней полосы Боллинджера
/// - Стоп-лосс: 2 * ATR
struct MomentumStrategy {
    rsi_period: usize,
    bb_period: usize,
    bb_std: f64,
    atr_period: usize,
    atr_multiplier: f64,
}

impl MomentumStrategy {
    fn new() -> Self {
        Self {
            rsi_period: 14,
            bb_period: 20,
            bb_std: 2.0,
            atr_period: 14,
            atr_multiplier: 2.0,
        }
    }

    fn generate_signals(
        &self,
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> Vec<TradeSignal> {
        let n = closes.len();

        // Рассчитываем индикаторы
        let rsi = factors::rsi(closes, self.rsi_period);
        let bb = factors::bollinger_bands(closes, self.bb_period, self.bb_std);
        let atr = factors::atr(highs, lows, closes, self.atr_period);

        let mut signals = vec![TradeSignal::Hold; n];

        for i in self.bb_period.max(self.rsi_period).max(self.atr_period)..n {
            let rsi_val = rsi[i];
            let price = closes[i];
            let bb_upper = bb.upper[i];
            let bb_lower = bb.lower[i];

            if rsi_val.is_nan() || bb_upper.is_nan() || bb_lower.is_nan() {
                continue;
            }

            // Условия покупки: RSI < 30 И цена ниже нижней полосы
            if rsi_val < 30.0 && price < bb_lower {
                signals[i] = TradeSignal::Buy;
            }
            // Условия продажи: RSI > 70 ИЛИ цена выше верхней полосы
            else if rsi_val > 70.0 || price > bb_upper {
                signals[i] = TradeSignal::Sell;
            }
        }

        signals
    }

    fn backtest(
        &self,
        timestamps: &[i64],
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> BacktestResult {
        let signals = self.generate_signals(highs, lows, closes);
        let atr = factors::atr(highs, lows, closes, self.atr_period);

        let mut trades: Vec<Trade> = Vec::new();
        let mut position: Option<Position> = None;
        let mut equity_curve = vec![10000.0]; // Начальный капитал $10,000

        for i in 1..closes.len() {
            let current_equity = *equity_curve.last().unwrap();

            match (&position, signals[i]) {
                // Открываем позицию
                (None, TradeSignal::Buy) => {
                    position = Some(Position {
                        entry_price: closes[i],
                        entry_time: timestamps[i],
                        size: current_equity / closes[i],
                    });
                }
                // Закрываем позицию
                (Some(pos), TradeSignal::Sell) => {
                    let pnl = (closes[i] - pos.entry_price) / pos.entry_price;
                    let new_equity = current_equity * (1.0 + pnl);

                    trades.push(Trade {
                        entry_price: pos.entry_price,
                        exit_price: closes[i],
                        pnl_percent: pnl * 100.0,
                        holding_period: timestamps[i] - pos.entry_time,
                    });

                    equity_curve.push(new_equity);
                    position = None;
                }
                // Проверяем стоп-лосс
                (Some(pos), _) => {
                    let stop_loss = pos.entry_price - self.atr_multiplier * atr[i];

                    if closes[i] < stop_loss {
                        let pnl = (closes[i] - pos.entry_price) / pos.entry_price;
                        let new_equity = current_equity * (1.0 + pnl);

                        trades.push(Trade {
                            entry_price: pos.entry_price,
                            exit_price: closes[i],
                            pnl_percent: pnl * 100.0,
                            holding_period: timestamps[i] - pos.entry_time,
                        });

                        equity_curve.push(new_equity);
                        position = None;
                    } else {
                        equity_curve.push(current_equity);
                    }
                }
                _ => {
                    equity_curve.push(current_equity);
                }
            }
        }

        // Закрываем открытую позицию в конце
        if let Some(pos) = position {
            let last_price = *closes.last().unwrap();
            let pnl = (last_price - pos.entry_price) / pos.entry_price;

            trades.push(Trade {
                entry_price: pos.entry_price,
                exit_price: last_price,
                pnl_percent: pnl * 100.0,
                holding_period: *timestamps.last().unwrap() - pos.entry_time,
            });
        }

        BacktestResult {
            trades,
            equity_curve,
            initial_capital: 10000.0,
        }
    }
}

/// Результат бэктестинга
struct BacktestResult {
    trades: Vec<Trade>,
    equity_curve: Vec<f64>,
    initial_capital: f64,
}

impl BacktestResult {
    fn total_return(&self) -> f64 {
        let final_equity = *self.equity_curve.last().unwrap_or(&self.initial_capital);
        ((final_equity - self.initial_capital) / self.initial_capital) * 100.0
    }

    fn num_trades(&self) -> usize {
        self.trades.len()
    }

    fn win_rate(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        let wins = self.trades.iter().filter(|t| t.pnl_percent > 0.0).count();
        (wins as f64 / self.trades.len() as f64) * 100.0
    }

    fn avg_profit(&self) -> f64 {
        let profits: Vec<f64> = self.trades.iter()
            .filter(|t| t.pnl_percent > 0.0)
            .map(|t| t.pnl_percent)
            .collect();

        if profits.is_empty() {
            return 0.0;
        }
        profits.iter().sum::<f64>() / profits.len() as f64
    }

    fn avg_loss(&self) -> f64 {
        let losses: Vec<f64> = self.trades.iter()
            .filter(|t| t.pnl_percent < 0.0)
            .map(|t| t.pnl_percent)
            .collect();

        if losses.is_empty() {
            return 0.0;
        }
        losses.iter().sum::<f64>() / losses.len() as f64
    }

    fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self.trades.iter()
            .filter(|t| t.pnl_percent > 0.0)
            .map(|t| t.pnl_percent)
            .sum();

        let gross_loss: f64 = self.trades.iter()
            .filter(|t| t.pnl_percent < 0.0)
            .map(|t| t.pnl_percent.abs())
            .sum();

        if gross_loss == 0.0 {
            return f64::INFINITY;
        }
        gross_profit / gross_loss
    }

    fn max_drawdown(&self) -> f64 {
        let mut max_equity = self.initial_capital;
        let mut max_dd = 0.0;

        for &equity in &self.equity_curve {
            if equity > max_equity {
                max_equity = equity;
            }
            let dd = (max_equity - equity) / max_equity * 100.0;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== Бэктестинг моментум-стратегии ===\n");

    // Получаем данные
    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    println!("Получаем исторические данные для {}...", symbol);

    // Получаем 500 часовых свечей (~20 дней)
    let klines = client
        .get_klines_with_interval(symbol, Interval::Hour1, 500)
        .await?;

    println!("Получено {} свечей\n", klines.len());

    // Извлекаем данные
    let timestamps = klines.timestamps();
    let highs = klines.highs();
    let lows = klines.lows();
    let closes = klines.closes();

    // Создаём и запускаем стратегию
    let strategy = MomentumStrategy::new();

    println!("Параметры стратегии:");
    println!("  RSI период: {}", strategy.rsi_period);
    println!("  Bollinger период: {}", strategy.bb_period);
    println!("  Bollinger std: {}", strategy.bb_std);
    println!("  ATR период: {}", strategy.atr_period);
    println!("  ATR множитель (стоп-лосс): {}", strategy.atr_multiplier);
    println!();

    let result = strategy.backtest(&timestamps, &highs, &lows, &closes);

    // Выводим результаты
    println!("=== Результаты бэктестинга ===\n");

    println!("Начальный капитал: ${:.2}", result.initial_capital);
    println!("Конечный капитал: ${:.2}", result.equity_curve.last().unwrap_or(&result.initial_capital));
    println!("Общая доходность: {:.2}%", result.total_return());
    println!();

    println!("Всего сделок: {}", result.num_trades());
    println!("Процент прибыльных: {:.1}%", result.win_rate());
    println!("Средняя прибыль: {:.2}%", result.avg_profit());
    println!("Средний убыток: {:.2}%", result.avg_loss());
    println!("Profit Factor: {:.2}", result.profit_factor());
    println!("Максимальная просадка: {:.2}%", result.max_drawdown());
    println!();

    // Выводим последние сделки
    if !result.trades.is_empty() {
        println!("=== Последние 5 сделок ===\n");

        for trade in result.trades.iter().rev().take(5) {
            let hours = trade.holding_period / 3_600_000;
            println!(
                "Вход: ${:.2} -> Выход: ${:.2} | P&L: {:+.2}% | Время: {}ч",
                trade.entry_price,
                trade.exit_price,
                trade.pnl_percent,
                hours
            );
        }
    }

    // Показываем текущие сигналы
    println!("\n=== Текущее состояние рынка ===\n");

    let rsi = factors::rsi(&closes, 14);
    let bb = factors::bollinger_bands(&closes, 20, 2.0);

    let current_price = *closes.last().unwrap();
    let current_rsi = *rsi.last().unwrap();
    let bb_upper = *bb.upper.last().unwrap();
    let bb_lower = *bb.lower.last().unwrap();

    println!("Цена: ${:.2}", current_price);
    println!("RSI: {:.1}", current_rsi);
    println!("BB верхняя: ${:.2}", bb_upper);
    println!("BB нижняя: ${:.2}", bb_lower);

    let signal = if current_rsi < 30.0 && current_price < bb_lower {
        "ПОКУПКА (RSI < 30 и цена ниже BB)"
    } else if current_rsi > 70.0 {
        "ПРОДАЖА (RSI > 70)"
    } else if current_price > bb_upper {
        "ПРОДАЖА (цена выше BB)"
    } else {
        "ОЖИДАНИЕ"
    };

    println!("\nТекущий сигнал: {}", signal);

    Ok(())
}
