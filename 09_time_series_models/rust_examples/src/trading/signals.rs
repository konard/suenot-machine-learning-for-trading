//! Торговые сигналы и позиции

/// Торговый сигнал
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Держать текущую позицию
    Hold,
    /// Купить спред (long asset1, short asset2)
    BuySpread,
    /// Продать спред (short asset1, long asset2)
    SellSpread,
    /// Выход из длинной позиции
    ExitLong,
    /// Выход из короткой позиции
    ExitShort,
    /// Стоп-лосс
    StopLoss,
}

impl Signal {
    pub fn as_str(&self) -> &'static str {
        match self {
            Signal::Hold => "HOLD",
            Signal::BuySpread => "BUY_SPREAD",
            Signal::SellSpread => "SELL_SPREAD",
            Signal::ExitLong => "EXIT_LONG",
            Signal::ExitShort => "EXIT_SHORT",
            Signal::StopLoss => "STOP_LOSS",
        }
    }

    pub fn is_entry(&self) -> bool {
        matches!(self, Signal::BuySpread | Signal::SellSpread)
    }

    pub fn is_exit(&self) -> bool {
        matches!(self, Signal::ExitLong | Signal::ExitShort | Signal::StopLoss)
    }
}

/// Текущая позиция
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    /// Нет позиции
    Flat,
    /// Длинная позиция по спреду (long asset1, short asset2)
    Long,
    /// Короткая позиция по спреду (short asset1, long asset2)
    Short,
}

impl Position {
    pub fn as_str(&self) -> &'static str {
        match self {
            Position::Flat => "FLAT",
            Position::Long => "LONG",
            Position::Short => "SHORT",
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, Position::Flat)
    }
}

/// Информация о сделке
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: usize,
    pub exit_time: usize,
    pub position: Position,
    pub entry_price1: f64,
    pub entry_price2: f64,
    pub exit_price1: f64,
    pub exit_price2: f64,
    pub hedge_ratio: f64,
    pub pnl: f64,
    pub exit_reason: Signal,
}

impl Trade {
    pub fn new(
        entry_time: usize,
        exit_time: usize,
        position: Position,
        entry_price1: f64,
        entry_price2: f64,
        exit_price1: f64,
        exit_price2: f64,
        hedge_ratio: f64,
        exit_reason: Signal,
    ) -> Self {
        let pnl = match position {
            Position::Long => {
                // Long asset1, short asset2
                (exit_price1 - entry_price1) - hedge_ratio * (exit_price2 - entry_price2)
            }
            Position::Short => {
                // Short asset1, long asset2
                (entry_price1 - exit_price1) - hedge_ratio * (entry_price2 - exit_price2)
            }
            Position::Flat => 0.0,
        };

        Self {
            entry_time,
            exit_time,
            position,
            entry_price1,
            entry_price2,
            exit_price1,
            exit_price2,
            hedge_ratio,
            pnl,
            exit_reason,
        }
    }

    pub fn holding_period(&self) -> usize {
        self.exit_time - self.entry_time
    }

    pub fn is_profitable(&self) -> bool {
        self.pnl > 0.0
    }

    pub fn display(&self) -> String {
        format!(
            "{:?} {} -> {}: PnL={:.4} ({:.2}%)",
            self.position,
            self.entry_time,
            self.exit_time,
            self.pnl,
            self.return_pct() * 100.0
        )
    }

    pub fn return_pct(&self) -> f64 {
        let entry_value = self.entry_price1 + self.hedge_ratio * self.entry_price2;
        if entry_value > 0.0 {
            self.pnl / entry_value
        } else {
            0.0
        }
    }
}

/// Извлечение сделок из сигналов
pub fn extract_trades(
    signals: &[Signal],
    prices1: &[f64],
    prices2: &[f64],
    hedge_ratio: f64,
) -> Vec<Trade> {
    let mut trades = Vec::new();
    let mut current_position = Position::Flat;
    let mut entry_time = 0;
    let mut entry_price1 = 0.0;
    let mut entry_price2 = 0.0;

    for (i, &signal) in signals.iter().enumerate() {
        match signal {
            Signal::BuySpread => {
                if current_position == Position::Flat {
                    current_position = Position::Long;
                    entry_time = i;
                    entry_price1 = prices1[i];
                    entry_price2 = prices2[i];
                }
            }
            Signal::SellSpread => {
                if current_position == Position::Flat {
                    current_position = Position::Short;
                    entry_time = i;
                    entry_price1 = prices1[i];
                    entry_price2 = prices2[i];
                }
            }
            Signal::ExitLong | Signal::ExitShort | Signal::StopLoss => {
                if current_position != Position::Flat {
                    trades.push(Trade::new(
                        entry_time,
                        i,
                        current_position,
                        entry_price1,
                        entry_price2,
                        prices1[i],
                        prices2[i],
                        hedge_ratio,
                        signal,
                    ));
                    current_position = Position::Flat;
                }
            }
            Signal::Hold => {}
        }
    }

    trades
}

/// Статистика по сделкам
#[derive(Debug, Clone)]
pub struct TradeStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_pnl: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub avg_holding_period: f64,
    pub max_drawdown: f64,
}

impl TradeStats {
    pub fn from_trades(trades: &[Trade]) -> Self {
        if trades.is_empty() {
            return Self {
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                win_rate: 0.0,
                total_pnl: 0.0,
                avg_pnl: 0.0,
                avg_win: 0.0,
                avg_loss: 0.0,
                profit_factor: 0.0,
                avg_holding_period: 0.0,
                max_drawdown: 0.0,
            };
        }

        let total_trades = trades.len();
        let winning: Vec<_> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing: Vec<_> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let winning_trades = winning.len();
        let losing_trades = losing.len();
        let win_rate = winning_trades as f64 / total_trades as f64;

        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let avg_pnl = total_pnl / total_trades as f64;

        let total_wins: f64 = winning.iter().map(|t| t.pnl).sum();
        let total_losses: f64 = losing.iter().map(|t| t.pnl.abs()).sum();

        let avg_win = if !winning.is_empty() {
            total_wins / winning.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing.is_empty() {
            total_losses / losing.len() as f64
        } else {
            0.0
        };

        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else {
            f64::INFINITY
        };

        let avg_holding_period = trades.iter().map(|t| t.holding_period() as f64).sum::<f64>()
            / total_trades as f64;

        // Max drawdown
        let mut equity = 0.0;
        let mut peak = 0.0;
        let mut max_drawdown = 0.0;

        for trade in trades {
            equity += trade.pnl;
            if equity > peak {
                peak = equity;
            }
            let drawdown = peak - equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        Self {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            total_pnl,
            avg_pnl,
            avg_win,
            avg_loss,
            profit_factor,
            avg_holding_period,
            max_drawdown,
        }
    }

    pub fn display(&self) -> String {
        format!(
            r#"Trade Statistics
================
Total Trades: {}
Winning: {} ({:.1}%)
Losing: {}
Total PnL: {:.4}
Avg PnL: {:.4}
Avg Win: {:.4}
Avg Loss: {:.4}
Profit Factor: {:.2}
Avg Holding Period: {:.1}
Max Drawdown: {:.4}"#,
            self.total_trades,
            self.winning_trades,
            self.win_rate * 100.0,
            self.losing_trades,
            self.total_pnl,
            self.avg_pnl,
            self.avg_win,
            self.avg_loss,
            self.profit_factor,
            self.avg_holding_period,
            self.max_drawdown
        )
    }
}
