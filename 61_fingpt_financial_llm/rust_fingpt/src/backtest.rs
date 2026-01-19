//! Backtesting module for FinGPT trading strategies
//!
//! This module provides backtesting capabilities to evaluate trading strategies
//! based on sentiment analysis.

use crate::data::OHLCVBar;
use crate::sentiment::{Sentiment, SentimentAnalyzer};
use crate::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size as percentage of portfolio
    pub position_size_pct: f64,
    /// Sentiment threshold for trading
    pub sentiment_threshold: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Maximum number of concurrent positions
    pub max_positions: usize,
    /// Commission per trade (as percentage)
    pub commission_pct: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size_pct: 0.1,
            sentiment_threshold: 0.3,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            max_positions: 3,
            commission_pct: 0.001,
        }
    }
}

/// A single trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Symbol traded
    pub symbol: String,
    /// Trade side (buy/sell)
    pub side: String,
    /// Position size
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: DateTime<Utc>,
    /// Profit/Loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
}

/// Equity curve point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Portfolio equity value
    pub equity: f64,
}

/// Backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return percentage
    pub total_return: f64,
    /// Final portfolio value
    pub final_value: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Annualized volatility
    pub volatility: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate percentage
    pub win_rate: f64,
    /// List of all trades
    pub trades: Vec<Trade>,
    /// Equity curve
    pub equity_curve: Vec<EquityPoint>,
}

/// Sentiment event for backtesting
#[derive(Debug, Clone)]
pub struct SentimentEvent {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Sentiment value
    pub sentiment: Sentiment,
    /// Sentiment score (-1 to 1)
    pub score: f64,
    /// Confidence
    pub confidence: f64,
}

/// Position tracking during backtest
#[derive(Debug, Clone)]
struct Position {
    entry_price: f64,
    size: f64,
    entry_time: DateTime<Utc>,
    stop_loss: f64,
    take_profit: f64,
}

/// Backtester for FinGPT strategies
pub struct Backtester {
    #[allow(dead_code)]
    analyzer: SentimentAnalyzer,
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(analyzer: SentimentAnalyzer, config: BacktestConfig) -> Self {
        Self { analyzer, config }
    }

    /// Run backtest on historical data
    pub fn run(
        &self,
        price_data: &[OHLCVBar],
        sentiment_data: &[SentimentEvent],
    ) -> Result<BacktestResult> {
        let mut cash = self.config.initial_capital;
        let mut position: Option<Position> = None;
        let mut trades = Vec::new();
        let mut equity_curve = Vec::new();
        let mut daily_returns = Vec::new();
        let mut peak_equity = self.config.initial_capital;
        let mut max_drawdown: f64 = 0.0;

        // Create sentiment lookup by date
        let mut sentiment_by_date = std::collections::HashMap::new();
        for event in sentiment_data {
            let date = event.timestamp.date_naive();
            sentiment_by_date.insert(date, event.clone());
        }

        for (i, bar) in price_data.iter().enumerate() {
            let date = bar.timestamp.date_naive();

            // Check stop loss / take profit
            if let Some(ref pos) = position {
                let current_pnl = (bar.close - pos.entry_price) / pos.entry_price;

                if bar.low <= pos.stop_loss || bar.high >= pos.take_profit {
                    // Close position
                    let exit_price = if bar.low <= pos.stop_loss {
                        pos.stop_loss
                    } else {
                        pos.take_profit
                    };

                    let pnl = (exit_price - pos.entry_price) * pos.size;
                    let return_pct = (exit_price - pos.entry_price) / pos.entry_price;

                    cash += pos.size * exit_price * (1.0 - self.config.commission_pct);

                    trades.push(Trade {
                        symbol: "ASSET".to_string(),
                        side: "buy".to_string(),
                        size: pos.size,
                        entry_price: pos.entry_price,
                        exit_price,
                        entry_time: pos.entry_time,
                        exit_time: bar.timestamp,
                        pnl,
                        return_pct,
                    });

                    position = None;
                } else if current_pnl < -self.config.stop_loss_pct {
                    // Emergency stop
                    let exit_price = bar.close;
                    let pnl = (exit_price - pos.entry_price) * pos.size;
                    let return_pct = (exit_price - pos.entry_price) / pos.entry_price;

                    cash += pos.size * exit_price * (1.0 - self.config.commission_pct);

                    trades.push(Trade {
                        symbol: "ASSET".to_string(),
                        side: "buy".to_string(),
                        size: pos.size,
                        entry_price: pos.entry_price,
                        exit_price,
                        entry_time: pos.entry_time,
                        exit_time: bar.timestamp,
                        pnl,
                        return_pct,
                    });

                    position = None;
                }
            }

            // Check for entry signal
            if position.is_none() {
                if let Some(sentiment) = sentiment_by_date.get(&date) {
                    if sentiment.score > self.config.sentiment_threshold
                        && sentiment.confidence > 0.6
                    {
                        // Enter long position
                        let position_value = cash * self.config.position_size_pct;
                        let size = position_value / bar.close;
                        let cost = position_value * (1.0 + self.config.commission_pct);

                        if cost <= cash {
                            cash -= cost;
                            position = Some(Position {
                                entry_price: bar.close,
                                size,
                                entry_time: bar.timestamp,
                                stop_loss: bar.close * (1.0 - self.config.stop_loss_pct),
                                take_profit: bar.close * (1.0 + self.config.take_profit_pct),
                            });
                        }
                    }
                }
            }

            // Calculate equity
            let equity = cash + position.as_ref().map_or(0.0, |p| p.size * bar.close);

            equity_curve.push(EquityPoint {
                timestamp: bar.timestamp,
                equity,
            });

            // Track drawdown
            peak_equity = peak_equity.max(equity);
            let drawdown = (peak_equity - equity) / peak_equity;
            max_drawdown = max_drawdown.max(drawdown);

            // Calculate daily return
            if i > 0 {
                let prev_equity = equity_curve[i - 1].equity;
                let daily_return = (equity - prev_equity) / prev_equity;
                daily_returns.push(daily_return);
            }
        }

        // Close any remaining position
        if let Some(ref pos) = position {
            if let Some(last_bar) = price_data.last() {
                let exit_price = last_bar.close;
                let pnl = (exit_price - pos.entry_price) * pos.size;
                let return_pct = (exit_price - pos.entry_price) / pos.entry_price;

                cash += pos.size * exit_price * (1.0 - self.config.commission_pct);

                trades.push(Trade {
                    symbol: "ASSET".to_string(),
                    side: "buy".to_string(),
                    size: pos.size,
                    entry_price: pos.entry_price,
                    exit_price,
                    entry_time: pos.entry_time,
                    exit_time: last_bar.timestamp,
                    pnl,
                    return_pct,
                });
            }
        }

        // Calculate statistics
        let final_value = equity_curve.last().map_or(self.config.initial_capital, |e| e.equity);
        let total_return = (final_value - self.config.initial_capital) / self.config.initial_capital;

        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = trades.iter().filter(|t| t.pnl <= 0.0).count();
        let win_rate = if trades.is_empty() {
            0.0
        } else {
            winning_trades as f64 / trades.len() as f64
        };

        // Calculate Sharpe and Sortino ratios
        let (sharpe_ratio, sortino_ratio, volatility) = if daily_returns.len() > 1 {
            let mean_return: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let variance: f64 = daily_returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (daily_returns.len() - 1) as f64;
            let std_dev = variance.sqrt();

            let downside_returns: Vec<f64> = daily_returns
                .iter()
                .filter(|r| **r < 0.0)
                .copied()
                .collect();
            let downside_variance = if downside_returns.len() > 1 {
                downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                    / downside_returns.len() as f64
            } else {
                variance
            };
            let downside_dev = downside_variance.sqrt();

            let annualized_return = mean_return * 252.0;
            let annualized_vol = std_dev * (252.0_f64).sqrt();
            let annualized_downside = downside_dev * (252.0_f64).sqrt();

            let sharpe = if annualized_vol > 0.0 {
                annualized_return / annualized_vol
            } else {
                0.0
            };

            let sortino = if annualized_downside > 0.0 {
                annualized_return / annualized_downside
            } else {
                0.0
            };

            (sharpe, sortino, annualized_vol)
        } else {
            (0.0, 0.0, 0.0)
        };

        Ok(BacktestResult {
            total_return,
            final_value,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            volatility,
            total_trades: trades.len(),
            winning_trades,
            losing_trades,
            win_rate,
            trades,
            equity_curve,
        })
    }
}

/// Generate mock sentiment events for backtesting
pub fn generate_mock_sentiment(days: usize) -> Vec<SentimentEvent> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut events = Vec::new();
    let now = Utc::now();

    for day in 0..days {
        // Generate 0-2 sentiment events per day
        let num_events = rng.gen_range(0..3);

        for _ in 0..num_events {
            let timestamp = now
                - chrono::Duration::days((days - day - 1) as i64)
                + chrono::Duration::hours(rng.gen_range(8..16));

            let score: f64 = rng.gen_range(-1.0..1.0);
            let sentiment = if score > 0.3 {
                Sentiment::Positive
            } else if score < -0.3 {
                Sentiment::Negative
            } else {
                Sentiment::Neutral
            };

            let confidence = rng.gen_range(0.5..0.95);

            events.push(SentimentEvent {
                timestamp,
                sentiment,
                score,
                confidence,
            });
        }
    }

    events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataSource, MarketDataLoader};

    #[tokio::test]
    async fn test_backtest_run() {
        let analyzer = SentimentAnalyzer::new_mock();
        let config = BacktestConfig::default();
        let backtester = Backtester::new(analyzer, config);

        let loader = MarketDataLoader::new_mock();
        let price_data = loader.load("TEST", DataSource::Stock, 60).await.unwrap();
        let sentiment_data = generate_mock_sentiment(60);

        let result = backtester.run(&price_data, &sentiment_data).unwrap();

        assert!(result.final_value > 0.0);
        assert!(result.equity_curve.len() == 60);
    }
}
