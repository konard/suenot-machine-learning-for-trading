//! Стратегия парной торговли

use crate::trading::cointegration::{compute_spread, spread_zscore, engle_granger_test};
use crate::trading::signals::{Signal, Position};
use crate::analysis::statistics::mean;

/// Параметры стратегии парной торговли
#[derive(Debug, Clone)]
pub struct PairsTradingParams {
    /// Порог Z-score для входа в позицию
    pub entry_threshold: f64,
    /// Порог Z-score для выхода из позиции
    pub exit_threshold: f64,
    /// Порог Z-score для стоп-лосса
    pub stop_loss_threshold: f64,
    /// Период для расчёта скользящего Z-score
    pub lookback_period: usize,
    /// Минимальный период ожидания между сделками
    pub min_holding_period: usize,
    /// Пересчитывать hedge ratio
    pub dynamic_hedge: bool,
    /// Период пересчёта hedge ratio
    pub hedge_recalc_period: usize,
}

impl Default for PairsTradingParams {
    fn default() -> Self {
        Self {
            entry_threshold: 2.0,
            exit_threshold: 0.5,
            stop_loss_threshold: 4.0,
            lookback_period: 20,
            min_holding_period: 1,
            dynamic_hedge: true,
            hedge_recalc_period: 60,
        }
    }
}

/// Стратегия парной торговли
#[derive(Debug)]
pub struct PairsTradingStrategy {
    pub params: PairsTradingParams,
    pub asset1: String,
    pub asset2: String,
    pub hedge_ratio: f64,
    current_position: Position,
    entry_index: Option<usize>,
    entry_zscore: f64,
}

impl PairsTradingStrategy {
    /// Создать стратегию с заданными параметрами
    pub fn new(
        asset1: &str,
        asset2: &str,
        hedge_ratio: f64,
        params: PairsTradingParams,
    ) -> Self {
        Self {
            params,
            asset1: asset1.to_string(),
            asset2: asset2.to_string(),
            hedge_ratio,
            current_position: Position::Flat,
            entry_index: None,
            entry_zscore: 0.0,
        }
    }

    /// Генерация торговых сигналов
    pub fn generate_signals(
        &mut self,
        prices1: &[f64],
        prices2: &[f64],
    ) -> Vec<Signal> {
        let n = prices1.len().min(prices2.len());
        if n < self.params.lookback_period + 10 {
            return vec![Signal::Hold; n];
        }

        let mut signals = vec![Signal::Hold; n];

        // Пересчитываем hedge ratio если нужно
        if self.params.dynamic_hedge {
            self.update_hedge_ratio(prices1, prices2);
        }

        // Вычисляем спред и Z-score
        let spread = compute_spread(prices1, prices2, self.hedge_ratio);
        let zscore = spread_zscore(&spread, self.params.lookback_period);

        for i in self.params.lookback_period..n {
            let z = zscore[i];
            let signal = self.process_signal(z, i);
            signals[i] = signal;
        }

        signals
    }

    /// Обработка одного сигнала
    fn process_signal(&mut self, zscore: f64, index: usize) -> Signal {
        match self.current_position {
            Position::Flat => {
                // Проверяем вход
                if zscore > self.params.entry_threshold {
                    // Спред выше нормы: продаём asset1, покупаем asset2
                    self.current_position = Position::Short;
                    self.entry_index = Some(index);
                    self.entry_zscore = zscore;
                    Signal::SellSpread
                } else if zscore < -self.params.entry_threshold {
                    // Спред ниже нормы: покупаем asset1, продаём asset2
                    self.current_position = Position::Long;
                    self.entry_index = Some(index);
                    self.entry_zscore = zscore;
                    Signal::BuySpread
                } else {
                    Signal::Hold
                }
            }
            Position::Long => {
                // Проверяем выход из длинной позиции
                let holding_time = index - self.entry_index.unwrap_or(index);

                if zscore > -self.params.exit_threshold && holding_time >= self.params.min_holding_period {
                    // Спред вернулся к норме
                    self.current_position = Position::Flat;
                    self.entry_index = None;
                    Signal::ExitLong
                } else if zscore > self.params.stop_loss_threshold {
                    // Стоп-лосс
                    self.current_position = Position::Flat;
                    self.entry_index = None;
                    Signal::StopLoss
                } else {
                    Signal::Hold
                }
            }
            Position::Short => {
                // Проверяем выход из короткой позиции
                let holding_time = index - self.entry_index.unwrap_or(index);

                if zscore < self.params.exit_threshold && holding_time >= self.params.min_holding_period {
                    // Спред вернулся к норме
                    self.current_position = Position::Flat;
                    self.entry_index = None;
                    Signal::ExitShort
                } else if zscore < -self.params.stop_loss_threshold {
                    // Стоп-лосс
                    self.current_position = Position::Flat;
                    self.entry_index = None;
                    Signal::StopLoss
                } else {
                    Signal::Hold
                }
            }
        }
    }

    /// Обновление hedge ratio
    fn update_hedge_ratio(&mut self, prices1: &[f64], prices2: &[f64]) {
        let n = prices1.len().min(prices2.len());
        if n < self.params.hedge_recalc_period {
            return;
        }

        let recent1 = &prices1[n - self.params.hedge_recalc_period..];
        let recent2 = &prices2[n - self.params.hedge_recalc_period..];

        if let Some(result) = engle_granger_test(recent1, recent2) {
            self.hedge_ratio = result.hedge_ratio;
        }
    }

    /// Текущая позиция
    pub fn position(&self) -> Position {
        self.current_position
    }

    /// Сброс стратегии
    pub fn reset(&mut self) {
        self.current_position = Position::Flat;
        self.entry_index = None;
        self.entry_zscore = 0.0;
    }
}

/// Результаты анализа пары
#[derive(Debug, Clone)]
pub struct PairAnalysis {
    pub asset1: String,
    pub asset2: String,
    pub hedge_ratio: f64,
    pub spread_mean: f64,
    pub spread_std: f64,
    pub current_zscore: f64,
    pub half_life: Option<f64>,
    pub correlation: f64,
    pub num_crossings: usize,
    pub avg_time_between_crossings: f64,
}

/// Анализ пары для торговли
pub fn analyze_pair(
    asset1: &str,
    asset2: &str,
    prices1: &[f64],
    prices2: &[f64],
) -> Option<PairAnalysis> {
    let n = prices1.len().min(prices2.len());
    if n < 50 {
        return None;
    }

    let result = engle_granger_test(prices1, prices2)?;
    let spread = compute_spread(prices1, prices2, result.hedge_ratio);
    let zscore = spread_zscore(&spread, 20);

    // Считаем пересечения нуля
    let mut crossings = 0;
    for i in 1..zscore.len() {
        if (zscore[i - 1] <= 0.0 && zscore[i] > 0.0) || (zscore[i - 1] >= 0.0 && zscore[i] < 0.0) {
            crossings += 1;
        }
    }

    let avg_time = if crossings > 0 {
        zscore.len() as f64 / crossings as f64
    } else {
        f64::INFINITY
    };

    let correlation = crate::analysis::statistics::correlation(prices1, prices2);

    Some(PairAnalysis {
        asset1: asset1.to_string(),
        asset2: asset2.to_string(),
        hedge_ratio: result.hedge_ratio,
        spread_mean: result.spread_mean,
        spread_std: result.spread_std,
        current_zscore: *zscore.last().unwrap_or(&0.0),
        half_life: result.half_life,
        correlation,
        num_crossings: crossings,
        avg_time_between_crossings: avg_time,
    })
}

/// Bollinger Bands для спреда
#[derive(Debug, Clone)]
pub struct SpreadBands {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub spread: Vec<f64>,
}

pub fn compute_spread_bands(
    prices1: &[f64],
    prices2: &[f64],
    hedge_ratio: f64,
    lookback: usize,
    num_std: f64,
) -> SpreadBands {
    let spread = compute_spread(prices1, prices2, hedge_ratio);
    let n = spread.len();

    let mut upper = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];

    for i in (lookback - 1)..n {
        let window = &spread[i + 1 - lookback..=i];
        let m = mean(window);
        let s = crate::analysis::statistics::std_dev(window);

        middle[i] = m;
        upper[i] = m + num_std * s;
        lower[i] = m - num_std * s;
    }

    SpreadBands {
        upper,
        middle,
        lower,
        spread,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairs_strategy() {
        // Создаём тестовые данные
        let n = 200;
        let mut prices1 = vec![100.0];
        let mut prices2 = vec![50.0];

        for i in 1..n {
            let noise1 = ((i * 7919) % 1000) as f64 / 5000.0 - 0.1;
            let noise2 = ((i * 1237) % 1000) as f64 / 5000.0 - 0.1;

            prices1.push(prices1[i - 1] + noise1);
            prices2.push(prices2[i - 1] * 0.5 + prices1[i - 1] * 0.5 + noise2);
        }

        let mut strategy = PairsTradingStrategy::new(
            "ASSET1",
            "ASSET2",
            1.0,
            PairsTradingParams::default(),
        );

        let signals = strategy.generate_signals(&prices1, &prices2);
        assert_eq!(signals.len(), n);
    }
}
