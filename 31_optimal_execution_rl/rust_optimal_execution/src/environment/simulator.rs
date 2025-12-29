//! Симулятор рынка для среды исполнения

use crate::api::Candle;
use crate::impact::{ImpactModel, ImpactParams, SquareRootImpact};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Симулятор рыночной динамики
pub struct MarketSimulator {
    /// Исторические свечи
    candles: Vec<Candle>,
    /// Текущий индекс в данных
    current_idx: usize,
    /// Модель market impact
    impact_model: Box<dyn ImpactModel>,
    /// Параметры impact
    impact_params: ImpactParams,
    /// Текущая цена
    pub current_price: f64,
    /// Цена при входе (arrival price)
    pub arrival_price: f64,
    /// Накопленный impact
    accumulated_impact: f64,
    /// Генератор случайных чисел
    rng: rand::rngs::ThreadRng,
}

impl MarketSimulator {
    /// Создать симулятор из исторических данных
    pub fn from_candles(candles: Vec<Candle>) -> Self {
        let current_price = candles.first().map(|c| c.close).unwrap_or(50000.0);
        let impact_params = Self::estimate_params(&candles);

        Self {
            candles,
            current_idx: 0,
            impact_model: Box::new(SquareRootImpact),
            impact_params,
            current_price,
            arrival_price: current_price,
            accumulated_impact: 0.0,
            rng: rand::thread_rng(),
        }
    }

    /// Создать симулятор для синтетических данных
    pub fn synthetic(
        initial_price: f64,
        volatility: f64,
        num_steps: usize,
    ) -> Self {
        let candles = Self::generate_synthetic_candles(initial_price, volatility, num_steps);
        let mut sim = Self::from_candles(candles);
        sim.impact_params.volatility = volatility;
        sim
    }

    /// Оценить параметры из данных
    fn estimate_params(candles: &[Candle]) -> ImpactParams {
        if candles.is_empty() {
            return ImpactParams::crypto_default();
        }

        // Среднедневной объём
        let adv: f64 = candles.iter()
            .map(|c| c.volume * c.close)
            .sum::<f64>() / candles.len().max(1) as f64;

        // Волатильность (из доходностей)
        let returns: Vec<f64> = candles.windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        let volatility = if returns.is_empty() {
            0.02
        } else {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        };

        ImpactParams::new(adv.max(1_000_000.0), volatility.max(0.001))
    }

    /// Генерировать синтетические свечи
    fn generate_synthetic_candles(
        initial_price: f64,
        volatility: f64,
        num_steps: usize,
    ) -> Vec<Candle> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, volatility).unwrap();
        let mut candles = Vec::with_capacity(num_steps);
        let mut price = initial_price;

        for i in 0..num_steps {
            let return_pct: f64 = normal.sample(&mut rng);
            let open = price;
            let close = open * (1.0 + return_pct);

            // Генерируем high/low с учётом волатильности
            let intra_vol = volatility * 0.5;
            let high = open.max(close) * (1.0 + rng.gen::<f64>() * intra_vol);
            let low = open.min(close) * (1.0 - rng.gen::<f64>() * intra_vol);

            // Объём с некоторой случайностью
            let base_volume = 1000.0;
            let volume = base_volume * (0.5 + rng.gen::<f64>());

            candles.push(Candle {
                timestamp: i as i64 * 60000, // 1-минутные свечи
                open,
                high,
                low,
                close,
                volume,
                turnover: volume * (open + close) / 2.0,
            });

            price = close;
        }

        candles
    }

    /// Сбросить симулятор
    pub fn reset(&mut self, start_idx: Option<usize>) {
        self.current_idx = start_idx.unwrap_or_else(|| {
            if self.candles.len() > 100 {
                self.rng.gen_range(0..self.candles.len() - 100)
            } else {
                0
            }
        });

        if let Some(candle) = self.candles.get(self.current_idx) {
            self.current_price = candle.close;
            self.arrival_price = candle.close;
        }

        self.accumulated_impact = 0.0;
    }

    /// Выполнить шаг симуляции
    pub fn step(&mut self, executed_quantity: f64) -> SimulationStep {
        // Получаем рыночные данные
        let market_data = self.get_market_data();

        // Рассчитываем impact
        let temp_impact = self.impact_model.temporary_impact(executed_quantity, &self.impact_params);
        let perm_impact = self.impact_model.permanent_impact(executed_quantity, &self.impact_params);

        // Цена исполнения с учётом impact
        let execution_price = self.current_price * (1.0 + temp_impact + perm_impact / 2.0);

        // Накапливаем постоянный impact
        self.accumulated_impact += perm_impact;

        // Переходим к следующему шагу
        self.current_idx = (self.current_idx + 1).min(self.candles.len().saturating_sub(1));

        // Обновляем цену (рыночное движение + накопленный impact)
        if let Some(candle) = self.candles.get(self.current_idx) {
            self.current_price = candle.close * (1.0 + self.accumulated_impact);
        }

        // Shortfall для этого шага
        let step_shortfall = (execution_price - self.arrival_price) * executed_quantity;

        SimulationStep {
            execution_price,
            market_data,
            temp_impact,
            perm_impact,
            step_shortfall,
            is_terminal: self.current_idx >= self.candles.len() - 1,
        }
    }

    /// Получить текущие рыночные данные
    pub fn get_market_data(&self) -> MarketData {
        let current = self.candles.get(self.current_idx);
        let prev = if self.current_idx > 0 {
            self.candles.get(self.current_idx - 1)
        } else {
            None
        };

        // Рассчитываем показатели
        let spread = self.impact_params.spread;

        let volatility = if self.current_idx >= 20 {
            let window: Vec<_> = self.candles[self.current_idx - 20..=self.current_idx]
                .iter()
                .collect();
            Self::calculate_volatility(&window)
        } else {
            self.impact_params.volatility
        };

        let momentum = match (current, prev) {
            (Some(c), Some(p)) if p.close > 0.0 => (c.close - p.close) / p.close,
            _ => 0.0,
        };

        let volume = current.map(|c| c.volume).unwrap_or(0.0);
        let avg_volume = self.impact_params.adv / (24.0 * 60.0); // Средний минутный объём

        MarketData {
            price: self.current_price,
            spread,
            volatility,
            momentum,
            volume_ratio: if avg_volume > 0.0 { volume / avg_volume } else { 1.0 },
            order_imbalance: self.rng.gen_range(-0.5..0.5), // Симулированный
        }
    }

    /// Рассчитать волатильность из свечей
    fn calculate_volatility(candles: &[&Candle]) -> f64 {
        if candles.len() < 2 {
            return 0.02;
        }

        let returns: Vec<f64> = candles.windows(2)
            .filter_map(|w| {
                if w[0].close > 0.0 {
                    Some((w[1].close / w[0].close).ln())
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return 0.02;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Количество оставшихся шагов
    pub fn remaining_steps(&self) -> usize {
        self.candles.len().saturating_sub(self.current_idx + 1)
    }

    /// Установить модель impact
    pub fn with_impact_model<M: ImpactModel + 'static>(mut self, model: M) -> Self {
        self.impact_model = Box::new(model);
        self
    }

    /// Установить параметры impact
    pub fn with_impact_params(mut self, params: ImpactParams) -> Self {
        self.impact_params = params;
        self
    }
}

/// Результат шага симуляции
#[derive(Debug, Clone)]
pub struct SimulationStep {
    /// Цена исполнения
    pub execution_price: f64,
    /// Рыночные данные
    pub market_data: MarketData,
    /// Временный impact
    pub temp_impact: f64,
    /// Постоянный impact
    pub perm_impact: f64,
    /// Shortfall на этом шаге
    pub step_shortfall: f64,
    /// Терминальное состояние
    pub is_terminal: bool,
}

/// Рыночные данные
#[derive(Debug, Clone, Default)]
pub struct MarketData {
    /// Текущая цена
    pub price: f64,
    /// Спред
    pub spread: f64,
    /// Волатильность
    pub volatility: f64,
    /// Momentum (изменение цены)
    pub momentum: f64,
    /// Отношение объёма к среднему
    pub volume_ratio: f64,
    /// Дисбаланс книги ордеров
    pub order_imbalance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_simulator() {
        let sim = MarketSimulator::synthetic(50000.0, 0.02, 100);
        assert_eq!(sim.candles.len(), 100);
        assert!(sim.current_price > 0.0);
    }

    #[test]
    fn test_simulation_step() {
        let mut sim = MarketSimulator::synthetic(50000.0, 0.02, 100);
        sim.reset(Some(0));

        let step = sim.step(100.0);

        assert!(step.execution_price > 0.0);
        assert!(step.temp_impact >= 0.0);
    }
}
