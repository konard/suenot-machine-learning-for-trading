//! Среда для обучения с подкреплением

use super::state::{ExecutionState, ExecutionAction};
use super::simulator::MarketSimulator;
use crate::api::Candle;
use serde::{Deserialize, Serialize};

/// Конфигурация среды
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    /// Общий объём для исполнения
    pub total_quantity: f64,
    /// Максимальное количество шагов
    pub max_steps: usize,
    /// Количество дискретных действий
    pub num_actions: usize,
    /// Коэффициент неприятия риска
    pub risk_aversion: f64,
    /// Комиссия за сделку (в долях)
    pub trading_cost: f64,
    /// Штраф за неисполнение
    pub non_execution_penalty: f64,
    /// Использовать дискретные действия
    pub discrete_actions: bool,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            total_quantity: 1000.0,
            max_steps: 60,
            num_actions: 11, // 0%, 10%, 20%, ..., 100%
            risk_aversion: 1e-6,
            trading_cost: 0.0001, // 1 bps
            non_execution_penalty: 0.01,
            discrete_actions: true,
        }
    }
}

/// Результат шага среды
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Новое состояние
    pub state: ExecutionState,
    /// Вознаграждение
    pub reward: f64,
    /// Эпизод завершён
    pub done: bool,
    /// Дополнительная информация
    pub info: StepInfo,
}

/// Дополнительная информация о шаге
#[derive(Debug, Clone, Default)]
pub struct StepInfo {
    /// Исполненный объём
    pub executed_quantity: f64,
    /// Цена исполнения
    pub execution_price: f64,
    /// Стоимость исполнения
    pub execution_cost: f64,
    /// Implementation shortfall на шаге
    pub step_shortfall: f64,
    /// Накопленный shortfall
    pub cumulative_shortfall: f64,
    /// Оставшийся объём
    pub remaining_quantity: f64,
    /// Оставшееся время
    pub remaining_time: usize,
}

/// Среда для оптимального исполнения
pub struct ExecutionEnv {
    /// Конфигурация
    config: EnvConfig,
    /// Симулятор рынка
    simulator: MarketSimulator,
    /// Общий объём для исполнения
    total_quantity: f64,
    /// Оставшийся объём
    remaining_quantity: f64,
    /// Текущий шаг
    current_step: usize,
    /// Цена при входе
    arrival_price: f64,
    /// Накопленный shortfall
    cumulative_shortfall: f64,
    /// Исполненный объём по шагам
    execution_history: Vec<f64>,
    /// Цены исполнения
    price_history: Vec<f64>,
}

impl ExecutionEnv {
    /// Создать среду из исторических данных
    pub fn from_candles(candles: Vec<Candle>, config: EnvConfig) -> Self {
        let simulator = MarketSimulator::from_candles(candles);
        Self::with_simulator(simulator, config)
    }

    /// Создать среду с синтетическими данными
    pub fn synthetic(initial_price: f64, volatility: f64, config: EnvConfig) -> Self {
        let simulator = MarketSimulator::synthetic(
            initial_price,
            volatility,
            config.max_steps + 100,
        );
        Self::with_simulator(simulator, config)
    }

    /// Создать среду с симулятором
    pub fn with_simulator(simulator: MarketSimulator, config: EnvConfig) -> Self {
        let total_quantity = config.total_quantity;

        Self {
            config,
            simulator,
            total_quantity,
            remaining_quantity: total_quantity,
            current_step: 0,
            arrival_price: 0.0,
            cumulative_shortfall: 0.0,
            execution_history: Vec::new(),
            price_history: Vec::new(),
        }
    }

    /// Размерность состояния
    pub fn state_dim(&self) -> usize {
        ExecutionState::state_dim()
    }

    /// Количество действий (для дискретных)
    pub fn action_dim(&self) -> usize {
        self.config.num_actions
    }

    /// Сбросить среду
    pub fn reset(&mut self) -> ExecutionState {
        self.simulator.reset(None);
        self.remaining_quantity = self.total_quantity;
        self.current_step = 0;
        self.arrival_price = self.simulator.current_price;
        self.cumulative_shortfall = 0.0;
        self.execution_history.clear();
        self.price_history.clear();

        self.get_state()
    }

    /// Сбросить с указанием начального индекса
    pub fn reset_with_idx(&mut self, start_idx: usize) -> ExecutionState {
        self.simulator.reset(Some(start_idx));
        self.remaining_quantity = self.total_quantity;
        self.current_step = 0;
        self.arrival_price = self.simulator.current_price;
        self.cumulative_shortfall = 0.0;
        self.execution_history.clear();
        self.price_history.clear();

        self.get_state()
    }

    /// Выполнить действие
    pub fn step(&mut self, action: ExecutionAction) -> StepResult {
        // Преобразуем действие в долю
        let fraction = action.to_fraction(self.config.num_actions);

        // Рассчитываем объём для исполнения
        let execute_qty = (fraction * self.remaining_quantity).min(self.remaining_quantity);

        // Симулируем исполнение
        let sim_step = self.simulator.step(execute_qty);

        // Обновляем состояние
        self.remaining_quantity -= execute_qty;
        self.current_step += 1;

        // Рассчитываем стоимость (shortfall + комиссии)
        let trading_cost = execute_qty * sim_step.execution_price * self.config.trading_cost;
        let step_shortfall = sim_step.step_shortfall + trading_cost;
        self.cumulative_shortfall += step_shortfall;

        // Сохраняем историю
        self.execution_history.push(execute_qty);
        self.price_history.push(sim_step.execution_price);

        // Проверяем завершение
        let done = self.is_done();

        // Рассчитываем вознаграждение
        let reward = self.calculate_reward(
            execute_qty,
            step_shortfall,
            done,
        );

        let info = StepInfo {
            executed_quantity: execute_qty,
            execution_price: sim_step.execution_price,
            execution_cost: step_shortfall,
            step_shortfall,
            cumulative_shortfall: self.cumulative_shortfall,
            remaining_quantity: self.remaining_quantity,
            remaining_time: self.remaining_steps(),
        };

        StepResult {
            state: self.get_state(),
            reward,
            done,
            info,
        }
    }

    /// Рассчитать вознаграждение
    fn calculate_reward(
        &self,
        executed_qty: f64,
        step_shortfall: f64,
        done: bool,
    ) -> f64 {
        // Основной reward: отрицательный shortfall (нормализованный)
        let normalized_shortfall = step_shortfall / (self.total_quantity * self.arrival_price);
        let mut reward = -normalized_shortfall * 1000.0; // Масштабируем

        // Штраф за риск (remaining quantity * volatility)
        let market_data = self.simulator.get_market_data();
        let risk_penalty = self.config.risk_aversion
            * self.remaining_quantity.powi(2)
            * market_data.volatility.powi(2);
        reward -= risk_penalty;

        // Штраф за неисполнение в конце
        if done && self.remaining_quantity > 0.0 {
            let non_exec_penalty = self.config.non_execution_penalty
                * (self.remaining_quantity / self.total_quantity);
            reward -= non_exec_penalty;
        }

        // Небольшой бонус за исполнение
        if executed_qty > 0.0 {
            reward += 0.001;
        }

        reward
    }

    /// Получить текущее состояние
    fn get_state(&self) -> ExecutionState {
        let market_data = self.simulator.get_market_data();

        let mut state = ExecutionState {
            remaining_fraction: self.remaining_quantity / self.total_quantity,
            time_fraction: self.remaining_steps() as f64 / self.config.max_steps as f64,
            spread: market_data.spread / 0.001, // Нормализация
            volatility: (market_data.volatility - 0.02) / 0.02, // Относительно 2%
            order_imbalance: market_data.order_imbalance,
            momentum: market_data.momentum * 100.0, // Масштабирование
            volume: (market_data.volume_ratio - 1.0).clamp(-2.0, 2.0),
            vwap_deviation: self.calculate_vwap_deviation(),
            execution_rate: self.calculate_execution_rate(),
            current_shortfall: self.normalized_shortfall(),
        };

        state.normalize();
        state
    }

    /// Проверить завершение эпизода
    fn is_done(&self) -> bool {
        self.current_step >= self.config.max_steps
            || self.remaining_quantity <= 0.0
            || self.simulator.remaining_steps() == 0
    }

    /// Оставшиеся шаги
    fn remaining_steps(&self) -> usize {
        self.config.max_steps.saturating_sub(self.current_step)
    }

    /// Нормализованный shortfall
    fn normalized_shortfall(&self) -> f64 {
        let total_value = self.total_quantity * self.arrival_price;
        if total_value > 0.0 {
            (self.cumulative_shortfall / total_value).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Рассчитать отклонение от VWAP
    fn calculate_vwap_deviation(&self) -> f64 {
        if self.execution_history.is_empty() || self.price_history.is_empty() {
            return 0.0;
        }

        let total_value: f64 = self.execution_history.iter()
            .zip(self.price_history.iter())
            .map(|(q, p)| q * p)
            .sum();

        let total_qty: f64 = self.execution_history.iter().sum();

        if total_qty > 0.0 && self.arrival_price > 0.0 {
            let vwap = total_value / total_qty;
            ((vwap - self.arrival_price) / self.arrival_price).clamp(-0.1, 0.1) * 10.0
        } else {
            0.0
        }
    }

    /// Рассчитать скорость исполнения
    fn calculate_execution_rate(&self) -> f64 {
        if self.current_step == 0 {
            return 0.0;
        }

        // Текущая скорость vs ожидаемая (TWAP)
        let executed = self.total_quantity - self.remaining_quantity;
        let expected = self.total_quantity * (self.current_step as f64 / self.config.max_steps as f64);

        if expected > 0.0 {
            ((executed - expected) / expected).clamp(-2.0, 2.0)
        } else {
            0.0
        }
    }

    /// Получить конфигурацию
    pub fn config(&self) -> &EnvConfig {
        &self.config
    }

    /// Получить историю исполнения
    pub fn execution_history(&self) -> &[f64] {
        &self.execution_history
    }

    /// Получить историю цен
    pub fn price_history(&self) -> &[f64] {
        &self.price_history
    }

    /// Получить текущий shortfall
    pub fn cumulative_shortfall(&self) -> f64 {
        self.cumulative_shortfall
    }

    /// Получить arrival price
    pub fn arrival_price(&self) -> f64 {
        self.arrival_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_creation() {
        let config = EnvConfig::default();
        let mut env = ExecutionEnv::synthetic(50000.0, 0.02, config);

        let state = env.reset();
        assert!((state.remaining_fraction - 1.0).abs() < 0.001);
        assert!((state.time_fraction - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_env_step() {
        let config = EnvConfig::default();
        let mut env = ExecutionEnv::synthetic(50000.0, 0.02, config);

        env.reset();

        // Выполняем действие (10% от оставшегося)
        let action = ExecutionAction::Discrete(1);
        let result = env.step(action);

        assert!(result.info.executed_quantity > 0.0);
        assert!(result.state.remaining_fraction < 1.0);
    }

    #[test]
    fn test_full_episode() {
        let config = EnvConfig {
            max_steps: 10,
            total_quantity: 100.0,
            ..Default::default()
        };
        let mut env = ExecutionEnv::synthetic(50000.0, 0.02, config);

        env.reset();

        let mut total_reward = 0.0;
        let mut done = false;

        while !done {
            let action = ExecutionAction::Discrete(5); // 50% каждый раз
            let result = env.step(action);
            total_reward += result.reward;
            done = result.done;
        }

        assert!(done);
        println!("Total reward: {}", total_reward);
    }
}
