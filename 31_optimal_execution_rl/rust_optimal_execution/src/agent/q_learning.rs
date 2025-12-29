//! Табличный Q-Learning агент

use super::traits::{Agent, Experience};
use crate::environment::{ExecutionState, ExecutionAction};
use rand::Rng;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Конфигурация Q-Learning агента
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLearningConfig {
    /// Количество действий
    pub num_actions: usize,
    /// Скорость обучения
    pub learning_rate: f64,
    /// Коэффициент дисконтирования
    pub gamma: f64,
    /// Начальное значение epsilon
    pub epsilon_start: f64,
    /// Конечное значение epsilon
    pub epsilon_end: f64,
    /// Скорость затухания epsilon
    pub epsilon_decay: f64,
    /// Количество бинов для дискретизации
    pub num_bins: usize,
}

impl Default for QLearningConfig {
    fn default() -> Self {
        Self {
            num_actions: 11,
            learning_rate: 0.1,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            num_bins: 10,
        }
    }
}

/// Табличный Q-Learning агент
///
/// Использует дискретизацию состояния для работы с непрерывным пространством.
pub struct QLearningAgent {
    /// Q-таблица
    q_table: HashMap<String, Vec<f64>>,
    /// Конфигурация
    config: QLearningConfig,
    /// Текущее значение epsilon
    epsilon: f64,
    /// Счётчик шагов
    step_count: usize,
}

impl QLearningAgent {
    /// Создать нового агента
    pub fn new(config: QLearningConfig) -> Self {
        Self {
            q_table: HashMap::new(),
            epsilon: config.epsilon_start,
            config,
            step_count: 0,
        }
    }

    /// Дискретизировать состояние в строковый ключ
    fn discretize_state(&self, state: &ExecutionState) -> String {
        let bins = self.config.num_bins;

        let discretize = |value: f64, min: f64, max: f64| -> usize {
            let normalized = (value - min) / (max - min);
            let bin = (normalized * bins as f64) as usize;
            bin.min(bins - 1)
        };

        format!(
            "{}_{}_{}_{}",
            discretize(state.remaining_fraction, 0.0, 1.0),
            discretize(state.time_fraction, 0.0, 1.0),
            discretize(state.volatility, -1.0, 1.0),
            discretize(state.momentum, -1.0, 1.0),
        )
    }

    /// Получить Q-значения для состояния
    fn get_q_values(&self, state: &ExecutionState) -> Vec<f64> {
        let key = self.discretize_state(state);
        self.q_table
            .get(&key)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.config.num_actions])
    }

    /// Получить или создать Q-значения
    fn get_or_create_q_values(&mut self, state: &ExecutionState) -> &mut Vec<f64> {
        let key = self.discretize_state(state);
        let num_actions = self.config.num_actions;
        self.q_table
            .entry(key)
            .or_insert_with(|| vec![0.0; num_actions])
    }

    /// Выбрать лучшее действие
    fn greedy_action(&self, state: &ExecutionState) -> usize {
        let q_values = self.get_q_values(state);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Обновить Q-значение
    fn update_q_value(&mut self, experience: &Experience) {
        let action_idx = match experience.action {
            ExecutionAction::Discrete(idx) => idx,
            ExecutionAction::Continuous(frac) => {
                (frac * (self.config.num_actions - 1) as f64) as usize
            }
        };

        // Текущее Q-значение
        let current_q = {
            let q_values = self.get_or_create_q_values(&experience.state);
            q_values[action_idx]
        };

        // Максимальное Q-значение следующего состояния
        let next_max_q = if experience.done {
            0.0
        } else {
            self.get_q_values(&experience.next_state)
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        };

        // TD update
        let td_target = experience.reward + self.config.gamma * next_max_q;
        let new_q = current_q + self.config.learning_rate * (td_target - current_q);

        // Обновляем Q-значение
        let q_values = self.get_or_create_q_values(&experience.state);
        q_values[action_idx] = new_q;
    }

    /// Получить размер Q-таблицы
    pub fn table_size(&self) -> usize {
        self.q_table.len()
    }

    /// Получить статистику
    pub fn stats(&self) -> QLearningStats {
        let mut total_q = 0.0;
        let mut count = 0;

        for values in self.q_table.values() {
            for &v in values {
                total_q += v;
                count += 1;
            }
        }

        QLearningStats {
            table_size: self.q_table.len(),
            avg_q_value: if count > 0 { total_q / count as f64 } else { 0.0 },
            epsilon: self.epsilon,
            step_count: self.step_count,
        }
    }
}

impl Agent for QLearningAgent {
    fn select_action(&self, state: &ExecutionState, epsilon: f64) -> ExecutionAction {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < epsilon {
            // Exploration: случайное действие
            ExecutionAction::Discrete(rng.gen_range(0..self.config.num_actions))
        } else {
            // Exploitation: лучшее действие
            ExecutionAction::Discrete(self.greedy_action(state))
        }
    }

    fn remember(
        &mut self,
        state: ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: ExecutionState,
        done: bool,
    ) {
        let experience = Experience::new(state, action, reward, next_state, done);
        self.update_q_value(&experience);
        self.step_count += 1;
    }

    fn train_step(&mut self) -> f64 {
        // Q-learning обновляет сразу в remember
        0.0
    }

    fn can_train(&self) -> bool {
        true
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay)
            .max(self.config.epsilon_end);
    }

    fn save(&self, path: &str) -> anyhow::Result<()> {
        let data = SavedQLearning {
            q_table: self.q_table.clone(),
            config: self.config.clone(),
            epsilon: self.epsilon,
            step_count: self.step_count,
        };
        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn load(&mut self, path: &str) -> anyhow::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let data: SavedQLearning = serde_json::from_str(&json)?;
        self.q_table = data.q_table;
        self.config = data.config;
        self.epsilon = data.epsilon;
        self.step_count = data.step_count;
        Ok(())
    }

    fn num_actions(&self) -> usize {
        self.config.num_actions
    }

    fn reset(&mut self) {
        // Не сбрасываем Q-таблицу, только epsilon
    }
}

/// Структура для сохранения агента
#[derive(Serialize, Deserialize)]
struct SavedQLearning {
    q_table: HashMap<String, Vec<f64>>,
    config: QLearningConfig,
    epsilon: f64,
    step_count: usize,
}

/// Статистика Q-Learning агента
#[derive(Debug, Clone)]
pub struct QLearningStats {
    pub table_size: usize,
    pub avg_q_value: f64,
    pub epsilon: f64,
    pub step_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_learning_agent() {
        let config = QLearningConfig::default();
        let mut agent = QLearningAgent::new(config);

        let state = ExecutionState::initial();
        let action = agent.select_action(&state, 0.0);

        assert!(matches!(action, ExecutionAction::Discrete(_)));
    }

    #[test]
    fn test_q_learning_update() {
        let config = QLearningConfig::default();
        let mut agent = QLearningAgent::new(config);

        let state = ExecutionState::initial();
        let action = ExecutionAction::Discrete(5);
        let next_state = ExecutionState::initial();

        agent.remember(state, action, 1.0, next_state, false);

        assert!(agent.table_size() > 0);
    }
}
