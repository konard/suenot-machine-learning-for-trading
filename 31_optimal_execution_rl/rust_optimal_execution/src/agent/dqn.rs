//! Deep Q-Network агент

use super::traits::{Agent, Experience};
use super::replay_buffer::ReplayBuffer;
use super::neural_network::NeuralNetwork;
use crate::environment::{ExecutionState, ExecutionAction};
use ndarray::Array1;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Конфигурация DQN агента
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DQNConfig {
    /// Размерность состояния
    pub state_dim: usize,
    /// Количество действий
    pub num_actions: usize,
    /// Размеры скрытых слоёв
    pub hidden_layers: Vec<usize>,
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
    /// Размер буфера воспроизведения
    pub buffer_size: usize,
    /// Размер батча
    pub batch_size: usize,
    /// Частота обновления целевой сети
    pub target_update_freq: usize,
    /// Tau для soft update
    pub tau: f64,
    /// Использовать Double DQN
    pub double_dqn: bool,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            state_dim: ExecutionState::state_dim(),
            num_actions: 11,
            hidden_layers: vec![128, 64],
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            buffer_size: 100000,
            batch_size: 64,
            target_update_freq: 100,
            tau: 0.005,
            double_dqn: true,
        }
    }
}

/// Deep Q-Network агент
pub struct DQNAgent {
    /// Основная сеть (Q-network)
    q_network: NeuralNetwork,
    /// Целевая сеть (Target network)
    target_network: NeuralNetwork,
    /// Буфер воспроизведения
    replay_buffer: ReplayBuffer,
    /// Конфигурация
    config: DQNConfig,
    /// Текущее значение epsilon
    epsilon: f64,
    /// Счётчик шагов обучения
    train_step_count: usize,
    /// Общий счётчик шагов
    total_steps: usize,
}

impl DQNAgent {
    /// Создать нового агента
    pub fn new(config: DQNConfig) -> Self {
        // Создаём архитектуру сети
        let mut layer_sizes = vec![config.state_dim];
        layer_sizes.extend(&config.hidden_layers);
        layer_sizes.push(config.num_actions);

        let q_network = NeuralNetwork::new(&layer_sizes, config.learning_rate);
        let mut target_network = NeuralNetwork::new(&layer_sizes, config.learning_rate);
        target_network.copy_weights_from(&q_network);

        let replay_buffer = ReplayBuffer::new(config.buffer_size);

        Self {
            q_network,
            target_network,
            replay_buffer,
            epsilon: config.epsilon_start,
            config,
            train_step_count: 0,
            total_steps: 0,
        }
    }

    /// Получить Q-значения для состояния
    fn get_q_values(&self, state: &ExecutionState) -> Array1<f64> {
        let features = state.to_features();
        self.q_network.predict(&features)
    }

    /// Получить Q-значения от целевой сети
    fn get_target_q_values(&self, state: &ExecutionState) -> Array1<f64> {
        let features = state.to_features();
        self.target_network.predict(&features)
    }

    /// Выбрать лучшее действие (greedy)
    fn greedy_action(&self, state: &ExecutionState) -> usize {
        let q_values = self.get_q_values(state);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Обучить на батче
    fn train_on_batch(&mut self, batch: &[Experience]) -> f64 {
        let batch_size = batch.len();
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for experience in batch {
            let state_features = experience.state.to_features();
            let next_state_features = experience.next_state.to_features();

            // Получаем текущие Q-значения
            let current_q = self.q_network.predict(&state_features);

            // Рассчитываем целевое Q-значение
            let target_q = if experience.done {
                experience.reward
            } else {
                let next_q = if self.config.double_dqn {
                    // Double DQN: выбираем действие основной сетью, оцениваем целевой
                    let online_q = self.q_network.predict(&next_state_features);
                    let best_action = online_q
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.target_network.predict(&next_state_features)[best_action]
                } else {
                    // Обычный DQN
                    let target_q_next = self.target_network.predict(&next_state_features);
                    target_q_next.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                };
                experience.reward + self.config.gamma * next_q
            };

            // Создаём целевой вектор
            let action_idx = match experience.action {
                ExecutionAction::Discrete(idx) => idx,
                ExecutionAction::Continuous(frac) => {
                    (frac * (self.config.num_actions - 1) as f64) as usize
                }
            };

            let mut target = current_q.clone();
            target[action_idx] = target_q;

            // Обучаем сеть
            let loss = self.q_network.train_single(&state_features, &target);
            total_loss += loss;
        }

        self.train_step_count += 1;

        // Обновляем целевую сеть
        if self.train_step_count % self.config.target_update_freq == 0 {
            self.target_network.soft_update(&self.q_network, self.config.tau);
        }

        total_loss / batch_size as f64
    }

    /// Получить статистику
    pub fn stats(&self) -> DQNStats {
        DQNStats {
            buffer_size: self.replay_buffer.len(),
            epsilon: self.epsilon,
            train_steps: self.train_step_count,
            total_steps: self.total_steps,
            num_parameters: self.q_network.num_parameters(),
        }
    }
}

impl Agent for DQNAgent {
    fn select_action(&self, state: &ExecutionState, epsilon: f64) -> ExecutionAction {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < epsilon {
            // Exploration
            ExecutionAction::Discrete(rng.gen_range(0..self.config.num_actions))
        } else {
            // Exploitation
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
        self.replay_buffer.add(state, action, reward, next_state, done);
        self.total_steps += 1;
    }

    fn train_step(&mut self) -> f64 {
        if !self.can_train() {
            return 0.0;
        }

        let batch = self.replay_buffer.sample(self.config.batch_size);
        self.train_on_batch(&batch)
    }

    fn can_train(&self) -> bool {
        self.replay_buffer.len() >= self.config.batch_size
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay)
            .max(self.config.epsilon_end);
    }

    fn save(&self, path: &str) -> anyhow::Result<()> {
        let data = SavedDQN {
            q_network_json: self.q_network.to_json()?,
            target_network_json: self.target_network.to_json()?,
            config: self.config.clone(),
            epsilon: self.epsilon,
            train_step_count: self.train_step_count,
            total_steps: self.total_steps,
        };
        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn load(&mut self, path: &str) -> anyhow::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let data: SavedDQN = serde_json::from_str(&json)?;

        self.q_network = NeuralNetwork::from_json(&data.q_network_json)?;
        self.target_network = NeuralNetwork::from_json(&data.target_network_json)?;
        self.config = data.config;
        self.epsilon = data.epsilon;
        self.train_step_count = data.train_step_count;
        self.total_steps = data.total_steps;

        Ok(())
    }

    fn num_actions(&self) -> usize {
        self.config.num_actions
    }

    fn reset(&mut self) {
        // Опционально: можно сбросить epsilon
    }
}

/// Структура для сохранения DQN
#[derive(Serialize, Deserialize)]
struct SavedDQN {
    q_network_json: String,
    target_network_json: String,
    config: DQNConfig,
    epsilon: f64,
    train_step_count: usize,
    total_steps: usize,
}

/// Статистика DQN агента
#[derive(Debug, Clone)]
pub struct DQNStats {
    pub buffer_size: usize,
    pub epsilon: f64,
    pub train_steps: usize,
    pub total_steps: usize,
    pub num_parameters: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dqn_creation() {
        let config = DQNConfig::default();
        let agent = DQNAgent::new(config);

        assert_eq!(agent.num_actions(), 11);
        assert!(!agent.can_train());
    }

    #[test]
    fn test_dqn_action_selection() {
        let config = DQNConfig::default();
        let agent = DQNAgent::new(config);

        let state = ExecutionState::initial();

        // С epsilon=0 должен выбирать greedy
        let action1 = agent.select_action(&state, 0.0);
        let action2 = agent.select_action(&state, 0.0);

        // Оба действия должны быть одинаковыми
        assert_eq!(
            matches!(action1, ExecutionAction::Discrete(_)),
            matches!(action2, ExecutionAction::Discrete(_))
        );
    }

    #[test]
    fn test_dqn_training() {
        let config = DQNConfig {
            batch_size: 10,
            buffer_size: 100,
            ..Default::default()
        };
        let mut agent = DQNAgent::new(config);

        // Добавляем опыт
        for _ in 0..20 {
            agent.remember(
                ExecutionState::initial(),
                ExecutionAction::Discrete(5),
                0.1,
                ExecutionState::initial(),
                false,
            );
        }

        assert!(agent.can_train());

        let loss = agent.train_step();
        assert!(loss >= 0.0);
    }
}
