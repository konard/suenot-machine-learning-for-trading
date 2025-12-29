//! Трейты для RL агентов

use crate::environment::{ExecutionState, ExecutionAction};

/// Трейт для RL агента
pub trait Agent: Send {
    /// Выбрать действие на основе состояния
    fn select_action(&self, state: &ExecutionState, epsilon: f64) -> ExecutionAction;

    /// Сохранить переход в буфер опыта
    fn remember(
        &mut self,
        state: ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: ExecutionState,
        done: bool,
    );

    /// Выполнить шаг обучения
    fn train_step(&mut self) -> f64;

    /// Проверить, можно ли обучаться
    fn can_train(&self) -> bool;

    /// Получить текущее значение epsilon
    fn get_epsilon(&self) -> f64;

    /// Уменьшить epsilon
    fn decay_epsilon(&mut self);

    /// Сохранить агента
    fn save(&self, path: &str) -> anyhow::Result<()>;

    /// Загрузить агента
    fn load(&mut self, path: &str) -> anyhow::Result<()>;

    /// Получить количество действий
    fn num_actions(&self) -> usize;

    /// Сбросить состояние агента
    fn reset(&mut self);
}

/// Опыт для буфера воспроизведения
#[derive(Debug, Clone)]
pub struct Experience {
    /// Состояние
    pub state: ExecutionState,
    /// Действие
    pub action: ExecutionAction,
    /// Вознаграждение
    pub reward: f64,
    /// Следующее состояние
    pub next_state: ExecutionState,
    /// Завершено ли
    pub done: bool,
}

impl Experience {
    /// Создать новый опыт
    pub fn new(
        state: ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: ExecutionState,
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}
