//! Буфер воспроизведения опыта (Experience Replay Buffer)

use super::traits::Experience;
use crate::environment::{ExecutionState, ExecutionAction};
use rand::seq::SliceRandom;
use std::collections::VecDeque;

/// Буфер воспроизведения опыта
pub struct ReplayBuffer {
    /// Буфер опытов
    buffer: VecDeque<Experience>,
    /// Максимальный размер
    capacity: usize,
}

impl ReplayBuffer {
    /// Создать новый буфер
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Добавить опыт в буфер
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Добавить переход
    pub fn add(
        &mut self,
        state: ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: ExecutionState,
        done: bool,
    ) {
        self.push(Experience::new(state, action, reward, next_state, done));
    }

    /// Выбрать случайный батч
    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut rng = rand::thread_rng();
        let indices: Vec<_> = (0..self.buffer.len()).collect();
        let sample_size = batch_size.min(self.buffer.len());

        indices
            .choose_multiple(&mut rng, sample_size)
            .map(|&i| self.buffer[i].clone())
            .collect()
    }

    /// Размер буфера
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Проверить, пуст ли буфер
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Очистить буфер
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Вместимость буфера
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Приоритетный буфер воспроизведения (PER)
pub struct PrioritizedReplayBuffer {
    /// Буфер опытов
    buffer: VecDeque<(Experience, f64)>, // (experience, priority)
    /// Максимальный размер
    capacity: usize,
    /// Альфа для приоритезации
    alpha: f64,
    /// Бета для importance sampling
    beta: f64,
    /// Минимальный приоритет
    min_priority: f64,
}

impl PrioritizedReplayBuffer {
    /// Создать новый приоритетный буфер
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            alpha,
            beta,
            min_priority: 1e-6,
        }
    }

    /// Добавить опыт с приоритетом
    pub fn push(&mut self, experience: Experience, priority: f64) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        let priority = priority.max(self.min_priority).powf(self.alpha);
        self.buffer.push_back((experience, priority));
    }

    /// Добавить переход с максимальным приоритетом
    pub fn add(
        &mut self,
        state: ExecutionState,
        action: ExecutionAction,
        reward: f64,
        next_state: ExecutionState,
        done: bool,
    ) {
        let max_priority = self.buffer
            .iter()
            .map(|(_, p)| *p)
            .fold(1.0_f64, f64::max);

        self.push(
            Experience::new(state, action, reward, next_state, done),
            max_priority,
        );
    }

    /// Выбрать батч с приоритетами
    pub fn sample(&self, batch_size: usize) -> (Vec<Experience>, Vec<f64>, Vec<usize>) {
        let mut rng = rand::thread_rng();
        let sample_size = batch_size.min(self.buffer.len());

        // Рассчитываем вероятности
        let total_priority: f64 = self.buffer.iter().map(|(_, p)| *p).sum();

        // Простая версия: выбираем случайно с весами
        let indices: Vec<usize> = (0..self.buffer.len()).collect();
        let selected: Vec<_> = indices
            .choose_multiple(&mut rng, sample_size)
            .cloned()
            .collect();

        let experiences: Vec<_> = selected.iter()
            .map(|&i| self.buffer[i].0.clone())
            .collect();

        // Importance sampling weights
        let weights: Vec<_> = selected.iter()
            .map(|&i| {
                let prob = self.buffer[i].1 / total_priority;
                (self.buffer.len() as f64 * prob).powf(-self.beta)
            })
            .collect();

        // Нормализация весов
        let max_weight = weights.iter().fold(0.0_f64, |a, &b| a.max(b));
        let weights: Vec<_> = weights.iter()
            .map(|w| w / max_weight)
            .collect();

        (experiences, weights, selected)
    }

    /// Обновить приоритеты
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f64]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            if idx < self.buffer.len() {
                self.buffer[idx].1 = priority.max(self.min_priority).powf(self.alpha);
            }
        }
    }

    /// Размер буфера
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Проверить, пуст ли буфер
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Увеличить beta (для annealing)
    pub fn increase_beta(&mut self, increment: f64) {
        self.beta = (self.beta + increment).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);

        for i in 0..50 {
            buffer.add(
                ExecutionState::initial(),
                ExecutionAction::Discrete(i % 11),
                i as f64 * 0.1,
                ExecutionState::initial(),
                false,
            );
        }

        assert_eq!(buffer.len(), 50);

        let batch = buffer.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut buffer = ReplayBuffer::new(10);

        for i in 0..20 {
            buffer.add(
                ExecutionState::initial(),
                ExecutionAction::Discrete(0),
                i as f64,
                ExecutionState::initial(),
                false,
            );
        }

        assert_eq!(buffer.len(), 10);
    }
}
