//! Memory buffer for experience replay.
//!
//! This module implements a memory buffer that stores experiences from past tasks
//! to enable experience replay during continual learning.

use rand::seq::SliceRandom;
use rand::thread_rng;

/// A single experience sample.
#[derive(Clone, Debug)]
pub struct Experience {
    /// Input features.
    pub input: Vec<f64>,
    /// Target output.
    pub target: Vec<f64>,
    /// Task identifier (e.g., market regime).
    pub task_id: usize,
    /// Importance weight for this sample.
    pub importance: f64,
}

impl Experience {
    /// Create a new experience.
    pub fn new(input: Vec<f64>, target: Vec<f64>, task_id: usize) -> Self {
        Self {
            input,
            target,
            task_id,
            importance: 1.0,
        }
    }

    /// Create a new experience with custom importance.
    pub fn with_importance(input: Vec<f64>, target: Vec<f64>, task_id: usize, importance: f64) -> Self {
        Self {
            input,
            target,
            task_id,
            importance,
        }
    }
}

/// Strategy for selecting which samples to remove when buffer is full.
#[derive(Clone, Copy, Debug, Default)]
pub enum ReplacementStrategy {
    /// Replace oldest samples first (FIFO).
    #[default]
    Oldest,
    /// Replace random samples.
    Random,
    /// Replace samples with lowest importance.
    LowestImportance,
    /// Reservoir sampling for uniform distribution.
    Reservoir,
}

/// Memory buffer for storing and replaying experiences.
#[derive(Debug)]
pub struct MemoryBuffer {
    /// Stored experiences.
    buffer: Vec<Experience>,
    /// Maximum buffer capacity.
    capacity: usize,
    /// Strategy for replacing samples.
    replacement_strategy: ReplacementStrategy,
    /// Total number of samples seen (for reservoir sampling).
    total_seen: usize,
}

impl MemoryBuffer {
    /// Create a new memory buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            replacement_strategy: ReplacementStrategy::default(),
            total_seen: 0,
        }
    }

    /// Create a new memory buffer with custom replacement strategy.
    pub fn with_strategy(capacity: usize, strategy: ReplacementStrategy) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            replacement_strategy: strategy,
            total_seen: 0,
        }
    }

    /// Add an experience to the buffer.
    pub fn add(&mut self, experience: Experience) {
        self.total_seen += 1;

        if self.buffer.len() < self.capacity {
            self.buffer.push(experience);
        } else {
            match self.replacement_strategy {
                ReplacementStrategy::Oldest => {
                    // Remove first (oldest) and append new
                    self.buffer.remove(0);
                    self.buffer.push(experience);
                }
                ReplacementStrategy::Random => {
                    let idx = rand::random::<usize>() % self.capacity;
                    self.buffer[idx] = experience;
                }
                ReplacementStrategy::LowestImportance => {
                    // Find index of lowest importance sample
                    let min_idx = self
                        .buffer
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.importance.partial_cmp(&b.importance).unwrap()
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    // Only replace if new sample is more important
                    if experience.importance > self.buffer[min_idx].importance {
                        self.buffer[min_idx] = experience;
                    }
                }
                ReplacementStrategy::Reservoir => {
                    // Reservoir sampling: probability capacity/total_seen
                    let prob = self.capacity as f64 / self.total_seen as f64;
                    if rand::random::<f64>() < prob {
                        let idx = rand::random::<usize>() % self.capacity;
                        self.buffer[idx] = experience;
                    }
                }
            }
        }
    }

    /// Add multiple experiences.
    pub fn add_batch(&mut self, experiences: Vec<Experience>) {
        for exp in experiences {
            self.add(exp);
        }
    }

    /// Sample a batch of experiences randomly.
    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(batch_size.min(self.buffer.len()))
            .map(|i| &self.buffer[i])
            .collect()
    }

    /// Sample experiences with importance weighting.
    pub fn sample_weighted(&self, batch_size: usize) -> Vec<&Experience> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let total_importance: f64 = self.buffer.iter().map(|e| e.importance).sum();
        if total_importance <= 0.0 {
            return self.sample(batch_size);
        }

        let mut sampled = Vec::with_capacity(batch_size);
        let _rng = thread_rng();

        for _ in 0..batch_size {
            let threshold = rand::random::<f64>() * total_importance;
            let mut cumsum = 0.0;

            for exp in &self.buffer {
                cumsum += exp.importance;
                if cumsum >= threshold {
                    sampled.push(exp);
                    break;
                }
            }
        }

        // If we didn't get enough samples, fill with random
        while sampled.len() < batch_size && sampled.len() < self.buffer.len() {
            let idx = rand::random::<usize>() % self.buffer.len();
            sampled.push(&self.buffer[idx]);
        }

        sampled
    }

    /// Sample experiences from a specific task.
    pub fn sample_task(&self, task_id: usize, batch_size: usize) -> Vec<&Experience> {
        let task_experiences: Vec<&Experience> = self
            .buffer
            .iter()
            .filter(|e| e.task_id == task_id)
            .collect();

        if task_experiences.is_empty() {
            return Vec::new();
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..task_experiences.len()).collect();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(batch_size.min(task_experiences.len()))
            .map(|i| task_experiences[i])
            .collect()
    }

    /// Get all experiences (for EWC Fisher computation).
    pub fn get_all(&self) -> &[Experience] {
        &self.buffer
    }

    /// Get experiences for a specific task.
    pub fn get_task(&self, task_id: usize) -> Vec<&Experience> {
        self.buffer.iter().filter(|e| e.task_id == task_id).collect()
    }

    /// Get the number of unique tasks in the buffer.
    pub fn num_tasks(&self) -> usize {
        let mut tasks: Vec<usize> = self.buffer.iter().map(|e| e.task_id).collect();
        tasks.sort();
        tasks.dedup();
        tasks.len()
    }

    /// Get task IDs present in the buffer.
    pub fn task_ids(&self) -> Vec<usize> {
        let mut tasks: Vec<usize> = self.buffer.iter().map(|e| e.task_id).collect();
        tasks.sort();
        tasks.dedup();
        tasks
    }

    /// Get current buffer size.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.total_seen = 0;
    }

    /// Update importance weights for all experiences using a scoring function.
    pub fn update_importance<F>(&mut self, scorer: F)
    where
        F: Fn(&Experience) -> f64,
    {
        for exp in &mut self.buffer {
            exp.importance = scorer(exp);
        }
    }

    /// Get statistics about the buffer.
    pub fn stats(&self) -> MemoryStats {
        let tasks = self.task_ids();
        let task_counts: Vec<(usize, usize)> = tasks
            .iter()
            .map(|&t| (t, self.buffer.iter().filter(|e| e.task_id == t).count()))
            .collect();

        let avg_importance = if self.buffer.is_empty() {
            0.0
        } else {
            self.buffer.iter().map(|e| e.importance).sum::<f64>() / self.buffer.len() as f64
        };

        MemoryStats {
            size: self.buffer.len(),
            capacity: self.capacity,
            num_tasks: tasks.len(),
            task_counts,
            avg_importance,
            total_seen: self.total_seen,
        }
    }
}

/// Statistics about the memory buffer.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current buffer size.
    pub size: usize,
    /// Buffer capacity.
    pub capacity: usize,
    /// Number of unique tasks.
    pub num_tasks: usize,
    /// Sample count per task.
    pub task_counts: Vec<(usize, usize)>,
    /// Average importance weight.
    pub avg_importance: f64,
    /// Total samples seen.
    pub total_seen: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_buffer_basic() {
        let mut buffer = MemoryBuffer::new(10);

        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 10);

        for i in 0..5 {
            let exp = Experience::new(vec![i as f64], vec![1.0], 0);
            buffer.add(exp);
        }

        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_memory_buffer_overflow() {
        let mut buffer = MemoryBuffer::new(5);

        for i in 0..10 {
            let exp = Experience::new(vec![i as f64], vec![1.0], 0);
            buffer.add(exp);
        }

        assert_eq!(buffer.len(), 5);
        // With oldest replacement, should have indices 5-9
        let first = &buffer.get_all()[0];
        assert_eq!(first.input[0], 5.0);
    }

    #[test]
    fn test_memory_buffer_sampling() {
        let mut buffer = MemoryBuffer::new(100);

        for i in 0..50 {
            let exp = Experience::new(vec![i as f64], vec![1.0], i % 3);
            buffer.add(exp);
        }

        let sample = buffer.sample(10);
        assert_eq!(sample.len(), 10);

        let task_sample = buffer.sample_task(0, 5);
        assert!(task_sample.len() <= 5);
        for exp in task_sample {
            assert_eq!(exp.task_id, 0);
        }
    }

    #[test]
    fn test_memory_buffer_stats() {
        let mut buffer = MemoryBuffer::new(100);

        for i in 0..30 {
            let exp = Experience::with_importance(
                vec![i as f64],
                vec![1.0],
                i % 3,
                (i % 10) as f64,
            );
            buffer.add(exp);
        }

        let stats = buffer.stats();
        assert_eq!(stats.size, 30);
        assert_eq!(stats.num_tasks, 3);
        assert_eq!(stats.total_seen, 30);
    }
}
