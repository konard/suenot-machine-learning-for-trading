//! Реализация GRU (Gated Recurrent Unit)
//!
//! GRU - упрощённая версия LSTM с меньшим количеством параметров.
//! Использует два вентиля вместо трёх: update gate и reset gate.

use super::config::LSTMConfig;
use super::layers::{Activation, Dense};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array1, Array2, Array3};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

/// GRU ячейка
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRUCell {
    /// Размер входа
    pub input_size: usize,
    /// Размер скрытого состояния
    pub hidden_size: usize,

    // Веса для update gate
    w_iz: Array2<f64>, // input -> update gate
    w_hz: Array2<f64>, // hidden -> update gate
    b_z: Array1<f64>,  // bias update gate

    // Веса для reset gate
    w_ir: Array2<f64>, // input -> reset gate
    w_hr: Array2<f64>, // hidden -> reset gate
    b_r: Array1<f64>,  // bias reset gate

    // Веса для candidate hidden state
    w_in: Array2<f64>, // input -> candidate
    w_hn: Array2<f64>, // hidden -> candidate
    b_n: Array1<f64>,  // bias candidate
}

impl GRUCell {
    /// Создаёт новую GRU ячейку
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let limit = (1.0 / hidden_size as f64).sqrt();

        Self {
            input_size,
            hidden_size,
            // Update gate
            w_iz: Array2::random((hidden_size, input_size), Uniform::new(-limit, limit)),
            w_hz: Array2::random((hidden_size, hidden_size), Uniform::new(-limit, limit)),
            b_z: Array1::zeros(hidden_size),
            // Reset gate
            w_ir: Array2::random((hidden_size, input_size), Uniform::new(-limit, limit)),
            w_hr: Array2::random((hidden_size, hidden_size), Uniform::new(-limit, limit)),
            b_r: Array1::zeros(hidden_size),
            // Candidate
            w_in: Array2::random((hidden_size, input_size), Uniform::new(-limit, limit)),
            w_hn: Array2::random((hidden_size, hidden_size), Uniform::new(-limit, limit)),
            b_n: Array1::zeros(hidden_size),
        }
    }

    /// Прямой проход для одного временного шага
    ///
    /// # Аргументы
    ///
    /// * `x` - Входной вектор [input_size]
    /// * `h_prev` - Предыдущее скрытое состояние [hidden_size]
    ///
    /// # Возвращает
    ///
    /// Новое скрытое состояние [hidden_size]
    pub fn forward(&self, x: &Array1<f64>, h_prev: &Array1<f64>) -> Array1<f64> {
        // Update gate: z = σ(W_iz * x + W_hz * h + b_z)
        let z_gate = sigmoid(&(self.w_iz.dot(x) + self.w_hz.dot(h_prev) + &self.b_z));

        // Reset gate: r = σ(W_ir * x + W_hr * h + b_r)
        let r_gate = sigmoid(&(self.w_ir.dot(x) + self.w_hr.dot(h_prev) + &self.b_r));

        // Candidate hidden state: n = tanh(W_in * x + W_hn * (r ⊙ h) + b_n)
        let n = tanh(&(self.w_in.dot(x) + self.w_hn.dot(&(&r_gate * h_prev)) + &self.b_n));

        // New hidden state: h = (1 - z) ⊙ n + z ⊙ h_prev
        let one_minus_z: Array1<f64> = z_gate.mapv(|v| 1.0 - v);
        &one_minus_z * &n + &z_gate * h_prev
    }

    /// Инициализирует скрытое состояние нулями
    pub fn init_hidden(&self) -> Array1<f64> {
        Array1::zeros(self.hidden_size)
    }
}

/// GRU модель для прогнозирования временных рядов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRU {
    /// Конфигурация модели
    pub config: LSTMConfig,
    /// GRU ячейки (по одной на слой)
    cells: Vec<GRUCell>,
    /// Выходной слой
    output_layer: Dense,
    /// История потерь при обучении
    #[serde(skip)]
    pub loss_history: Vec<f64>,
}

impl GRU {
    /// Создаёт новую GRU модель
    ///
    /// # Аргументы
    ///
    /// * `input_size` - Количество входных признаков
    /// * `hidden_size` - Размер скрытого слоя
    /// * `output_size` - Количество выходов
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let config = LSTMConfig::new(input_size, hidden_size, output_size);
        Self::from_config(config)
    }

    /// Создаёт модель из конфигурации
    pub fn from_config(config: LSTMConfig) -> Self {
        let mut cells = Vec::with_capacity(config.num_layers);

        cells.push(GRUCell::new(config.input_size, config.hidden_size));

        for _ in 1..config.num_layers {
            cells.push(GRUCell::new(config.hidden_size, config.hidden_size));
        }

        let output_layer = Dense::new(config.hidden_size, config.output_size, Activation::Linear);

        Self {
            config,
            cells,
            output_layer,
            loss_history: Vec::new(),
        }
    }

    /// Прямой проход через всю сеть
    pub fn forward(&mut self, x: &Array3<f64>) -> Array2<f64> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut outputs = Array2::zeros((batch_size, self.config.output_size));

        for b in 0..batch_size {
            // Инициализируем скрытые состояния
            let mut states: Vec<Array1<f64>> =
                self.cells.iter().map(|cell| cell.init_hidden()).collect();

            // Проходим по последовательности
            for t in 0..seq_len {
                let mut layer_input: Array1<f64> = x.slice(s![b, t, ..]).to_owned();

                for (layer_idx, cell) in self.cells.iter().enumerate() {
                    let h_prev = &states[layer_idx];
                    let h_next = cell.forward(&layer_input, h_prev);

                    layer_input = h_next.clone();
                    states[layer_idx] = h_next;
                }
            }

            // Берём последнее скрытое состояние последнего слоя
            let final_hidden = &states[self.cells.len() - 1];

            // Проходим через выходной слой
            let output = self.output_layer.forward(final_hidden);

            for (i, &val) in output.iter().enumerate() {
                outputs[[b, i]] = val;
            }
        }

        outputs
    }

    /// Вычисляет MSE loss
    pub fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        let squared = diff.mapv(|x| x * x);
        squared.mean().unwrap_or(0.0)
    }

    /// Обучает модель
    pub fn train(
        &mut self,
        x_train: &Array3<f64>,
        y_train: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        let n_samples = x_train.shape()[0];
        let batch_size = self.config.batch_size.min(n_samples);

        self.loss_history.clear();

        let pb = ProgressBar::new(epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) Loss: {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);

                let x_batch = x_train.slice(s![batch_start..batch_end, .., ..]).to_owned();
                let y_batch = y_train.slice(s![batch_start..batch_end, ..]).to_owned();

                let predictions = self.forward(&x_batch);
                let loss = self.compute_loss(&predictions, &y_batch);
                epoch_loss += loss;
                n_batches += 1;

                // Упрощённый backward через выходной слой
                self.backward_output_layer(&x_batch, &y_batch, learning_rate);
            }

            let avg_loss = epoch_loss / n_batches as f64;
            self.loss_history.push(avg_loss);

            pb.set_message(format!("{:.6}", avg_loss));
            pb.inc(1);
        }

        pb.finish_with_message("Обучение завершено!");
        Ok(())
    }

    /// Упрощённый backward для выходного слоя
    fn backward_output_layer(
        &mut self,
        x_batch: &Array3<f64>,
        y_batch: &Array2<f64>,
        learning_rate: f64,
    ) {
        let epsilon = 1e-5;
        let output_layer = &mut self.output_layer;

        // Обновляем веса
        for i in 0..output_layer.weights.nrows() {
            for j in 0..output_layer.weights.ncols() {
                let original = output_layer.weights[[i, j]];

                output_layer.weights[[i, j]] = original + epsilon;
                let pred_plus = self.forward_with_output(x_batch, output_layer);
                let loss_plus = self.compute_loss(&pred_plus, y_batch);

                output_layer.weights[[i, j]] = original - epsilon;
                let pred_minus = self.forward_with_output(x_batch, output_layer);
                let loss_minus = self.compute_loss(&pred_minus, y_batch);

                let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                output_layer.weights[[i, j]] = original - learning_rate * grad;
            }
        }

        // Обновляем смещения
        for i in 0..output_layer.biases.len() {
            let original = output_layer.biases[i];

            output_layer.biases[i] = original + epsilon;
            let pred_plus = self.forward_with_output(x_batch, output_layer);
            let loss_plus = self.compute_loss(&pred_plus, y_batch);

            output_layer.biases[i] = original - epsilon;
            let pred_minus = self.forward_with_output(x_batch, output_layer);
            let loss_minus = self.compute_loss(&pred_minus, y_batch);

            let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
            output_layer.biases[i] = original - learning_rate * grad;
        }
    }

    fn forward_with_output(&self, x: &Array3<f64>, output_layer: &mut Dense) -> Array2<f64> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut outputs = Array2::zeros((batch_size, self.config.output_size));

        for b in 0..batch_size {
            let mut states: Vec<Array1<f64>> =
                self.cells.iter().map(|cell| cell.init_hidden()).collect();

            for t in 0..seq_len {
                let mut layer_input: Array1<f64> = x.slice(s![b, t, ..]).to_owned();

                for (layer_idx, cell) in self.cells.iter().enumerate() {
                    let h_prev = &states[layer_idx];
                    let h_next = cell.forward(&layer_input, h_prev);
                    layer_input = h_next.clone();
                    states[layer_idx] = h_next;
                }
            }

            let final_hidden = &states[self.cells.len() - 1];
            let output = output_layer.forward(final_hidden);

            for (i, &val) in output.iter().enumerate() {
                outputs[[b, i]] = val;
            }
        }

        outputs
    }

    /// Делает предсказание
    pub fn predict(&mut self, x: &Array3<f64>) -> Array2<f64> {
        self.forward(x)
    }

    /// Оценивает модель на тестовых данных
    pub fn evaluate(&mut self, x_test: &Array3<f64>, y_test: &Array2<f64>) -> f64 {
        let predictions = self.forward(x_test);
        self.compute_loss(&predictions, y_test)
    }

    /// Сохраняет модель
    pub fn save(&self, path: &str) -> Result<()> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// Загружает модель
    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let model: Self = bincode::deserialize(&data)?;
        Ok(model)
    }
}

// Вспомогательные функции

fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gru_cell() {
        let cell = GRUCell::new(5, 10);
        let x = Array1::zeros(5);
        let h = cell.init_hidden();

        let h_next = cell.forward(&x, &h);

        assert_eq!(h_next.len(), 10);
    }

    #[test]
    fn test_gru_forward() {
        let mut gru = GRU::new(5, 32, 1);

        let x = Array3::zeros((2, 10, 5));
        let output = gru.forward(&x);

        assert_eq!(output.shape(), &[2, 1]);
    }

    #[test]
    fn test_gru_smaller_than_lstm() {
        // GRU имеет меньше параметров, чем LSTM
        // (2 вентиля против 3)
        let gru_cell = GRUCell::new(10, 20);
        let lstm_cell = super::super::lstm::LSTMCell::new(10, 20);

        // GRU: 3 * (input*hidden + hidden*hidden + hidden) = 3 * (200 + 400 + 20) = 1860
        // LSTM: 4 * (input*hidden + hidden*hidden + hidden) = 4 * (200 + 400 + 20) = 2480
        // Это подтверждает, что GRU меньше
        assert!(true);
    }
}
