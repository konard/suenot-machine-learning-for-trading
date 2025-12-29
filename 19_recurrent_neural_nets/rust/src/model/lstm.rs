//! Реализация LSTM (Long Short-Term Memory)

use super::config::LSTMConfig;
use super::layers::{Activation, Dense};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array1, Array2, Array3};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

/// LSTM ячейка
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMCell {
    /// Размер входа
    pub input_size: usize,
    /// Размер скрытого состояния
    pub hidden_size: usize,

    // Веса для входного вентиля (input gate)
    w_ii: Array2<f64>, // input -> input gate
    w_hi: Array2<f64>, // hidden -> input gate
    b_i: Array1<f64>,  // bias input gate

    // Веса для вентиля забывания (forget gate)
    w_if: Array2<f64>, // input -> forget gate
    w_hf: Array2<f64>, // hidden -> forget gate
    b_f: Array1<f64>,  // bias forget gate

    // Веса для кандидата ячейки (cell candidate)
    w_ig: Array2<f64>, // input -> cell candidate
    w_hg: Array2<f64>, // hidden -> cell candidate
    b_g: Array1<f64>,  // bias cell candidate

    // Веса для выходного вентиля (output gate)
    w_io: Array2<f64>, // input -> output gate
    w_ho: Array2<f64>, // hidden -> output gate
    b_o: Array1<f64>,  // bias output gate

    // Градиенты (для обучения)
    #[serde(skip)]
    gradients: Option<LSTMGradients>,
}

#[derive(Debug, Clone)]
struct LSTMGradients {
    dw_ii: Array2<f64>,
    dw_hi: Array2<f64>,
    db_i: Array1<f64>,
    dw_if: Array2<f64>,
    dw_hf: Array2<f64>,
    db_f: Array1<f64>,
    dw_ig: Array2<f64>,
    dw_hg: Array2<f64>,
    db_g: Array1<f64>,
    dw_io: Array2<f64>,
    dw_ho: Array2<f64>,
    db_o: Array1<f64>,
}

impl LSTMCell {
    /// Создаёт новую LSTM ячейку
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let limit = (1.0 / hidden_size as f64).sqrt();

        Self {
            input_size,
            hidden_size,
            // Input gate
            w_ii: Array2::random((hidden_size, input_size), Uniform::new(-limit, limit)),
            w_hi: Array2::random((hidden_size, hidden_size), Uniform::new(-limit, limit)),
            b_i: Array1::zeros(hidden_size),
            // Forget gate
            w_if: Array2::random((hidden_size, input_size), Uniform::new(-limit, limit)),
            w_hf: Array2::random((hidden_size, hidden_size), Uniform::new(-limit, limit)),
            b_f: Array1::from_elem(hidden_size, 1.0), // Инициализация 1 для забывания
            // Cell candidate
            w_ig: Array2::random((hidden_size, input_size), Uniform::new(-limit, limit)),
            w_hg: Array2::random((hidden_size, hidden_size), Uniform::new(-limit, limit)),
            b_g: Array1::zeros(hidden_size),
            // Output gate
            w_io: Array2::random((hidden_size, input_size), Uniform::new(-limit, limit)),
            w_ho: Array2::random((hidden_size, hidden_size), Uniform::new(-limit, limit)),
            b_o: Array1::zeros(hidden_size),
            gradients: None,
        }
    }

    /// Прямой проход для одного временного шага
    ///
    /// # Аргументы
    ///
    /// * `x` - Входной вектор [input_size]
    /// * `h_prev` - Предыдущее скрытое состояние [hidden_size]
    /// * `c_prev` - Предыдущее состояние ячейки [hidden_size]
    ///
    /// # Возвращает
    ///
    /// (h_next, c_next) - Новые скрытое состояние и состояние ячейки
    pub fn forward(
        &self,
        x: &Array1<f64>,
        h_prev: &Array1<f64>,
        c_prev: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        // Input gate: i = σ(W_ii * x + W_hi * h + b_i)
        let i_gate = sigmoid(&(self.w_ii.dot(x) + self.w_hi.dot(h_prev) + &self.b_i));

        // Forget gate: f = σ(W_if * x + W_hf * h + b_f)
        let f_gate = sigmoid(&(self.w_if.dot(x) + self.w_hf.dot(h_prev) + &self.b_f));

        // Cell candidate: g = tanh(W_ig * x + W_hg * h + b_g)
        let g = tanh(&(self.w_ig.dot(x) + self.w_hg.dot(h_prev) + &self.b_g));

        // Output gate: o = σ(W_io * x + W_ho * h + b_o)
        let o_gate = sigmoid(&(self.w_io.dot(x) + self.w_ho.dot(h_prev) + &self.b_o));

        // New cell state: c = f * c_prev + i * g
        let c_next = &f_gate * c_prev + &i_gate * &g;

        // New hidden state: h = o * tanh(c)
        let h_next = &o_gate * &tanh(&c_next);

        (h_next, c_next)
    }

    /// Инициализирует скрытое состояние нулями
    pub fn init_hidden(&self) -> (Array1<f64>, Array1<f64>) {
        (
            Array1::zeros(self.hidden_size),
            Array1::zeros(self.hidden_size),
        )
    }
}

/// LSTM модель для прогнозирования временных рядов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTM {
    /// Конфигурация модели
    pub config: LSTMConfig,
    /// LSTM ячейки (по одной на слой)
    cells: Vec<LSTMCell>,
    /// Выходной слой
    output_layer: Dense,
    /// История потерь при обучении
    #[serde(skip)]
    pub loss_history: Vec<f64>,
}

impl LSTM {
    /// Создаёт новую LSTM модель
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

        // Первый слой принимает входные данные
        cells.push(LSTMCell::new(config.input_size, config.hidden_size));

        // Последующие слои принимают выход предыдущего слоя
        for _ in 1..config.num_layers {
            cells.push(LSTMCell::new(config.hidden_size, config.hidden_size));
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
    ///
    /// # Аргументы
    ///
    /// * `x` - Входная последовательность [batch_size, seq_len, input_size]
    ///
    /// # Возвращает
    ///
    /// Выходные значения [batch_size, output_size]
    pub fn forward(&mut self, x: &Array3<f64>) -> Array2<f64> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut outputs = Array2::zeros((batch_size, self.config.output_size));

        for b in 0..batch_size {
            // Инициализируем скрытые состояния для каждого слоя
            let mut states: Vec<(Array1<f64>, Array1<f64>)> =
                self.cells.iter().map(|cell| cell.init_hidden()).collect();

            // Проходим по последовательности
            for t in 0..seq_len {
                let mut layer_input: Array1<f64> = x.slice(s![b, t, ..]).to_owned();

                // Проходим через все LSTM слои
                for (layer_idx, cell) in self.cells.iter().enumerate() {
                    let (h_prev, c_prev) = &states[layer_idx];
                    let (h_next, c_next) = cell.forward(&layer_input, h_prev, c_prev);

                    layer_input = h_next.clone();
                    states[layer_idx] = (h_next, c_next);
                }
            }

            // Берём последнее скрытое состояние последнего слоя
            let final_hidden = &states[self.cells.len() - 1].0;

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
    ///
    /// # Аргументы
    ///
    /// * `x_train` - Обучающие данные [samples, seq_len, features]
    /// * `y_train` - Целевые значения [samples, output_size]
    /// * `epochs` - Количество эпох
    /// * `learning_rate` - Скорость обучения
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

            // Простой SGD по батчам
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);

                let x_batch = x_train.slice(s![batch_start..batch_end, .., ..]).to_owned();
                let y_batch = y_train.slice(s![batch_start..batch_end, ..]).to_owned();

                // Forward pass
                let predictions = self.forward(&x_batch);

                // Compute loss
                let loss = self.compute_loss(&predictions, &y_batch);
                epoch_loss += loss;
                n_batches += 1;

                // Backward pass (упрощённый вариант через конечные разности)
                self.backward_finite_diff(&x_batch, &y_batch, learning_rate);
            }

            let avg_loss = epoch_loss / n_batches as f64;
            self.loss_history.push(avg_loss);

            pb.set_message(format!("{:.6}", avg_loss));
            pb.inc(1);
        }

        pb.finish_with_message("Обучение завершено!");
        Ok(())
    }

    /// Упрощённый backward pass через конечные разности
    fn backward_finite_diff(
        &mut self,
        x_batch: &Array3<f64>,
        y_batch: &Array2<f64>,
        learning_rate: f64,
    ) {
        let epsilon = 1e-5;

        // Обновляем веса выходного слоя
        let output_layer = &mut self.output_layer;

        // Веса выходного слоя
        for i in 0..output_layer.weights.nrows() {
            for j in 0..output_layer.weights.ncols() {
                let original = output_layer.weights[[i, j]];

                // f(w + eps)
                output_layer.weights[[i, j]] = original + epsilon;
                let pred_plus = self.forward_with_output_layer(x_batch, output_layer);
                let loss_plus = self.compute_loss(&pred_plus, y_batch);

                // f(w - eps)
                output_layer.weights[[i, j]] = original - epsilon;
                let pred_minus = self.forward_with_output_layer(x_batch, output_layer);
                let loss_minus = self.compute_loss(&pred_minus, y_batch);

                // Gradient
                let grad = (loss_plus - loss_minus) / (2.0 * epsilon);

                // Update
                output_layer.weights[[i, j]] = original - learning_rate * grad;
            }
        }

        // Смещения выходного слоя
        for i in 0..output_layer.biases.len() {
            let original = output_layer.biases[i];

            output_layer.biases[i] = original + epsilon;
            let pred_plus = self.forward_with_output_layer(x_batch, output_layer);
            let loss_plus = self.compute_loss(&pred_plus, y_batch);

            output_layer.biases[i] = original - epsilon;
            let pred_minus = self.forward_with_output_layer(x_batch, output_layer);
            let loss_minus = self.compute_loss(&pred_minus, y_batch);

            let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
            output_layer.biases[i] = original - learning_rate * grad;
        }
    }

    /// Forward с заданным output layer (для численного градиента)
    fn forward_with_output_layer(
        &self,
        x: &Array3<f64>,
        output_layer: &mut Dense,
    ) -> Array2<f64> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut outputs = Array2::zeros((batch_size, self.config.output_size));

        for b in 0..batch_size {
            let mut states: Vec<(Array1<f64>, Array1<f64>)> =
                self.cells.iter().map(|cell| cell.init_hidden()).collect();

            for t in 0..seq_len {
                let mut layer_input: Array1<f64> = x.slice(s![b, t, ..]).to_owned();

                for (layer_idx, cell) in self.cells.iter().enumerate() {
                    let (h_prev, c_prev) = &states[layer_idx];
                    let (h_next, c_next) = cell.forward(&layer_input, h_prev, c_prev);

                    layer_input = h_next.clone();
                    states[layer_idx] = (h_next, c_next);
                }
            }

            let final_hidden = &states[self.cells.len() - 1].0;
            let output = output_layer.forward(final_hidden);

            for (i, &val) in output.iter().enumerate() {
                outputs[[b, i]] = val;
            }
        }

        outputs
    }

    /// Делает предсказание для одной последовательности
    pub fn predict(&mut self, x: &Array3<f64>) -> Array2<f64> {
        self.forward(x)
    }

    /// Оценивает модель на тестовых данных
    pub fn evaluate(&mut self, x_test: &Array3<f64>, y_test: &Array2<f64>) -> f64 {
        let predictions = self.forward(x_test);
        self.compute_loss(&predictions, y_test)
    }

    /// Сохраняет модель в файл
    pub fn save(&self, path: &str) -> Result<()> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// Загружает модель из файла
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
    fn test_lstm_cell() {
        let cell = LSTMCell::new(5, 10);
        let x = Array1::zeros(5);
        let (h, c) = cell.init_hidden();

        let (h_next, c_next) = cell.forward(&x, &h, &c);

        assert_eq!(h_next.len(), 10);
        assert_eq!(c_next.len(), 10);
    }

    #[test]
    fn test_lstm_forward() {
        let mut lstm = LSTM::new(5, 32, 1);

        let x = Array3::zeros((2, 10, 5)); // batch=2, seq=10, features=5
        let output = lstm.forward(&x);

        assert_eq!(output.shape(), &[2, 1]);
    }

    #[test]
    fn test_lstm_from_config() {
        let config = LSTMConfig::new(5, 64, 1).with_layers(2);
        let lstm = LSTM::from_config(config);

        assert_eq!(lstm.cells.len(), 2);
    }
}
