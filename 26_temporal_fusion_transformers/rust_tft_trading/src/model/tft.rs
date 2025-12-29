//! Temporal Fusion Transformer Model
//!
//! Полная реализация TFT для многогоризонтного прогнозирования.

use super::{
    Attention, GatedResidualNetwork, InterpretableMultiHeadAttention, QuantileLoss,
    QuantilePrediction, TFTConfig, VariableSelectionNetwork,
};
use crate::data::TFTSample;
use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Temporal Fusion Transformer Model
#[derive(Debug)]
pub struct TFTModel {
    /// Конфигурация модели
    pub config: TFTConfig,

    /// Variable Selection для encoder
    encoder_vsn: VariableSelectionNetwork,

    /// Variable Selection для decoder
    decoder_vsn: VariableSelectionNetwork,

    /// GRN для статических признаков -> context
    static_context_grn: Option<GatedResidualNetwork>,

    /// LSTM encoder веса (упрощенная версия)
    lstm_encoder: SimpleLSTM,

    /// LSTM decoder веса
    lstm_decoder: SimpleLSTM,

    /// Temporal Self-Attention
    attention: InterpretableMultiHeadAttention,

    /// Output GRN
    output_grn: GatedResidualNetwork,

    /// Quantile output weights
    quantile_weights: Array2<f64>,
    quantile_bias: Array1<f64>,

    /// Loss function
    loss_fn: QuantileLoss,

    /// Сохраненные веса важности переменных (для интерпретации)
    last_encoder_importance: Option<Array1<f64>>,
    last_decoder_importance: Option<Array1<f64>>,
}

impl TFTModel {
    /// Создает новую модель TFT
    pub fn new(config: TFTConfig) -> Self {
        // Validate config
        config.validate().expect("Invalid TFT configuration");

        let hidden = config.hidden_size;
        let dropout = config.dropout;

        // Variable Selection Networks
        let encoder_vsn = VariableSelectionNetwork::new(
            config.num_encoder_features,
            1, // каждый признак как скаляр
            hidden,
            None,
            dropout,
        );

        let decoder_vsn = VariableSelectionNetwork::new(
            config.num_decoder_features,
            1,
            hidden,
            None,
            dropout,
        );

        // Static context GRN
        let static_context_grn = if config.num_static_features > 0 {
            Some(GatedResidualNetwork::new(
                config.num_static_features,
                hidden,
                hidden,
                None,
                dropout,
            ))
        } else {
            None
        };

        // LSTM layers
        let lstm_encoder = SimpleLSTM::new(hidden, hidden);
        let lstm_decoder = SimpleLSTM::new(hidden, hidden);

        // Temporal Attention
        let attention =
            InterpretableMultiHeadAttention::new(hidden, config.num_attention_heads, dropout);

        // Output layers
        let output_grn = GatedResidualNetwork::new(hidden, hidden, hidden, None, dropout);

        let num_quantiles = config.quantiles.len();
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (hidden + num_quantiles) as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();

        let quantile_weights =
            Array2::from_shape_fn((num_quantiles, hidden), |_| rng.sample(normal));
        let quantile_bias = Array1::zeros(num_quantiles);

        let loss_fn = QuantileLoss::with_quantiles(config.quantiles.clone());

        Self {
            config,
            encoder_vsn,
            decoder_vsn,
            static_context_grn,
            lstm_encoder,
            lstm_decoder,
            attention,
            output_grn,
            quantile_weights,
            quantile_bias,
            loss_fn,
            last_encoder_importance: None,
            last_decoder_importance: None,
        }
    }

    /// Forward pass для одного sample
    pub fn forward(&mut self, sample: &TFTSample) -> QuantilePrediction {
        let encoder_len = self.config.encoder_length;
        let pred_len = self.config.prediction_length;
        let hidden = self.config.hidden_size;

        // Обрабатываем encoder input через VSN
        let mut encoder_outputs = Array2::zeros((encoder_len, hidden));
        let mut total_encoder_weights = Array1::zeros(self.config.num_encoder_features);

        for t in 0..encoder_len {
            // Преобразуем строку в матрицу (num_features, 1)
            let row = sample.encoder_input.row(t);
            let inputs = Array2::from_shape_fn((row.len(), 1), |(i, _)| row[i]);

            let (output, weights) = self.encoder_vsn.forward(&inputs, None);
            encoder_outputs.row_mut(t).assign(&output);
            total_encoder_weights = &total_encoder_weights + &weights;
        }

        // Сохраняем средние веса для интерпретации
        self.last_encoder_importance = Some(&total_encoder_weights / encoder_len as f64);

        // LSTM encoder
        let (encoder_hidden, encoder_cell) = self.lstm_encoder.forward_sequence(&encoder_outputs);

        // Обрабатываем decoder input через VSN
        let mut decoder_outputs = Array2::zeros((pred_len, hidden));
        let mut total_decoder_weights = Array1::zeros(self.config.num_decoder_features);

        for t in 0..pred_len {
            let row = sample.decoder_input.row(t);
            let inputs = Array2::from_shape_fn((row.len(), 1), |(i, _)| row[i]);

            let (output, weights) = self.decoder_vsn.forward(&inputs, None);
            decoder_outputs.row_mut(t).assign(&output);
            total_decoder_weights = &total_decoder_weights + &weights;
        }

        self.last_decoder_importance = Some(&total_decoder_weights / pred_len as f64);

        // LSTM decoder с начальным состоянием от encoder
        let (decoder_hidden, _) =
            self.lstm_decoder
                .forward_sequence_with_state(&decoder_outputs, &encoder_hidden, &encoder_cell);

        // Temporal Self-Attention
        // Объединяем encoder и decoder hidden states
        let total_len = encoder_len + pred_len;
        let mut combined = Array2::zeros((total_len, hidden));
        for t in 0..encoder_len {
            combined.row_mut(t).assign(&encoder_hidden.row(t));
        }
        for t in 0..pred_len {
            combined
                .row_mut(encoder_len + t)
                .assign(&decoder_hidden.row(t));
        }

        // Causal mask для attention
        let mask = Attention::create_causal_mask(total_len);

        // Self-attention
        let attended = self.attention.forward(&combined, &combined, &combined, Some(&mask));

        // Берем только decoder часть для прогноза
        let decoder_attended = attended.slice(ndarray::s![encoder_len.., ..]).to_owned();

        // Output через GRN и quantile projection
        let mut predictions = Array2::zeros((pred_len, self.config.quantiles.len()));

        for t in 0..pred_len {
            let h = decoder_attended.row(t).to_owned();
            let out = self.output_grn.forward(&h, None);

            // Quantile projection
            let quantiles = self.quantile_weights.dot(&out) + &self.quantile_bias;
            predictions.row_mut(t).assign(&quantiles);
        }

        QuantilePrediction::new(self.config.quantiles.clone(), predictions)
    }

    /// Predict для batch
    pub fn predict_batch(&mut self, samples: &[TFTSample]) -> Vec<QuantilePrediction> {
        samples.iter().map(|s| self.forward(s)).collect()
    }

    /// Вычисляет loss для одного sample
    pub fn compute_loss(&mut self, sample: &TFTSample) -> f64 {
        let prediction = self.forward(sample);
        self.loss_fn.loss_single(&sample.target, &prediction.values)
    }

    /// Возвращает важность encoder переменных
    pub fn get_encoder_importance(&self) -> Option<&Array1<f64>> {
        self.last_encoder_importance.as_ref()
    }

    /// Возвращает важность decoder переменных
    pub fn get_decoder_importance(&self) -> Option<&Array1<f64>> {
        self.last_decoder_importance.as_ref()
    }

    /// Возвращает attention weights
    pub fn get_attention_weights(&self) -> Option<&Array2<f64>> {
        self.attention.get_attention_weights()
    }

    /// Возвращает количество параметров
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.encoder_vsn.num_parameters();
        count += self.decoder_vsn.num_parameters();
        if let Some(grn) = &self.static_context_grn {
            count += grn.num_parameters();
        }
        count += self.lstm_encoder.num_parameters();
        count += self.lstm_decoder.num_parameters();
        // attention params
        count += self.output_grn.num_parameters();
        count += self.quantile_weights.len() + self.quantile_bias.len();
        count
    }
}

/// Упрощенная LSTM реализация
#[derive(Debug)]
pub struct SimpleLSTM {
    input_size: usize,
    hidden_size: usize,

    // Gates weights: [input_gate, forget_gate, cell_gate, output_gate]
    weight_ih: Array2<f64>, // (4*hidden, input)
    weight_hh: Array2<f64>, // (4*hidden, hidden)
    bias: Array1<f64>,      // (4*hidden)
}

impl SimpleLSTM {
    /// Создает новый LSTM слой
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / hidden_size as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();

        let weight_ih =
            Array2::from_shape_fn((4 * hidden_size, input_size), |_| rng.sample(normal));
        let weight_hh =
            Array2::from_shape_fn((4 * hidden_size, hidden_size), |_| rng.sample(normal));
        let bias = Array1::zeros(4 * hidden_size);

        Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias,
        }
    }

    /// Forward pass для последовательности
    pub fn forward_sequence(&self, inputs: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let seq_len = inputs.nrows();
        let h0 = Array1::zeros(self.hidden_size);
        let c0 = Array1::zeros(self.hidden_size);

        self.forward_with_state(inputs, &h0, &c0)
    }

    /// Forward с начальным состоянием
    pub fn forward_sequence_with_state(
        &self,
        inputs: &Array2<f64>,
        h0: &Array2<f64>,
        c0: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        // Берем последнее состояние из h0/c0
        let h = h0.row(h0.nrows() - 1).to_owned();
        let c = c0.row(c0.nrows() - 1).to_owned();

        self.forward_with_state(inputs, &h, &c)
    }

    fn forward_with_state(
        &self,
        inputs: &Array2<f64>,
        h0: &Array1<f64>,
        c0: &Array1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let seq_len = inputs.nrows();
        let h = self.hidden_size;

        let mut hidden_states = Array2::zeros((seq_len, h));
        let mut cell_states = Array2::zeros((seq_len, h));

        let mut ht = h0.clone();
        let mut ct = c0.clone();

        for t in 0..seq_len {
            let xt = inputs.row(t);

            // Gates = W_ih * x + W_hh * h + b
            let gates = self.weight_ih.dot(&xt.to_owned())
                + self.weight_hh.dot(&ht)
                + &self.bias;

            // Split into 4 gates
            let i = Self::sigmoid(&gates.slice(ndarray::s![0..h]).to_owned());
            let f = Self::sigmoid(&gates.slice(ndarray::s![h..2 * h]).to_owned());
            let g = Self::tanh(&gates.slice(ndarray::s![2 * h..3 * h]).to_owned());
            let o = Self::sigmoid(&gates.slice(ndarray::s![3 * h..4 * h]).to_owned());

            // Update cell state
            ct = &f * &ct + &i * &g;

            // Update hidden state
            ht = &o * &Self::tanh(&ct);

            hidden_states.row_mut(t).assign(&ht);
            cell_states.row_mut(t).assign(&ct);
        }

        (hidden_states, cell_states)
    }

    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn tanh(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.tanh())
    }

    /// Количество параметров
    pub fn num_parameters(&self) -> usize {
        self.weight_ih.len() + self.weight_hh.len() + self.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_sample(encoder_len: usize, pred_len: usize, num_enc: usize, num_dec: usize) -> TFTSample {
        TFTSample {
            encoder_input: Array2::from_elem((encoder_len, num_enc), 0.5),
            decoder_input: Array2::from_elem((pred_len, num_dec), 0.3),
            target: Array1::from_elem(pred_len, 0.1),
            static_features: Array1::zeros(0),
            timestamp_start: 0,
            timestamp_prediction: 1000,
        }
    }

    #[test]
    fn test_tft_forward() {
        let config = TFTConfig {
            hidden_size: 16,
            num_attention_heads: 2,
            encoder_length: 10,
            prediction_length: 5,
            num_encoder_features: 8,
            num_decoder_features: 4,
            quantiles: vec![0.1, 0.5, 0.9],
            ..Default::default()
        };

        let mut model = TFTModel::new(config);
        let sample = create_sample(10, 5, 8, 4);

        let prediction = model.forward(&sample);

        assert_eq!(prediction.values.shape(), &[5, 3]);
        assert_eq!(prediction.quantiles.len(), 3);
    }

    #[test]
    fn test_tft_interpretability() {
        let config = TFTConfig {
            hidden_size: 16,
            num_attention_heads: 2,
            encoder_length: 10,
            prediction_length: 5,
            num_encoder_features: 8,
            num_decoder_features: 4,
            ..Default::default()
        };

        let mut model = TFTModel::new(config);
        let sample = create_sample(10, 5, 8, 4);

        let _ = model.forward(&sample);

        // Должны быть доступны веса важности
        assert!(model.get_encoder_importance().is_some());
        assert!(model.get_decoder_importance().is_some());
        assert!(model.get_attention_weights().is_some());
    }

    #[test]
    fn test_simple_lstm() {
        let lstm = SimpleLSTM::new(16, 32);

        let inputs = Array2::from_elem((10, 16), 0.5);
        let (hidden, cell) = lstm.forward_sequence(&inputs);

        assert_eq!(hidden.shape(), &[10, 32]);
        assert_eq!(cell.shape(), &[10, 32]);
    }

    #[test]
    fn test_tft_loss() {
        let config = TFTConfig {
            hidden_size: 16,
            num_attention_heads: 2,
            encoder_length: 10,
            prediction_length: 5,
            num_encoder_features: 8,
            num_decoder_features: 4,
            ..Default::default()
        };

        let mut model = TFTModel::new(config);
        let sample = create_sample(10, 5, 8, 4);

        let loss = model.compute_loss(&sample);
        assert!(loss >= 0.0);
    }
}
