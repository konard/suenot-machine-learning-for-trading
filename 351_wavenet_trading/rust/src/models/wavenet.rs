//! WaveNet модель для прогнозирования временных рядов

use super::activations::gated_activation;
use super::layers::{Conv1D, Dense, DilatedConv1D};

/// Блок WaveNet с расширенными свёртками
#[derive(Debug, Clone)]
pub struct WaveNetBlock {
    pub filter_conv: DilatedConv1D,
    pub gate_conv: DilatedConv1D,
    pub residual_conv: Conv1D,
    pub skip_conv: Conv1D,
    pub dilation: usize,
}

impl WaveNetBlock {
    pub fn new(channels: usize, kernel_size: usize, dilation: usize) -> Self {
        Self {
            filter_conv: DilatedConv1D::new(channels, channels, kernel_size, dilation),
            gate_conv: DilatedConv1D::new(channels, channels, kernel_size, dilation),
            residual_conv: Conv1D::new(channels, channels, 1),
            skip_conv: Conv1D::new(channels, channels, 1),
            dilation,
        }
    }

    /// Прямой проход блока
    /// Возвращает (residual_output, skip_output)
    pub fn forward(&self, input: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Расширенные свёртки
        let filter = self.filter_conv.forward(input);
        let gate = self.gate_conv.forward(input);

        // Вентильная активация
        let activated: Vec<Vec<f64>> = filter
            .iter()
            .zip(gate.iter())
            .map(|(f, g)| {
                f.iter()
                    .zip(g.iter())
                    .map(|(&fv, &gv)| gated_activation(fv, gv))
                    .collect()
            })
            .collect();

        // 1x1 свёртки для residual и skip
        let residual_out = self.residual_conv.forward(&activated);
        let skip = self.skip_conv.forward(&activated);

        // Остаточное соединение
        let residual_with_input: Vec<Vec<f64>> = input
            .iter()
            .zip(residual_out.iter())
            .map(|(i, r)| i.iter().zip(r.iter()).map(|(a, b)| a + b).collect())
            .collect();

        (residual_with_input, skip)
    }
}

/// Конфигурация WaveNet
#[derive(Debug, Clone)]
pub struct WaveNetConfig {
    pub input_channels: usize,
    pub residual_channels: usize,
    pub skip_channels: usize,
    pub output_channels: usize,
    pub kernel_size: usize,
    pub num_blocks: usize,
    pub num_stacks: usize,
}

impl Default for WaveNetConfig {
    fn default() -> Self {
        Self {
            input_channels: 5,       // OHLCV
            residual_channels: 32,
            skip_channels: 32,
            output_channels: 1,      // Предсказание возврата
            kernel_size: 2,
            num_blocks: 10,          // 10 блоков = рецептивное поле 1024
            num_stacks: 1,
        }
    }
}

impl WaveNetConfig {
    /// Рассчитать рецептивное поле
    pub fn receptive_field(&self) -> usize {
        let dilations: usize = (0..self.num_blocks).map(|i| 2_usize.pow(i as u32)).sum();
        (self.kernel_size - 1) * dilations * self.num_stacks + 1
    }
}

/// WaveNet модель
#[derive(Debug)]
pub struct WaveNet {
    pub config: WaveNetConfig,
    pub input_conv: Conv1D,
    pub blocks: Vec<WaveNetBlock>,
    pub output_conv1: Conv1D,
    pub output_conv2: Conv1D,
    pub output_dense: Dense,
}

impl WaveNet {
    pub fn new(config: WaveNetConfig) -> Self {
        // Входной слой
        let input_conv = Conv1D::new(config.input_channels, config.residual_channels, 1);

        // Блоки WaveNet
        let mut blocks = Vec::new();
        for stack in 0..config.num_stacks {
            for block in 0..config.num_blocks {
                let dilation = 2_usize.pow(block as u32);
                blocks.push(WaveNetBlock::new(
                    config.residual_channels,
                    config.kernel_size,
                    dilation,
                ));
                let _ = stack; // suppress unused warning
            }
        }

        // Выходные слои
        let output_conv1 = Conv1D::new(config.skip_channels, config.skip_channels, 1);
        let output_conv2 = Conv1D::new(config.skip_channels, config.output_channels, 1);
        let output_dense = Dense::new(config.output_channels, config.output_channels);

        Self {
            config,
            input_conv,
            blocks,
            output_conv1,
            output_conv2,
            output_dense,
        }
    }

    /// Прямой проход модели
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<f64> {
        let seq_len = input[0].len();

        // Входная проекция
        let mut x = self.input_conv.forward(input);

        // Накопление skip-соединений
        let mut skip_sum = vec![vec![0.0; seq_len]; self.config.skip_channels];

        // Проход через все блоки
        for block in &self.blocks {
            let (residual, skip) = block.forward(&x);
            x = residual;

            // Суммируем skip-соединения
            for (ss, s) in skip_sum.iter_mut().zip(skip.iter()) {
                for (ssv, sv) in ss.iter_mut().zip(s.iter()) {
                    *ssv += sv;
                }
            }
        }

        // Применяем ReLU к сумме skip
        for ch in &mut skip_sum {
            for v in ch.iter_mut() {
                *v = v.max(0.0);
            }
        }

        // Выходные свёртки
        let y = self.output_conv1.forward(&skip_sum);
        let y: Vec<Vec<f64>> = y
            .iter()
            .map(|ch| ch.iter().map(|v| v.max(0.0)).collect())
            .collect();

        let y = self.output_conv2.forward(&y);

        // Берём последнее значение как прогноз
        let last_values: Vec<f64> = y.iter().map(|ch| *ch.last().unwrap_or(&0.0)).collect();

        self.output_dense.forward(&last_values)
    }

    /// Прогноз для одного временного окна
    pub fn predict(&self, window: &[Vec<f64>]) -> f64 {
        let output = self.forward(window);
        output.first().copied().unwrap_or(0.0)
    }

    /// Прогноз для всей последовательности (скользящее окно)
    pub fn predict_sequence(&self, data: &[Vec<f64>], window_size: usize) -> Vec<f64> {
        let seq_len = data[0].len();
        if seq_len < window_size {
            return Vec::new();
        }

        let mut predictions = Vec::new();
        for i in window_size..=seq_len {
            let window: Vec<Vec<f64>> = data
                .iter()
                .map(|ch| ch[i - window_size..i].to_vec())
                .collect();

            predictions.push(self.predict(&window));
        }

        predictions
    }

    /// Информация о модели
    pub fn summary(&self) {
        println!("=== WaveNet Model Summary ===");
        println!("Input channels:    {}", self.config.input_channels);
        println!("Residual channels: {}", self.config.residual_channels);
        println!("Skip channels:     {}", self.config.skip_channels);
        println!("Output channels:   {}", self.config.output_channels);
        println!("Kernel size:       {}", self.config.kernel_size);
        println!("Number of blocks:  {}", self.blocks.len());
        println!("Receptive field:   {}", self.config.receptive_field());
        println!("=============================");
    }
}

/// Простой WaveNet для демонстрации
pub struct SimpleWaveNet {
    pub dilated_convs: Vec<DilatedConv1D>,
    pub output_dense: Dense,
}

impl SimpleWaveNet {
    /// Создать простую модель с заданным количеством слоёв
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut dilated_convs = Vec::new();

        // Первый слой
        dilated_convs.push(DilatedConv1D::new(input_size, hidden_size, 2, 1));

        // Остальные слои с экспоненциально растущим dilation
        for i in 1..num_layers {
            let dilation = 2_usize.pow(i as u32);
            dilated_convs.push(DilatedConv1D::new(hidden_size, hidden_size, 2, dilation));
        }

        let output_dense = Dense::new(hidden_size, 1);

        Self {
            dilated_convs,
            output_dense,
        }
    }

    /// Рецептивное поле модели
    pub fn receptive_field(&self) -> usize {
        self.dilated_convs
            .iter()
            .map(|c| c.receptive_field())
            .sum::<usize>()
            - self.dilated_convs.len()
            + 1
    }

    /// Прямой проход
    pub fn forward(&self, input: &[Vec<f64>]) -> f64 {
        let mut x = input.to_vec();

        for conv in &self.dilated_convs {
            x = conv.forward(&x);
            // ReLU активация
            for ch in &mut x {
                for v in ch.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        }

        // Берём последнее значение каждого канала
        let last_values: Vec<f64> = x.iter().map(|ch| *ch.last().unwrap_or(&0.0)).collect();

        let output = self.output_dense.forward(&last_values);
        output.first().copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenet_block() {
        let block = WaveNetBlock::new(8, 2, 4);
        let input = vec![vec![1.0; 20]; 8];
        let (residual, skip) = block.forward(&input);

        assert_eq!(residual.len(), 8);
        assert_eq!(skip.len(), 8);
        assert_eq!(residual[0].len(), 20);
    }

    #[test]
    fn test_wavenet_receptive_field() {
        let config = WaveNetConfig {
            num_blocks: 10,
            kernel_size: 2,
            num_stacks: 1,
            ..Default::default()
        };

        // С 10 блоками и ядром 2: (2-1) * (1+2+4+...+512) = 1023
        assert_eq!(config.receptive_field(), 1024);
    }

    #[test]
    fn test_wavenet_forward() {
        let config = WaveNetConfig {
            input_channels: 5,
            residual_channels: 8,
            skip_channels: 8,
            output_channels: 1,
            kernel_size: 2,
            num_blocks: 4,
            num_stacks: 1,
        };

        let model = WaveNet::new(config);
        let input = vec![vec![1.0; 50]; 5];
        let output = model.forward(&input);

        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_simple_wavenet() {
        let model = SimpleWaveNet::new(3, 16, 5);
        let input = vec![vec![1.0; 100]; 3];
        let output = model.forward(&input);

        // Должно вернуть одно число
        assert!(output.is_finite());
    }
}
