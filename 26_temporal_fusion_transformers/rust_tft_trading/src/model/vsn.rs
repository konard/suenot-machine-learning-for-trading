//! Variable Selection Network (VSN)
//!
//! Автоматически определяет важность переменных и создает
//! взвешенную комбинацию признаков.

use super::GatedResidualNetwork;
use ndarray::{Array1, Array2, Axis};

/// Variable Selection Network
///
/// Выбирает важные переменные и создает взвешенное представление.
#[derive(Debug)]
pub struct VariableSelectionNetwork {
    /// Количество входных переменных
    pub num_inputs: usize,

    /// Размер каждой переменной
    pub input_size: usize,

    /// Размер скрытого слоя
    pub hidden_size: usize,

    /// Размер контекста
    pub context_size: Option<usize>,

    /// GRN для каждой переменной
    variable_grns: Vec<GatedResidualNetwork>,

    /// GRN для выбора переменных (softmax weights)
    selection_grn: GatedResidualNetwork,

    /// Веса для финального объединения
    output_grn: GatedResidualNetwork,
}

impl VariableSelectionNetwork {
    /// Создает новый VSN
    pub fn new(
        num_inputs: usize,
        input_size: usize,
        hidden_size: usize,
        context_size: Option<usize>,
        dropout: f64,
    ) -> Self {
        // GRN для каждой переменной
        let variable_grns: Vec<_> = (0..num_inputs)
            .map(|_| GatedResidualNetwork::new(input_size, hidden_size, hidden_size, None, dropout))
            .collect();

        // GRN для выбора переменных
        // Вход: конкатенация всех переменных + опциональный контекст
        let flattened_size = num_inputs * input_size;
        let selection_grn = GatedResidualNetwork::new(
            flattened_size,
            hidden_size,
            num_inputs,
            context_size,
            dropout,
        );

        // GRN для финального выхода
        let output_grn =
            GatedResidualNetwork::new(hidden_size, hidden_size, hidden_size, context_size, dropout);

        Self {
            num_inputs,
            input_size,
            hidden_size,
            context_size,
            variable_grns,
            selection_grn,
            output_grn,
        }
    }

    /// Forward pass
    ///
    /// # Аргументы
    /// * `inputs` - Входные переменные shape (num_inputs, input_size)
    /// * `context` - Опциональный контекст
    ///
    /// # Возвращает
    /// * (output, weights) - Взвешенный выход и веса важности
    pub fn forward(
        &self,
        inputs: &Array2<f64>,
        context: Option<&Array1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        // Проверяем размеры
        assert_eq!(inputs.nrows(), self.num_inputs);
        assert_eq!(inputs.ncols(), self.input_size);

        // Обрабатываем каждую переменную через свой GRN
        let mut processed_vars: Vec<Array1<f64>> = Vec::with_capacity(self.num_inputs);
        for (i, grn) in self.variable_grns.iter().enumerate() {
            let var_input = inputs.row(i).to_owned();
            let processed = grn.forward(&var_input, None);
            processed_vars.push(processed);
        }

        // Конкатенируем все входы для selection
        let flattened: Vec<f64> = inputs.iter().copied().collect();
        let flattened = Array1::from_vec(flattened);

        // Получаем веса выбора через softmax
        let selection_logits = self.selection_grn.forward(&flattened, context);
        let weights = Self::softmax(&selection_logits);

        // Взвешенная сумма обработанных переменных
        let mut weighted_sum = Array1::zeros(self.hidden_size);
        for (i, var) in processed_vars.iter().enumerate() {
            weighted_sum = weighted_sum + var * weights[i];
        }

        // Финальная обработка
        let output = self.output_grn.forward(&weighted_sum, context);

        (output, weights)
    }

    /// Forward pass для временного ряда (все временные шаги)
    ///
    /// # Аргументы
    /// * `inputs` - Shape (time_steps, num_inputs, input_size)
    /// * `context` - Опциональный контекст
    ///
    /// # Возвращает
    /// * (outputs, weights) - Shape (time_steps, hidden_size) и средние веса
    pub fn forward_temporal(
        &self,
        inputs: &[Array2<f64>],
        context: Option<&Array1<f64>>,
    ) -> (Array2<f64>, Array1<f64>) {
        let time_steps = inputs.len();
        let mut outputs = Array2::zeros((time_steps, self.hidden_size));
        let mut all_weights = Array2::zeros((time_steps, self.num_inputs));

        for (t, input) in inputs.iter().enumerate() {
            let (output, weights) = self.forward(input, context);
            outputs.row_mut(t).assign(&output);
            all_weights.row_mut(t).assign(&weights);
        }

        // Средние веса по времени
        let mean_weights = all_weights.mean_axis(Axis(0)).unwrap();

        (outputs, mean_weights)
    }

    /// Softmax функция
    fn softmax(x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Array1<f64> = x.mapv(|v| (v - max_val).exp());
        let sum: f64 = exp_vals.sum();
        exp_vals / sum
    }

    /// Возвращает количество параметров
    pub fn num_parameters(&self) -> usize {
        let var_params: usize = self.variable_grns.iter().map(|g| g.num_parameters()).sum();
        var_params + self.selection_grn.num_parameters() + self.output_grn.num_parameters()
    }
}

/// Интерпретация весов VSN
#[derive(Debug, Clone)]
pub struct VariableImportance {
    /// Имена переменных
    pub names: Vec<String>,

    /// Веса важности
    pub weights: Vec<f64>,
}

impl VariableImportance {
    /// Создает из весов и имен
    pub fn new(names: Vec<String>, weights: Array1<f64>) -> Self {
        Self {
            names,
            weights: weights.to_vec(),
        }
    }

    /// Сортирует по важности (убывание)
    pub fn sorted(&self) -> Vec<(String, f64)> {
        let mut pairs: Vec<_> = self
            .names
            .iter()
            .zip(self.weights.iter())
            .map(|(n, w)| (n.clone(), *w))
            .collect();

        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs
    }

    /// Возвращает топ-N важных переменных
    pub fn top_n(&self, n: usize) -> Vec<(String, f64)> {
        self.sorted().into_iter().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsn_forward() {
        let vsn = VariableSelectionNetwork::new(4, 8, 16, None, 0.1);

        let inputs = Array2::from_elem((4, 8), 0.5);
        let (output, weights) = vsn.forward(&inputs, None);

        assert_eq!(output.len(), 16);
        assert_eq!(weights.len(), 4);

        // Веса должны суммироваться в 1
        let sum: f64 = weights.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vsn_with_context() {
        let vsn = VariableSelectionNetwork::new(4, 8, 16, Some(10), 0.1);

        let inputs = Array2::from_elem((4, 8), 0.5);
        let context = Array1::from_elem(10, 0.3);
        let (output, weights) = vsn.forward(&inputs, Some(&context));

        assert_eq!(output.len(), 16);
        assert_eq!(weights.len(), 4);
    }

    #[test]
    fn test_vsn_temporal() {
        let vsn = VariableSelectionNetwork::new(4, 8, 16, None, 0.1);

        let inputs: Vec<Array2<f64>> = (0..10)
            .map(|_| Array2::from_elem((4, 8), 0.5))
            .collect();

        let (outputs, weights) = vsn.forward_temporal(&inputs, None);

        assert_eq!(outputs.shape(), &[10, 16]);
        assert_eq!(weights.len(), 4);
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = VariableSelectionNetwork::softmax(&x);

        // Сумма должна быть 1
        assert!((result.sum() - 1.0).abs() < 1e-6);

        // Большие значения должны иметь большие вероятности
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_variable_importance() {
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let weights = Array1::from_vec(vec![0.1, 0.5, 0.4]);

        let importance = VariableImportance::new(names, weights);
        let sorted = importance.sorted();

        assert_eq!(sorted[0].0, "b");
        assert_eq!(sorted[1].0, "c");
        assert_eq!(sorted[2].0, "a");
    }
}
