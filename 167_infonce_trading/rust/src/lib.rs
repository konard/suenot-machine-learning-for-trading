use ndarray::{Array1, Array2};

/// Computes the L2 norm of a 1D array.
fn l2_normalize(v: &Array1<f32>) -> Array1<f32> {
    let norm = v.dot(v).sqrt().max(1e-8);
    v / norm
}

/// Normalizes each row of a 2D array to unit length.
fn row_normalize(mat: &Array2<f32>) -> Array2<f32> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let mut result = Array2::zeros((nrows, ncols));
    for i in 0..nrows {
        let row = mat.row(i).to_owned();
        let normed = l2_normalize(&row);
        result.row_mut(i).assign(&normed);
    }
    result
}

/// Computes InfoNCE loss for a batch of anchor-positive pairs using in-batch negatives.
///
/// # Arguments
/// * `anchors` - (batch_size, embed_dim) anchor embeddings
/// * `positives` - (batch_size, embed_dim) positive embeddings
/// * `temperature` - temperature scaling parameter
///
/// # Returns
/// Mean InfoNCE loss (scalar)
pub fn infonce_loss(
    anchors: &Array2<f32>,
    positives: &Array2<f32>,
    temperature: f32,
) -> f32 {
    let batch_size = anchors.nrows();
    assert_eq!(anchors.nrows(), positives.nrows());
    assert_eq!(anchors.ncols(), positives.ncols());

    // L2 normalize
    let a_norm = row_normalize(anchors);
    let p_norm = row_normalize(positives);

    // Similarity matrix: (batch_size, batch_size)
    // sim[i][j] = dot(anchor_i, positive_j) / temperature
    let sim_matrix = a_norm.dot(&p_norm.t()) / temperature;

    // For each row i, label is i (diagonal is the positive)
    let mut total_loss = 0.0f32;

    for i in 0..batch_size {
        let row = sim_matrix.row(i);

        // LogSumExp for numerical stability
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logsumexp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;

        // Loss for sample i: -log(exp(sim[i][i]) / sum_j exp(sim[i][j]))
        //                   = -sim[i][i] + logsumexp
        let loss_i = -row[i] + logsumexp;
        total_loss += loss_i;
    }

    total_loss / batch_size as f32
}

/// Computes InfoNCE loss and accuracy.
///
/// Accuracy = fraction of samples where argmax(sim[i]) == i.
pub fn infonce_loss_with_accuracy(
    anchors: &Array2<f32>,
    positives: &Array2<f32>,
    temperature: f32,
) -> (f32, f32) {
    let batch_size = anchors.nrows();

    let a_norm = row_normalize(anchors);
    let p_norm = row_normalize(positives);

    let sim_matrix = a_norm.dot(&p_norm.t()) / temperature;

    let mut total_loss = 0.0f32;
    let mut correct = 0usize;

    for i in 0..batch_size {
        let row = sim_matrix.row(i);

        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logsumexp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;

        total_loss += -row[i] + logsumexp;

        // Accuracy: check if the positive (index i) has the highest score
        let pred = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if pred == i {
            correct += 1;
        }
    }

    let loss = total_loss / batch_size as f32;
    let accuracy = correct as f32 / batch_size as f32;

    (loss, accuracy)
}

/// Scores similarity between a query embedding and a set of candidate embeddings.
/// Returns softmax probabilities (useful for inference / nearest-neighbor retrieval).
pub fn score_candidates(
    query: &Array1<f32>,
    candidates: &Array2<f32>,
    temperature: f32,
) -> Array1<f32> {
    let q_norm = l2_normalize(query);
    let c_norm = row_normalize(candidates);

    // Similarities: (num_candidates,)
    let sims = c_norm.dot(&q_norm) / temperature;

    // Softmax
    let max_val = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps = sims.mapv(|x| (x - max_val).exp());
    let sum_exp: f32 = exps.sum();

    exps / sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_l2_normalize() {
        let v = array![3.0, 4.0];
        let normed = l2_normalize(&v);
        let norm = normed.dot(&normed).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Normalized vector should have unit norm");
    }

    #[test]
    fn test_infonce_identical_pairs() {
        // When anchors == positives, the diagonal has highest similarity
        // so the loss should be relatively low
        let embeddings = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let loss = infonce_loss(&embeddings, &embeddings, 0.1);
        // With orthogonal embeddings as both anchor and positive,
        // the diagonal similarity = 1.0 / 0.1 = 10.0 and off-diag = 0.0
        // loss ≈ -10.0 + log(exp(10.0) + exp(0.0) + exp(0.0)) ≈ 0.0
        assert!(loss < 0.1, "Loss should be near zero for identical orthogonal pairs, got {}", loss);
    }

    #[test]
    fn test_infonce_accuracy() {
        let anchors = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ];
        let positives = array![
            [0.9, 0.1],
            [0.1, 0.9],
            [-0.9, 0.1],
            [0.1, -0.9],
        ];
        let (loss, acc) = infonce_loss_with_accuracy(&anchors, &positives, 0.1);
        assert!(acc > 0.5, "Accuracy should be > 50% for well-separated pairs, got {}", acc);
        println!("InfoNCE Loss: {:.4}, Accuracy: {:.2}%", loss, acc * 100.0);
    }

    #[test]
    fn test_score_candidates() {
        let query = array![1.0, 0.0, 0.0];
        let candidates = array![
            [1.0, 0.0, 0.0],   // most similar
            [0.0, 1.0, 0.0],   // orthogonal
            [-1.0, 0.0, 0.0],  // opposite
        ];
        let probs = score_candidates(&query, &candidates, 0.1);
        assert!(probs[0] > probs[1], "First candidate should have highest probability");
        assert!(probs[1] > probs[2], "Orthogonal should rank above opposite");
        println!("Candidate probabilities: {:?}", probs);
    }

    #[test]
    fn test_batch_size_one() {
        // Edge case: batch size of 1
        let anchors = array![[1.0, 2.0, 3.0]];
        let positives = array![[1.0, 2.0, 3.0]];
        let (loss, acc) = infonce_loss_with_accuracy(&anchors, &positives, 0.1);
        assert!(loss.is_finite(), "Loss should be finite");
        assert!((acc - 1.0).abs() < 1e-6, "Accuracy should be 1.0 with batch size 1");
    }
}
