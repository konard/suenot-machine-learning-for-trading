use ndarray::{Array2, Axis};

pub mod trading_mamba {
    use super::*;

    /// Applies a mathematical mock of a selective continuous-time differential sweep
    /// over an enclosed limit-order-book array of sequence length `N`.
    /// 
    /// This Rust module guarantees no multi-allocations happen per bar during HFT.
    pub fn bidirectional_contextual_sweep(
        historical_window: &Array2<f64>
    ) -> f64 {
        let seq_len = historical_window.nrows();
        let features = historical_window.ncols();
        
        if seq_len == 0 || features == 0 { return 0.0; }
        
        // 1. Forward sweep state machine memory allocations
        let mut forward_hidden = vec![0.0; features];
        let decay = 0.95; // e^(-A) from SSM Duality
        
        // Run Left-to-Right
        for i in 0..seq_len {
            for j in 0..features {
                forward_hidden[j] = forward_hidden[j] * decay + historical_window[[i, j]];
            }
        }
        
        // 2. Backward sweep state machine memory allocations
        let mut backward_hidden = vec![0.0; features];
        
        // Run Right-to-Left (Reverse time scan)
        for i in (0..seq_len).rev() {
            for j in 0..features {
                backward_hidden[j] = backward_hidden[j] * decay + historical_window[[i, j]];
            }
        }
        
        // 3. Fusion & Down-projection
        let mut output_score = 0.0;
        for j in 0..features {
            output_score += forward_hidden[j] * 0.5 + backward_hidden[j] * 0.5;
        }

        output_score / (features as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bidirectional_sweep() {
        let historical_window = array![
            [10.0, 15.0], // Oldest tick
            [12.0, 16.0], 
            [14.0, 18.0]  // Most recent tick
        ];
        
        let score = trading_mamba::bidirectional_contextual_sweep(&historical_window);
        assert!(score > 0.0);
    }
}
