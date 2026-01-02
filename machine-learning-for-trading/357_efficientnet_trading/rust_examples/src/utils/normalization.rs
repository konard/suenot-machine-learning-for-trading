//! Data normalization utilities

/// Normalize data to [0, 1] range using min-max scaling
pub fn normalize_minmax(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let min = data.iter().cloned().fold(f64::MAX, f64::min);
    let max = data.iter().cloned().fold(f64::MIN, f64::max);
    let range = max - min;

    if range == 0.0 {
        return vec![0.5; data.len()];
    }

    data.iter().map(|&x| (x - min) / range).collect()
}

/// Normalize data to [-1, 1] range
pub fn normalize(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let min = data.iter().cloned().fold(f64::MAX, f64::min);
    let max = data.iter().cloned().fold(f64::MIN, f64::max);
    let range = max - min;

    if range == 0.0 {
        return vec![0.0; data.len()];
    }

    data.iter().map(|&x| 2.0 * (x - min) / range - 1.0).collect()
}

/// Standardize data (z-score normalization)
pub fn standardize(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![0.0; data.len()];
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = {
        let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (data.len() - 1) as f64;
        variance.sqrt()
    };

    if std == 0.0 {
        return vec![0.0; data.len()];
    }

    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Normalize 2D data (image) to [0, 1]
pub fn normalize_image(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut min = f64::MAX;
    let mut max = f64::MIN;

    for row in data {
        for &val in row {
            min = min.min(val);
            max = max.max(val);
        }
    }

    let range = max - min;
    if range == 0.0 {
        return data.iter().map(|row| vec![0.5; row.len()]).collect();
    }

    data.iter()
        .map(|row| row.iter().map(|&x| (x - min) / range).collect())
        .collect()
}

/// Apply ImageNet normalization (for pre-trained models)
pub fn imagenet_normalize(pixel: [u8; 3]) -> [f64; 3] {
    // ImageNet mean and std
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    [
        (pixel[0] as f64 / 255.0 - mean[0]) / std[0],
        (pixel[1] as f64 / 255.0 - mean[1]) / std[1],
        (pixel[2] as f64 / 255.0 - mean[2]) / std[2],
    ]
}

/// Reverse ImageNet normalization
pub fn imagenet_denormalize(normalized: [f64; 3]) -> [u8; 3] {
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    [
        ((normalized[0] * std[0] + mean[0]) * 255.0).clamp(0.0, 255.0) as u8,
        ((normalized[1] * std[1] + mean[1]) * 255.0).clamp(0.0, 255.0) as u8,
        ((normalized[2] * std[2] + mean[2]) * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_minmax() {
        let data = vec![0.0, 5.0, 10.0];
        let result = normalize_minmax(&data);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 0.001);
        assert!((result[1] - 0.5).abs() < 0.001);
        assert!((result[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize() {
        let data = vec![0.0, 5.0, 10.0];
        let result = normalize(&data);

        assert_eq!(result.len(), 3);
        assert!((result[0] - (-1.0)).abs() < 0.001);
        assert!((result[1] - 0.0).abs() < 0.001);
        assert!((result[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_standardize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = standardize(&data);

        // Mean should be ~0
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!(mean.abs() < 0.001);
    }

    #[test]
    fn test_imagenet_normalize_roundtrip() {
        let pixel = [128u8, 64, 200];
        let normalized = imagenet_normalize(pixel);
        let denormalized = imagenet_denormalize(normalized);

        assert_eq!(pixel, denormalized);
    }
}
