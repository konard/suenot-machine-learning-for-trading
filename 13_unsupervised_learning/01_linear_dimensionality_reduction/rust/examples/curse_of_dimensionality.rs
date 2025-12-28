//! Example: Curse of Dimensionality
//!
//! This example demonstrates how distance metrics behave
//! as the number of dimensions increases.

use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("===========================================");
    println!("  The Curse of Dimensionality");
    println!("===========================================");
    println!();
    println!("This example shows how the average distance between");
    println!("random points in a unit hypercube increases with dimension.");
    println!();

    let n_points = 500;
    let dimensions = vec![1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000];

    println!(
        "{:>6} {:>12} {:>12} {:>12}",
        "Dims", "Avg Dist", "Min Dist", "Max Dist"
    );
    println!("{:-<50}", "");

    let mut rng = rand::thread_rng();

    for &dim in &dimensions {
        // Generate random points uniformly in [0, 1]^dim
        let mut points = Array2::zeros((n_points, dim));
        for i in 0..n_points {
            for j in 0..dim {
                points[[i, j]] = rng.gen::<f64>();
            }
        }

        // Calculate pairwise distances
        let (avg_dist, min_dist, max_dist) = calculate_distance_stats(&points);

        println!(
            "{:>6} {:>12.4} {:>12.4} {:>12.4}",
            dim, avg_dist, min_dist, max_dist
        );
    }

    println!();
    println!("Key Observations:");
    println!("-----------------");
    println!("1. Average distance grows approximately as sqrt(dim/6)");
    println!("2. Even minimum distances increase substantially");
    println!("3. Points become 'equidistant' - all far from each other");
    println!();
    println!("Implications for ML:");
    println!("- Distance-based algorithms (k-NN) become less effective");
    println!("- Need exponentially more data to maintain density");
    println!("- Dimensionality reduction (PCA) helps mitigate this");
}

fn calculate_distance_stats(points: &Array2<f64>) -> (f64, f64, f64) {
    let n = points.nrows();
    let mut distances = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_distance(
                &points.row(i).to_owned(),
                &points.row(j).to_owned(),
            );
            distances.push(dist);
        }
    }

    let avg = distances.iter().sum::<f64>() / distances.len() as f64;
    let min = distances.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    (avg, min, max)
}

fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
