//! Benchmarks for conformal prediction algorithms

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array1;
use rand::Rng;

// Import from the library
use conformal_trading::conformal::predictor::SplitConformalPredictor;
use conformal_trading::conformal::scores::{compute_quantile, compute_median};

/// Benchmark calibration with different sample sizes
fn bench_calibration(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration");

    for size in [100, 500, 1000, 5000].iter() {
        let mut rng = rand::thread_rng();

        // Generate calibration data
        let y_cal: Vec<f64> = (0..*size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let y_pred_cal: Vec<f64> = y_cal.iter()
            .map(|&y| y + rng.gen_range(-0.1..0.1))
            .collect();

        let y_cal_arr = Array1::from_vec(y_cal);
        let y_pred_cal_arr = Array1::from_vec(y_pred_cal);

        group.bench_with_input(
            BenchmarkId::new("split_conformal", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut cp = SplitConformalPredictor::new(0.1);
                    cp.calibrate(
                        black_box(&y_cal_arr),
                        black_box(&y_pred_cal_arr)
                    );
                    cp
                })
            }
        );
    }

    group.finish();
}

/// Benchmark prediction interval generation
fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    let mut rng = rand::thread_rng();

    // Setup: create calibrated predictor
    let cal_scores: Vec<f64> = (0..500)
        .map(|_| rng.gen_range(0.0..0.1))
        .collect();

    let mut cp = SplitConformalPredictor::new(0.1);
    cp.calibrate_from_scores(cal_scores);

    for n_preds in [1, 10, 100, 1000].iter() {
        let predictions: Vec<f64> = (0..*n_preds)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let pred_arr = Array1::from_vec(predictions);

        group.bench_with_input(
            BenchmarkId::new("predict_intervals", n_preds),
            n_preds,
            |b, _| {
                b.iter(|| {
                    cp.predict_intervals(black_box(&pred_arr))
                })
            }
        );
    }

    group.finish();
}

/// Benchmark quantile computation
fn bench_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile");
    let mut rng = rand::thread_rng();

    for size in [100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("compute_quantile", size),
            &data,
            |b, data| {
                b.iter(|| {
                    compute_quantile(black_box(data), 0.9)
                })
            }
        );

        group.bench_with_input(
            BenchmarkId::new("compute_median", size),
            &data,
            |b, data| {
                b.iter(|| {
                    compute_median(black_box(data))
                })
            }
        );
    }

    group.finish();
}

/// Benchmark coverage evaluation
fn bench_coverage_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("coverage_evaluation");
    let mut rng = rand::thread_rng();

    // Setup calibrated predictor
    let cal_scores: Vec<f64> = (0..500)
        .map(|_| rng.gen_range(0.0..0.1))
        .collect();

    let mut cp = SplitConformalPredictor::new(0.1);
    cp.calibrate_from_scores(cal_scores);

    for size in [100, 500, 1000].iter() {
        let y_test: Vec<f64> = (0..*size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let y_pred_test: Vec<f64> = y_test.iter()
            .map(|&y| y + rng.gen_range(-0.1..0.1))
            .collect();

        let y_test_arr = Array1::from_vec(y_test);
        let y_pred_test_arr = Array1::from_vec(y_pred_test);

        group.bench_with_input(
            BenchmarkId::new("evaluate_coverage", size),
            size,
            |b, _| {
                b.iter(|| {
                    cp.evaluate_coverage(
                        black_box(&y_test_arr),
                        black_box(&y_pred_test_arr)
                    )
                })
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_calibration,
    bench_prediction,
    bench_quantile,
    bench_coverage_evaluation
);
criterion_main!(benches);
