//! Benchmarks for alpha factor evaluation.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llm_alpha_mining::data::generate_synthetic_data;
use llm_alpha_mining::alpha::{AlphaEvaluator, predefined_factors};

fn benchmark_factor_evaluation(c: &mut Criterion) {
    let data = generate_synthetic_data("BTCUSDT", 1000, 42);
    let evaluator = AlphaEvaluator::new(&data);
    let factors = predefined_factors();

    c.bench_function("evaluate_momentum_5d", |b| {
        b.iter(|| {
            evaluator.evaluate(black_box(&factors[0])).unwrap()
        })
    });

    c.bench_function("evaluate_mean_reversion", |b| {
        b.iter(|| {
            evaluator.evaluate(black_box(&factors[1])).unwrap()
        })
    });

    c.bench_function("evaluate_all_predefined", |b| {
        b.iter(|| {
            for factor in &factors {
                let _ = evaluator.evaluate(black_box(factor));
            }
        })
    });
}

fn benchmark_data_generation(c: &mut Criterion) {
    c.bench_function("generate_100_candles", |b| {
        b.iter(|| generate_synthetic_data("TEST", black_box(100), 42))
    });

    c.bench_function("generate_1000_candles", |b| {
        b.iter(|| generate_synthetic_data("TEST", black_box(1000), 42))
    });
}

criterion_group!(benches, benchmark_factor_evaluation, benchmark_data_generation);
criterion_main!(benches);
