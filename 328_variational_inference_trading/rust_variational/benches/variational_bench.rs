//! Benchmarks for Variational Inference Trading
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use variational_inference_trading::variational::{VAEConfig, VAE};

fn benchmark_vae_forward(c: &mut Criterion) {
    let config = VAEConfig::new(13, 8, vec![64, 32])
        .with_sequence_length(60);
    let vae = VAE::new(config);

    let input = Array1::from_vec(vec![0.1; 13 * 60]);

    c.bench_function("vae_forward", |b| {
        b.iter(|| {
            let output = vae.forward(black_box(&input));
            black_box(output)
        })
    });
}

fn benchmark_vae_predict(c: &mut Criterion) {
    let config = VAEConfig::new(13, 8, vec![64, 32])
        .with_sequence_length(60);
    let vae = VAE::new(config);

    let input = Array1::from_vec(vec![0.1; 13 * 60]);

    c.bench_function("vae_predict_50_samples", |b| {
        b.iter(|| {
            let prediction = vae.predict(black_box(&input), 50);
            black_box(prediction)
        })
    });
}

fn benchmark_vae_encode(c: &mut Criterion) {
    let config = VAEConfig::new(13, 8, vec![64, 32])
        .with_sequence_length(60);
    let vae = VAE::new(config);

    let input = Array1::from_vec(vec![0.1; 13 * 60]);

    c.bench_function("vae_encode", |b| {
        b.iter(|| {
            let (mu, log_var) = vae.encode(black_box(&input));
            black_box((mu, log_var))
        })
    });
}

criterion_group!(
    benches,
    benchmark_vae_forward,
    benchmark_vae_predict,
    benchmark_vae_encode
);
criterion_main!(benches);
