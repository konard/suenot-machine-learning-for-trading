//! Benchmarks for normalizing flow operations.
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use normalizing_flows_finance::flows::config::FlowConfig;
use normalizing_flows_finance::flows::realnvp::RealNVP;
use normalizing_flows_finance::flows::traits::NormalizingFlow;
use normalizing_flows_finance::risk::metrics::{compute_cvar, compute_risk_metrics, compute_var};
use normalizing_flows_finance::utils::returns::generate_synthetic_returns;
use ndarray::Array1;

fn bench_flow_forward(c: &mut Criterion) {
    let config = FlowConfig::new(1).with_n_layers(4).with_hidden_dim(32);
    let flow = RealNVP::new(config);

    let z = Array1::from_vec(vec![0.5]);

    c.bench_function("flow_forward_1d", |b| {
        b.iter(|| flow.forward(black_box(&z)))
    });
}

fn bench_flow_inverse(c: &mut Criterion) {
    let config = FlowConfig::new(1).with_n_layers(4).with_hidden_dim(32);
    let flow = RealNVP::new(config);

    let x = Array1::from_vec(vec![0.02]);

    c.bench_function("flow_inverse_1d", |b| {
        b.iter(|| flow.inverse(black_box(&x)))
    });
}

fn bench_flow_log_prob(c: &mut Criterion) {
    let config = FlowConfig::new(1).with_n_layers(4).with_hidden_dim(32);
    let flow = RealNVP::new(config);

    let x = Array1::from_vec(vec![0.02]);

    c.bench_function("flow_log_prob_1d", |b| {
        b.iter(|| flow.log_prob(black_box(&x)))
    });
}

fn bench_flow_sample(c: &mut Criterion) {
    let config = FlowConfig::new(1).with_n_layers(4).with_hidden_dim(32);
    let flow = RealNVP::new(config);

    c.bench_function("flow_sample_1000", |b| {
        b.iter(|| flow.sample(black_box(1000)))
    });
}

fn bench_var_calculation(c: &mut Criterion) {
    let config = FlowConfig::new(1).with_n_layers(4).with_hidden_dim(32);
    let flow = RealNVP::new(config);

    c.bench_function("compute_var_10k_samples", |b| {
        b.iter(|| compute_var(black_box(&flow), black_box(0.05), black_box(10000)))
    });
}

fn bench_cvar_calculation(c: &mut Criterion) {
    let config = FlowConfig::new(1).with_n_layers(4).with_hidden_dim(32);
    let flow = RealNVP::new(config);

    c.bench_function("compute_cvar_10k_samples", |b| {
        b.iter(|| compute_cvar(black_box(&flow), black_box(0.05), black_box(10000)))
    });
}

fn bench_risk_metrics(c: &mut Criterion) {
    let config = FlowConfig::new(1).with_n_layers(4).with_hidden_dim(32);
    let flow = RealNVP::new(config);

    c.bench_function("compute_risk_metrics_10k", |b| {
        b.iter(|| compute_risk_metrics(black_box(&flow), black_box(10000)))
    });
}

fn bench_synthetic_generation(c: &mut Criterion) {
    c.bench_function("generate_synthetic_returns_1000", |b| {
        b.iter(|| generate_synthetic_returns(black_box(1000), black_box(4.0)))
    });
}

criterion_group!(
    benches,
    bench_flow_forward,
    bench_flow_inverse,
    bench_flow_log_prob,
    bench_flow_sample,
    bench_var_calculation,
    bench_cvar_calculation,
    bench_risk_metrics,
    bench_synthetic_generation,
);

criterion_main!(benches);
