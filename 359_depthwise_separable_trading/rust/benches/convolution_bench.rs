//! Criterion benchmarks for convolution operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use dsc_trading::convolution::{
    DepthwiseConv1d, DepthwiseSeparableConv1d, PointwiseConv1d,
};
use ndarray::Array2;

fn bench_depthwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("Depthwise Convolution");

    for channels in [16, 32, 64, 128].iter() {
        let conv = DepthwiseConv1d::new(*channels, 3).unwrap();
        let input = Array2::from_elem((*channels, 100), 1.0);

        group.bench_with_input(
            BenchmarkId::new("channels", channels),
            &input,
            |b, input| {
                b.iter(|| conv.forward(black_box(input)));
            },
        );
    }

    group.finish();
}

fn bench_pointwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pointwise Convolution");

    for (in_ch, out_ch) in [(16, 32), (32, 64), (64, 128), (128, 256)].iter() {
        let conv = PointwiseConv1d::new(*in_ch, *out_ch).unwrap();
        let input = Array2::from_elem((*in_ch, 100), 1.0);

        group.bench_with_input(
            BenchmarkId::new("channels", format!("{}_{}", in_ch, out_ch)),
            &input,
            |b, input| {
                b.iter(|| conv.forward(black_box(input)));
            },
        );
    }

    group.finish();
}

fn bench_dsc(c: &mut Criterion) {
    let mut group = c.benchmark_group("Depthwise Separable Convolution");

    for (in_ch, out_ch) in [(16, 32), (32, 64), (64, 128)].iter() {
        let conv = DepthwiseSeparableConv1d::new(*in_ch, *out_ch, 3).unwrap();
        let input = Array2::from_elem((*in_ch, 100), 1.0);

        group.bench_with_input(
            BenchmarkId::new("channels", format!("{}_{}", in_ch, out_ch)),
            &input,
            |b, input| {
                b.iter(|| conv.forward(black_box(input)));
            },
        );
    }

    group.finish();
}

fn bench_sequence_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sequence Length Scaling");

    let conv = DepthwiseSeparableConv1d::new(64, 64, 3).unwrap();

    for seq_len in [50, 100, 200, 500, 1000].iter() {
        let input = Array2::from_elem((64, *seq_len), 1.0);

        group.bench_with_input(
            BenchmarkId::new("length", seq_len),
            &input,
            |b, input| {
                b.iter(|| conv.forward(black_box(input)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_depthwise,
    bench_pointwise,
    bench_dsc,
    bench_sequence_length
);
criterion_main!(benches);
