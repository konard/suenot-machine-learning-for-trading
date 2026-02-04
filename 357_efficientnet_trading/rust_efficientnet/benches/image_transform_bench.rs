//! Benchmarks for image transformation performance
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use efficientnet_trading::image::gaf::GAFTransformer;
use efficientnet_trading::image::mtf::MTFTransformer;
use efficientnet_trading::image::stack::ImageStackGenerator;

/// Generate sample price data
fn generate_prices(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 45000.0 + 500.0 * (i as f64 * 0.1).sin() + 100.0 * (i as f64 * 0.3).cos())
        .collect()
}

fn bench_gaf_transform(c: &mut Criterion) {
    let prices = generate_prices(120);

    let mut group = c.benchmark_group("GAF Transform");

    for size in [64, 128, 224].iter() {
        let transformer = GAFTransformer::new(*size);

        group.bench_with_input(
            BenchmarkId::new("GASF", size),
            size,
            |b, _| b.iter(|| transformer.transform_gasf(black_box(&prices))),
        );

        group.bench_with_input(
            BenchmarkId::new("GADF", size),
            size,
            |b, _| b.iter(|| transformer.transform_gadf(black_box(&prices))),
        );
    }

    group.finish();
}

fn bench_mtf_transform(c: &mut Criterion) {
    let prices = generate_prices(120);

    let mut group = c.benchmark_group("MTF Transform");

    for (bins, size) in [(4, 64), (8, 128), (16, 224)].iter() {
        let transformer = MTFTransformer::new(*bins, *size);

        group.bench_with_input(
            BenchmarkId::new("MTF", format!("{}bins_{}px", bins, size)),
            &(*bins, *size),
            |b, _| b.iter(|| transformer.transform(black_box(&prices))),
        );
    }

    group.finish();
}

fn bench_image_stack(c: &mut Criterion) {
    let prices = generate_prices(120);

    let mut group = c.benchmark_group("Image Stack");

    for size in [64, 128, 224].iter() {
        let generator = ImageStackGenerator::new(*size);

        group.bench_with_input(
            BenchmarkId::new("MultiTransform", size),
            size,
            |b, _| b.iter(|| generator.create_multi_transform_stack(black_box(&prices))),
        );
    }

    group.finish();
}

fn bench_different_series_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("Series Length");

    for length in [60, 120, 240, 480].iter() {
        let prices = generate_prices(*length);
        let transformer = GAFTransformer::new(64);

        group.bench_with_input(
            BenchmarkId::new("GASF", length),
            length,
            |b, _| b.iter(|| transformer.transform_gasf(black_box(&prices))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gaf_transform,
    bench_mtf_transform,
    bench_image_stack,
    bench_different_series_lengths,
);

criterion_main!(benches);
