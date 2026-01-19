//! Benchmarks for sentiment analysis performance

use bloomberggpt_trading::sentiment::MockSentimentAnalyzer;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_mock_analyzer(c: &mut Criterion) {
    let analyzer = MockSentimentAnalyzer::new();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let texts = vec![
        "Apple reports record earnings",
        "Tesla stock crashes after disappointing guidance",
        "Microsoft Azure growth exceeds expectations amid strong cloud demand",
        "Federal Reserve maintains interest rates unchanged, markets rally",
    ];

    c.bench_function("analyze_single_text", |b| {
        b.iter(|| {
            rt.block_on(async {
                analyzer.analyze(black_box("Apple reports record earnings")).await.unwrap()
            })
        })
    });

    let mut group = c.benchmark_group("analyze_batch");
    for size in [1, 5, 10, 50].iter() {
        let batch: Vec<String> = texts.iter()
            .cycle()
            .take(*size)
            .map(|s| s.to_string())
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &batch, |b, batch| {
            b.iter(|| {
                rt.block_on(async {
                    use bloomberggpt_trading::sentiment::SentimentAnalyzerTrait;
                    analyzer.analyze_batch(black_box(batch)).await.unwrap()
                })
            })
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_mock_analyzer);
criterion_main!(benches);
