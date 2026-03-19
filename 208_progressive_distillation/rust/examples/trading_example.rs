use anyhow::Result;
use progressive_distillation::*;

fn main() -> Result<()> {
    println!("=== Chapter 208: Progressive Distillation for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch data from Bybit or fall back to synthetic data
    // -----------------------------------------------------------------------
    let (x_train, y_train) = match fetch_bybit_klines("BTCUSDT", "5", 200) {
        Ok(candles) => {
            println!(
                "Fetched {} candles from Bybit (BTCUSDT, 5m interval)",
                candles.len()
            );
            if candles.len() > 7 {
                let (x, y) = candles_to_features(&candles);
                println!("Generated {} samples with {} features\n", x.nrows(), x.ncols());
                (x, y)
            } else {
                println!("Not enough candles, using synthetic data\n");
                generate_synthetic_data(200, 4)
            }
        }
        Err(e) => {
            println!("Bybit API unavailable ({}), using synthetic data\n", e);
            generate_synthetic_data(200, 4)
        }
    };

    let n_features = x_train.ncols();

    // -----------------------------------------------------------------------
    // Step 2: Train teacher model (large)
    // -----------------------------------------------------------------------
    println!("--- Training Teacher Model ---");
    let teacher_arch = vec![n_features, 64, 32, 16, 1];
    let mut teacher = FlexibleNetwork::new(&teacher_arch);
    println!(
        "Teacher architecture: {:?} ({} params)",
        teacher_arch,
        teacher.param_count()
    );

    let teacher_loss = teacher.train_supervised(&x_train, &y_train, 3, 0.001);
    let teacher_r2 = teacher.r_squared(&x_train, &y_train);
    println!("Teacher trained — MSE: {:.6}, R²: {:.4}\n", teacher_loss, teacher_r2);

    // -----------------------------------------------------------------------
    // Step 3: Progressive distillation (4 stages: teacher → medium → small → tiny)
    // -----------------------------------------------------------------------
    println!("--- Progressive Distillation Pipeline ---");
    let stages = vec![
        vec![n_features, 32, 16, 1],    // Stage 1: medium
        vec![n_features, 16, 8, 1],     // Stage 2: small
        vec![n_features, 8, 1],         // Stage 3: tiny
    ];

    println!("Stage architectures:");
    for (i, arch) in stages.iter().enumerate() {
        let tmp = FlexibleNetwork::new(arch);
        println!("  Stage {}: {:?} ({} params)", i + 1, arch, tmp.param_count());
    }
    println!();

    let mut distiller = ProgressiveDistiller::new(stages);
    distiller.run(&teacher, &x_train, &y_train, 3, 0.001);
    distiller.print_summary();

    // -----------------------------------------------------------------------
    // Step 4: One-shot distillation for comparison
    // -----------------------------------------------------------------------
    println!("--- One-Shot Distillation (for comparison) ---");
    let tiny_arch = vec![n_features, 8, 1]; // Same as final progressive stage
    println!(
        "One-shot target architecture: {:?} ({} params)",
        tiny_arch,
        FlexibleNetwork::new(&tiny_arch).param_count()
    );

    let (_, oneshot_metrics) = one_shot_distill(
        &teacher,
        &tiny_arch,
        &x_train,
        &y_train,
        3, // same total epochs as one progressive stage
        0.001,
    );
    println!("  {}\n", oneshot_metrics);

    // -----------------------------------------------------------------------
    // Step 5: Compare progressive vs one-shot
    // -----------------------------------------------------------------------
    println!("--- Comparison: Progressive vs One-Shot ---");
    let progressive_final = distiller.stage_metrics.last().unwrap();

    println!(
        "Progressive (final stage): R² = {:.4}, Retention = {:.1}%",
        progressive_final.r_squared,
        progressive_final.accuracy_retention * 100.0
    );
    println!(
        "One-shot:                  R² = {:.4}, Retention = {:.1}%",
        oneshot_metrics.r_squared,
        oneshot_metrics.accuracy_retention * 100.0
    );
    println!(
        "Both at compression ratio: {:.1}x\n",
        progressive_final.compression_ratio
    );

    if progressive_final.accuracy_retention >= oneshot_metrics.accuracy_retention {
        println!("Result: Progressive distillation retained more accuracy.");
    } else {
        println!("Result: One-shot distillation performed better in this run.");
        println!("(This can happen with small datasets or few epochs — progressive");
        println!(" distillation typically shines at higher compression ratios.)");
    }

    // -----------------------------------------------------------------------
    // Step 6: Model size tracking across stages
    // -----------------------------------------------------------------------
    println!("\n--- Model Size Tracking ---");
    println!("{:<10} {:<12} {:<15} {:<12}", "Stage", "Params", "Compression", "Size (KB)");
    println!("{}", "-".repeat(49));
    for m in &distiller.stage_metrics {
        let size_kb = (m.param_count * 8) as f64 / 1024.0; // 8 bytes per f64
        println!(
            "{:<10} {:<12} {:<15.1} {:<12.2}",
            m.stage,
            m.param_count,
            m.compression_ratio,
            size_kb
        );
    }

    println!("\nDone.");
    Ok(())
}
