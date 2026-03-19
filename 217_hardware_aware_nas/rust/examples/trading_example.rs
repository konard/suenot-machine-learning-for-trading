use hardware_aware_nas::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Hardware-Aware NAS for Trading ===\n");

    // ------------------------------------------------------------------
    // 1. Fetch BTCUSDT data from Bybit
    // ------------------------------------------------------------------
    println!("Fetching BTCUSDT 15m candles from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "15", 200).await {
        Ok(c) => {
            println!("Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("Failed to fetch from Bybit ({}), using synthetic data", e);
            generate_synthetic_candles(200)
        }
    };

    let returns = compute_returns(&candles);
    let volatility = compute_volatility(&returns, 20);
    println!(
        "Returns: {} data points, mean={:.6}, std={:.6}",
        returns.len(),
        mean(&returns),
        std_dev(&returns)
    );
    if !volatility.is_empty() {
        println!(
            "Volatility (20-period): mean={:.6}\n",
            mean(&volatility)
        );
    }

    // Use returns as proxy data for architecture evaluation
    let eval_data: Vec<f64> = returns.clone();

    // ------------------------------------------------------------------
    // 2. Define hardware profiles
    // ------------------------------------------------------------------
    let fpga = HardwareProfile::fpga_hft();
    let gpu = HardwareProfile::gpu_batch();
    let cpu = HardwareProfile::cpu_mobile();

    println!("Hardware Profiles:");
    println!(
        "  {} - Memory: {} MB, Peak: {:.0} GFLOP/s",
        fpga.name,
        fpga.memory_budget_bytes / (1024 * 1024),
        fpga.peak_flops / 1e9
    );
    println!(
        "  {} - Memory: {} MB, Peak: {:.0} TFLOP/s",
        gpu.name,
        gpu.memory_budget_bytes / (1024 * 1024),
        gpu.peak_flops / 1e12
    );
    println!(
        "  {} - Memory: {} MB, Peak: {:.0} GFLOP/s",
        cpu.name,
        cpu.memory_budget_bytes / (1024 * 1024),
        cpu.peak_flops / 1e9
    );
    println!();

    // ------------------------------------------------------------------
    // 3. Compare pre-built architecture templates
    // ------------------------------------------------------------------
    let input_features = 16;
    let output_classes = 3; // buy / hold / sell

    println!("=== Architecture Templates Comparison ===\n");

    let templates = vec![
        ("FPGA-HFT", ArchitectureTemplates::fpga_hft_template(input_features, output_classes)),
        ("GPU-Batch", ArchitectureTemplates::gpu_batch_template(input_features, output_classes)),
        ("CPU-Mobile", ArchitectureTemplates::cpu_mobile_template(input_features, output_classes)),
    ];

    let hardware_profiles = vec![&fpga, &gpu, &cpu];

    for (name, arch) in &templates {
        println!("--- {} ---", name);
        println!("  Layers: {}", arch.layers.len());
        println!("  Params: {}", compute_total_params(arch));
        println!("  FLOPs (seq=64): {}", compute_architecture_flops(arch, 64));
        println!("  Memory (FP32): {} bytes", compute_total_memory(arch, 64, 4));

        for hw in &hardware_profiles {
            let lat = estimate_latency(arch, hw);
            println!("  Latency on {}: {:.2} us", hw.name, lat);
        }
        println!();
    }

    // ------------------------------------------------------------------
    // 4. Run Hardware-Aware NAS for each target
    // ------------------------------------------------------------------
    let num_iterations = 500;

    println!("=== Hardware-Aware NAS Search ({} iterations each) ===\n", num_iterations);

    // FPGA search: strict 20 us latency budget
    let fpga_budget = 20.0;
    println!("--- FPGA Search (latency budget: {} us) ---", fpga_budget);
    let fpga_space = SearchSpace::fpga_space(input_features, output_classes);
    let fpga_results = search_constrained(&fpga_space, &fpga, fpga_budget, num_iterations, &eval_data, 42);
    println!("  Found {} architectures meeting constraint", fpga_results.len());
    if let Some((best_arch, best_metrics)) = fpga_results.first() {
        println!("  Best (by latency): {}", best_metrics);
        println!("{}", best_arch.summary());
    }

    // GPU search: 500 us latency budget
    let gpu_budget = 500.0;
    println!("--- GPU Search (latency budget: {} us) ---", gpu_budget);
    let gpu_space = SearchSpace::gpu_space(input_features, output_classes);
    let gpu_results = search_constrained(&gpu_space, &gpu, gpu_budget, num_iterations, &eval_data, 43);
    println!("  Found {} architectures meeting constraint", gpu_results.len());
    if let Some((best_arch, best_metrics)) = gpu_results.first() {
        println!("  Best (by latency): {}", best_metrics);
        println!("{}", best_arch.summary());
    }

    // CPU search: 1000 us latency budget
    let cpu_budget = 1000.0;
    println!("--- CPU Search (latency budget: {} us) ---", cpu_budget);
    let cpu_space = SearchSpace::cpu_space(input_features, output_classes);
    let cpu_results = search_constrained(&cpu_space, &cpu, cpu_budget, num_iterations, &eval_data, 44);
    println!("  Found {} architectures meeting constraint", cpu_results.len());
    if let Some((best_arch, best_metrics)) = cpu_results.first() {
        println!("  Best (by latency): {}", best_metrics);
        println!("{}", best_arch.summary());
    }

    // ------------------------------------------------------------------
    // 5. Show Pareto front for GPU (accuracy vs latency)
    // ------------------------------------------------------------------
    println!("=== Pareto Front (GPU, accuracy vs latency) ===\n");
    let gpu_pareto = search(&gpu_space, &gpu, gpu_budget, num_iterations, &eval_data, 45);
    println!(
        "  {:>6}  {:>10}  {:>10}  {:>8}  {:>10}",
        "Acc", "Lat (us)", "FLOPs", "Params", "Memory"
    );
    println!("  {}", "-".repeat(52));
    for (_, m) in gpu_pareto.iter().take(10) {
        println!(
            "  {:>6.4}  {:>10.2}  {:>10}  {:>8}  {:>10}",
            m.accuracy_proxy, m.latency_us, m.flops, m.params, m.memory_bytes
        );
    }

    // ------------------------------------------------------------------
    // 6. Cross-hardware comparison for a single architecture
    // ------------------------------------------------------------------
    println!("\n=== Cross-Hardware Latency for FPGA Template ===\n");
    let reference_arch = ArchitectureTemplates::fpga_hft_template(input_features, output_classes);
    for hw in &hardware_profiles {
        let lat = estimate_latency(&reference_arch, hw);
        let meets_budget = match hw.device_type {
            DeviceType::FPGA => lat <= fpga_budget,
            DeviceType::GPU => lat <= gpu_budget,
            DeviceType::CPU => lat <= cpu_budget,
        };
        println!(
            "  {} -> {:.2} us  [{}]",
            hw.name,
            lat,
            if meets_budget { "PASS" } else { "FAIL" }
        );
    }

    println!("\nDone.");
    Ok(())
}

// Helper: generate synthetic candles when API is unavailable
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64;
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    use rand::{Rng, SeedableRng};
    for i in 0..n {
        let ret = rng.gen_range(-0.02..0.02);
        price *= 1.0 + ret;
        let high = price * (1.0 + rng.gen_range(0.0..0.01));
        let low = price * (1.0 - rng.gen_range(0.0..0.01));
        candles.push(Candle {
            timestamp: 1700000000 + (i as u64) * 900,
            open: price / (1.0 + ret * 0.5),
            high,
            low,
            close: price,
            volume: rng.gen_range(100.0..10000.0),
        });
    }
    candles
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}
