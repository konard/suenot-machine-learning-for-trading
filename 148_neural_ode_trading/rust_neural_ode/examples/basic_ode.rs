//! Basic ODE Solver Example
//!
//! Demonstrates the ODE solvers by solving classical differential equations:
//! 1. Exponential decay: dy/dt = -y
//! 2. Simple harmonic oscillator: dx/dt = v, dv/dt = -x
//! 3. Van der Pol oscillator: dx/dt = v, dv/dt = mu*(1-x^2)*v - x
//!
//! Run: cargo run --example basic_ode

use ndarray::Array1;
use neural_ode_trading::ode_solver::{DormandPrinceSolver, EulerSolver, ODESolver, RK4Solver};

fn main() {
    println!("═══════════════════════════════════════════");
    println!("  Neural ODE - Basic ODE Solver Examples");
    println!("═══════════════════════════════════════════\n");

    // ── Example 1: Exponential Decay ──
    println!("1. Exponential Decay: dy/dt = -y, y(0) = 1");
    println!("   Exact solution: y(t) = exp(-t)");
    println!("   ─────────────────────────────────────");

    let y0 = Array1::from_vec(vec![1.0]);
    let decay = |_t: f64, y: &Array1<f64>| -y.clone();

    let rk4 = RK4Solver;
    let t_points: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
    let trajectory = rk4.solve_trajectory(&decay, &y0, &t_points, 0.01);

    println!("   {:>6} {:>12} {:>12} {:>12}", "t", "RK4", "Exact", "Error");
    for (t, y) in &trajectory {
        let exact = (-t).exp();
        println!(
            "   {:>6.1} {:>12.8} {:>12.8} {:>12.2e}",
            t,
            y[0],
            exact,
            (y[0] - exact).abs()
        );
    }

    // ── Example 2: Simple Harmonic Oscillator ──
    println!("\n2. Simple Harmonic Oscillator");
    println!("   dx/dt = v,  dv/dt = -x");
    println!("   x(0) = 1, v(0) = 0 => x(t) = cos(t), v(t) = -sin(t)");
    println!("   ─────────────────────────────────────");

    let y0_sho = Array1::from_vec(vec![1.0, 0.0]); // [x, v]
    let sho = |_t: f64, y: &Array1<f64>| {
        Array1::from_vec(vec![y[1], -y[0]])
    };

    let t_sho: Vec<f64> = (0..=20).map(|i| i as f64 * std::f64::consts::PI / 10.0).collect();
    let traj_sho = rk4.solve_trajectory(&sho, &y0_sho, &t_sho, 0.01);

    println!("   {:>6} {:>10} {:>10} {:>10} {:>10}", "t", "x(t)", "cos(t)", "v(t)", "-sin(t)");
    for (t, y) in &traj_sho {
        println!(
            "   {:>6.2} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            t,
            y[0],
            t.cos(),
            y[1],
            -t.sin()
        );
    }

    // ── Example 3: Van der Pol Oscillator ──
    println!("\n3. Van der Pol Oscillator (nonlinear)");
    println!("   dx/dt = v,  dv/dt = mu*(1-x^2)*v - x,  mu = 2.0");
    println!("   ─────────────────────────────────────");

    let mu = 2.0;
    let vdp = move |_t: f64, y: &Array1<f64>| {
        let dx = y[1];
        let dv = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
        Array1::from_vec(vec![dx, dv])
    };

    let y0_vdp = Array1::from_vec(vec![2.0, 0.0]);
    let t_vdp: Vec<f64> = (0..=100).map(|i| i as f64 * 0.2).collect();

    // Compare solvers
    let dopri = DormandPrinceSolver::new(1e-8, 1e-10);

    let start = std::time::Instant::now();
    let traj_rk4 = rk4.solve_trajectory(&vdp, &y0_vdp, &t_vdp, 0.01);
    let time_rk4 = start.elapsed();

    let start = std::time::Instant::now();
    let traj_dopri = dopri.solve_trajectory(&vdp, &y0_vdp, &t_vdp, 0.1);
    let time_dopri = start.elapsed();

    println!("   {:>6} {:>10} {:>10} {:>10} {:>10}", "t", "x(RK4)", "v(RK4)", "x(DOPRI)", "v(DOPRI)");
    for i in (0..traj_rk4.len()).step_by(20) {
        let (t, y_rk4) = &traj_rk4[i];
        let (_, y_dp) = &traj_dopri[i];
        println!(
            "   {:>6.1} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            t, y_rk4[0], y_rk4[1], y_dp[0], y_dp[1]
        );
    }

    println!("\n   Solver timing:");
    println!("   RK4:           {:>8}us", time_rk4.as_micros());
    println!("   Dormand-Prince: {:>8}us", time_dopri.as_micros());

    // ── Summary ──
    println!("\n═══════════════════════════════════════════");
    println!("  Key takeaways for Neural ODEs:");
    println!("  - RK4: fixed step, reliable, good baseline");
    println!("  - Dormand-Prince: adaptive step, efficient");
    println!("  - ODE dynamics capture continuous evolution");
    println!("  - These solvers form the backbone of Neural ODEs");
    println!("═══════════════════════════════════════════");
}
