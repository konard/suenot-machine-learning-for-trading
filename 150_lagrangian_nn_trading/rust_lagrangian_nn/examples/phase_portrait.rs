//! Phase Portrait Example for Lagrangian Neural Networks
//!
//! Demonstrates how to:
//! 1. Create a Lagrangian NN
//! 2. Compute accelerations via Euler-Lagrange equations
//! 3. Integrate trajectories
//! 4. Check energy conservation
//! 5. Compare with dissipative LNN
//!
//! Usage:
//!   cargo run --example phase_portrait

use lagrangian_nn_trading::nn::{LagrangianNN, DissipativeLNN};
use lagrangian_nn_trading::integrator;
use lagrangian_nn_trading::utils;

fn main() {
    println!("=== Lagrangian Neural Network: Phase Portrait Demo ===\n");

    // 1. Create a conservative LNN
    let coord_dim = 1;
    let hidden_dim = 32;
    let num_layers = 2;
    let mass_reg = 1e-4;

    let model = LagrangianNN::new(coord_dim, hidden_dim, num_layers, mass_reg);
    println!("Created LNN with {} parameters", model.num_parameters());
    println!("Coord dim: {}, Hidden dim: {}, Layers: {}", coord_dim, hidden_dim, num_layers);

    // 2. Test Lagrangian evaluation
    let q = vec![0.5];
    let qdot = vec![0.3];

    let l = model.lagrangian(&q, &qdot);
    let e = model.energy(&q, &qdot);
    println!("\nAt q={:.2}, q-dot={:.2}:", q[0], qdot[0]);
    println!("  L(q, q-dot) = {:.6}", l);
    println!("  E = q-dot * dL/dq-dot - L = {:.6}", e);

    // 3. Compute acceleration
    let qddot = model.acceleration(&q, &qdot);
    println!("  q-ddot = {:.6}", qddot[0]);

    // 4. Integrate a trajectory
    println!("\n--- Conservative Trajectory ---");
    let q0 = vec![1.0];
    let qdot0 = vec![0.0];
    let dt = 0.05;
    let n_steps = 200;

    let (traj_q, traj_v) = integrator::integrate_trajectory(&model, &q0, &qdot0, dt, n_steps);

    println!("Integrated {} steps with dt={}", n_steps, dt);
    println!("Start: q={:.4}, q-dot={:.4}", q0[0], qdot0[0]);
    println!("End:   q={:.4}, q-dot={:.4}", traj_q.last().unwrap()[0], traj_v.last().unwrap()[0]);

    // 5. Energy conservation check
    let energies = utils::energy_along_trajectory(&model, &traj_q, &traj_v);
    let e_initial = energies[0];
    let e_final = *energies.last().unwrap();
    let e_min = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let e_max = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let drift_pct = (e_final - e_initial) / (e_initial.abs() + 1e-10) * 100.0;

    println!("\nEnergy Conservation:");
    println!("  Initial: {:.6}", e_initial);
    println!("  Final:   {:.6}", e_final);
    println!("  Min:     {:.6}", e_min);
    println!("  Max:     {:.6}", e_max);
    println!("  Drift:   {:.4}%", drift_pct);

    // 6. Print phase portrait sample
    println!("\nPhase Portrait (first 20 points):");
    println!("  {:>8}  {:>10}  {:>10}  {:>10}", "step", "q", "q-dot", "E");
    for i in 0..20.min(traj_q.len()) {
        println!(
            "  {:>8}  {:>10.4}  {:>10.4}  {:>10.6}",
            i, traj_q[i][0], traj_v[i][0], energies[i]
        );
    }

    // 7. Dissipative LNN comparison
    println!("\n--- Dissipative Trajectory ---");
    let dissip_model = DissipativeLNN::new(coord_dim, hidden_dim, num_layers, mass_reg);
    println!("Created Dissipative LNN with {} parameters", dissip_model.num_parameters());

    let (dissip_traj_q, dissip_traj_v) = integrator::integrate_trajectory_dissipative(
        &dissip_model, &q0, &qdot0, dt, n_steps,
    );

    println!("Start: q={:.4}, q-dot={:.4}", q0[0], qdot0[0]);
    println!(
        "End:   q={:.4}, q-dot={:.4}",
        dissip_traj_q.last().unwrap()[0],
        dissip_traj_v.last().unwrap()[0]
    );

    // Dissipative energy (should decrease)
    let dissip_energies: Vec<f64> = dissip_traj_q.iter()
        .zip(dissip_traj_v.iter())
        .map(|(q, v)| dissip_model.energy(q, v))
        .collect();

    let de_initial = dissip_energies[0];
    let de_final = *dissip_energies.last().unwrap();
    println!("\nDissipative Energy:");
    println!("  Initial: {:.6}", de_initial);
    println!("  Final:   {:.6}", de_final);
    println!("  Change:  {:.6} (should decrease due to dissipation)", de_final - de_initial);

    // 8. Compare integration methods
    println!("\n--- Integration Method Comparison ---");

    let (euler_q, euler_v) = {
        let mut q = q0.clone();
        let mut v = qdot0.clone();
        let mut tq = vec![q.clone()];
        let mut tv = vec![v.clone()];
        for _ in 0..n_steps {
            let (qn, vn) = integrator::euler_step(&model, &q, &v, dt);
            tq.push(qn.clone());
            tv.push(vn.clone());
            q = qn;
            v = vn;
        }
        (tq, tv)
    };

    let euler_energies = utils::energy_along_trajectory(&model, &euler_q, &euler_v);
    let euler_drift = (euler_energies.last().unwrap() - euler_energies[0])
        / (euler_energies[0].abs() + 1e-10) * 100.0;

    let (leapfrog_q, leapfrog_v) = {
        let mut q = q0.clone();
        let mut v = qdot0.clone();
        let mut tq = vec![q.clone()];
        let mut tv = vec![v.clone()];
        for _ in 0..n_steps {
            let (qn, vn) = integrator::leapfrog_step(&model, &q, &v, dt);
            tq.push(qn.clone());
            tv.push(vn.clone());
            q = qn;
            v = vn;
        }
        (tq, tv)
    };

    let leapfrog_energies = utils::energy_along_trajectory(&model, &leapfrog_q, &leapfrog_v);
    let leapfrog_drift = (leapfrog_energies.last().unwrap() - leapfrog_energies[0])
        / (leapfrog_energies[0].abs() + 1e-10) * 100.0;

    println!("  Euler:    energy drift = {:.4}%", euler_drift);
    println!("  Leapfrog: energy drift = {:.4}%", leapfrog_drift);
    println!("  RK4:      energy drift = {:.4}%", drift_pct);

    // 9. Serialization test
    println!("\n--- Serialization ---");
    let json = serde_json::to_string(&model).unwrap();
    let restored: LagrangianNN = serde_json::from_str(&json).unwrap();
    let l_orig = model.lagrangian(&q, &qdot);
    let l_restored = restored.lagrangian(&q, &qdot);
    println!("  Original L:  {:.6}", l_orig);
    println!("  Restored L:  {:.6}", l_restored);
    println!("  Match: {}", (l_orig - l_restored).abs() < 1e-10);

    println!("\nPhase portrait demo complete!");
}
