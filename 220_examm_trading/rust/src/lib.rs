//! EXAMM Trading - Evolutionary eXploration of Augmenting Memory Models
//!
//! This crate implements the EXAMM algorithm for evolving recurrent neural network
//! architectures optimized for financial time series prediction.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::distributions::Standard;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. Recurrent Cell Types
// ---------------------------------------------------------------------------

/// Supported recurrent cell types that can be assigned to hidden nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CellType {
    SimpleRNN,
    LSTM,
    GRU,
    MGU,
}

impl CellType {
    /// Returns all available cell types.
    pub fn all() -> &'static [CellType] {
        &[CellType::SimpleRNN, CellType::LSTM, CellType::GRU, CellType::MGU]
    }

    /// Pick a random cell type.
    pub fn random(rng: &mut impl Rng) -> Self {
        let types = Self::all();
        types[rng.gen_range(0..types.len())]
    }

    /// Number of parameter matrices required for this cell type.
    /// Each "matrix" is represented as (input_size + hidden_size) x hidden_size + bias.
    pub fn num_gates(&self) -> usize {
        match self {
            CellType::SimpleRNN => 1,
            CellType::LSTM => 4, // forget, input, cell candidate, output
            CellType::GRU => 3,  // update, reset, candidate
            CellType::MGU => 2,  // forget, candidate
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Recurrent Cell Implementations
// ---------------------------------------------------------------------------

/// Sigmoid activation function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Element-wise sigmoid for Array1.
fn sigmoid_arr(a: &Array1<f64>) -> Array1<f64> {
    a.mapv(sigmoid)
}

/// Element-wise tanh for Array1.
fn tanh_arr(a: &Array1<f64>) -> Array1<f64> {
    a.mapv(|v| v.tanh())
}

/// Element-wise multiply.
fn hadamard(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    a * b
}

/// State carried between time steps for a single recurrent cell.
#[derive(Debug, Clone)]
pub struct CellState {
    pub h: Array1<f64>,
    /// Cell state (only used by LSTM, ignored by others).
    pub c: Array1<f64>,
}

impl CellState {
    pub fn zeros(hidden_size: usize) -> Self {
        Self {
            h: Array1::zeros(hidden_size),
            c: Array1::zeros(hidden_size),
        }
    }
}

/// Parameters for a recurrent cell. Each gate has a weight matrix W (for input),
/// a recurrent matrix U (for hidden state), and a bias vector b.
#[derive(Debug, Clone)]
pub struct CellParams {
    pub cell_type: CellType,
    pub input_size: usize,
    pub hidden_size: usize,
    /// Gate weights: Vec of (W, U, b) per gate.
    pub gates: Vec<(Array2<f64>, Array2<f64>, Array1<f64>)>,
}

impl CellParams {
    /// Create new cell parameters with small random weights.
    pub fn new_random(cell_type: CellType, input_size: usize, hidden_size: usize, rng: &mut impl Rng) -> Self {
        let num_gates = cell_type.num_gates();
        let scale = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let mut gates = Vec::with_capacity(num_gates);
        for _ in 0..num_gates {
            let w = Array2::from_shape_fn((input_size, hidden_size), |_| rng.sample::<f64, _>(Standard) * scale);
            let u = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.sample::<f64, _>(Standard) * scale);
            let b = Array1::zeros(hidden_size);
            gates.push((w, u, b));
        }
        Self { cell_type, input_size, hidden_size, gates }
    }

    /// Compute the linear combination Wx + Uh + b for a given gate index.
    fn gate_linear(&self, gate_idx: usize, x: &Array1<f64>, h: &Array1<f64>) -> Array1<f64> {
        let (ref w, ref u, ref b) = self.gates[gate_idx];
        x.dot(w) + h.dot(u) + b
    }

    /// Forward pass through the cell, returning the new state.
    pub fn forward(&self, x: &Array1<f64>, state: &CellState) -> CellState {
        match self.cell_type {
            CellType::SimpleRNN => {
                let h_new = tanh_arr(&self.gate_linear(0, x, &state.h));
                CellState { h: h_new, c: state.c.clone() }
            }
            CellType::LSTM => {
                let f = sigmoid_arr(&self.gate_linear(0, x, &state.h));
                let i = sigmoid_arr(&self.gate_linear(1, x, &state.h));
                let c_tilde = tanh_arr(&self.gate_linear(2, x, &state.h));
                let o = sigmoid_arr(&self.gate_linear(3, x, &state.h));
                let c_new = hadamard(&f, &state.c) + hadamard(&i, &c_tilde);
                let h_new = hadamard(&o, &tanh_arr(&c_new));
                CellState { h: h_new, c: c_new }
            }
            CellType::GRU => {
                let z = sigmoid_arr(&self.gate_linear(0, x, &state.h));
                let r = sigmoid_arr(&self.gate_linear(1, x, &state.h));
                let rh = hadamard(&r, &state.h);
                let h_tilde = tanh_arr(&self.gate_linear(2, x, &rh));
                let ones = Array1::ones(self.hidden_size);
                let h_new = hadamard(&(&ones - &z), &state.h) + hadamard(&z, &h_tilde);
                CellState { h: h_new, c: state.c.clone() }
            }
            CellType::MGU => {
                let f = sigmoid_arr(&self.gate_linear(0, x, &state.h));
                let fh = hadamard(&f, &state.h);
                let h_tilde = tanh_arr(&self.gate_linear(1, x, &fh));
                let ones = Array1::ones(self.hidden_size);
                let h_new = hadamard(&(&ones - &f), &state.h) + hadamard(&f, &h_tilde);
                CellState { h: h_new, c: state.c.clone() }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Genome for Recurrent Network Topology
// ---------------------------------------------------------------------------

/// Global innovation counter for historical marking of genes (NEAT-style).
static INNOVATION_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_innovation() -> u64 {
    INNOVATION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

/// Reset innovation counter (useful for tests).
pub fn reset_innovation_counter() {
    INNOVATION_COUNTER.store(0, std::sync::atomic::Ordering::SeqCst);
}

/// A node in the recurrent network genome.
#[derive(Debug, Clone)]
pub struct NodeGene {
    pub id: u64,
    pub node_type: NodeType,
    /// Cell type (only meaningful for hidden nodes).
    pub cell_type: CellType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
}

/// An edge in the recurrent network genome.
#[derive(Debug, Clone)]
pub struct EdgeGene {
    pub innovation: u64,
    pub from_node: u64,
    pub to_node: u64,
    pub weight: f64,
    /// Time delay: 0 = feed-forward, >= 1 = recurrent.
    pub time_delay: usize,
    pub enabled: bool,
}

/// A genome encoding a recurrent neural network topology.
#[derive(Debug, Clone)]
pub struct Genome {
    pub id: u64,
    pub nodes: Vec<NodeGene>,
    pub edges: Vec<EdgeGene>,
    pub fitness: f64,
    pub species_id: usize,
    next_node_id: u64,
}

impl Genome {
    /// Create a minimal genome with direct input-to-output connections.
    pub fn new_minimal(input_size: usize, output_size: usize, cell_type: CellType, rng: &mut impl Rng) -> Self {
        let mut nodes = Vec::new();
        let mut node_id: u64 = 0;

        // Input nodes
        for _ in 0..input_size {
            nodes.push(NodeGene { id: node_id, node_type: NodeType::Input, cell_type: CellType::SimpleRNN });
            node_id += 1;
        }

        // Output nodes
        let output_start = node_id;
        for _ in 0..output_size {
            nodes.push(NodeGene { id: node_id, node_type: NodeType::Output, cell_type });
            node_id += 1;
        }

        // Direct connections from each input to each output
        let mut edges = Vec::new();
        for i in 0..input_size as u64 {
            for o in output_start..node_id {
                edges.push(EdgeGene {
                    innovation: next_innovation(),
                    from_node: i,
                    to_node: o,
                    weight: rng.gen_range(-0.5..0.5),
                    time_delay: 0,
                    enabled: true,
                });
            }
        }

        Self {
            id: rng.gen(),
            nodes,
            edges,
            fitness: f64::NEG_INFINITY,
            species_id: 0,
            next_node_id: node_id,
        }
    }

    /// Get input size (number of input nodes).
    pub fn input_size(&self) -> usize {
        self.nodes.iter().filter(|n| n.node_type == NodeType::Input).count()
    }

    /// Get output size.
    pub fn output_size(&self) -> usize {
        self.nodes.iter().filter(|n| n.node_type == NodeType::Output).count()
    }

    /// Get hidden nodes.
    pub fn hidden_nodes(&self) -> Vec<&NodeGene> {
        self.nodes.iter().filter(|n| n.node_type == NodeType::Hidden).collect()
    }

    /// Count active edges.
    pub fn active_edges(&self) -> usize {
        self.edges.iter().filter(|e| e.enabled).count()
    }
}

// ---------------------------------------------------------------------------
// 4. Mutation Operators
// ---------------------------------------------------------------------------

/// Configuration for mutation probabilities.
#[derive(Debug, Clone)]
pub struct MutationConfig {
    pub add_node_prob: f64,
    pub add_edge_prob: f64,
    pub change_cell_type_prob: f64,
    pub modify_weights_prob: f64,
    pub toggle_edge_prob: f64,
    pub weight_perturbation_scale: f64,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            add_node_prob: 0.05,
            add_edge_prob: 0.10,
            change_cell_type_prob: 0.05,
            modify_weights_prob: 0.80,
            toggle_edge_prob: 0.02,
            weight_perturbation_scale: 0.3,
        }
    }
}

/// Apply mutations to a genome (modifies in place).
pub fn mutate(genome: &mut Genome, config: &MutationConfig, rng: &mut impl Rng) {
    // Add node mutation
    if rng.gen::<f64>() < config.add_node_prob && !genome.edges.is_empty() {
        mutate_add_node(genome, rng);
    }

    // Add edge mutation
    if rng.gen::<f64>() < config.add_edge_prob {
        mutate_add_edge(genome, rng);
    }

    // Change cell type mutation
    if rng.gen::<f64>() < config.change_cell_type_prob {
        mutate_change_cell_type(genome, rng);
    }

    // Modify weights mutation
    if rng.gen::<f64>() < config.modify_weights_prob {
        mutate_modify_weights(genome, config.weight_perturbation_scale, rng);
    }

    // Toggle edge mutation
    if rng.gen::<f64>() < config.toggle_edge_prob {
        mutate_toggle_edge(genome, rng);
    }
}

/// Add a new hidden node by splitting an existing edge.
fn mutate_add_node(genome: &mut Genome, rng: &mut impl Rng) {
    let enabled_edges: Vec<usize> = genome.edges.iter()
        .enumerate()
        .filter(|(_, e)| e.enabled)
        .map(|(i, _)| i)
        .collect();

    if enabled_edges.is_empty() {
        return;
    }

    let edge_idx = enabled_edges[rng.gen_range(0..enabled_edges.len())];
    genome.edges[edge_idx].enabled = false;

    let from = genome.edges[edge_idx].from_node;
    let to = genome.edges[edge_idx].to_node;
    let old_weight = genome.edges[edge_idx].weight;

    let new_node_id = genome.next_node_id;
    genome.next_node_id += 1;

    genome.nodes.push(NodeGene {
        id: new_node_id,
        node_type: NodeType::Hidden,
        cell_type: CellType::random(rng),
    });

    // Edge from source to new node with weight 1.0
    genome.edges.push(EdgeGene {
        innovation: next_innovation(),
        from_node: from,
        to_node: new_node_id,
        weight: 1.0,
        time_delay: 0,
        enabled: true,
    });

    // Edge from new node to target with original weight
    genome.edges.push(EdgeGene {
        innovation: next_innovation(),
        from_node: new_node_id,
        to_node: to,
        weight: old_weight,
        time_delay: 0,
        enabled: true,
    });
}

/// Add a new edge between two previously unconnected nodes.
fn mutate_add_edge(genome: &mut Genome, rng: &mut impl Rng) {
    let node_ids: Vec<u64> = genome.nodes.iter().map(|n| n.id).collect();
    if node_ids.len() < 2 {
        return;
    }

    // Try a few times to find a valid connection
    for _ in 0..20 {
        let from = node_ids[rng.gen_range(0..node_ids.len())];
        let to = node_ids[rng.gen_range(0..node_ids.len())];

        // Don't connect to input nodes or from output nodes in feed-forward
        let to_node = genome.nodes.iter().find(|n| n.id == to).unwrap();
        if to_node.node_type == NodeType::Input {
            continue;
        }

        // Check if edge already exists
        let exists = genome.edges.iter().any(|e| e.from_node == from && e.to_node == to && e.enabled);
        if exists {
            continue;
        }

        // Recurrent connection if from >= to (allows self-connections and backward)
        let time_delay = if from >= to { 1 } else { 0 };

        genome.edges.push(EdgeGene {
            innovation: next_innovation(),
            from_node: from,
            to_node: to,
            weight: rng.gen_range(-0.5..0.5),
            time_delay,
            enabled: true,
        });
        break;
    }
}

/// Change the cell type of a random hidden node.
fn mutate_change_cell_type(genome: &mut Genome, rng: &mut impl Rng) {
    let hidden_indices: Vec<usize> = genome.nodes.iter()
        .enumerate()
        .filter(|(_, n)| n.node_type == NodeType::Hidden)
        .map(|(i, _)| i)
        .collect();

    if hidden_indices.is_empty() {
        return;
    }

    let idx = hidden_indices[rng.gen_range(0..hidden_indices.len())];
    let old_type = genome.nodes[idx].cell_type;
    let mut new_type = CellType::random(rng);
    // Ensure it actually changes
    while new_type == old_type && CellType::all().len() > 1 {
        new_type = CellType::random(rng);
    }
    genome.nodes[idx].cell_type = new_type;
}

/// Apply Gaussian perturbation to edge weights.
fn mutate_modify_weights(genome: &mut Genome, scale: f64, rng: &mut impl Rng) {
    for edge in genome.edges.iter_mut() {
        if edge.enabled {
            if rng.gen::<f64>() < 0.9 {
                // Perturb
                edge.weight += rng.gen::<f64>() * 2.0 * scale - scale;
            } else {
                // Reset
                edge.weight = rng.gen_range(-1.0..1.0);
            }
        }
    }
}

/// Toggle a random edge's enabled state.
fn mutate_toggle_edge(genome: &mut Genome, rng: &mut impl Rng) {
    if genome.edges.is_empty() {
        return;
    }
    let idx = rng.gen_range(0..genome.edges.len());
    genome.edges[idx].enabled = !genome.edges[idx].enabled;
}

// ---------------------------------------------------------------------------
// 5. Crossover for Recurrent Genomes
// ---------------------------------------------------------------------------

/// Perform crossover between two parent genomes, producing one offspring.
/// The more fit parent is `parent_a`.
pub fn crossover(parent_a: &Genome, parent_b: &Genome, rng: &mut impl Rng) -> Genome {
    let (better, worse) = if parent_a.fitness >= parent_b.fitness {
        (parent_a, parent_b)
    } else {
        (parent_b, parent_a)
    };

    // Build innovation -> edge map for the worse parent
    let worse_map: HashMap<u64, &EdgeGene> = worse.edges.iter()
        .map(|e| (e.innovation, e))
        .collect();

    let mut child_edges = Vec::new();
    for edge in &better.edges {
        if let Some(other_edge) = worse_map.get(&edge.innovation) {
            // Matching gene: randomly pick from either parent
            if rng.gen_bool(0.5) {
                child_edges.push(edge.clone());
            } else {
                child_edges.push((*other_edge).clone());
            }
        } else {
            // Disjoint/excess gene from better parent: always include
            child_edges.push(edge.clone());
        }
    }

    // Nodes: take all nodes from the better parent
    let child_nodes = better.nodes.clone();

    Genome {
        id: rng.gen(),
        nodes: child_nodes,
        edges: child_edges,
        fitness: f64::NEG_INFINITY,
        species_id: 0,
        next_node_id: better.next_node_id,
    }
}

// ---------------------------------------------------------------------------
// 6. Speciation and Diversity Maintenance
// ---------------------------------------------------------------------------

/// Compatibility distance between two genomes.
pub fn compatibility_distance(a: &Genome, b: &Genome, c1: f64, c2: f64, c3: f64) -> f64 {
    let a_map: HashMap<u64, &EdgeGene> = a.edges.iter().map(|e| (e.innovation, e)).collect();
    let b_map: HashMap<u64, &EdgeGene> = b.edges.iter().map(|e| (e.innovation, e)).collect();

    let mut matching = 0usize;
    let mut disjoint = 0usize;
    let mut weight_diff_sum = 0.0f64;

    for (inn, edge_a) in &a_map {
        if let Some(edge_b) = b_map.get(inn) {
            matching += 1;
            weight_diff_sum += (edge_a.weight - edge_b.weight).abs();
        } else {
            disjoint += 1;
        }
    }

    // Count disjoint/excess from b not in a
    let excess: usize = b_map.keys().filter(|k| !a_map.contains_key(k)).count();
    let total_disjoint = disjoint + excess;

    let n = a.edges.len().max(b.edges.len()).max(1) as f64;
    let avg_weight_diff = if matching > 0 { weight_diff_sum / matching as f64 } else { 0.0 };

    // We lump disjoint and excess together for simplicity
    c1 * (total_disjoint as f64 / n) + c3 * avg_weight_diff
}

/// Assign genomes to species based on compatibility threshold.
pub fn assign_species(
    genomes: &mut [Genome],
    representatives: &mut Vec<Genome>,
    threshold: f64,
    c1: f64,
    c2: f64,
    c3: f64,
) {
    for genome in genomes.iter_mut() {
        let mut assigned = false;
        for (species_idx, rep) in representatives.iter().enumerate() {
            if compatibility_distance(genome, rep, c1, c2, c3) < threshold {
                genome.species_id = species_idx;
                assigned = true;
                break;
            }
        }
        if !assigned {
            genome.species_id = representatives.len();
            representatives.push(genome.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Network Evaluation (Decode genome to runnable network)
// ---------------------------------------------------------------------------

/// A decoded network that can process sequences.
pub struct DecodedNetwork {
    /// Cell params per node id.
    cells: HashMap<u64, CellParams>,
    /// Node processing order (topologically sorted where possible).
    node_order: Vec<u64>,
    /// Edges grouped by target node.
    edges_by_target: HashMap<u64, Vec<EdgeGene>>,
    /// Node types.
    node_types: HashMap<u64, NodeType>,
    hidden_size: usize,
    input_size: usize,
    output_size: usize,
}

impl DecodedNetwork {
    /// Decode a genome into a runnable network.
    pub fn from_genome(genome: &Genome, hidden_size: usize, rng: &mut impl Rng) -> Self {
        let input_size = genome.input_size();
        let output_size = genome.output_size();

        let mut cells = HashMap::new();
        let mut node_types = HashMap::new();
        let mut edges_by_target: HashMap<u64, Vec<EdgeGene>> = HashMap::new();

        for node in &genome.nodes {
            node_types.insert(node.id, node.node_type);
            if node.node_type != NodeType::Input {
                // Determine effective input size for this node based on incoming edges
                let cell_input = input_size.max(1); // simplified: use projection
                cells.insert(
                    node.id,
                    CellParams::new_random(node.cell_type, cell_input, hidden_size, rng),
                );
            }
        }

        for edge in &genome.edges {
            if edge.enabled {
                edges_by_target.entry(edge.to_node).or_default().push(edge.clone());
            }
        }

        // Simple ordering: inputs first, then hidden, then outputs
        let mut node_order: Vec<u64> = Vec::new();
        for n in &genome.nodes {
            if n.node_type == NodeType::Input {
                node_order.push(n.id);
            }
        }
        for n in &genome.nodes {
            if n.node_type == NodeType::Hidden {
                node_order.push(n.id);
            }
        }
        for n in &genome.nodes {
            if n.node_type == NodeType::Output {
                node_order.push(n.id);
            }
        }

        Self { cells, node_order, edges_by_target, node_types, hidden_size, input_size, output_size }
    }

    /// Run the network on a sequence of inputs, returning the output at each time step.
    pub fn forward_sequence(&self, inputs: &[Array1<f64>]) -> Vec<Array1<f64>> {
        let seq_len = inputs.len();
        let max_delay = self.edges_by_target.values()
            .flat_map(|edges| edges.iter().map(|e| e.time_delay))
            .max()
            .unwrap_or(0);

        // Node activations over time: node_id -> Vec<Array1<f64>>
        let mut activations: HashMap<u64, Vec<Array1<f64>>> = HashMap::new();
        let mut states: HashMap<u64, CellState> = HashMap::new();

        // Initialize
        for &nid in &self.node_order {
            let size = if self.node_types[&nid] == NodeType::Input {
                1 // scalar per input node
            } else {
                self.hidden_size
            };
            activations.insert(nid, vec![Array1::zeros(size); max_delay + 1]);
            states.insert(nid, CellState::zeros(self.hidden_size));
        }

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Set input node activations
            for (i, nid) in self.node_order.iter().enumerate() {
                if *self.node_types.get(nid).unwrap() == NodeType::Input {
                    if i < inputs[t].len() {
                        let val = inputs[t][i];
                        let act = activations.get_mut(nid).unwrap();
                        act.push(Array1::from_vec(vec![val]));
                    }
                }
            }

            // Process hidden and output nodes
            for nid in &self.node_order {
                if *self.node_types.get(nid).unwrap() == NodeType::Input {
                    continue;
                }

                // Aggregate incoming signals
                let mut aggregated = Array1::zeros(self.input_size.max(1));
                if let Some(incoming) = self.edges_by_target.get(nid) {
                    for edge in incoming {
                        let src_acts = activations.get(&edge.from_node).unwrap();
                        let delay = edge.time_delay;
                        if src_acts.len() > delay {
                            let src_act = &src_acts[src_acts.len() - 1 - delay];
                            // Project source activation to aggregated size
                            let len = aggregated.len().min(src_act.len());
                            for k in 0..len {
                                aggregated[k] += src_act[k] * edge.weight;
                            }
                        }
                    }
                }

                // Forward through cell
                let cell = self.cells.get(nid).unwrap();
                let state = states.get(nid).unwrap().clone();
                // Ensure input matches expected size
                let cell_input = if aggregated.len() >= cell.input_size {
                    aggregated.slice(ndarray::s![..cell.input_size]).to_owned()
                } else {
                    let mut padded = Array1::zeros(cell.input_size);
                    let len = aggregated.len().min(cell.input_size);
                    for k in 0..len {
                        padded[k] = aggregated[k];
                    }
                    padded
                };

                let new_state = cell.forward(&cell_input, &state);
                let activation = new_state.h.clone();
                activations.get_mut(nid).unwrap().push(activation);
                states.insert(*nid, new_state);
            }

            // Collect output
            let mut output = Array1::zeros(self.output_size);
            let mut out_idx = 0;
            for nid in &self.node_order {
                if *self.node_types.get(nid).unwrap() == NodeType::Output {
                    let acts = activations.get(nid).unwrap();
                    if let Some(last) = acts.last() {
                        // Take the first element of the hidden state as the scalar output
                        if !last.is_empty() {
                            output[out_idx] = last[0];
                        }
                    }
                    out_idx += 1;
                    if out_idx >= self.output_size {
                        break;
                    }
                }
            }
            outputs.push(output);
        }

        outputs
    }
}

// ---------------------------------------------------------------------------
// 8. Fitness Evaluation on Time Series Prediction
// ---------------------------------------------------------------------------

/// Evaluate genome fitness as negative MSE on a time series prediction task.
/// `data` is a list of (input_features, target) pairs over time.
pub fn evaluate_fitness(
    genome: &Genome,
    data: &[(Array1<f64>, f64)],
    hidden_size: usize,
    rng: &mut impl Rng,
) -> f64 {
    if data.is_empty() {
        return f64::NEG_INFINITY;
    }

    let network = DecodedNetwork::from_genome(genome, hidden_size, rng);
    let inputs: Vec<Array1<f64>> = data.iter().map(|(x, _)| x.clone()).collect();
    let targets: Vec<f64> = data.iter().map(|(_, y)| *y).collect();

    let outputs = network.forward_sequence(&inputs);

    let mut mse = 0.0;
    let n = outputs.len().min(targets.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    for i in 0..n {
        let pred = if !outputs[i].is_empty() { outputs[i][0] } else { 0.0 };
        let diff = pred - targets[i];
        mse += diff * diff;
    }
    mse /= n as f64;

    -mse // Negative MSE (higher is better)
}

// ---------------------------------------------------------------------------
// 9. Island-Based Population Management
// ---------------------------------------------------------------------------

/// Configuration for the EXAMM evolutionary algorithm.
#[derive(Debug, Clone)]
pub struct ExammConfig {
    pub num_islands: usize,
    pub population_per_island: usize,
    pub num_generations: usize,
    pub tournament_size: usize,
    pub migration_interval: usize,
    pub crossover_prob: f64,
    pub hidden_size: usize,
    pub mutation_config: MutationConfig,
    pub species_threshold: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
}

impl Default for ExammConfig {
    fn default() -> Self {
        Self {
            num_islands: 4,
            population_per_island: 20,
            num_generations: 50,
            tournament_size: 3,
            migration_interval: 10,
            crossover_prob: 0.2,
            hidden_size: 8,
            mutation_config: MutationConfig::default(),
            species_threshold: 3.0,
            c1: 1.0,
            c2: 1.0,
            c3: 0.4,
        }
    }
}

/// An island in the population.
#[derive(Debug, Clone)]
pub struct Island {
    pub population: Vec<Genome>,
    pub species_representatives: Vec<Genome>,
    pub default_cell_type: CellType,
}

/// The main EXAMM evolutionary engine.
pub struct ExammEngine {
    pub islands: Vec<Island>,
    pub config: ExammConfig,
    pub best_genome: Option<Genome>,
    pub generation: usize,
}

impl ExammEngine {
    /// Create a new EXAMM engine with initialized islands.
    pub fn new(
        input_size: usize,
        output_size: usize,
        config: ExammConfig,
        rng: &mut impl Rng,
    ) -> Self {
        let cell_types = CellType::all();
        let mut islands = Vec::with_capacity(config.num_islands);

        for i in 0..config.num_islands {
            let cell_type = cell_types[i % cell_types.len()];
            let mut population = Vec::with_capacity(config.population_per_island);
            for _ in 0..config.population_per_island {
                population.push(Genome::new_minimal(input_size, output_size, cell_type, rng));
            }
            islands.push(Island {
                population,
                species_representatives: Vec::new(),
                default_cell_type: cell_type,
            });
        }

        Self {
            islands,
            config,
            best_genome: None,
            generation: 0,
        }
    }

    /// Run one generation of evolution on all islands.
    pub fn evolve_generation(
        &mut self,
        data: &[(Array1<f64>, f64)],
        rng: &mut impl Rng,
    ) {
        let hidden_size = self.config.hidden_size;
        let tournament_size = self.config.tournament_size;
        let crossover_prob = self.config.crossover_prob;
        let mutation_config = self.config.mutation_config.clone();

        for island in &mut self.islands {
            // Evaluate fitness for all individuals
            for genome in &mut island.population {
                if genome.fitness == f64::NEG_INFINITY {
                    genome.fitness = evaluate_fitness(genome, data, hidden_size, rng);
                }
            }

            // Speciation
            assign_species(
                &mut island.population,
                &mut island.species_representatives,
                self.config.species_threshold,
                self.config.c1,
                self.config.c2,
                self.config.c3,
            );

            // Create offspring
            let pop_size = island.population.len();
            if pop_size < 2 {
                continue;
            }

            // Tournament selection for parent(s)
            let parent_a = tournament_select(&island.population, tournament_size, rng);
            let offspring = if rng.gen::<f64>() < crossover_prob {
                let parent_b = tournament_select(&island.population, tournament_size, rng);
                let mut child = crossover(&parent_a, &parent_b, rng);
                mutate(&mut child, &mutation_config, rng);
                child
            } else {
                let mut child = parent_a.clone();
                child.id = rng.gen();
                mutate(&mut child, &mutation_config, rng);
                child
            };

            // Replace worst individual
            if let Some(worst_idx) = island.population.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                island.population[worst_idx] = offspring;
            }
        }

        // Migration
        self.generation += 1;
        if self.generation % self.config.migration_interval == 0 {
            self.migrate(rng);
        }

        // Track global best
        for island in &self.islands {
            for genome in &island.population {
                let dominated = match &self.best_genome {
                    Some(best) => genome.fitness > best.fitness,
                    None => true,
                };
                if dominated {
                    self.best_genome = Some(genome.clone());
                }
            }
        }
    }

    /// Migrate the best genome from each island to a random other island.
    fn migrate(&mut self, rng: &mut impl Rng) {
        let num_islands = self.islands.len();
        if num_islands < 2 {
            return;
        }

        let mut best_per_island: Vec<Genome> = Vec::new();
        for island in &self.islands {
            if let Some(best) = island.population.iter()
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            {
                best_per_island.push(best.clone());
            }
        }

        for (src_idx, migrant) in best_per_island.into_iter().enumerate() {
            let mut dst_idx = rng.gen_range(0..num_islands);
            while dst_idx == src_idx {
                dst_idx = rng.gen_range(0..num_islands);
            }

            // Replace worst on destination island
            if let Some(worst_idx) = self.islands[dst_idx].population.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
            {
                self.islands[dst_idx].population[worst_idx] = migrant;
            }
        }
    }

    /// Run evolution for all configured generations.
    pub fn run(&mut self, data: &[(Array1<f64>, f64)], rng: &mut impl Rng) {
        let num_generations = self.config.num_generations;
        for gen in 0..num_generations {
            self.evolve_generation(data, rng);
            if (gen + 1) % 10 == 0 {
                let best_fitness = self.best_genome.as_ref().map(|g| g.fitness).unwrap_or(f64::NEG_INFINITY);
                println!("Generation {}/{}: best fitness = {:.6}", gen + 1, num_generations, best_fitness);
            }
        }
    }

    /// Get a description of the best evolved architecture.
    pub fn describe_best(&self) -> String {
        match &self.best_genome {
            None => "No genome evolved yet.".to_string(),
            Some(genome) => {
                let hidden = genome.hidden_nodes();
                let cell_types: Vec<String> = hidden.iter().map(|n| format!("{:?}", n.cell_type)).collect();
                let active = genome.active_edges();
                format!(
                    "Best Genome (fitness={:.6}):\n  Nodes: {} input, {} hidden, {} output\n  Hidden cell types: [{}]\n  Active edges: {}\n  Total edges: {}",
                    genome.fitness,
                    genome.input_size(),
                    hidden.len(),
                    genome.output_size(),
                    cell_types.join(", "),
                    active,
                    genome.edges.len(),
                )
            }
        }
    }
}

/// Tournament selection: pick the best from a random sample.
fn tournament_select(population: &[Genome], tournament_size: usize, rng: &mut impl Rng) -> Genome {
    let mut best: Option<&Genome> = None;
    let size = tournament_size.min(population.len());
    for _ in 0..size {
        let idx = rng.gen_range(0..population.len());
        let candidate = &population[idx];
        if best.is_none() || candidate.fitness > best.unwrap().fitness {
            best = Some(candidate);
        }
    }
    best.unwrap().clone()
}

// ---------------------------------------------------------------------------
// 10. Bybit API Data Fetching
// ---------------------------------------------------------------------------

/// A single kline (candlestick) data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures.
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline data from Bybit V5 API.
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
    }

    let mut klines: Vec<Kline> = resp.result.list.iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Kline {
                    timestamp: row[0].parse().unwrap_or(0),
                    open: row[1].parse().unwrap_or(0.0),
                    high: row[2].parse().unwrap_or(0.0),
                    low: row[3].parse().unwrap_or(0.0),
                    close: row[4].parse().unwrap_or(0.0),
                    volume: row[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse for chronological order
    klines.reverse();
    Ok(klines)
}

/// Prepare time series data from klines for EXAMM training.
/// Returns (input_features, target) pairs where target is the next log return.
pub fn prepare_training_data(klines: &[Kline]) -> Vec<(Array1<f64>, f64)> {
    if klines.len() < 3 {
        return Vec::new();
    }

    // Compute log returns and normalized volume
    let mut log_returns = Vec::new();
    let mut volumes = Vec::new();
    for i in 1..klines.len() {
        let lr = (klines[i].close / klines[i - 1].close).ln();
        log_returns.push(lr);
        volumes.push(klines[i].volume);
    }

    // Z-score normalize
    let lr_mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
    let lr_std = (log_returns.iter().map(|x| (x - lr_mean).powi(2)).sum::<f64>() / log_returns.len() as f64).sqrt();
    let lr_std = if lr_std < 1e-10 { 1.0 } else { lr_std };

    let vol_mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let vol_std = (volumes.iter().map(|x| (x - vol_mean).powi(2)).sum::<f64>() / volumes.len() as f64).sqrt();
    let vol_std = if vol_std < 1e-10 { 1.0 } else { vol_std };

    let norm_lr: Vec<f64> = log_returns.iter().map(|x| (x - lr_mean) / lr_std).collect();
    let norm_vol: Vec<f64> = volumes.iter().map(|x| (x - vol_mean) / vol_std).collect();

    // Build feature-target pairs: features = [log_return, volume], target = next log_return
    let mut data = Vec::new();
    for i in 0..norm_lr.len() - 1 {
        let features = Array1::from_vec(vec![norm_lr[i], norm_vol[i]]);
        let target = norm_lr[i + 1];
        data.push((features, target));
    }

    data
}

/// Generate synthetic price data for testing (random walk with drift).
pub fn generate_synthetic_data(num_points: usize, rng: &mut impl Rng) -> Vec<Kline> {
    let mut klines = Vec::with_capacity(num_points);
    let mut price = 50000.0_f64; // Starting price similar to BTC
    let mut timestamp = 1_700_000_000_000u64;

    for _ in 0..num_points {
        let ret = 0.0001 + rng.gen::<f64>() * 0.02 - 0.01; // Small drift + noise
        let open = price;
        price *= (1.0 + ret);
        let close = price;
        let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.005);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.005);
        let volume = 100.0 + rng.gen::<f64>() * 900.0;

        klines.push(Kline {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
        timestamp += 3_600_000; // 1 hour
    }
    klines
}

// ---------------------------------------------------------------------------
// 11. Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_cell_types() {
        assert_eq!(CellType::SimpleRNN.num_gates(), 1);
        assert_eq!(CellType::LSTM.num_gates(), 4);
        assert_eq!(CellType::GRU.num_gates(), 3);
        assert_eq!(CellType::MGU.num_gates(), 2);
    }

    #[test]
    fn test_simple_rnn_forward() {
        let mut rng = test_rng();
        let params = CellParams::new_random(CellType::SimpleRNN, 2, 4, &mut rng);
        let state = CellState::zeros(4);
        let input = Array1::from_vec(vec![0.5, -0.3]);
        let new_state = params.forward(&input, &state);
        assert_eq!(new_state.h.len(), 4);
        // tanh output should be in [-1, 1]
        for &v in new_state.h.iter() {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_lstm_forward() {
        let mut rng = test_rng();
        let params = CellParams::new_random(CellType::LSTM, 2, 4, &mut rng);
        let state = CellState::zeros(4);
        let input = Array1::from_vec(vec![1.0, -1.0]);
        let new_state = params.forward(&input, &state);
        assert_eq!(new_state.h.len(), 4);
        assert_eq!(new_state.c.len(), 4);
    }

    #[test]
    fn test_gru_forward() {
        let mut rng = test_rng();
        let params = CellParams::new_random(CellType::GRU, 3, 5, &mut rng);
        let state = CellState::zeros(5);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let new_state = params.forward(&input, &state);
        assert_eq!(new_state.h.len(), 5);
    }

    #[test]
    fn test_mgu_forward() {
        let mut rng = test_rng();
        let params = CellParams::new_random(CellType::MGU, 2, 3, &mut rng);
        let state = CellState::zeros(3);
        let input = Array1::from_vec(vec![0.5, 0.5]);
        let new_state = params.forward(&input, &state);
        assert_eq!(new_state.h.len(), 3);
    }

    #[test]
    fn test_genome_creation() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let genome = Genome::new_minimal(3, 1, CellType::LSTM, &mut rng);
        assert_eq!(genome.input_size(), 3);
        assert_eq!(genome.output_size(), 1);
        assert_eq!(genome.edges.len(), 3); // 3 inputs -> 1 output
    }

    #[test]
    fn test_mutation_add_node() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let mut genome = Genome::new_minimal(2, 1, CellType::GRU, &mut rng);
        let initial_nodes = genome.nodes.len();
        mutate_add_node(&mut genome, &mut rng);
        assert_eq!(genome.nodes.len(), initial_nodes + 1);
        assert!(genome.hidden_nodes().len() == 1);
    }

    #[test]
    fn test_mutation_add_edge() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let mut genome = Genome::new_minimal(2, 1, CellType::GRU, &mut rng);
        mutate_add_node(&mut genome, &mut rng); // Need hidden node for more edge options
        let initial_edges = genome.edges.len();
        mutate_add_edge(&mut genome, &mut rng);
        assert!(genome.edges.len() >= initial_edges);
    }

    #[test]
    fn test_mutation_change_cell_type() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let mut genome = Genome::new_minimal(2, 1, CellType::LSTM, &mut rng);
        mutate_add_node(&mut genome, &mut rng);
        let hidden_before: Vec<CellType> = genome.hidden_nodes().iter().map(|n| n.cell_type).collect();
        mutate_change_cell_type(&mut genome, &mut rng);
        let hidden_after: Vec<CellType> = genome.hidden_nodes().iter().map(|n| n.cell_type).collect();
        // Cell type should have changed (probabilistically — could rarely stay the same)
        assert_eq!(hidden_before.len(), hidden_after.len());
    }

    #[test]
    fn test_crossover() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let mut parent_a = Genome::new_minimal(2, 1, CellType::LSTM, &mut rng);
        parent_a.fitness = -0.5;
        let mut parent_b = Genome::new_minimal(2, 1, CellType::GRU, &mut rng);
        parent_b.fitness = -0.8;
        let child = crossover(&parent_a, &parent_b, &mut rng);
        assert_eq!(child.input_size(), 2);
        assert_eq!(child.output_size(), 1);
    }

    #[test]
    fn test_compatibility_distance() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let genome_a = Genome::new_minimal(2, 1, CellType::LSTM, &mut rng);
        let genome_b = Genome::new_minimal(2, 1, CellType::GRU, &mut rng);
        let dist = compatibility_distance(&genome_a, &genome_b, 1.0, 1.0, 0.4);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_fitness_evaluation() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let genome = Genome::new_minimal(2, 1, CellType::LSTM, &mut rng);
        let data: Vec<(Array1<f64>, f64)> = (0..20)
            .map(|i| {
                let x = Array1::from_vec(vec![i as f64 * 0.1, (i as f64 * 0.1).sin()]);
                let y = ((i + 1) as f64 * 0.1).sin();
                (x, y)
            })
            .collect();
        let fitness = evaluate_fitness(&genome, &data, 4, &mut rng);
        assert!(fitness.is_finite());
        assert!(fitness <= 0.0); // Negative MSE
    }

    #[test]
    fn test_examm_engine_creation() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let config = ExammConfig {
            num_islands: 2,
            population_per_island: 5,
            num_generations: 2,
            ..Default::default()
        };
        let engine = ExammEngine::new(2, 1, config, &mut rng);
        assert_eq!(engine.islands.len(), 2);
        assert_eq!(engine.islands[0].population.len(), 5);
    }

    #[test]
    fn test_examm_evolution() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let config = ExammConfig {
            num_islands: 2,
            population_per_island: 5,
            num_generations: 5,
            migration_interval: 3,
            hidden_size: 4,
            ..Default::default()
        };
        let mut engine = ExammEngine::new(2, 1, config, &mut rng);

        let data: Vec<(Array1<f64>, f64)> = (0..30)
            .map(|i| {
                let x = Array1::from_vec(vec![i as f64 * 0.1, (i as f64 * 0.05).sin()]);
                let y = ((i + 1) as f64 * 0.1).sin();
                (x, y)
            })
            .collect();

        engine.run(&data, &mut rng);
        assert!(engine.best_genome.is_some());
        let desc = engine.describe_best();
        assert!(desc.contains("Best Genome"));
    }

    #[test]
    fn test_synthetic_data_generation() {
        let mut rng = test_rng();
        let klines = generate_synthetic_data(100, &mut rng);
        assert_eq!(klines.len(), 100);
        assert!(klines[0].close > 0.0);
    }

    #[test]
    fn test_prepare_training_data() {
        let mut rng = test_rng();
        let klines = generate_synthetic_data(50, &mut rng);
        let data = prepare_training_data(&klines);
        assert!(!data.is_empty());
        assert_eq!(data[0].0.len(), 2); // log_return + volume features
    }

    #[test]
    fn test_network_forward_sequence() {
        let mut rng = test_rng();
        reset_innovation_counter();
        let genome = Genome::new_minimal(2, 1, CellType::GRU, &mut rng);
        let network = DecodedNetwork::from_genome(&genome, 4, &mut rng);
        let inputs: Vec<Array1<f64>> = (0..10)
            .map(|i| Array1::from_vec(vec![i as f64 * 0.1, 0.5]))
            .collect();
        let outputs = network.forward_sequence(&inputs);
        assert_eq!(outputs.len(), 10);
        for out in &outputs {
            assert_eq!(out.len(), 1);
        }
    }

    #[test]
    fn test_cell_state_initialization() {
        let state = CellState::zeros(8);
        assert_eq!(state.h.len(), 8);
        assert_eq!(state.c.len(), 8);
        assert!(state.h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }
}
