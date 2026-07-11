//! Shared helpers for the simulation test suite.

use extropic_sim as sim;

/// Exact Boltzmann distribution of a small model, by enumeration.
///
/// Returns the probability of every joint assignment, indexed by
/// mixed-radix encoding of the node values (first node varies fastest).
pub struct Exact {
    pub probabilities: Vec<f64>,
    state_counts: Vec<u32>,
}

impl Exact {
    pub fn new(graph: &sim::Graph, factors: &[sim::DiscreteFactor]) -> Self {
        let state_counts: Vec<u32> = graph
            .nodes()
            .map(|node| graph.node_kind(node).state_count())
            .collect();
        let total: u64 = state_counts.iter().map(|&count| count as u64).product();
        assert!(total <= 1 << 24, "model too large for enumeration");

        let mut values = vec![0u32; state_counts.len()];
        let mut probabilities = Vec::with_capacity(total as usize);
        for index in 0..total {
            let mut remainder = index;
            for (value, &count) in values.iter_mut().zip(state_counts.iter()) {
                *value = (remainder % count as u64) as u32;
                remainder /= count as u64;
            }
            let energy = sim::total_energy(factors, graph, &values);
            probabilities.push((-energy).exp());
        }
        let normalizer: f64 = probabilities.iter().sum();
        for probability in probabilities.iter_mut() {
            *probability /= normalizer;
        }
        Self {
            probabilities,
            state_counts,
        }
    }

    fn decode(&self, mut index: usize, node: sim::Node) -> u32 {
        for &count in self.state_counts[..node.index()].iter() {
            index /= count as usize;
        }
        (index % self.state_counts[node.index()] as usize) as u32
    }

    /// Expected value of the product of the spins (±1) of `nodes`.
    pub fn moment(&self, nodes: &[sim::Node]) -> f64 {
        self.probabilities
            .iter()
            .enumerate()
            .map(|(index, probability)| {
                let product: f64 = nodes
                    .iter()
                    .map(|&node| self.decode(index, node) as f64 * 2.0 - 1.0)
                    .product();
                probability * product
            })
            .sum()
    }

    /// Probability that `node` takes the given value.
    pub fn marginal(&self, node: sim::Node, value: u32) -> f64 {
        self.probabilities
            .iter()
            .enumerate()
            .filter(|&(index, _)| self.decode(index, node) == value)
            .map(|(_, probability)| probability)
            .sum()
    }

    /// Probability of the given values of `nodes`, with all other
    /// nodes marginalized out.
    pub fn joint(&self, nodes: &[sim::Node], values: &[u32]) -> f64 {
        self.probabilities
            .iter()
            .enumerate()
            .filter(|&(index, _)| {
                nodes
                    .iter()
                    .zip(values.iter())
                    .all(|(&node, &value)| self.decode(index, node) == value)
            })
            .map(|(_, probability)| probability)
            .sum()
    }
}

/// Empirical marginal of one observed node over all samples and chains.
pub fn empirical_marginal(
    samples: &sim::Samples,
    block_index: usize,
    position: u32,
    value: u32,
) -> f64 {
    let mut hits = 0u64;
    for sample in 0..samples.sample_count() {
        for chain in 0..samples.chain_count() {
            if samples.value(block_index, sample, chain, position) == value {
                hits += 1;
            }
        }
    }
    hits as f64 / (samples.sample_count() as u64 * samples.chain_count() as u64) as f64
}

pub fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{what}: actual {actual:.4} vs expected {expected:.4} (tolerance {tolerance})",
    );
}
