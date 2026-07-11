//! Statistical simulation suite.
//!
//! Small models are checked against exact Boltzmann distributions
//! obtained by enumeration. The checks are shared between the CPU
//! and GPU samplers; GPU tests are `#[ignore]`d by default and run
//! where a GPU (or a software Vulkan driver) is available:
//!
//! ```text
//! cargo test -- --ignored
//! ```

mod common;

use common::{Exact, assert_close, empirical_marginal};
use extropic_sim::{self as sim, Sampler as _};

/// A 5-node Ising chain with non-uniform biases and couplings,
/// sampled with two-color block Gibbs.
fn ising_chain(sampler: &mut dyn sim::Sampler) {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(5);
    let biases = vec![0.3, -0.2, 0.0, 0.4, -0.5];
    let weights = vec![0.5, -0.7, 0.3, 0.6];

    let factors = [
        sim::DiscreteFactor::bias(nodes.clone(), biases),
        sim::DiscreteFactor::coupling(&nodes[..4], &nodes[1..], weights),
    ];
    let free_blocks = [
        sim::Block::new([nodes[0], nodes[2], nodes[4]]),
        sim::Block::new([nodes[1], nodes[3]]),
    ];
    let program = sim::Program::compile(&graph, &free_blocks, &[], &factors).unwrap();
    let exact = Exact::new(&graph, &factors);

    let schedule = sim::Schedule {
        n_warmup: 200,
        n_samples: 4000,
        steps_per_sample: 2,
    };
    let mut state = sim::State::random(&program, 32, 1);

    let mut moments: Vec<Vec<sim::Node>> = nodes.iter().map(|&node| vec![node]).collect();
    for pair in nodes.windows(2) {
        moments.push(pair.to_vec());
    }
    let estimated = sampler.accumulate_moments(&program, &schedule, &mut state, 2, &moments);

    for (tuple, &estimate) in moments.iter().zip(estimated.iter()) {
        assert_close(
            estimate,
            exact.moment(tuple),
            0.02,
            &format!("moment of {tuple:?}"),
        );
    }
}

/// A 4-node Potts-like chain of 3-state categorical variables
/// with asymmetric couplings and biases.
fn potts_chain(sampler: &mut dyn sim::Sampler) {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_categoricals(4, 3);

    // Per-node bias tables of shape [4, 3].
    let bias_weights = vec![
        0.2, 0.0, -0.3, //
        0.0, 0.5, 0.0, //
        -0.2, 0.1, 0.3, //
        0.0, 0.0, 0.4,
    ];
    // Coupling tables of shape [3, 3, 3]: favor equal neighbors.
    let mut coupling_weights = vec![0.0f32; 3 * 3 * 3];
    for edge in 0..3 {
        for state in 0..3 {
            coupling_weights[edge * 9 + state * 3 + state] = 0.8;
        }
        coupling_weights[edge * 9 + 2] = -0.4;
    }

    let factors = [
        sim::DiscreteFactor::cat_bias(nodes.clone(), bias_weights),
        sim::DiscreteFactor::cat_coupling(&nodes[..3], &nodes[1..], coupling_weights),
    ];
    let free_blocks = [
        sim::Block::new([nodes[0], nodes[2]]),
        sim::Block::new([nodes[1], nodes[3]]),
    ];
    let program = sim::Program::compile(&graph, &free_blocks, &[], &factors).unwrap();
    let exact = Exact::new(&graph, &factors);

    let schedule = sim::Schedule {
        n_warmup: 200,
        n_samples: 4000,
        steps_per_sample: 2,
    };
    let mut state = sim::State::random(&program, 32, 3);
    let observed = [sim::Block::new(nodes.clone())];
    let samples = sampler.sample_states(&program, &schedule, &mut state, 4, &observed);

    for (position, &node) in nodes.iter().enumerate() {
        for value in 0..3 {
            assert_close(
                empirical_marginal(&samples, 0, position as u32, value),
                exact.marginal(node, value),
                0.02,
                &format!("marginal of node {position} state {value}"),
            );
        }
    }
}

/// A model with a three-spin interaction on top of pairwise terms,
/// checking higher-order factors and third moments.
fn higher_order(sampler: &mut dyn sim::Sampler) {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(3);

    let factors = [
        sim::DiscreteFactor::bias(nodes.clone(), vec![0.2, -0.1, 0.3]),
        sim::DiscreteFactor::coupling([nodes[0]], [nodes[1]], vec![0.4]),
        sim::DiscreteFactor {
            spin_groups: vec![
                sim::Block::new([nodes[0]]),
                sim::Block::new([nodes[1]]),
                sim::Block::new([nodes[2]]),
            ],
            cat_groups: Vec::new(),
            weights: vec![-0.6],
        },
    ];
    // Every node interacts with every other, so blocks are single-node.
    let free_blocks = [
        sim::Block::new([nodes[0]]),
        sim::Block::new([nodes[1]]),
        sim::Block::new([nodes[2]]),
    ];
    let program = sim::Program::compile(&graph, &free_blocks, &[], &factors).unwrap();
    let exact = Exact::new(&graph, &factors);

    let schedule = sim::Schedule {
        n_warmup: 200,
        n_samples: 4000,
        steps_per_sample: 2,
    };
    let mut state = sim::State::random(&program, 32, 5);

    let moments = vec![
        vec![nodes[0]],
        vec![nodes[1]],
        vec![nodes[2]],
        vec![nodes[0], nodes[1]],
        vec![nodes[0], nodes[1], nodes[2]],
    ];
    let estimated = sampler.accumulate_moments(&program, &schedule, &mut state, 6, &moments);

    for (tuple, &estimate) in moments.iter().zip(estimated.iter()) {
        assert_close(
            estimate,
            exact.moment(tuple),
            0.02,
            &format!("moment of {tuple:?}"),
        );
    }
}

/// A mixed model where a categorical variable selects which bias
/// a pair of spins feels.
fn mixed_spin_categorical(sampler: &mut dyn sim::Sampler) {
    let mut graph = sim::Graph::new();
    let spins = graph.add_spins(2);
    let selector = graph.add_categorical(3);

    // Weight tables of shape [2, 3]: per spin, per selector state.
    let mixed_weights = vec![
        0.8, -0.8, 0.1, //
        -0.5, 0.5, 0.9,
    ];
    let factors = [
        sim::DiscreteFactor {
            spin_groups: vec![sim::Block::new(spins.clone())],
            cat_groups: vec![sim::Block::new([selector, selector])],
            weights: mixed_weights,
        },
        sim::DiscreteFactor::coupling([spins[0]], [spins[1]], vec![0.3]),
        sim::DiscreteFactor::cat_bias([selector], vec![0.0, 0.2, -0.1]),
    ];
    let free_blocks = [
        sim::Block::new([spins[0]]),
        sim::Block::new([spins[1]]),
        sim::Block::new([selector]),
    ];
    let program = sim::Program::compile(&graph, &free_blocks, &[], &factors).unwrap();
    let exact = Exact::new(&graph, &factors);

    let schedule = sim::Schedule {
        n_warmup: 200,
        n_samples: 4000,
        steps_per_sample: 2,
    };
    let mut state = sim::State::random(&program, 32, 7);
    let observed = [sim::Block::new(spins.clone()), sim::Block::new([selector])];
    let samples = sampler.sample_states(&program, &schedule, &mut state, 8, &observed);

    for (position, &node) in spins.iter().enumerate() {
        assert_close(
            empirical_marginal(&samples, 0, position as u32, 1),
            exact.marginal(node, 1),
            0.02,
            &format!("marginal of spin {position}"),
        );
    }
    for value in 0..3 {
        assert_close(
            empirical_marginal(&samples, 1, 0, value),
            exact.marginal(selector, value),
            0.02,
            &format!("marginal of selector state {value}"),
        );
    }
}

/// Clamped nodes must stay fixed, and the free nodes must follow
/// the conditional distribution given the clamped values.
fn clamping(sampler: &mut dyn sim::Sampler) {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(4);

    let factors = [
        sim::DiscreteFactor::bias(nodes.clone(), vec![0.1, -0.2, 0.3, 0.0]),
        sim::DiscreteFactor::coupling(&nodes[..3], &nodes[1..], vec![0.6, -0.4, 0.5]),
    ];
    let free_blocks = [sim::Block::new([nodes[1]]), sim::Block::new([nodes[2]])];
    let clamped_block = sim::Block::new([nodes[0], nodes[3]]);
    let clamped_values = [1u32, 0u32];
    let program = sim::Program::compile(
        &graph,
        &free_blocks,
        std::slice::from_ref(&clamped_block),
        &factors,
    )
    .unwrap();

    let schedule = sim::Schedule {
        n_warmup: 200,
        n_samples: 4000,
        steps_per_sample: 2,
    };
    let mut state = sim::State::random(&program, 32, 9);
    state.write_block(&clamped_block, &clamped_values);

    let observed = [sim::Block::new(nodes.clone())];
    let samples = sampler.sample_states(&program, &schedule, &mut state, 10, &observed);

    // Clamped nodes never move.
    assert_close(
        empirical_marginal(&samples, 0, 0, clamped_values[0]),
        1.0,
        1.0e-9,
        "clamped node 0",
    );
    assert_close(
        empirical_marginal(&samples, 0, 3, clamped_values[1]),
        1.0,
        1.0e-9,
        "clamped node 3",
    );

    // Free nodes follow the conditional distribution.
    let exact = Exact::new(&graph, &factors);
    let condition = exact.joint(&[nodes[0], nodes[3]], &clamped_values);
    for (position, node) in [(1u32, nodes[1]), (2, nodes[2])] {
        let joint = exact.joint(&[nodes[0], node, nodes[3]], &[1, 1, 0]);
        assert_close(
            empirical_marginal(&samples, 0, position, 1),
            joint / condition,
            0.02,
            &format!("conditional marginal of node {position}"),
        );
    }
}

#[test]
fn cpu_ising_chain() {
    ising_chain(&mut sim::CpuSampler::new());
}

#[test]
fn cpu_potts_chain() {
    potts_chain(&mut sim::CpuSampler::new());
}

#[test]
fn cpu_higher_order() {
    higher_order(&mut sim::CpuSampler::new());
}

#[test]
fn cpu_mixed_spin_categorical() {
    mixed_spin_categorical(&mut sim::CpuSampler::new());
}

#[test]
fn cpu_clamping() {
    clamping(&mut sim::CpuSampler::new());
}

#[test]
fn cpu_determinism() {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(6);
    let factors = [sim::DiscreteFactor::coupling(
        &nodes[..5],
        &nodes[1..],
        vec![0.5; 5],
    )];
    let free_blocks = [
        sim::Block::new(nodes.iter().copied().step_by(2).collect::<Vec<_>>()),
        sim::Block::new(nodes.iter().copied().skip(1).step_by(2).collect::<Vec<_>>()),
    ];
    let program = sim::Program::compile(&graph, &free_blocks, &[], &factors).unwrap();
    let schedule = sim::Schedule {
        n_warmup: 10,
        n_samples: 50,
        steps_per_sample: 1,
    };
    let observed = [sim::Block::new(nodes)];

    let run = |seed: u32| {
        let mut state = sim::State::zeros(&program, 4);
        let samples =
            sim::CpuSampler::new().sample_states(&program, &schedule, &mut state, seed, &observed);
        let mut values = Vec::new();
        for sample in 0..50 {
            for chain in 0..4 {
                for position in 0..6 {
                    values.push(samples.value(0, sample, chain, position));
                }
            }
        }
        values
    };

    assert_eq!(run(42), run(42), "same seed must reproduce");
    assert_ne!(run(42), run(43), "different seeds must diverge");
}
