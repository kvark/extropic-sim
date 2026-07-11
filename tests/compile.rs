//! Validation of program compilation errors.

use extropic_sim as sim;

fn chain_factors(nodes: &[sim::Node]) -> Vec<sim::DiscreteFactor> {
    vec![sim::DiscreteFactor::coupling(
        &nodes[..nodes.len() - 1],
        &nodes[1..],
        vec![1.0; nodes.len() - 1],
    )]
}

#[test]
fn rejects_empty_block() {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(2);
    let result = sim::Program::compile(
        &graph,
        &[sim::Block::new(nodes), sim::Block::new([])],
        &[],
        &[],
    );
    assert_eq!(result.err(), Some(sim::Error::EmptyBlock));
}

#[test]
fn rejects_mixed_block() {
    let mut graph = sim::Graph::new();
    let spin = graph.add_spin();
    let cat = graph.add_categorical(3);
    let result = sim::Program::compile(&graph, &[sim::Block::new([spin, cat])], &[], &[]);
    assert_eq!(result.err(), Some(sim::Error::MixedBlock(spin, cat)));
}

#[test]
fn rejects_duplicate_node() {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(2);
    let result = sim::Program::compile(
        &graph,
        &[sim::Block::new(nodes.clone())],
        &[sim::Block::new([nodes[1]])],
        &[],
    );
    assert_eq!(result.err(), Some(sim::Error::DuplicateNode(nodes[1])));
}

#[test]
fn rejects_uncovered_node() {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(3);
    let result = sim::Program::compile(
        &graph,
        &[sim::Block::new([nodes[0]]), sim::Block::new([nodes[1]])],
        &[],
        &chain_factors(&nodes),
    );
    assert_eq!(result.err(), Some(sim::Error::UncoveredNode(nodes[2])));
}

#[test]
fn rejects_intra_block_interaction() {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(3);
    let result = sim::Program::compile(
        &graph,
        &[sim::Block::new(nodes.clone())],
        &[],
        &chain_factors(&nodes),
    );
    assert!(matches!(
        result.err(),
        Some(sim::Error::IntraBlockInteraction(..))
    ));
}

#[test]
fn rejects_weight_length_mismatch() {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(3);
    let factor = sim::DiscreteFactor::bias(nodes.clone(), vec![0.0; 2]);
    let blocks: Vec<_> = nodes.iter().map(|&node| sim::Block::new([node])).collect();
    let result = sim::Program::compile(&graph, &blocks, &[], &[factor]);
    assert_eq!(
        result.err(),
        Some(sim::Error::WeightLengthMismatch {
            expected: 3,
            actual: 2,
        })
    );
}

#[test]
fn rejects_group_kind_mismatch() {
    let mut graph = sim::Graph::new();
    let spin = graph.add_spin();
    let cat = graph.add_categorical(3);
    let factor = sim::DiscreteFactor::coupling([spin], [cat], vec![1.0]);
    let result = sim::Program::compile(
        &graph,
        &[sim::Block::new([spin]), sim::Block::new([cat])],
        &[],
        &[factor],
    );
    assert_eq!(result.err(), Some(sim::Error::GroupKindMismatch(cat)));
}

#[test]
fn rejects_bad_state_count() {
    let mut graph = sim::Graph::new();
    let cat = graph.add_categorical(1);
    let result = sim::Program::compile(&graph, &[sim::Block::new([cat])], &[], &[]);
    assert_eq!(result.err(), Some(sim::Error::BadStateCount(cat)));
}

#[test]
fn clamped_head_records_are_skipped() {
    // A factor whose head is clamped produces no updates for it,
    // but still influences the free nodes.
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(2);
    let program = sim::Program::compile(
        &graph,
        &[sim::Block::new([nodes[0]])],
        &[sim::Block::new([nodes[1]])],
        &chain_factors(&nodes),
    )
    .unwrap();
    assert_eq!(program.block_count(), 1);
}

#[test]
fn coloring_produces_valid_blocks() {
    // A random-ish graph: coloring must produce a partition that
    // compiles, i.e. with no intra-block interactions.
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(20);
    let mut edges = Vec::new();
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if (i * 7 + j * 13) % 5 == 0 {
                edges.push((nodes[i], nodes[j]));
            }
        }
    }
    let weights = vec![0.1; edges.len()];
    let heads: Vec<_> = edges.iter().map(|&(a, _)| a).collect();
    let tails: Vec<_> = edges.iter().map(|&(_, b)| b).collect();
    let factors = [sim::DiscreteFactor::coupling(heads, tails, weights)];

    let blocks = sim::color_blocks(&graph, &nodes, &edges);
    let total: usize = blocks.iter().map(|block| block.len()).sum();
    assert_eq!(total, nodes.len());
    sim::Program::compile(&graph, &blocks, &[], &factors).unwrap();
}

#[test]
fn rejects_mixed_state_cat_group() {
    // Categorical group members must agree on the state count,
    // since it shapes the weight tensor.
    let mut graph = sim::Graph::new();
    let c3 = graph.add_categorical(3);
    let c5 = graph.add_categorical(5);
    let factor = sim::DiscreteFactor::cat_bias(vec![c3, c5], vec![0.0; 6]);
    let result = sim::Program::compile(
        &graph,
        &[sim::Block::new([c3]), sim::Block::new([c5])],
        &[],
        &[factor],
    );
    assert_eq!(result.err(), Some(sim::Error::MixedBlock(c3, c5)));
}

#[test]
fn rejects_spin_in_cat_group_tail() {
    let mut graph = sim::Graph::new();
    let c3 = graph.add_categorical(3);
    let spin = graph.add_spin();
    let factor = sim::DiscreteFactor::cat_bias(vec![c3, spin], vec![0.0; 6]);
    let result = sim::Program::compile(
        &graph,
        &[sim::Block::new([c3]), sim::Block::new([spin])],
        &[],
        &[factor],
    );
    assert_eq!(result.err(), Some(sim::Error::GroupKindMismatch(spin)));
}

#[test]
fn updates_weights_in_place() {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(2);
    let factors = [sim::DiscreteFactor::coupling(
        [nodes[0]],
        [nodes[1]],
        vec![0.5],
    )];
    let blocks = [sim::Block::new([nodes[0]]), sim::Block::new([nodes[1]])];
    let mut program = sim::Program::compile(&graph, &blocks, &[], &factors).unwrap();

    let updated = [sim::DiscreteFactor::coupling(
        [nodes[0]],
        [nodes[1]],
        vec![-0.5],
    )];
    program.update_weights(&updated).unwrap();

    let bad = [
        sim::DiscreteFactor::coupling(&nodes[..1], &nodes[1..], vec![0.1; 1]),
        sim::DiscreteFactor::bias(nodes, vec![0.0; 2]),
    ];
    assert!(program.update_weights(&bad).is_err());
}

#[test]
#[should_panic(expected = "out of range")]
fn rejects_invalid_state_values() {
    use sim::Sampler as _;
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(2);
    let factors = chain_factors(&nodes);
    let blocks = [sim::Block::new([nodes[0]]), sim::Block::new([nodes[1]])];
    let program = sim::Program::compile(&graph, &blocks, &[], &factors).unwrap();
    let schedule = sim::Schedule {
        n_warmup: 1,
        n_samples: 1,
        steps_per_sample: 1,
    };
    let mut state = sim::State::zeros(&program, 1);
    state.set(0, nodes[0], 7);
    let _ = sim::CpuSampler::new().sample_states(&program, &schedule, &mut state, 0, &[]);
}

#[test]
#[should_panic(expected = "mixes spin and categorical")]
fn rejects_mixed_observed_block() {
    use sim::Sampler as _;
    let mut graph = sim::Graph::new();
    let spin = graph.add_spin();
    let cat = graph.add_categorical(3);
    let program = sim::Program::compile(
        &graph,
        &[sim::Block::new([spin]), sim::Block::new([cat])],
        &[],
        &[],
    )
    .unwrap();
    let schedule = sim::Schedule {
        n_warmup: 0,
        n_samples: 1,
        steps_per_sample: 1,
    };
    let mut state = sim::State::zeros(&program, 1);
    let _ = sim::CpuSampler::new().sample_states(
        &program,
        &schedule,
        &mut state,
        0,
        &[sim::Block::new([spin, cat])],
    );
}
