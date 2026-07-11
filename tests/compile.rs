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
