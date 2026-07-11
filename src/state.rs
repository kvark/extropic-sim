use super::{Block, Node, NodeKind, Program, rng};

/// Number of Gibbs steps to run and samples to collect.
///
/// One *step* updates every free block of the program once.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    /// Steps to run before the first sample is recorded.
    pub n_warmup: u32,
    /// Number of samples to record.
    pub n_samples: u32,
    /// Steps between consecutive samples.
    pub steps_per_sample: u32,
}

/// One entry of the unrolled run: advance the chains by one Gibbs
/// step, or record the sample with the given index.
///
/// Both sampler backends execute the schedule through [`run_ops`],
/// so their chains stay in lockstep by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RunOp {
    Step,
    Record(u32),
}

/// Unroll a schedule into the shared operation sequence: warmup
/// steps, a sample right after the warmup, and `steps_per_sample`
/// steps before each subsequent sample.
pub(crate) fn run_ops(schedule: &Schedule) -> impl Iterator<Item = RunOp> {
    let warmup = (0..schedule.n_warmup).map(|_| RunOp::Step);
    let steps_per_sample = schedule.steps_per_sample;
    let sampling = (0..schedule.n_samples).flat_map(move |sample| {
        let steps = if sample == 0 { 0 } else { steps_per_sample };
        (0..steps)
            .map(|_| RunOp::Step)
            .chain(Some(RunOp::Record(sample)))
    });
    warmup.chain(sampling)
}

/// Dense state of a batch of independent sampling chains.
///
/// Holds one value per node per chain: 0/1 for spin nodes,
/// the category index for categorical nodes.
#[derive(Clone, Debug)]
pub struct State {
    pub(crate) n_chains: u32,
    pub(crate) n_nodes: u32,
    pub(crate) values: Vec<u32>,
}

impl State {
    /// Create an all-zero state for `n_chains` chains.
    ///
    /// # Panics
    ///
    /// Panics if `n_chains` is zero, or if the total number of values
    /// doesn't fit in 32 bits of addressing.
    pub fn zeros(program: &Program, n_chains: u32) -> Self {
        let n_nodes = program.node_count() as u32;
        assert!(n_chains > 0, "at least one chain is required");
        assert!(
            n_nodes as u64 * n_chains as u64 <= u32::MAX as u64,
            "the state of {n_nodes} nodes times {n_chains} chains exceeds 32-bit addressing",
        );
        Self {
            n_chains,
            n_nodes,
            values: vec![0; n_nodes as usize * n_chains as usize],
        }
    }

    /// Create a uniformly random state for `n_chains` chains.
    pub fn random(program: &Program, n_chains: u32, seed: u32) -> Self {
        let mut state = Self::zeros(program, n_chains);
        for chain in 0..n_chains {
            for (index, kind) in program.node_kinds.iter().enumerate() {
                let u = rng::uniform(seed, rng::STEP_RANDOM_INIT, chain, index as u32);
                let value = (u * kind.state_count() as f32) as u32;
                state.set(chain, Node(index as u32), value);
            }
        }
        state
    }

    /// Number of chains.
    pub fn chain_count(&self) -> u32 {
        self.n_chains
    }

    fn index(&self, chain: u32, node: Node) -> usize {
        assert!(node.0 < self.n_nodes, "{node:?} is not part of the state");
        assert!(chain < self.n_chains, "chain {chain} is out of range");
        (chain * self.n_nodes + node.0) as usize
    }

    /// Value of a node in one chain.
    pub fn get(&self, chain: u32, node: Node) -> u32 {
        self.values[self.index(chain, node)]
    }

    /// Set the value of a node in one chain.
    ///
    /// The value must be valid for the kind of the node: 0/1 for
    /// spins, less than the state count for categorical nodes.
    /// Samplers reject states holding out-of-range values.
    pub fn set(&mut self, chain: u32, node: Node, value: u32) {
        let index = self.index(chain, node);
        self.values[index] = value;
    }

    /// Assign per-node values of a block, identically in every chain.
    ///
    /// Handy for initializing clamped blocks.
    pub fn write_block(&mut self, block: &Block, values: &[u32]) {
        assert_eq!(block.len(), values.len());
        for chain in 0..self.n_chains {
            for (&node, &value) in block.nodes.iter().zip(values.iter()) {
                self.set(chain, node, value);
            }
        }
    }

    pub(crate) fn chain_values(&self, chain: u32) -> &[u32] {
        let base = (chain * self.n_nodes) as usize;
        &self.values[base..base + self.n_nodes as usize]
    }
}

/// Check that a state is compatible with a program and all its
/// values are in range, so that both backends see the same input.
pub(crate) fn validate_state(program: &Program, state: &State) {
    assert_eq!(
        state.n_nodes as usize,
        program.node_count(),
        "the state doesn't match the node count of the program",
    );
    for chain in 0..state.n_chains {
        let values = state.chain_values(chain);
        for (index, (&value, kind)) in values.iter().zip(program.node_kinds.iter()).enumerate() {
            assert!(
                value < kind.state_count(),
                "value {value} of node {index} in chain {chain} is out of range for {kind:?}",
            );
        }
    }
}

/// How the values of one observed block are packed in sample storage.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ObservedBlockLayout {
    /// Offset of the block region in the storage, in words.
    pub base: u32,
    /// Words per (sample, chain) frame: `ceil(len / 32)` for spin
    /// blocks (one bit per node), `len` for categorical blocks.
    pub words_per_frame: u32,
    pub is_spin: bool,
    pub node_count: u32,
}

/// Packed storage layout for a set of observed blocks.
#[derive(Clone, Debug)]
pub(crate) struct ObservedLayout {
    pub blocks: Vec<ObservedBlockLayout>,
    pub total_words: u32,
}

impl ObservedLayout {
    /// Lay out storage for the observed blocks.
    ///
    /// # Panics
    ///
    /// Panics if an observed block is empty, mixes spin and
    /// categorical nodes, references a node outside the program,
    /// or if the total storage exceeds 32-bit addressing.
    pub fn new(program: &Program, observed: &[Block], n_samples: u32, n_chains: u32) -> Self {
        let n_frames = n_samples as u64 * n_chains as u64;
        let mut base = 0u64;
        let blocks = observed
            .iter()
            .map(|block| {
                let &first = block
                    .nodes
                    .first()
                    .expect("observed blocks must not be empty");
                let is_spin = program.node_kinds[first.index()] == NodeKind::Spin;
                for &node in block.nodes.iter() {
                    assert_eq!(
                        program.node_kinds[node.index()] == NodeKind::Spin,
                        is_spin,
                        "observed block mixes spin and categorical nodes",
                    );
                }
                let node_count = block.len() as u32;
                let words_per_frame = if is_spin {
                    node_count.div_ceil(32)
                } else {
                    node_count
                };
                let layout = ObservedBlockLayout {
                    base: base as u32,
                    words_per_frame,
                    is_spin,
                    node_count,
                };
                base += words_per_frame as u64 * n_frames;
                assert!(
                    base <= u32::MAX as u64,
                    "sample storage exceeds 32-bit addressing; \
                     reduce the sample count, chains, or observed nodes",
                );
                layout
            })
            .collect();
        Self {
            blocks,
            total_words: base as u32,
        }
    }
}

/// Samples recorded from a run, for a list of observed blocks.
///
/// Spin values are packed one bit per node, categorical values take
/// a full word each.
pub struct Samples {
    pub(crate) layout: ObservedLayout,
    pub(crate) n_samples: u32,
    pub(crate) n_chains: u32,
    pub(crate) data: Vec<u32>,
}

impl Samples {
    /// Number of recorded samples.
    pub fn sample_count(&self) -> u32 {
        self.n_samples
    }

    /// Number of chains.
    pub fn chain_count(&self) -> u32 {
        self.n_chains
    }

    /// Value of node `position` of observed block `block_index`,
    /// in the given sample and chain: 0/1 for spins, the category
    /// index for categorical nodes.
    pub fn value(&self, block_index: usize, sample: u32, chain: u32, position: u32) -> u32 {
        let block = &self.layout.blocks[block_index];
        let frame = sample * self.n_chains + chain;
        let frame_base = block.base + frame * block.words_per_frame;
        if block.is_spin {
            let word = self.data[(frame_base + position / 32) as usize];
            (word >> (position % 32)) & 1
        } else {
            self.data[(frame_base + position) as usize]
        }
    }

    /// Value of node `position` as a spin in {-1, +1}.
    pub fn spin(&self, block_index: usize, sample: u32, chain: u32, position: u32) -> i32 {
        self.value(block_index, sample, chain, position) as i32 * 2 - 1
    }

    /// Mean spin of a node over all samples and chains.
    pub fn mean_spin(&self, block_index: usize, position: u32) -> f64 {
        let mut total = 0i64;
        for sample in 0..self.n_samples {
            for chain in 0..self.n_chains {
                total += self.spin(block_index, sample, chain, position) as i64;
            }
        }
        total as f64 / (self.n_samples as u64 * self.n_chains as u64) as f64
    }
}

pub(crate) fn write_sample_frame(
    layout: &ObservedLayout,
    observed: &[Block],
    data: &mut [u32],
    chain_values: &[u32],
    sample: u32,
    n_chains: u32,
    chain: u32,
) {
    let frame = sample * n_chains + chain;
    for (block, block_layout) in observed.iter().zip(layout.blocks.iter()) {
        let frame_base = (block_layout.base + frame * block_layout.words_per_frame) as usize;
        if block_layout.is_spin {
            for (position, &node) in block.nodes.iter().enumerate() {
                let word = &mut data[frame_base + position / 32];
                *word |= (chain_values[node.index()] & 1) << (position % 32);
            }
        } else {
            for (position, &node) in block.nodes.iter().enumerate() {
                data[frame_base + position] = chain_values[node.index()];
            }
        }
    }
}
