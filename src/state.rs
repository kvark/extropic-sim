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
    pub fn zeros(program: &Program, n_chains: u32) -> Self {
        Self {
            n_chains,
            n_nodes: program.node_count() as u32,
            values: vec![0; program.node_count() * n_chains as usize],
        }
    }

    /// Create a uniformly random state for `n_chains` chains.
    pub fn random(program: &Program, n_chains: u32, seed: u32) -> Self {
        let mut state = Self::zeros(program, n_chains);
        for chain in 0..n_chains {
            for (index, kind) in program.node_kinds.iter().enumerate() {
                let u = rng::uniform(seed, !0, chain, index as u32);
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

    /// Value of a node in one chain.
    pub fn get(&self, chain: u32, node: Node) -> u32 {
        self.values[(chain * self.n_nodes + node.0) as usize]
    }

    /// Set the value of a node in one chain.
    pub fn set(&mut self, chain: u32, node: Node, value: u32) {
        self.values[(chain * self.n_nodes + node.0) as usize] = value;
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
    pub fn new(program: &Program, observed: &[Block], n_samples: u32, n_chains: u32) -> Self {
        let n_frames = n_samples * n_chains;
        let mut base = 0;
        let blocks = observed
            .iter()
            .map(|block| {
                let is_spin = program.node_kinds[block.nodes[0].index()] == NodeKind::Spin;
                let node_count = block.len() as u32;
                let words_per_frame = if is_spin {
                    node_count.div_ceil(32)
                } else {
                    node_count
                };
                let layout = ObservedBlockLayout {
                    base,
                    words_per_frame,
                    is_spin,
                    node_count,
                };
                base += words_per_frame * n_frames;
                layout
            })
            .collect();
        Self {
            blocks,
            total_words: base,
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
