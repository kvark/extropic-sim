use super::{Block, DiscreteFactor, Graph, Node, NodeKind};
use std::{collections::HashMap, fmt};

/// Maximum number of states of a categorical node.
///
/// Bounded by the fixed-size scratch array in the GPU kernel.
pub const MAX_CATEGORICAL_STATES: u32 = 256;

/// An error produced when compiling a [`Program`].
#[derive(Debug, PartialEq)]
pub enum Error {
    /// A block has no nodes.
    EmptyBlock,
    /// A block mixes nodes of different kinds.
    MixedBlock(Node, Node),
    /// A node appears more than once across the free and clamped blocks.
    DuplicateNode(Node),
    /// A factor references a node that is in no free or clamped block.
    UncoveredNode(Node),
    /// Node groups of one factor differ in length.
    GroupLengthMismatch,
    /// The length of a factor weight tensor doesn't match its node groups.
    WeightLengthMismatch { expected: usize, actual: usize },
    /// A spin group contains a categorical node, or the other way around.
    GroupKindMismatch(Node),
    /// Two nodes of one free block interact, breaking parallel updates.
    IntraBlockInteraction(Node, Node),
    /// A categorical node has fewer than 2 or too many states.
    BadStateCount(Node),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::EmptyBlock => write!(formatter, "block is empty"),
            Self::MixedBlock(a, b) => write!(
                formatter,
                "block mixes nodes of different kinds ({a:?} vs {b:?})"
            ),
            Self::DuplicateNode(node) => write!(
                formatter,
                "{node:?} appears more than once across the free and clamped blocks"
            ),
            Self::UncoveredNode(node) => write!(
                formatter,
                "{node:?} is referenced by a factor but not covered by any block"
            ),
            Self::GroupLengthMismatch => {
                write!(formatter, "node groups of a factor differ in length")
            }
            Self::WeightLengthMismatch { expected, actual } => write!(
                formatter,
                "weight tensor length {actual} doesn't match the expected {expected}"
            ),
            Self::GroupKindMismatch(node) => write!(
                formatter,
                "{node:?} has the wrong kind for its factor group"
            ),
            Self::IntraBlockInteraction(head, tail) => write!(
                formatter,
                "{head:?} and {tail:?} interact but belong to the same free block"
            ),
            Self::BadStateCount(node) => write!(
                formatter,
                "{node:?} must have between 2 and {MAX_CATEGORICAL_STATES} states"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Kind shared by all nodes of one block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BlockKind {
    Spin,
    Categorical { states: u32 },
}

/// Fixed header size of one record in the update stream, in words.
pub(crate) const RECORD_HEADER_WORDS: usize = 3;

/// Compiled update schedule of a single free block.
///
/// For every node of the block, `records` holds the contributions of all
/// factor instances that involve the node, as a flat `u32` stream indexed
/// by CSR-style `offsets`. Each record is laid out as:
///
/// ```text
/// [0]              weight base index
/// [1]              head stride (categorical head only, 0 for spins)
/// [2]              spin tail count | categorical tail count << 16
/// [3 ..]           spin tail node ids
/// [3 + n_spin ..]  (categorical tail node id, stride) pairs
/// ```
///
/// A spin head accumulates `gamma += prod(spin tails) * weights[base + dot(cat tails, strides)]`
/// and samples `P(up) = sigmoid(2 * gamma)`. A categorical head accumulates
/// `theta[k] += prod(spin tails) * weights[base + k * head_stride + dot(cat tails, strides)]`
/// and samples the softmax of `theta`.
#[derive(Debug)]
pub(crate) struct BlockProgram {
    pub node_ids: Vec<u32>,
    pub kind: BlockKind,
    pub offsets: Vec<u32>,
    pub records: Vec<u32>,
}

/// A model compiled into a block Gibbs sampling schedule.
///
/// Produced by [`Program::compile`] from a set of factors and a partition
/// of the interacting nodes into *free* blocks (updated in order, one
/// full pass per step) and *clamped* blocks (read but never updated).
pub struct Program {
    pub(crate) node_kinds: Vec<NodeKind>,
    pub(crate) weights: Vec<f32>,
    pub(crate) blocks: Vec<BlockProgram>,
    pub(crate) max_states: u32,
}

impl Program {
    /// Compile `factors` into a sampling program.
    ///
    /// Every step of the resulting program updates the free blocks
    /// sequentially, in the given order. Nodes within one block are
    /// updated in parallel, so they must not interact with each other.
    /// Nodes of `clamped_blocks` keep whatever state they were assigned.
    pub fn compile(
        graph: &Graph,
        free_blocks: &[Block],
        clamped_blocks: &[Block],
        factors: &[DiscreteFactor],
    ) -> Result<Self, Error> {
        // Locate every covered node: free ones by (block, position).
        let mut free_locations = HashMap::<Node, (usize, usize)>::new();
        let mut clamped = HashMap::<Node, ()>::new();
        for (block_index, block) in free_blocks.iter().enumerate() {
            for (position, &node) in block.nodes.iter().enumerate() {
                if free_locations
                    .insert(node, (block_index, position))
                    .is_some()
                {
                    return Err(Error::DuplicateNode(node));
                }
            }
        }
        for block in clamped_blocks.iter() {
            for &node in block.nodes.iter() {
                if free_locations.contains_key(&node) || clamped.insert(node, ()).is_some() {
                    return Err(Error::DuplicateNode(node));
                }
            }
        }

        let mut blocks = free_blocks
            .iter()
            .map(|block| {
                let kind = block_kind(graph, block)?;
                Ok(BlockProgram {
                    node_ids: block.nodes.iter().map(|node| node.0).collect(),
                    kind,
                    offsets: Vec::new(),
                    records: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        for block in clamped_blocks.iter() {
            let _ = block_kind(graph, block)?;
        }

        // Gather the records of each free node.
        let mut node_records = vec![Vec::<u32>::new(); graph.node_count()];
        let mut weights = Vec::<f32>::new();
        for factor in factors.iter() {
            compile_factor(
                graph,
                factor,
                weights.len() as u32,
                &free_locations,
                &clamped,
                &mut node_records,
            )?;
            weights.extend_from_slice(&factor.weights);
        }

        // Flatten per-node records into per-block CSR streams,
        // rejecting interactions within one block.
        for (block_index, block) in blocks.iter_mut().enumerate() {
            block.offsets.push(0);
            for &node_id in block.node_ids.iter() {
                let records = &node_records[node_id as usize];
                for tail_id in record_tail_ids(records) {
                    if let Some(&(tail_block, _)) = free_locations.get(&Node(tail_id))
                        && tail_block == block_index
                    {
                        return Err(Error::IntraBlockInteraction(Node(node_id), Node(tail_id)));
                    }
                }
                block.records.extend_from_slice(records);
                block.offsets.push(block.records.len() as u32);
            }
        }

        let max_states = blocks
            .iter()
            .map(|block| match block.kind {
                BlockKind::Spin => 2,
                BlockKind::Categorical { states } => states,
            })
            .max()
            .unwrap_or(2);

        Ok(Self {
            node_kinds: graph.nodes().map(|node| graph.node_kind(node)).collect(),
            weights,
            blocks,
            max_states,
        })
    }

    /// Replace the weight values of the program without recompiling.
    ///
    /// The factors must have the same node structure as the ones the
    /// program was compiled from; only the weight values may differ.
    /// The record streams reference weights by index, so this is all
    /// that a training loop needs between epochs.
    pub fn update_weights(&mut self, factors: &[DiscreteFactor]) -> Result<(), Error> {
        let total = factors.iter().map(|factor| factor.weights.len()).sum();
        if self.weights.len() != total {
            return Err(Error::WeightLengthMismatch {
                expected: self.weights.len(),
                actual: total,
            });
        }
        self.weights.clear();
        for factor in factors.iter() {
            self.weights.extend_from_slice(&factor.weights);
        }
        Ok(())
    }

    /// Total number of nodes in the source graph.
    pub fn node_count(&self) -> usize {
        self.node_kinds.len()
    }

    /// Number of free blocks updated per sampling step.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

fn block_kind(graph: &Graph, block: &Block) -> Result<BlockKind, Error> {
    let &first = block.nodes.first().ok_or(Error::EmptyBlock)?;
    let kind = match graph.node_kind(first) {
        NodeKind::Spin => BlockKind::Spin,
        NodeKind::Categorical { states } => {
            if !(2..=MAX_CATEGORICAL_STATES).contains(&states) {
                return Err(Error::BadStateCount(first));
            }
            BlockKind::Categorical { states }
        }
    };
    for &node in block.nodes[1..].iter() {
        let node_kind = match graph.node_kind(node) {
            NodeKind::Spin => BlockKind::Spin,
            NodeKind::Categorical { states } => BlockKind::Categorical { states },
        };
        if node_kind != kind {
            return Err(Error::MixedBlock(first, node));
        }
    }
    Ok(kind)
}

/// A single decoded record of the update stream.
pub(crate) struct Record<'a> {
    pub weight_base: u32,
    pub head_stride: u32,
    pub spin_tails: &'a [u32],
    /// Pairs of (node id, weight stride).
    pub cat_tails: &'a [u32],
}

impl Record<'_> {
    /// The spin product of the tails and the resolved weight index,
    /// given the dense per-node state of one chain.
    pub fn evaluate(&self, values: &[u32]) -> (f32, u32) {
        let mut product = 1.0f32;
        for &tail in self.spin_tails.iter() {
            product *= values[tail as usize] as f32 * 2.0 - 1.0;
        }
        let mut index = self.weight_base;
        for pair in self.cat_tails.chunks(2) {
            index += values[pair[0] as usize] * pair[1];
        }
        (product, index)
    }
}

/// Decode a record stream. The canonical CPU-side parser of the
/// format documented on [`BlockProgram`]; the GPU-side counterpart
/// lives in `shaders/gibbs.wgsl`.
pub(crate) fn parse_records<'a>(records: &'a [u32]) -> impl Iterator<Item = Record<'a>> {
    let mut cursor = 0;
    std::iter::from_fn(move || {
        if cursor >= records.len() {
            return None;
        }
        let spin_count = (records[cursor + 2] & 0xFFFF) as usize;
        let cat_count = (records[cursor + 2] >> 16) as usize;
        let tail_base = cursor + RECORD_HEADER_WORDS;
        let record = Record {
            weight_base: records[cursor],
            head_stride: records[cursor + 1],
            spin_tails: &records[tail_base..tail_base + spin_count],
            cat_tails: &records[tail_base + spin_count..tail_base + spin_count + 2 * cat_count],
        };
        cursor = tail_base + spin_count + 2 * cat_count;
        Some(record)
    })
}

/// Iterate over the tail node ids of a record stream.
fn record_tail_ids(records: &[u32]) -> impl Iterator<Item = u32> + '_ {
    parse_records(records).flat_map(|record| {
        record
            .spin_tails
            .iter()
            .copied()
            .chain(record.cat_tails.chunks(2).map(|pair| pair[0]))
            .collect::<Vec<_>>()
    })
}

fn compile_factor(
    graph: &Graph,
    factor: &DiscreteFactor,
    weight_base: u32,
    free_locations: &HashMap<Node, (usize, usize)>,
    clamped: &HashMap<Node, ()>,
    node_records: &mut [Vec<u32>],
) -> Result<(), Error> {
    let batch_len = factor.batch_len();
    if batch_len == 0 {
        return Err(Error::EmptyBlock);
    }
    for block in factor.spin_groups.iter().chain(factor.cat_groups.iter()) {
        if block.len() != batch_len {
            return Err(Error::GroupLengthMismatch);
        }
        for &node in block.nodes.iter() {
            if !free_locations.contains_key(&node) && !clamped.contains_key(&node) {
                return Err(Error::UncoveredNode(node));
            }
        }
    }
    for block in factor.spin_groups.iter() {
        for &node in block.nodes.iter() {
            if graph.node_kind(node) != NodeKind::Spin {
                return Err(Error::GroupKindMismatch(node));
            }
        }
    }

    // Row-major strides of the categorical axes of the weight tensor.
    // Nodes sharing an axis must agree on the state count, since it
    // determines the shape of the weight tensor.
    let mut cat_strides = vec![0u32; factor.cat_groups.len()];
    let mut instance_stride = 1u32;
    for (stride, block) in cat_strides.iter_mut().zip(factor.cat_groups.iter()).rev() {
        *stride = instance_stride;
        let &first = block.nodes.first().unwrap();
        let group_states = match graph.node_kind(first) {
            NodeKind::Categorical { states } if (2..=MAX_CATEGORICAL_STATES).contains(&states) => {
                states
            }
            NodeKind::Categorical { .. } => return Err(Error::BadStateCount(first)),
            NodeKind::Spin => return Err(Error::GroupKindMismatch(first)),
        };
        for &node in block.nodes[1..].iter() {
            match graph.node_kind(node) {
                NodeKind::Categorical { states } if states == group_states => {}
                NodeKind::Categorical { .. } => return Err(Error::MixedBlock(first, node)),
                NodeKind::Spin => return Err(Error::GroupKindMismatch(node)),
            }
        }
        instance_stride *= group_states;
    }
    let expected = batch_len * instance_stride as usize;
    if factor.weights.len() != expected {
        return Err(Error::WeightLengthMismatch {
            expected,
            actual: factor.weights.len(),
        });
    }

    for i in 0..batch_len {
        let base = weight_base + i as u32 * instance_stride;
        // One record for each spin head.
        for (head_group, head_block) in factor.spin_groups.iter().enumerate() {
            let head = head_block.nodes[i];
            if !free_locations.contains_key(&head) {
                continue;
            }
            let records = &mut node_records[head.index()];
            records.push(base);
            records.push(0);
            let spin_count = factor.spin_groups.len() as u32 - 1;
            records.push(spin_count | ((factor.cat_groups.len() as u32) << 16));
            for (tail_group, tail_block) in factor.spin_groups.iter().enumerate() {
                if tail_group != head_group {
                    records.push(tail_block.nodes[i].0);
                }
            }
            for (tail_block, &stride) in factor.cat_groups.iter().zip(cat_strides.iter()) {
                records.push(tail_block.nodes[i].0);
                records.push(stride);
            }
        }
        // One record for each categorical head.
        for (head_group, head_block) in factor.cat_groups.iter().enumerate() {
            let head = head_block.nodes[i];
            if !free_locations.contains_key(&head) {
                continue;
            }
            let records = &mut node_records[head.index()];
            records.push(base);
            records.push(cat_strides[head_group]);
            let spin_count = factor.spin_groups.len() as u32;
            records.push(spin_count | ((factor.cat_groups.len() as u32 - 1) << 16));
            for tail_block in factor.spin_groups.iter() {
                records.push(tail_block.nodes[i].0);
            }
            for (tail_group, (tail_block, &stride)) in
                factor.cat_groups.iter().zip(cat_strides.iter()).enumerate()
            {
                if tail_group != head_group {
                    records.push(tail_block.nodes[i].0);
                    records.push(stride);
                }
            }
        }
    }
    Ok(())
}
