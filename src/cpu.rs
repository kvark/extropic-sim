use super::{
    Block, Node, NodeKind, Program, Sampler, Samples, Schedule, State,
    program::{BlockKind, BlockProgram, RECORD_HEADER_WORDS},
    rng,
    state::{ObservedLayout, write_sample_frame},
};

/// A portable reference sampler that runs on the CPU.
///
/// Implements exactly the same algorithm as [`GpuSampler`](crate::GpuSampler),
/// including the random number generation, so the two produce statistically
/// identical chains for a given program and seed.
#[derive(Default)]
pub struct CpuSampler;

impl CpuSampler {
    /// Create a CPU sampler.
    pub fn new() -> Self {
        Self
    }
}

impl Sampler for CpuSampler {
    fn sample_states(
        &mut self,
        program: &Program,
        schedule: &Schedule,
        state: &mut State,
        seed: u32,
        observed: &[Block],
    ) -> Samples {
        let layout = ObservedLayout::new(program, observed, schedule.n_samples, state.n_chains);
        let mut data = vec![0u32; layout.total_words as usize];
        run(program, schedule, state, seed, |state, sample| {
            for chain in 0..state.n_chains {
                write_sample_frame(
                    &layout,
                    observed,
                    &mut data,
                    state.chain_values(chain),
                    sample,
                    state.n_chains,
                    chain,
                );
            }
        });
        Samples {
            layout,
            n_samples: schedule.n_samples,
            n_chains: state.n_chains,
            data,
        }
    }

    fn accumulate_moments(
        &mut self,
        program: &Program,
        schedule: &Schedule,
        state: &mut State,
        seed: u32,
        moments: &[Vec<Node>],
    ) -> Vec<f64> {
        for tuple in moments.iter() {
            for &node in tuple.iter() {
                assert_eq!(
                    program.node_kinds[node.index()],
                    NodeKind::Spin,
                    "moments are only defined for spin nodes",
                );
            }
        }
        let mut sums = vec![0i64; moments.len()];
        run(program, schedule, state, seed, |state, _| {
            for chain in 0..state.n_chains {
                let values = state.chain_values(chain);
                for (sum, tuple) in sums.iter_mut().zip(moments.iter()) {
                    let mut product = 1i64;
                    for &node in tuple.iter() {
                        product *= values[node.index()] as i64 * 2 - 1;
                    }
                    *sum += product;
                }
            }
        });
        let count = (schedule.n_samples * state.n_chains) as f64;
        sums.iter().map(|&sum| sum as f64 / count).collect()
    }
}

fn run(
    program: &Program,
    schedule: &Schedule,
    state: &mut State,
    seed: u32,
    mut on_sample: impl FnMut(&State, u32),
) {
    let mut counter = 0;
    for _ in 0..schedule.n_warmup {
        step(program, state, seed, &mut counter);
    }
    for sample in 0..schedule.n_samples {
        if sample != 0 {
            for _ in 0..schedule.steps_per_sample {
                step(program, state, seed, &mut counter);
            }
        }
        on_sample(state, sample);
    }
}

fn step(program: &Program, state: &mut State, seed: u32, counter: &mut u32) {
    for block in program.blocks.iter() {
        for chain in 0..state.n_chains {
            update_block(program, block, state, seed, *counter, chain);
        }
        *counter += 1;
    }
}

fn update_block(
    program: &Program,
    block: &BlockProgram,
    state: &mut State,
    seed: u32,
    counter: u32,
    chain: u32,
) {
    let base = (chain * state.n_nodes) as usize;
    // The borrow of `values` is split manually: records never reference
    // nodes of the block being updated, which is validated at compile time.
    for (position, &node_id) in block.node_ids.iter().enumerate() {
        let records =
            &block.records[block.offsets[position] as usize..block.offsets[position + 1] as usize];
        let values = &mut state.values[base..base + state.n_nodes as usize];
        let u = rng::uniform(seed, counter, chain, node_id);
        match block.kind {
            BlockKind::Spin => {
                let gamma = accumulate_gamma(records, values, &program.weights);
                let p_up = 1.0 / (1.0 + (-2.0 * gamma).exp());
                values[node_id as usize] = (u < p_up) as u32;
            }
            BlockKind::Categorical { states } => {
                let mut theta = vec![0.0f32; states as usize];
                accumulate_theta(records, values, &program.weights, &mut theta);
                values[node_id as usize] = sample_softmax(&theta, u);
            }
        }
    }
}

struct Record<'a> {
    weight_base: u32,
    head_stride: u32,
    spin_tails: &'a [u32],
    cat_tails: &'a [u32],
}

fn parse_records<'a>(records: &'a [u32]) -> impl Iterator<Item = Record<'a>> {
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

fn accumulate_gamma(records: &[u32], values: &[u32], weights: &[f32]) -> f32 {
    let mut gamma = 0.0;
    for record in parse_records(records) {
        let mut product = 1.0f32;
        for &tail in record.spin_tails.iter() {
            product *= values[tail as usize] as f32 * 2.0 - 1.0;
        }
        let mut index = record.weight_base;
        for pair in record.cat_tails.chunks(2) {
            index += values[pair[0] as usize] * pair[1];
        }
        gamma += product * weights[index as usize];
    }
    gamma
}

fn accumulate_theta(records: &[u32], values: &[u32], weights: &[f32], theta: &mut [f32]) {
    for record in parse_records(records) {
        let mut product = 1.0f32;
        for &tail in record.spin_tails.iter() {
            product *= values[tail as usize] as f32 * 2.0 - 1.0;
        }
        let mut index = record.weight_base;
        for pair in record.cat_tails.chunks(2) {
            index += values[pair[0] as usize] * pair[1];
        }
        for (k, value) in theta.iter_mut().enumerate() {
            *value += product * weights[(index + k as u32 * record.head_stride) as usize];
        }
    }
}

fn sample_softmax(theta: &[f32], u: f32) -> u32 {
    let mut max = theta[0];
    for &value in theta[1..].iter() {
        max = max.max(value);
    }
    let mut total = 0.0f32;
    for &value in theta.iter() {
        total += (value - max).exp();
    }
    let mut remaining = u * total;
    for (k, &value) in theta.iter().enumerate() {
        remaining -= (value - max).exp();
        if remaining <= 0.0 {
            return k as u32;
        }
    }
    theta.len() as u32 - 1
}
