use super::{
    Block, Node, NodeKind, Program, Sampler, Samples, Schedule, State,
    program::{BlockKind, BlockProgram, parse_records},
    rng,
    state::{ObservedLayout, RunOp, run_ops, validate_state, write_sample_frame},
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
        let count = (schedule.n_samples as u64 * state.n_chains as u64) as f64;
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
    validate_state(program, state);
    let mut theta_scratch = vec![0.0f32; program.max_states as usize];
    let mut counter = 0;
    for op in run_ops(schedule) {
        match op {
            RunOp::Step => {
                for block in program.blocks.iter() {
                    for chain in 0..state.n_chains {
                        update_block(
                            program,
                            block,
                            state,
                            seed,
                            counter,
                            chain,
                            &mut theta_scratch,
                        );
                    }
                    counter += 1;
                }
            }
            RunOp::Record(sample) => on_sample(state, sample),
        }
    }
}

fn update_block(
    program: &Program,
    block: &BlockProgram,
    state: &mut State,
    seed: u32,
    counter: u32,
    chain: u32,
    theta_scratch: &mut [f32],
) {
    let base = (chain * state.n_nodes) as usize;
    // Records never reference nodes of the block being updated,
    // which is validated at compile time, so in-place updates match
    // the snapshot semantics of a parallel block update.
    for (position, &node_id) in block.node_ids.iter().enumerate() {
        let records =
            &block.records[block.offsets[position] as usize..block.offsets[position + 1] as usize];
        let values = &mut state.values[base..base + state.n_nodes as usize];
        let u = rng::uniform(seed, counter, chain, node_id);
        match block.kind {
            BlockKind::Spin => {
                let mut gamma = 0.0f32;
                for record in parse_records(records) {
                    let (product, index) = record.evaluate(values);
                    gamma += product * program.weights[index as usize];
                }
                let p_up = 1.0 / (1.0 + (-2.0 * gamma).exp());
                values[node_id as usize] = (u < p_up) as u32;
            }
            BlockKind::Categorical { states } => {
                let theta = &mut theta_scratch[..states as usize];
                theta.fill(0.0);
                for record in parse_records(records) {
                    let (product, index) = record.evaluate(values);
                    for (k, value) in theta.iter_mut().enumerate() {
                        *value += product
                            * program.weights[(index + k as u32 * record.head_stride) as usize];
                    }
                }
                values[node_id as usize] = sample_softmax(theta, u);
            }
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
