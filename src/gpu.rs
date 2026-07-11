use blade_graphics as gpu;
use gpu::ShaderData as _;

use super::{
    Block, Node, NodeKind, Program, Sampler, Samples, Schedule, State,
    program::BlockKind,
    state::{ObservedLayout, RunOp, run_ops, validate_state},
};

/// Number of compute dispatches encoded per command buffer submission.
const OPS_PER_SUBMIT: usize = 512;
/// Workgroup size of all kernels, matching `gibbs.wgsl`.
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct UpdateParams {
    seed: u32,
    counter: u32,
    n_nodes: u32,
    n_chains: u32,
    node_count: u32,
    states: u32,
    offsets_base: u32,
    records_base: u32,
    node_ids_base: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct RecordParams {
    n_nodes: u32,
    n_chains: u32,
    sample: u32,
    node_count: u32,
    out_base: u32,
    words_per_frame: u32,
    node_ids_base: u32,
    is_spin: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct MomentParams {
    n_nodes: u32,
    n_chains: u32,
    n_moments: u32,
    pad0: u32,
}

#[derive(blade_macros::ShaderData)]
struct UpdateData {
    state: gpu::BufferPiece,
    records: gpu::BufferPiece,
    offsets: gpu::BufferPiece,
    node_ids: gpu::BufferPiece,
    weights: gpu::BufferPiece,
    params: UpdateParams,
}

#[derive(blade_macros::ShaderData)]
struct RecordData {
    state: gpu::BufferPiece,
    obs_node_ids: gpu::BufferPiece,
    out_samples: gpu::BufferPiece,
    record_params: RecordParams,
}

#[derive(blade_macros::ShaderData)]
struct MomentData {
    state: gpu::BufferPiece,
    moment_offsets: gpu::BufferPiece,
    moment_nodes: gpu::BufferPiece,
    accum: gpu::BufferPiece,
    moment_params: MomentParams,
}

/// GPU copies of the compiled program data, cached across runs.
struct ProgramBuffers {
    program_id: u64,
    weights_generation: u64,
    records: gpu::Buffer,
    offsets: gpu::Buffer,
    node_ids: gpu::Buffer,
    weights: gpu::Buffer,
    bases: Vec<BlockBases>,
}

struct Pipelines {
    max_states: u32,
    update_spins: gpu::ComputePipeline,
    update_categorical: gpu::ComputePipeline,
    record_states: gpu::ComputePipeline,
    accumulate_moments: gpu::ComputePipeline,
}

/// A sampler running on the GPU via `blade-graphics`.
///
/// Runs the same algorithm as [`CpuSampler`](crate::CpuSampler): programs
/// produce statistically equivalent chains on both backends. All chains
/// advance in parallel on the GPU, and each block update is one compute
/// dispatch, so wide models with many chains make the best use of it.
///
/// Sampler methods panic on invalid input (out-of-range state values,
/// malformed observed blocks) and when the GPU device is lost.
pub struct GpuSampler {
    context: gpu::Context,
    pipelines: Option<Pipelines>,
    program: Option<ProgramBuffers>,
}

impl GpuSampler {
    /// Create a sampler on the best available GPU.
    pub fn new() -> Result<Self, gpu::NotSupportedError> {
        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                validation: cfg!(debug_assertions),
                ..Default::default()
            })?
        };
        Ok(Self {
            context,
            pipelines: None,
            program: None,
        })
    }

    /// Name of the device backing this sampler.
    pub fn device_name(&self) -> String {
        self.context.device_information().device_name.clone()
    }

    fn ensure_pipelines(&mut self, max_states: u32) {
        if let Some(ref pipelines) = self.pipelines
            && pipelines.max_states == max_states
        {
            return;
        }
        if let Some(mut pipelines) = self.pipelines.take() {
            self.destroy_pipelines(&mut pipelines);
        }

        let source =
            include_str!("shaders/gibbs.wgsl").replace("MAX_STATES", &max_states.to_string());
        let shader = self.context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        self.pipelines = Some(Pipelines {
            max_states,
            update_spins: self
                .context
                .create_compute_pipeline(gpu::ComputePipelineDesc {
                    name: "update_spins",
                    data_layouts: &[&UpdateData::layout()],
                    compute: shader.at("update_spins"),
                }),
            update_categorical: self
                .context
                .create_compute_pipeline(gpu::ComputePipelineDesc {
                    name: "update_categorical",
                    data_layouts: &[&UpdateData::layout()],
                    compute: shader.at("update_categorical"),
                }),
            record_states: self
                .context
                .create_compute_pipeline(gpu::ComputePipelineDesc {
                    name: "record_states",
                    data_layouts: &[&RecordData::layout()],
                    compute: shader.at("record_states"),
                }),
            accumulate_moments: self
                .context
                .create_compute_pipeline(gpu::ComputePipelineDesc {
                    name: "accumulate_moments",
                    data_layouts: &[&MomentData::layout()],
                    compute: shader.at("accumulate_moments"),
                }),
        });
    }

    /// Upload the program data, or reuse the copy of the previous run.
    ///
    /// Weight updates between runs re-upload only the weights buffer.
    fn ensure_program(&mut self, program: &Program) {
        if let Some(ref mut buffers) = self.program
            && buffers.program_id == program.id
        {
            if buffers.weights_generation != program.weights_generation {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        program.weights.as_ptr() as *const u8,
                        buffers.weights.data(),
                        program.weights.len() * 4,
                    );
                }
                buffers.weights_generation = program.weights_generation;
            }
            return;
        }
        if let Some(mut buffers) = self.program.take() {
            self.destroy_program(&mut buffers);
        }

        // Concatenate the per-block program data.
        let mut offsets = Vec::new();
        let mut records = Vec::new();
        let mut node_ids = Vec::new();
        let bases: Vec<BlockBases> = program
            .blocks
            .iter()
            .map(|block| {
                let bases = BlockBases {
                    offsets: offsets.len() as u32,
                    records: records.len() as u32,
                    node_ids: node_ids.len() as u32,
                };
                offsets.extend_from_slice(&block.offsets);
                records.extend_from_slice(&block.records);
                node_ids.extend_from_slice(&block.node_ids);
                bases
            })
            .collect();

        let context = &self.context;
        self.program = Some(ProgramBuffers {
            program_id: program.id,
            weights_generation: program.weights_generation,
            records: create_upload_buffer(context, "records", bytemuck::cast_slice(&records)),
            offsets: create_upload_buffer(context, "offsets", bytemuck::cast_slice(&offsets)),
            node_ids: create_upload_buffer(context, "node_ids", bytemuck::cast_slice(&node_ids)),
            weights: create_upload_buffer(
                context,
                "weights",
                bytemuck::cast_slice(&program.weights),
            ),
            bases,
        });
    }

    fn destroy_program(&mut self, buffers: &mut ProgramBuffers) {
        self.context.destroy_buffer(buffers.records);
        self.context.destroy_buffer(buffers.offsets);
        self.context.destroy_buffer(buffers.node_ids);
        self.context.destroy_buffer(buffers.weights);
    }

    fn destroy_pipelines(&mut self, pipelines: &mut Pipelines) {
        self.context
            .destroy_compute_pipeline(&mut pipelines.update_spins);
        self.context
            .destroy_compute_pipeline(&mut pipelines.update_categorical);
        self.context
            .destroy_compute_pipeline(&mut pipelines.record_states);
        self.context
            .destroy_compute_pipeline(&mut pipelines.accumulate_moments);
    }
}

impl Drop for GpuSampler {
    fn drop(&mut self) {
        if let Some(mut pipelines) = self.pipelines.take() {
            self.destroy_pipelines(&mut pipelines);
        }
        if let Some(mut buffers) = self.program.take() {
            self.destroy_program(&mut buffers);
        }
    }
}

/// Submit the encoder once a chunk of dispatches has been recorded,
/// and start recording the next chunk while the GPU executes.
///
/// The encoder keeps two command buffers alive, so only the chunk
/// before the one just submitted needs to be finished.
fn flush_chunk(
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    pending: &mut Option<gpu::SyncPoint>,
    ops_in_chunk: &mut usize,
) {
    if *ops_in_chunk < OPS_PER_SUBMIT {
        return;
    }
    let sync_point = context.submit(encoder);
    if let Some(previous) = pending.replace(sync_point) {
        context
            .wait_for(&previous, !0)
            .expect("lost the GPU device while sampling");
    }
    encoder.start();
    *ops_in_chunk = 0;
}

/// Split a thread count into a dispatch grid that stays within the
/// guaranteed per-dimension workgroup limit.
fn dispatch_grid(threads: u32) -> [u32; 3] {
    const MAX_GROUPS: u32 = 0xFFFF;
    let groups = threads.div_ceil(WORKGROUP_SIZE);
    if groups <= MAX_GROUPS {
        [groups, 1, 1]
    } else {
        [MAX_GROUPS, groups.div_ceil(MAX_GROUPS), 1]
    }
}

fn create_upload_buffer(context: &gpu::Context, name: &str, contents: &[u8]) -> gpu::Buffer {
    let buffer = context.create_buffer(gpu::BufferDesc {
        name,
        size: contents.len().max(4) as u64,
        memory: gpu::Memory::Shared,
    });
    unsafe {
        std::ptr::copy_nonoverlapping(contents.as_ptr(), buffer.data(), contents.len());
    }
    buffer
}

/// Per-block base indices into the concatenated program buffers.
struct BlockBases {
    offsets: u32,
    records: u32,
    node_ids: u32,
}

#[derive(Clone, Copy)]
enum Observation<'a> {
    States {
        layout: &'a ObservedLayout,
        obs_node_ids: gpu::BufferPiece,
        out_samples: gpu::BufferPiece,
    },
    Moments {
        count: u32,
        offsets: gpu::BufferPiece,
        nodes: gpu::BufferPiece,
        accum: gpu::BufferPiece,
    },
}

impl Sampler for GpuSampler {
    fn sample_states(
        &mut self,
        program: &Program,
        schedule: &Schedule,
        state: &mut State,
        seed: u32,
        observed: &[Block],
    ) -> Samples {
        let layout = ObservedLayout::new(program, observed, schedule.n_samples, state.n_chains);
        let obs_node_ids: Vec<u32> = observed
            .iter()
            .flat_map(|block| block.nodes.iter().map(|node| node.0))
            .collect();
        let obs_buffer = create_upload_buffer(
            &self.context,
            "obs_node_ids",
            bytemuck::cast_slice(&obs_node_ids),
        );
        let out_buffer = self.context.create_buffer(gpu::BufferDesc {
            name: "out_samples",
            size: (layout.total_words.max(1) as u64) * 4,
            memory: gpu::Memory::Shared,
        });

        self.run(
            program,
            schedule,
            state,
            seed,
            Observation::States {
                layout: &layout,
                obs_node_ids: obs_buffer.into(),
                out_samples: out_buffer.into(),
            },
        );

        let mut data = vec![0u32; layout.total_words as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buffer.data() as *const u32,
                data.as_mut_ptr(),
                data.len(),
            );
        }
        self.context.destroy_buffer(obs_buffer);
        self.context.destroy_buffer(out_buffer);
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
        assert!(
            schedule.n_samples as u64 * state.n_chains as u64 <= i32::MAX as u64,
            "the GPU moment accumulator is 32-bit; reduce samples or chains",
        );
        let mut moment_offsets = vec![0u32];
        let mut moment_nodes = Vec::new();
        for tuple in moments.iter() {
            for &node in tuple.iter() {
                assert_eq!(
                    program.node_kinds[node.index()],
                    NodeKind::Spin,
                    "moments are only defined for spin nodes",
                );
                moment_nodes.push(node.0);
            }
            moment_offsets.push(moment_nodes.len() as u32);
        }
        let offsets_buffer = create_upload_buffer(
            &self.context,
            "moment_offsets",
            bytemuck::cast_slice(&moment_offsets),
        );
        let nodes_buffer = create_upload_buffer(
            &self.context,
            "moment_nodes",
            bytemuck::cast_slice(&moment_nodes),
        );
        let accum_buffer =
            create_upload_buffer(&self.context, "accum", &vec![0u8; moments.len().max(1) * 4]);

        self.run(
            program,
            schedule,
            state,
            seed,
            Observation::Moments {
                count: moments.len() as u32,
                offsets: offsets_buffer.into(),
                nodes: nodes_buffer.into(),
                accum: accum_buffer.into(),
            },
        );

        let mut sums = vec![0i32; moments.len()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                accum_buffer.data() as *const i32,
                sums.as_mut_ptr(),
                sums.len(),
            );
        }
        self.context.destroy_buffer(offsets_buffer);
        self.context.destroy_buffer(nodes_buffer);
        self.context.destroy_buffer(accum_buffer);

        let count = (schedule.n_samples as u64 * state.n_chains as u64) as f64;
        sums.iter().map(|&sum| sum as f64 / count).collect()
    }
}

impl GpuSampler {
    fn run(
        &mut self,
        program: &Program,
        schedule: &Schedule,
        state: &mut State,
        seed: u32,
        observation: Observation,
    ) {
        validate_state(program, state);
        self.ensure_pipelines(program.max_states.max(2));
        self.ensure_program(program);

        let context = &self.context;
        let state_buffer =
            create_upload_buffer(context, "state", bytemuck::cast_slice(&state.values));

        let pipelines = self.pipelines.as_ref().unwrap();
        let buffers = self.program.as_ref().unwrap();
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "gibbs",
            buffer_count: 2,
        });

        let mut counter = 0u32;
        let mut ops_in_chunk = 0usize;
        let mut pending = None;
        encoder.start();

        for op in run_ops(schedule) {
            if let RunOp::Step = op {
                for (block, bases) in program.blocks.iter().zip(buffers.bases.iter()) {
                    flush_chunk(context, &mut encoder, &mut pending, &mut ops_in_chunk);
                    let (pipeline, states) = match block.kind {
                        BlockKind::Spin => (&pipelines.update_spins, 2),
                        BlockKind::Categorical { states } => {
                            (&pipelines.update_categorical, states)
                        }
                    };
                    let node_count = block.node_ids.len() as u32;
                    let mut pass = encoder.compute("update");
                    let mut pc = pass.with(pipeline);
                    pc.bind(
                        0,
                        &UpdateData {
                            state: state_buffer.into(),
                            records: buffers.records.into(),
                            offsets: buffers.offsets.into(),
                            node_ids: buffers.node_ids.into(),
                            weights: buffers.weights.into(),
                            params: UpdateParams {
                                seed,
                                counter,
                                n_nodes: state.n_nodes,
                                n_chains: state.n_chains,
                                node_count,
                                states,
                                offsets_base: bases.offsets,
                                records_base: bases.records,
                                node_ids_base: bases.node_ids,
                                pad0: 0,
                                pad1: 0,
                                pad2: 0,
                            },
                        },
                    );
                    pc.dispatch(dispatch_grid(node_count * state.n_chains));
                    counter += 1;
                    ops_in_chunk += 1;
                }
            }

            let RunOp::Record(sample) = op else {
                continue;
            };
            match observation {
                Observation::States {
                    layout,
                    obs_node_ids,
                    out_samples,
                } => {
                    let mut node_ids_base = 0;
                    for block_layout in layout.blocks.iter() {
                        flush_chunk(context, &mut encoder, &mut pending, &mut ops_in_chunk);
                        let mut pass = encoder.compute("record");
                        let mut pc = pass.with(&pipelines.record_states);
                        pc.bind(
                            0,
                            &RecordData {
                                state: state_buffer.into(),
                                obs_node_ids,
                                out_samples,
                                record_params: RecordParams {
                                    n_nodes: state.n_nodes,
                                    n_chains: state.n_chains,
                                    sample,
                                    node_count: block_layout.node_count,
                                    out_base: block_layout.base,
                                    words_per_frame: block_layout.words_per_frame,
                                    node_ids_base,
                                    is_spin: block_layout.is_spin as u32,
                                },
                            },
                        );
                        pc.dispatch(dispatch_grid(block_layout.words_per_frame * state.n_chains));
                        node_ids_base += block_layout.node_count;
                        ops_in_chunk += 1;
                    }
                }
                Observation::Moments {
                    count,
                    offsets,
                    nodes,
                    accum,
                } => {
                    flush_chunk(context, &mut encoder, &mut pending, &mut ops_in_chunk);
                    let mut pass = encoder.compute("moments");
                    let mut pc = pass.with(&pipelines.accumulate_moments);
                    pc.bind(
                        0,
                        &MomentData {
                            state: state_buffer.into(),
                            moment_offsets: offsets,
                            moment_nodes: nodes,
                            accum,
                            moment_params: MomentParams {
                                n_nodes: state.n_nodes,
                                n_chains: state.n_chains,
                                n_moments: count,
                                pad0: 0,
                            },
                        },
                    );
                    pc.dispatch(dispatch_grid(count * state.n_chains));
                    ops_in_chunk += 1;
                }
            }
        }

        let sync_point = context.submit(&mut encoder);
        if let Some(previous) = pending {
            context
                .wait_for(&previous, !0)
                .expect("lost the GPU device while sampling");
        }
        context
            .wait_for(&sync_point, !0)
            .expect("lost the GPU device while sampling");

        // Read the final chain state back.
        unsafe {
            std::ptr::copy_nonoverlapping(
                state_buffer.data() as *const u32,
                state.values.as_mut_ptr(),
                state.values.len(),
            );
        }

        context.destroy_command_encoder(&mut encoder);
        context.destroy_buffer(state_buffer);
    }
}
