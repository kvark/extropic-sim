// Block Gibbs sampling kernels.
//
// The host substitutes MAX_STATES with the largest categorical state
// count of the program before compiling the module.
//
// All kernels operate on a dense `state` array holding one value per
// node per chain: 0/1 for spins, the category index for categorical
// nodes. Update records are documented in `src/program.rs`.

struct UpdateParams {
    seed: u32,
    counter: u32,
    n_nodes: u32,
    n_chains: u32,
    // The block being updated.
    node_count: u32,
    states: u32,
    offsets_base: u32,
    records_base: u32,
    node_ids_base: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

var<storage, read_write> state: array<u32>;
var<storage, read> records: array<u32>;
var<storage, read> offsets: array<u32>;
var<storage, read> node_ids: array<u32>;
var<storage, read> weights: array<f32>;
var<uniform> params: UpdateParams;

// From Jarzynski & Olano, "Hash Functions for GPU Rendering" (JCGT 2020).
fn pcg4d(input: vec4<u32>) -> vec4<u32> {
    var v = input * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> vec4<u32>(16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

fn random_uniform(seed: u32, step: u32, chain: u32, node: u32) -> f32 {
    let hash = pcg4d(vec4<u32>(seed, step, chain, node));
    return f32(hash.x >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(64)
fn update_spins(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) group_counts: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Workgroups form a 2D grid to stay within per-dimension limits.
    let thread = (group_id.y * group_counts.x + group_id.x) * 64u + local_index;
    if (thread >= params.node_count * params.n_chains) {
        return;
    }
    let chain = thread / params.node_count;
    let position = thread % params.node_count;
    let node_id = node_ids[params.node_ids_base + position];
    let chain_base = chain * params.n_nodes;

    var gamma = 0.0;
    var cursor = params.records_base + offsets[params.offsets_base + position];
    let end = params.records_base + offsets[params.offsets_base + position + 1u];
    while (cursor < end) {
        var index = records[cursor];
        let counts = records[cursor + 2u];
        var product = 1.0;
        cursor += 3u;
        for (var i = 0u; i < (counts & 0xFFFFu); i += 1u) {
            let value = state[chain_base + records[cursor]];
            product *= f32(value) * 2.0 - 1.0;
            cursor += 1u;
        }
        for (var i = 0u; i < (counts >> 16u); i += 1u) {
            index += state[chain_base + records[cursor]] * records[cursor + 1u];
            cursor += 2u;
        }
        gamma += product * weights[index];
    }

    let p_up = 1.0 / (1.0 + exp(-2.0 * gamma));
    let u = random_uniform(params.seed, params.counter, chain, node_id);
    state[chain_base + node_id] = u32(u < p_up);
}

@compute @workgroup_size(64)
fn update_categorical(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) group_counts: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Workgroups form a 2D grid to stay within per-dimension limits.
    let thread = (group_id.y * group_counts.x + group_id.x) * 64u + local_index;
    if (thread >= params.node_count * params.n_chains) {
        return;
    }
    let chain = thread / params.node_count;
    let position = thread % params.node_count;
    let node_id = node_ids[params.node_ids_base + position];
    let chain_base = chain * params.n_nodes;

    var theta: array<f32, MAX_STATES>;
    for (var k = 0u; k < params.states; k += 1u) {
        theta[k] = 0.0;
    }

    var cursor = params.records_base + offsets[params.offsets_base + position];
    let end = params.records_base + offsets[params.offsets_base + position + 1u];
    while (cursor < end) {
        var index = records[cursor];
        let head_stride = records[cursor + 1u];
        let counts = records[cursor + 2u];
        var product = 1.0;
        cursor += 3u;
        for (var i = 0u; i < (counts & 0xFFFFu); i += 1u) {
            let value = state[chain_base + records[cursor]];
            product *= f32(value) * 2.0 - 1.0;
            cursor += 1u;
        }
        for (var i = 0u; i < (counts >> 16u); i += 1u) {
            index += state[chain_base + records[cursor]] * records[cursor + 1u];
            cursor += 2u;
        }
        for (var k = 0u; k < params.states; k += 1u) {
            theta[k] += product * weights[index + k * head_stride];
        }
    }

    var max_theta = theta[0];
    for (var k = 1u; k < params.states; k += 1u) {
        max_theta = max(max_theta, theta[k]);
    }
    var total = 0.0;
    for (var k = 0u; k < params.states; k += 1u) {
        total += exp(theta[k] - max_theta);
    }

    let u = random_uniform(params.seed, params.counter, chain, node_id);
    var remaining = u * total;
    var value = params.states - 1u;
    for (var k = 0u; k < params.states; k += 1u) {
        remaining -= exp(theta[k] - max_theta);
        if (remaining <= 0.0) {
            value = k;
            break;
        }
    }
    state[chain_base + node_id] = value;
}

struct RecordParams {
    n_nodes: u32,
    n_chains: u32,
    sample: u32,
    // The block being recorded.
    node_count: u32,
    out_base: u32,
    words_per_frame: u32,
    node_ids_base: u32,
    is_spin: u32,
}

var<storage, read> obs_node_ids: array<u32>;
var<storage, read_write> out_samples: array<u32>;
var<uniform> record_params: RecordParams;

@compute @workgroup_size(64)
fn record_states(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) group_counts: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Workgroups form a 2D grid to stay within per-dimension limits.
    let thread = (group_id.y * group_counts.x + group_id.x) * 64u + local_index;
    let words = record_params.words_per_frame;
    if (thread >= words * record_params.n_chains) {
        return;
    }
    let chain = thread / words;
    let word_index = thread % words;
    let frame = record_params.sample * record_params.n_chains + chain;
    let out_index = record_params.out_base + frame * words + word_index;
    let chain_base = chain * record_params.n_nodes;

    if (record_params.is_spin != 0u) {
        // One thread packs the spins of 32 nodes into an output word.
        let start = word_index * 32u;
        let count = min(32u, record_params.node_count - start);
        var word = 0u;
        for (var i = 0u; i < count; i += 1u) {
            let node_id = obs_node_ids[record_params.node_ids_base + start + i];
            word |= (state[chain_base + node_id] & 1u) << i;
        }
        out_samples[out_index] = word;
    } else {
        let node_id = obs_node_ids[record_params.node_ids_base + word_index];
        out_samples[out_index] = state[chain_base + node_id];
    }
}

struct MomentParams {
    n_nodes: u32,
    n_chains: u32,
    n_moments: u32,
    pad0: u32,
}

var<storage, read> moment_offsets: array<u32>;
var<storage, read> moment_nodes: array<u32>;
var<storage, read_write> accum: array<atomic<i32>>;
var<uniform> moment_params: MomentParams;

@compute @workgroup_size(64)
fn accumulate_moments(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) group_counts: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Workgroups form a 2D grid to stay within per-dimension limits.
    let thread = (group_id.y * group_counts.x + group_id.x) * 64u + local_index;
    if (thread >= moment_params.n_moments * moment_params.n_chains) {
        return;
    }
    let chain = thread / moment_params.n_moments;
    let moment = thread % moment_params.n_moments;
    let chain_base = chain * moment_params.n_nodes;

    var product = 1;
    for (var i = moment_offsets[moment]; i < moment_offsets[moment + 1u]; i += 1u) {
        product *= i32(state[chain_base + moment_nodes[i]]) * 2 - 1;
    }
    atomicAdd(&accum[moment], product);
}
