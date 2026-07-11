//! Python bindings for `extropic-sim`, in the flavor of THRML.
//!
//! The module mirrors the parts of THRML's API that map onto
//! `extropic-sim`: nodes are created as free-standing objects,
//! grouped into blocks, and combined with an `IsingEBM` into an
//! `IsingSamplingProgram` that `sample_states` runs on the GPU
//! (or the CPU when no GPU is available).
//!
//! Divergences from THRML: states are plain nested Python lists
//! instead of JAX arrays, the PRNG key is an integer seed, and
//! batching is controlled by the shape of the initial state.

use pyo3::{
    IntoPyObjectExt as _,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use extropic_sim as sim;

static NODE_COUNT: AtomicU32 = AtomicU32::new(0);
static GPU_SAMPLER: Mutex<Option<sim::GpuSampler>> = Mutex::new(None);

/// A random variable taking values in {-1, +1}.
#[pyclass(frozen, from_py_object)]
#[derive(Clone, Copy)]
struct SpinNode {
    id: u32,
}

#[pymethods]
impl SpinNode {
    #[new]
    fn new() -> Self {
        Self {
            id: NODE_COUNT.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn __repr__(&self) -> String {
        format!("SpinNode({})", self.id)
    }
}

/// A random variable taking integer values in `[0, num_states)`.
///
/// Unlike THRML, the number of states is a property of the node
/// rather than of a shape/dtype map passed around separately.
#[pyclass(frozen, from_py_object)]
#[derive(Clone, Copy)]
struct CategoricalNode {
    id: u32,
    num_states: u32,
}

#[pymethods]
impl CategoricalNode {
    #[new]
    fn new(num_states: u32) -> Self {
        Self {
            id: NODE_COUNT.fetch_add(1, Ordering::Relaxed),
            num_states,
        }
    }

    fn __repr__(&self) -> String {
        format!("CategoricalNode({}, {})", self.id, self.num_states)
    }
}

/// Either kind of node, as accepted from Python.
#[derive(Clone, Copy, FromPyObject)]
enum PyNode {
    Spin(SpinNode),
    Categorical(CategoricalNode),
}

impl PyNode {
    fn id(&self) -> u32 {
        match *self {
            Self::Spin(node) => node.id,
            Self::Categorical(node) => node.id,
        }
    }
}

/// An ordered group of nodes of one kind.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Block {
    nodes: Vec<PyNode>,
}

#[pymethods]
impl Block {
    #[new]
    fn new(nodes: Vec<PyNode>) -> Self {
        Self { nodes }
    }

    fn __len__(&self) -> usize {
        self.nodes.len()
    }
}

/// An Ising energy-based model:
/// `E(s) = -beta * (sum_i b_i s_i + sum_ij J_ij s_i s_j)`.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct IsingEBM {
    nodes: Vec<PyNode>,
    edges: Vec<(PyNode, PyNode)>,
    biases: Vec<f32>,
    weights: Vec<f32>,
    beta: f32,
}

#[pymethods]
impl IsingEBM {
    #[new]
    fn new(
        nodes: Vec<PyNode>,
        edges: Vec<(PyNode, PyNode)>,
        biases: Vec<f32>,
        weights: Vec<f32>,
        beta: f32,
    ) -> PyResult<Self> {
        if nodes.len() != biases.len() {
            return Err(PyValueError::new_err("one bias per node is required"));
        }
        if edges.len() != weights.len() {
            return Err(PyValueError::new_err("one weight per edge is required"));
        }
        if nodes
            .iter()
            .chain(edges.iter().flat_map(|(a, b)| [a, b].into_iter()))
            .any(|node| !matches!(node, PyNode::Spin(_)))
        {
            return Err(PyValueError::new_err("IsingEBM only supports spin nodes"));
        }
        Ok(Self {
            nodes,
            edges,
            biases,
            weights,
            beta,
        })
    }
}

/// Warm-up length, number of samples, and steps between samples.
#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
struct SamplingSchedule {
    #[pyo3(get, set)]
    n_warmup: u32,
    #[pyo3(get, set)]
    n_samples: u32,
    #[pyo3(get, set)]
    steps_per_sample: u32,
}

#[pymethods]
impl SamplingSchedule {
    #[new]
    fn new(n_warmup: u32, n_samples: u32, steps_per_sample: u32) -> Self {
        Self {
            n_warmup,
            n_samples,
            steps_per_sample,
        }
    }
}

impl From<SamplingSchedule> for sim::Schedule {
    fn from(schedule: SamplingSchedule) -> Self {
        Self {
            n_warmup: schedule.n_warmup,
            n_samples: schedule.n_samples,
            steps_per_sample: schedule.steps_per_sample,
        }
    }
}

/// An `IsingEBM` compiled for block Gibbs sampling over the given
/// free and clamped blocks.
#[pyclass]
struct IsingSamplingProgram {
    program: sim::Program,
    free_blocks: Vec<Vec<sim::Node>>,
    clamped_blocks: Vec<Vec<sim::Node>>,
    node_map: HashMap<u32, sim::Node>,
}

#[pymethods]
impl IsingSamplingProgram {
    #[new]
    fn new(ebm: IsingEBM, free_blocks: Vec<Block>, clamped_blocks: Vec<Block>) -> PyResult<Self> {
        // Build a graph covering every node of the blocks.
        let mut graph = sim::Graph::new();
        let mut node_map = HashMap::new();
        for block in free_blocks.iter().chain(clamped_blocks.iter()) {
            for &node in block.nodes.iter() {
                node_map.entry(node.id()).or_insert_with(|| match node {
                    PyNode::Spin(_) => graph.add_spin(),
                    PyNode::Categorical(node) => graph.add_categorical(node.num_states),
                });
            }
        }
        let map_node = |node: &PyNode| -> PyResult<sim::Node> {
            node_map
                .get(&node.id())
                .copied()
                .ok_or_else(|| PyValueError::new_err("node is not covered by any block"))
        };
        let map_nodes =
            |nodes: &[PyNode]| -> PyResult<Vec<sim::Node>> { nodes.iter().map(map_node).collect() };

        let model = sim::models::IsingModel {
            nodes: map_nodes(&ebm.nodes)?,
            biases: ebm.biases.clone(),
            edges: ebm
                .edges
                .iter()
                .map(|(a, b)| Ok((map_node(a)?, map_node(b)?)))
                .collect::<PyResult<Vec<_>>>()?,
            weights: ebm.weights.clone(),
            beta: ebm.beta,
        };

        let free: Vec<Vec<sim::Node>> = free_blocks
            .iter()
            .map(|block| map_nodes(&block.nodes))
            .collect::<PyResult<_>>()?;
        let clamped: Vec<Vec<sim::Node>> = clamped_blocks
            .iter()
            .map(|block| map_nodes(&block.nodes))
            .collect::<PyResult<_>>()?;

        let free_sim: Vec<sim::Block> = free.iter().cloned().map(sim::Block::new).collect();
        let clamped_sim: Vec<sim::Block> = clamped.iter().cloned().map(sim::Block::new).collect();
        let program = model
            .compile(&graph, &free_sim, &clamped_sim)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        Ok(Self {
            program,
            free_blocks: free,
            clamped_blocks: clamped,
            node_map,
        })
    }
}

/// Per-block state values: either one value per node, or a batch
/// of per-node values (one row per chain).
#[derive(Clone, FromPyObject)]
enum BlockValues {
    Flat(Vec<u32>),
    Batched(Vec<Vec<u32>>),
}

/// Infer the chain count from state values; `None` means unbatched.
fn infer_chains(state: &[BlockValues]) -> PyResult<Option<u32>> {
    let mut n_chains = None;
    for values in state.iter() {
        if let BlockValues::Batched(ref rows) = *values {
            let count = rows.len() as u32;
            if n_chains.is_some_and(|previous| previous != count) {
                return Err(PyValueError::new_err(
                    "inconsistent batch sizes across blocks",
                ));
            }
            n_chains = Some(count);
        }
    }
    Ok(n_chains)
}

fn write_values(
    state: &mut sim::State,
    nodes: &[sim::Node],
    values: &BlockValues,
    n_chains: u32,
) -> PyResult<()> {
    let rows: Vec<&[u32]> = match *values {
        BlockValues::Flat(ref row) => vec![row; n_chains as usize],
        BlockValues::Batched(ref rows) => {
            if rows.len() != n_chains as usize {
                return Err(PyValueError::new_err(
                    "inconsistent batch sizes across blocks",
                ));
            }
            rows.iter().map(|row| row.as_slice()).collect()
        }
    };
    for (chain, row) in rows.into_iter().enumerate() {
        if row.len() != nodes.len() {
            return Err(PyValueError::new_err("state length mismatch"));
        }
        for (&node, &value) in nodes.iter().zip(row.iter()) {
            state.set(chain as u32, node, value);
        }
    }
    Ok(())
}

fn initial_state(
    program: &IsingSamplingProgram,
    init_state: &[BlockValues],
    clamped_data: &[BlockValues],
) -> PyResult<(sim::State, Option<u32>)> {
    if init_state.len() != program.free_blocks.len() {
        return Err(PyValueError::new_err("one init state per free block"));
    }
    if clamped_data.len() != program.clamped_blocks.len() {
        return Err(PyValueError::new_err("one data entry per clamped block"));
    }
    let batched = match (infer_chains(init_state)?, infer_chains(clamped_data)?) {
        (Some(a), Some(b)) if a != b => {
            return Err(PyValueError::new_err(
                "inconsistent batch sizes across blocks",
            ));
        }
        (Some(count), _) | (None, Some(count)) => Some(count),
        (None, None) => None,
    };
    let n_chains = batched.unwrap_or(1);
    let mut state = sim::State::zeros(&program.program, n_chains);
    for (nodes, values) in program.free_blocks.iter().zip(init_state.iter()) {
        write_values(&mut state, nodes, values, n_chains)?;
    }
    for (nodes, values) in program.clamped_blocks.iter().zip(clamped_data.iter()) {
        write_values(&mut state, nodes, values, n_chains)?;
    }
    Ok((state, batched))
}

fn with_sampler<R>(device: &str, run: impl FnOnce(&mut dyn sim::Sampler) -> R) -> PyResult<R> {
    match device {
        "cpu" => Ok(run(&mut sim::CpuSampler::new())),
        "gpu" | "auto" => {
            // A panic while sampling poisons the mutex; the sampler
            // itself holds no state across runs, so keep using it.
            let mut guard = GPU_SAMPLER
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if guard.is_none() {
                match sim::GpuSampler::new() {
                    Ok(sampler) => *guard = Some(sampler),
                    Err(error) if device == "gpu" => {
                        return Err(PyRuntimeError::new_err(format!(
                            "failed to initialize the GPU: {error:?}"
                        )));
                    }
                    Err(_) => return Ok(run(&mut sim::CpuSampler::new())),
                }
            }
            Ok(run(guard.as_mut().unwrap()))
        }
        _ => Err(PyValueError::new_err(
            "device must be 'auto', 'gpu', or 'cpu'",
        )),
    }
}

/// Initialize block states from the marginal biases of the model:
/// every spin is up with probability `sigmoid(2 * beta * b_i)`.
///
/// Draws are keyed by the creation order of the nodes, so a script
/// that builds its nodes deterministically gets reproducible states.
///
/// Returns one entry per block: a list of 0/1 values, or a batch of
/// `n_chains` such lists when `n_chains` is given.
#[pyfunction]
#[pyo3(signature = (seed, ebm, blocks, n_chains=None))]
fn hinton_init(
    py: Python,
    seed: u32,
    ebm: &IsingEBM,
    blocks: Vec<Block>,
    n_chains: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let biases: HashMap<u32, f32> = ebm
        .nodes
        .iter()
        .zip(ebm.biases.iter())
        .map(|(node, &bias)| (node.id(), bias))
        .collect();
    let draw = |chain: u32, node: PyNode| -> u32 {
        let bias = biases.get(&node.id()).copied().unwrap_or_default();
        sim::models::hinton_draw(seed, chain, node.id(), ebm.beta, bias)
    };

    let result: Vec<Py<PyAny>> = blocks
        .iter()
        .map(|block| match n_chains {
            None => block
                .nodes
                .iter()
                .map(|&node| draw(0, node))
                .collect::<Vec<_>>()
                .into_py_any(py),
            Some(count) => (0..count)
                .map(|chain| {
                    block
                        .nodes
                        .iter()
                        .map(|&node| draw(chain, node))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
                .into_py_any(py),
        })
        .collect::<PyResult<_>>()?;
    result.into_py_any(py)
}

/// Run the schedule and return the recorded states of
/// `observed_blocks`.
///
/// `init_state` holds one entry per free block of the program, and
/// `clamped_data` one entry per clamped block. Passing batched values
/// (lists of lists) runs that many independent chains in parallel and
/// returns samples of shape `[n_samples][n_chains][len]` per block;
/// unbatched values return `[n_samples][len]`.
#[pyfunction]
#[pyo3(signature = (seed, program, schedule, init_state, clamped_data, observed_blocks, device="auto"))]
#[allow(clippy::too_many_arguments)]
fn sample_states(
    py: Python,
    seed: u32,
    program: &IsingSamplingProgram,
    schedule: &SamplingSchedule,
    init_state: Vec<BlockValues>,
    clamped_data: Vec<BlockValues>,
    observed_blocks: Vec<Block>,
    device: &str,
) -> PyResult<Py<PyAny>> {
    let (mut state, batched) = initial_state(program, &init_state, &clamped_data)?;
    let n_chains = batched.unwrap_or(1);

    let observed: Vec<sim::Block> = observed_blocks
        .iter()
        .map(|block| {
            block
                .nodes
                .iter()
                .map(|node| {
                    program.node_map.get(&node.id()).copied().ok_or_else(|| {
                        PyValueError::new_err("observed node is not part of the program")
                    })
                })
                .collect::<PyResult<Vec<_>>>()
                .map(sim::Block::new)
        })
        .collect::<PyResult<_>>()?;

    let samples = py.detach(|| {
        with_sampler(device, |sampler| {
            sampler.sample_states(
                &program.program,
                &(*schedule).into(),
                &mut state,
                seed,
                &observed,
            )
        })
    })?;

    // Convert to nested lists, dropping the chain axis for unbatched runs.
    let result: Vec<Py<PyAny>> = observed
        .iter()
        .enumerate()
        .map(|(block_index, block)| {
            let per_sample: Vec<Py<PyAny>> = (0..samples.sample_count())
                .map(|sample| {
                    let per_chain: Vec<Vec<u32>> = (0..n_chains)
                        .map(|chain| {
                            (0..block.len() as u32)
                                .map(|position| samples.value(block_index, sample, chain, position))
                                .collect()
                        })
                        .collect();
                    match batched {
                        Some(_) => per_chain.into_py_any(py),
                        None => per_chain[0].clone().into_py_any(py),
                    }
                })
                .collect::<PyResult<_>>()?;
            per_sample.into_py_any(py)
        })
        .collect::<PyResult<_>>()?;
    result.into_py_any(py)
}

/// Estimate moments of tuples of spin nodes by sampling: returns the
/// average product of each tuple's spins (±1) over samples and chains.
#[pyfunction]
#[pyo3(signature = (seed, program, schedule, init_state, clamped_data, moment_nodes, device="auto"))]
#[allow(clippy::too_many_arguments)]
fn estimate_moments(
    py: Python,
    seed: u32,
    program: &IsingSamplingProgram,
    schedule: &SamplingSchedule,
    init_state: Vec<BlockValues>,
    clamped_data: Vec<BlockValues>,
    moment_nodes: Vec<Vec<PyNode>>,
    device: &str,
) -> PyResult<Vec<f64>> {
    let (mut state, _) = initial_state(program, &init_state, &clamped_data)?;

    let tuples: Vec<Vec<sim::Node>> = moment_nodes
        .iter()
        .map(|tuple| {
            tuple
                .iter()
                .map(|node| {
                    program.node_map.get(&node.id()).copied().ok_or_else(|| {
                        PyValueError::new_err("moment node is not part of the program")
                    })
                })
                .collect()
        })
        .collect::<PyResult<_>>()?;

    py.detach(|| {
        with_sampler(device, |sampler| {
            sampler.accumulate_moments(
                &program.program,
                &(*schedule).into(),
                &mut state,
                seed,
                &tuples,
            )
        })
    })
}

#[pymodule]
#[pyo3(name = "extropic_sim")]
fn extropic_sim_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SpinNode>()?;
    module.add_class::<CategoricalNode>()?;
    module.add_class::<Block>()?;
    module.add_class::<IsingEBM>()?;
    module.add_class::<SamplingSchedule>()?;
    module.add_class::<IsingSamplingProgram>()?;
    module.add_function(wrap_pyfunction!(hinton_init, module)?)?;
    module.add_function(wrap_pyfunction!(sample_states, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_moments, module)?)?;
    Ok(())
}
