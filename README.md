# extropic-sim

Portable simulator for thermo-dynamic computing in Rust.

`extropic-sim` builds and samples probabilistic graphical models with
block Gibbs sampling, in the spirit of [THRML], with full GPU
acceleration via [blade-graphics] (Vulkan/Metal/OpenGL ES) and a
portable CPU fallback. Extropic is developing hardware that samples
certain discrete PGMs natively; this library is a small,
dependency-light place to prototype such models today.

[THRML]: https://github.com/extropic-ai/thrml
[blade-graphics]: https://crates.io/crates/blade-graphics

## Features

- Spin and categorical random variables in one heterogeneous graph
- Discrete energy factors of the form `-W[c_1..c_N] * s_1..s_M`:
  biases, pairwise couplings, higher-order and mixed interactions
- Block Gibbs sampling compiled into flat GPU-friendly update streams
- WGSL compute kernels, one dispatch per block update, batched chains
- Statistically identical CPU and GPU backends behind one `Sampler` trait
- Counter-based random numbers: reproducible runs for a fixed seed
- Ising/Boltzmann-machine helpers: greedy graph coloring, Hinton
  initialization, moment estimation, sampling-based KL gradients,
  in-place weight updates for training loops
- Validated against exact Boltzmann distributions, Onsager's 2D Ising
  solution, and generative RBM training

## Example

Sampling a small Ising chain with two-color block Gibbs:

```rust
use extropic_sim as sim;
use sim::Sampler as _;

let mut graph = sim::Graph::new();
let nodes = graph.add_spins(5);

let factors = [
    sim::DiscreteFactor::bias(nodes.clone(), vec![0.0; 5]),
    sim::DiscreteFactor::coupling(&nodes[..4], &nodes[1..], vec![0.5; 4]),
];
let free_blocks = [
    sim::Block::new(nodes.iter().copied().step_by(2).collect::<Vec<_>>()),
    sim::Block::new(nodes.iter().copied().skip(1).step_by(2).collect::<Vec<_>>()),
];
let program = sim::Program::compile(&graph, &free_blocks, &[], &factors).unwrap();

let schedule = sim::Schedule {
    n_warmup: 100,
    n_samples: 1000,
    steps_per_sample: 2,
};
let mut state = sim::State::random(&program, 8, 123);
let mut sampler = sim::GpuSampler::new().unwrap();
let samples = sampler.sample_states(
    &program,
    &schedule,
    &mut state,
    0,
    &[sim::Block::new(nodes)],
);
```

## Examples

- `cargo run --release --example ising2d` — 2D lattice magnetization
  across the phase transition, checked against Onsager's solution
- `cargo run --release --example rbm` — a restricted Boltzmann machine
  learning Bars-and-Stripes with sampling-based gradients

## Testing

CPU tests, including statistical checks against exact distributions:

```bash
cargo test
```

The same statistical suite on the GPU (requires Vulkan or Metal; a
software driver like lavapipe works):

```bash
cargo test --release -- --ignored
```

## Python

THRML-flavored Python bindings live in [`py/`](py/), exposing nodes,
blocks, `IsingEBM`, and `sample_states` backed by the same samplers.
