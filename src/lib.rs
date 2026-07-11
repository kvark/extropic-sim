//! Portable simulator for thermodynamic computing.
//!
//! `extropic-sim` builds and samples probabilistic graphical models with
//! block Gibbs sampling, in the spirit of [THRML], with full GPU
//! acceleration via [blade-graphics] and a portable CPU fallback.
//!
//! [THRML]: https://github.com/extropic-ai/thrml
//! [blade-graphics]: https://crates.io/crates/blade-graphics
//!
//! # Overview
//!
//! - Describe the random variables of a model with a [`Graph`]:
//!   spin variables in {-1, +1} and categorical variables in `[0, K)`.
//! - Describe the energy function as a list of [`DiscreteFactor`]s.
//!   An instance of a factor contributes `-W[c_1, .., c_N] * s_1 * .. * s_M`
//!   to the energy, covering biases, pairwise and higher-order couplings
//!   of both spin and categorical variables.
//! - Partition the nodes into [`Block`]s of conditionally independent
//!   variables and compile a [`Program`].
//! - Run a [`Schedule`] on a [`CpuSampler`] or a [`GpuSampler`], recording
//!   [`Samples`] or accumulating moments across a batch of parallel chains.
//!
//! # Example
//!
//! Sampling a small Ising chain with two-color block Gibbs:
//!
//! ```
//! use extropic_sim as sim;
//! use sim::Sampler as _;
//!
//! let mut graph = sim::Graph::new();
//! let nodes = graph.add_spins(5);
//! let biases = vec![0.0; 5];
//! let weights = vec![0.5; 4];
//!
//! let factors = [
//!     sim::DiscreteFactor::bias(nodes.clone(), biases),
//!     sim::DiscreteFactor::coupling(&nodes[..4], &nodes[1..], weights),
//! ];
//! let free_blocks = [
//!     sim::Block::new(nodes.iter().copied().step_by(2).collect::<Vec<_>>()),
//!     sim::Block::new(nodes.iter().copied().skip(1).step_by(2).collect::<Vec<_>>()),
//! ];
//! let program = sim::Program::compile(&graph, &free_blocks, &[], &factors).unwrap();
//!
//! let schedule = sim::Schedule {
//!     n_warmup: 100,
//!     n_samples: 1000,
//!     steps_per_sample: 2,
//! };
//! let mut state = sim::State::random(&program, 8, 123);
//! let samples = sim::CpuSampler::new().sample_states(
//!     &program,
//!     &schedule,
//!     &mut state,
//!     0,
//!     &[sim::Block::new(nodes)],
//! );
//! assert_eq!(samples.sample_count(), 1000);
//! ```

#![allow(
    // We don't want to use `Doc(hidden)` and `SAFETY` comments.
    clippy::missing_safety_doc,
)]

mod cpu;
mod factor;
mod gpu;
mod graph;
pub mod models;
mod program;
pub mod rng;
mod state;

pub use cpu::CpuSampler;
pub use factor::{DiscreteFactor, total_energy};
pub use gpu::GpuSampler;
pub use graph::{Block, Graph, Node, NodeKind};
pub use models::color_blocks;
pub use program::{Error, MAX_CATEGORICAL_STATES, Program};
pub use state::{Samples, Schedule, State};

/// A backend that can run block Gibbs sampling programs.
pub trait Sampler {
    /// Run the schedule, recording the state of `observed` blocks
    /// at every sample.
    ///
    /// The chains start from `state`, which is left at the final state
    /// when the run completes.
    fn sample_states(
        &mut self,
        program: &Program,
        schedule: &Schedule,
        state: &mut State,
        seed: u32,
        observed: &[Block],
    ) -> Samples;

    /// Run the schedule, accumulating the moments of the given tuples
    /// of spin nodes.
    ///
    /// Returns one value per tuple: the average of the product of its
    /// spins (each ±1) over all samples and chains.
    fn accumulate_moments(
        &mut self,
        program: &Program,
        schedule: &Schedule,
        state: &mut State,
        seed: u32,
        moments: &[Vec<Node>],
    ) -> Vec<f64>;
}
