//! Throughput benchmark: spin updates per second on 2D lattices.
//!
//! Compares the CPU and GPU backends over a range of model sizes and
//! chain counts. The GPU pulls ahead as the number of parallel updates
//! per dispatch (nodes per block times chains) grows.

use extropic_sim::{self as sim, models::IsingModel};
use std::time::Instant;

const STEPS: u32 = 200;

struct Lattice {
    graph: sim::Graph,
    nodes: Vec<sim::Node>,
    edges: Vec<(sim::Node, sim::Node)>,
    blocks: Vec<sim::Block>,
}

fn build(size: usize) -> Lattice {
    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(size * size);
    let mut edges = Vec::with_capacity(2 * size * size);
    for y in 0..size {
        for x in 0..size {
            let here = nodes[y * size + x];
            edges.push((here, nodes[y * size + (x + 1) % size]));
            edges.push((here, nodes[((y + 1) % size) * size + x]));
        }
    }
    let blocks = sim::color_blocks(&graph, &nodes, &edges);
    Lattice {
        graph,
        nodes,
        edges,
        blocks,
    }
}

fn run(sampler: &mut dyn sim::Sampler, program: &sim::Program, n_chains: u32, nodes: usize) -> f64 {
    let schedule = sim::Schedule {
        n_warmup: STEPS,
        n_samples: 1,
        steps_per_sample: 1,
    };
    let mut state = sim::State::zeros(program, n_chains);
    let start = Instant::now();
    let _ = sampler.accumulate_moments(program, &schedule, &mut state, 0, &[]);
    let elapsed = start.elapsed().as_secs_f64();
    nodes as f64 * STEPS as f64 * n_chains as f64 / elapsed
}

fn main() {
    env_logger::init();

    let mut cpu = sim::CpuSampler::new();
    let mut gpu = sim::GpuSampler::new().ok();
    match gpu {
        Some(ref sampler) => println!("GPU: {}", sampler.device_name()),
        None => println!("GPU: none"),
    }

    println!("lattice   chains    CPU updates/s    GPU updates/s");
    for &(size, n_chains) in [(32usize, 16u32), (64, 64), (128, 128)].iter() {
        let lattice = build(size);
        let model = IsingModel {
            nodes: lattice.nodes.clone(),
            biases: vec![0.1; lattice.nodes.len()],
            edges: lattice.edges.clone(),
            weights: vec![0.4; lattice.edges.len()],
            beta: 1.0,
        };
        let program = model.compile(&lattice.graph, &lattice.blocks, &[]).unwrap();

        let cpu_rate = run(&mut cpu, &program, n_chains, lattice.nodes.len());
        let gpu_rate = match gpu {
            Some(ref mut sampler) => run(sampler, &program, n_chains, lattice.nodes.len()),
            None => 0.0,
        };
        println!(
            "{:3}x{size:<3}   {n_chains:4}     {:12.3e}     {:12.3e}",
            size, cpu_rate, gpu_rate
        );
    }
}
