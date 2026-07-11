//! Ferromagnetic 2D Ising model on a periodic lattice.
//!
//! Sweeps the inverse temperature across the critical point and
//! compares the measured magnetization with Onsager's exact solution
//! for the infinite lattice. Runs on the GPU when one is available.

use extropic_sim::{self as sim, models::IsingModel};

const SIZE: usize = 32;
const N_CHAINS: u32 = 32;

/// Onsager's spontaneous magnetization of the infinite 2D lattice.
fn onsager_magnetization(beta: f32) -> f64 {
    let x = (2.0 * beta as f64).sinh().powi(-4);
    if x >= 1.0 { 0.0 } else { (1.0 - x).powf(0.125) }
}

fn main() {
    env_logger::init();

    let mut cpu_sampler = sim::CpuSampler::new();
    let mut gpu_sampler = sim::GpuSampler::new().ok();
    let sampler: &mut dyn sim::Sampler = match gpu_sampler {
        Some(ref mut sampler) => {
            println!("Sampling on {}", sampler.device_name());
            sampler
        }
        None => {
            println!("No GPU available, sampling on the CPU");
            &mut cpu_sampler
        }
    };

    let mut graph = sim::Graph::new();
    let nodes = graph.add_spins(SIZE * SIZE);
    let mut edges = Vec::with_capacity(2 * SIZE * SIZE);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let here = nodes[y * SIZE + x];
            edges.push((here, nodes[y * SIZE + (x + 1) % SIZE]));
            edges.push((here, nodes[((y + 1) % SIZE) * SIZE + x]));
        }
    }
    // The periodic checkerboard needs an even size for two colors.
    let free_blocks = sim::color_blocks(&graph, &nodes, &edges);
    assert_eq!(free_blocks.len(), 2);

    let schedule = sim::Schedule {
        n_warmup: 500,
        n_samples: 200,
        steps_per_sample: 5,
    };
    let observed = [sim::Block::new(nodes.clone())];

    println!("{SIZE}x{SIZE} periodic lattice, {N_CHAINS} chains");
    println!("beta      <|m|>     Onsager");
    for step in 0..9 {
        let beta = 0.30 + step as f32 * 0.03;
        let model = IsingModel {
            nodes: nodes.clone(),
            biases: vec![0.0; nodes.len()],
            edges: edges.clone(),
            weights: vec![1.0; edges.len()],
            beta,
        };
        let program = model.compile(&graph, &free_blocks, &[]).unwrap();
        // Start from the ordered state to avoid trapping domain walls.
        let mut state = sim::State::zeros(&program, N_CHAINS);
        let samples = sampler.sample_states(&program, &schedule, &mut state, step, &observed);

        let mut total = 0.0;
        for sample in 0..samples.sample_count() {
            for chain in 0..samples.chain_count() {
                let mut sum = 0i64;
                for position in 0..nodes.len() as u32 {
                    sum += samples.spin(0, sample, chain, position) as i64;
                }
                total += (sum as f64 / nodes.len() as f64).abs();
            }
        }
        let magnetization = total / (samples.sample_count() * samples.chain_count()) as f64;
        println!(
            "{beta:.2}    {magnetization:8.4}    {:8.4}",
            onsager_magnetization(beta)
        );
    }
}
