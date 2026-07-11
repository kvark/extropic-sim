//! Training a restricted Boltzmann machine on Bars-and-Stripes.
//!
//! A spin RBM with 16 visible and 24 hidden units learns the 4x4
//! Bars-and-Stripes distribution with the classic two-phase gradient:
//! positive moments with the data clamped, negative moments from the
//! free-running model. After training, the model is sampled freely
//! and generated images are checked for validity.
//!
//! Runs on the GPU when one is available.

use extropic_sim::{self as sim, models::IsingModel};

const VISIBLE: usize = 16;
const HIDDEN: usize = 24;
const SIDE: usize = 4;
const EPOCHS: u32 = 300;
const LEARNING_RATE: f64 = 0.1;

/// All 4x4 bars-and-stripes images: each row (or each column) is
/// either all up or all down.
fn bars_and_stripes() -> Vec<[u32; VISIBLE]> {
    let mut patterns = Vec::new();
    for choice in 0..1u32 << SIDE {
        let mut rows = [0u32; VISIBLE];
        let mut columns = [0u32; VISIBLE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                rows[y * SIDE + x] = (choice >> y) & 1;
                columns[y * SIDE + x] = (choice >> x) & 1;
            }
        }
        patterns.push(rows);
        if choice != 0 && choice != (1 << SIDE) - 1 {
            patterns.push(columns);
        }
    }
    patterns
}

fn is_valid(image: &[u32]) -> bool {
    let rows = (0..SIDE).all(|y| (1..SIDE).all(|x| image[y * SIDE + x] == image[y * SIDE]));
    let columns = (0..SIDE).all(|x| (1..SIDE).all(|y| image[y * SIDE + x] == image[x]));
    rows || columns
}

fn main() {
    env_logger::init();

    let mut cpu_sampler = sim::CpuSampler::new();
    let mut gpu_sampler = sim::GpuSampler::new().ok();
    let sampler: &mut dyn sim::Sampler = match gpu_sampler {
        Some(ref mut sampler) => {
            println!("Training on {}", sampler.device_name());
            sampler
        }
        None => {
            println!("No GPU available, training on the CPU");
            &mut cpu_sampler
        }
    };

    let mut graph = sim::Graph::new();
    let visible = graph.add_spins(VISIBLE);
    let hidden = graph.add_spins(HIDDEN);
    let mut edges = Vec::with_capacity(VISIBLE * HIDDEN);
    for &v in visible.iter() {
        for &h in hidden.iter() {
            edges.push((v, h));
        }
    }
    let mut model = IsingModel {
        nodes: visible.iter().chain(hidden.iter()).copied().collect(),
        biases: vec![0.0; VISIBLE + HIDDEN],
        edges,
        weights: vec![0.0; VISIBLE * HIDDEN],
        beta: 1.0,
    };

    let patterns = bars_and_stripes();
    println!(
        "{} patterns, {VISIBLE} visible + {HIDDEN} hidden units, {EPOCHS} epochs",
        patterns.len()
    );

    let visible_block = sim::Block::new(visible.clone());
    let hidden_block = sim::Block::new(hidden.clone());
    // Positive phase: hidden units relax against clamped data.
    let positive_schedule = sim::Schedule {
        n_warmup: 10,
        n_samples: 20,
        steps_per_sample: 1,
    };
    // Negative phase: the whole model runs free.
    let negative_schedule = sim::Schedule {
        n_warmup: 20,
        n_samples: 20,
        steps_per_sample: 2,
    };

    for epoch in 0..EPOCHS {
        let seed = epoch * 2;
        let positive_program = model
            .compile(
                &graph,
                std::slice::from_ref(&hidden_block),
                std::slice::from_ref(&visible_block),
            )
            .unwrap();
        // One chain per training pattern, visible units clamped to it.
        let mut positive_state = model.hinton_init(&positive_program, patterns.len() as u32, seed);
        for (chain, pattern) in patterns.iter().enumerate() {
            for (&node, &value) in visible.iter().zip(pattern.iter()) {
                positive_state.set(chain as u32, node, value);
            }
        }
        let positive = model.estimate_moments(
            sampler,
            &positive_program,
            &positive_schedule,
            &mut positive_state,
            seed,
        );

        let negative_program = model
            .compile(&graph, &[visible_block.clone(), hidden_block.clone()], &[])
            .unwrap();
        let mut negative_state = model.hinton_init(&negative_program, 64, seed + 1);
        let negative = model.estimate_moments(
            sampler,
            &negative_program,
            &negative_schedule,
            &mut negative_state,
            seed + 1,
        );

        let (bias_grad, weight_grad) = model.kl_gradient(&positive, &negative);
        for (bias, grad) in model.biases.iter_mut().zip(bias_grad.iter()) {
            *bias -= (LEARNING_RATE * grad) as f32;
        }
        for (weight, grad) in model.weights.iter_mut().zip(weight_grad.iter()) {
            *weight -= (LEARNING_RATE * grad) as f32;
        }

        if (epoch + 1) % 50 == 0 {
            let valid = generate_and_check(sampler, &graph, &model, &visible, epoch);
            println!("epoch {:3}: {valid:5.1}% valid generations", epoch + 1);
        }
    }

    // A random 16-bit image is valid with probability 30/65536 = 0.05%.
    let valid = generate_and_check(sampler, &graph, &model, &visible, !0);
    println!("final: {valid:5.1}% valid generations (random baseline 0.05%)");
}

/// Sample the trained model freely and return the percentage of
/// generated visible images that are valid bars-and-stripes.
fn generate_and_check(
    sampler: &mut dyn sim::Sampler,
    graph: &sim::Graph,
    model: &IsingModel,
    visible: &[sim::Node],
    seed: u32,
) -> f64 {
    let free_blocks = [
        sim::Block::new(visible.to_vec()),
        sim::Block::new(model.nodes[VISIBLE..].to_vec()),
    ];
    let program = model.compile(graph, &free_blocks, &[]).unwrap();
    let schedule = sim::Schedule {
        n_warmup: 100,
        n_samples: 50,
        steps_per_sample: 5,
    };
    let mut state = model.hinton_init(&program, 64, seed);
    let samples = sampler.sample_states(
        &program,
        &schedule,
        &mut state,
        seed,
        &[sim::Block::new(visible.to_vec())],
    );

    let mut valid = 0u32;
    let mut total = 0u32;
    let mut image = [0u32; VISIBLE];
    for sample in 0..samples.sample_count() {
        for chain in 0..samples.chain_count() {
            for (position, pixel) in image.iter_mut().enumerate() {
                *pixel = samples.value(0, sample, chain, position as u32);
            }
            valid += is_valid(&image) as u32;
            total += 1;
        }
    }
    valid as f64 * 100.0 / total as f64
}
