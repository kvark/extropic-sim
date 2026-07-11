use crate::{Block, DiscreteFactor, Graph, Node, Program, Sampler, Schedule, State, rng};

/// The counter value reserved for state initialization draws.
const INIT_STEP: u32 = !1;

/// First and second moments of an Ising model, estimated by sampling.
pub struct IsingMoments {
    /// `<s_i>` for every node of the model, in order.
    pub nodes: Vec<f64>,
    /// `<s_i * s_j>` for every edge of the model, in order.
    pub edges: Vec<f64>,
}

/// An Ising model: spins with biases, coupled over a set of edges.
///
/// The energy function is
///
/// ```text
/// E(s) = -beta * (sum_i b_i s_i + sum_(i,j) J_ij s_i s_j)
/// ```
///
/// with all spins in {-1, +1}, matching THRML's `IsingEBM`. Boltzmann
/// machines in their spin formulation are the same model, so this type
/// doubles as the workhorse for sampling-based training.
#[derive(Clone, Debug)]
pub struct IsingModel {
    /// The spins of the model.
    pub nodes: Vec<Node>,
    /// The bias `b_i` of every node, in order.
    pub biases: Vec<f32>,
    /// Pairs of distinct coupled nodes.
    pub edges: Vec<(Node, Node)>,
    /// The coupling `J_ij` of every edge, in order.
    pub weights: Vec<f32>,
    /// Inverse temperature.
    pub beta: f32,
}

impl IsingModel {
    /// The energy terms of the model, with `beta` folded in.
    pub fn factors(&self) -> Vec<DiscreteFactor> {
        assert_eq!(self.nodes.len(), self.biases.len());
        assert_eq!(self.edges.len(), self.weights.len());
        let mut factors = Vec::with_capacity(2);
        if !self.nodes.is_empty() {
            factors.push(DiscreteFactor::bias(
                self.nodes.clone(),
                self.biases.iter().map(|&bias| self.beta * bias).collect(),
            ));
        }
        if !self.edges.is_empty() {
            factors.push(DiscreteFactor::coupling(
                self.edges.iter().map(|&(a, _)| a).collect::<Vec<_>>(),
                self.edges.iter().map(|&(_, b)| b).collect::<Vec<_>>(),
                self.weights
                    .iter()
                    .map(|&weight| self.beta * weight)
                    .collect(),
            ));
        }
        factors
    }

    /// Compile the model into a sampling program.
    ///
    /// The union of `free_blocks` and `clamped_blocks` must cover all
    /// the nodes of the model; see [`color_blocks`](crate::color_blocks)
    /// for building a valid partition automatically.
    pub fn compile(
        &self,
        graph: &Graph,
        free_blocks: &[Block],
        clamped_blocks: &[Block],
    ) -> Result<Program, crate::Error> {
        Program::compile(graph, free_blocks, clamped_blocks, &self.factors())
    }

    /// Initialize chains by sampling every spin from its isolated
    /// marginal `P(up) = sigmoid(2 * beta * b_i)`, ignoring couplings.
    ///
    /// This is the classic initialization heuristic of Hinton's
    /// "A Practical Guide to Training Restricted Boltzmann Machines".
    pub fn hinton_init(&self, program: &Program, n_chains: u32, seed: u32) -> State {
        let mut state = State::zeros(program, n_chains);
        for chain in 0..n_chains {
            for (&node, &bias) in self.nodes.iter().zip(self.biases.iter()) {
                let p_up = 1.0 / (1.0 + (-2.0 * self.beta * bias).exp());
                let u = rng::uniform(seed, INIT_STEP, chain, node.0);
                state.set(chain, node, (u < p_up) as u32);
            }
        }
        state
    }

    /// Estimate the first moments of all nodes and the second moments
    /// of all edges by sampling.
    pub fn estimate_moments(
        &self,
        sampler: &mut dyn Sampler,
        program: &Program,
        schedule: &Schedule,
        state: &mut State,
        seed: u32,
    ) -> IsingMoments {
        let mut tuples: Vec<Vec<Node>> = self.nodes.iter().map(|&node| vec![node]).collect();
        tuples.extend(self.edges.iter().map(|&(a, b)| vec![a, b]));
        let moments = sampler.accumulate_moments(program, schedule, state, seed, &tuples);
        let (node_moments, edge_moments) = moments.split_at(self.nodes.len());
        IsingMoments {
            nodes: node_moments.to_vec(),
            edges: edge_moments.to_vec(),
        }
    }

    /// Monte Carlo estimate of the gradient of the KL divergence between
    /// a data distribution and the model, with respect to the biases and
    /// the coupling weights.
    ///
    /// Takes moments estimated in the *positive* phase (data clamped)
    /// and the *negative* phase (free-running model):
    ///
    /// ```text
    /// dKL/db_i = -beta * (<s_i>+ - <s_i>-)
    /// dKL/dJ_ij = -beta * (<s_i s_j>+ - <s_i s_j>-)
    /// ```
    ///
    /// Descending this gradient trains Boltzmann machines; see
    /// `examples/rbm.rs`.
    pub fn kl_gradient(
        &self,
        positive: &IsingMoments,
        negative: &IsingMoments,
    ) -> (Vec<f64>, Vec<f64>) {
        let beta = self.beta as f64;
        let bias_grad = positive
            .nodes
            .iter()
            .zip(negative.nodes.iter())
            .map(|(&pos, &neg)| -beta * (pos - neg))
            .collect();
        let weight_grad = positive
            .edges
            .iter()
            .zip(negative.edges.iter())
            .map(|(&pos, &neg)| -beta * (pos - neg))
            .collect();
        (bias_grad, weight_grad)
    }
}

/// Partition `nodes` into blocks of non-interacting nodes with a greedy
/// graph coloring over `edges`.
///
/// Bipartite graphs (chains, lattices, RBMs) get two blocks; general
/// graphs may get more. The result is a valid free-block partition for
/// [`Program::compile`]. All nodes must be of the same kind.
pub fn color_blocks(graph: &Graph, nodes: &[Node], edges: &[(Node, Node)]) -> Vec<Block> {
    let kind = graph.node_kind(nodes[0]);
    let mut order = vec![usize::MAX; graph.node_count()];
    for (index, &node) in nodes.iter().enumerate() {
        assert_eq!(graph.node_kind(node), kind, "nodes must be of one kind");
        order[node.index()] = index;
    }

    let mut adjacency = vec![Vec::new(); nodes.len()];
    for &(a, b) in edges.iter() {
        adjacency[order[a.index()]].push(order[b.index()]);
        adjacency[order[b.index()]].push(order[a.index()]);
    }

    let mut colors = vec![usize::MAX; nodes.len()];
    let mut block_nodes: Vec<Vec<Node>> = Vec::new();
    for (index, &node) in nodes.iter().enumerate() {
        let mut used = adjacency[index]
            .iter()
            .map(|&neighbor| colors[neighbor])
            .collect::<Vec<_>>();
        used.sort_unstable();
        let mut color = 0;
        for value in used {
            if value == color {
                color += 1;
            }
        }
        colors[index] = color;
        if color == block_nodes.len() {
            block_nodes.push(Vec::new());
        }
        block_nodes[color].push(node);
    }
    block_nodes.into_iter().map(Block::new).collect()
}
