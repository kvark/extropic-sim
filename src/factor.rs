use super::{Block, Graph, NodeKind};

/// A batch of energy terms coupling spin and categorical variables.
///
/// Instance `i` of the batch contributes the energy
///
/// ```text
/// E_i(state) = -W_i[c_1, ..., c_N] * s_1 * ... * s_M
/// ```
///
/// where `s_m` is the value (interpreted as -1 or +1) of node
/// `spin_groups[m][i]`, and `c_n` is the value of node `cat_groups[n][i]`,
/// used to index into the weight tensor `W_i = weights[i, ...]`.
///
/// This is the same family of factors as THRML's `DiscreteEBMFactor`:
/// biases (M=1, N=0), pairwise couplings (M=2, N=0), higher-order spin
/// products, categorical tables and mixed spin/categorical terms are all
/// special cases.
///
/// No node should appear twice within one instance of the factor,
/// otherwise Gibbs sampling does not target the Boltzmann distribution
/// of the resulting energy.
#[derive(Clone, Debug)]
pub struct DiscreteFactor {
    /// Groups of spin nodes, each of the batch length.
    pub spin_groups: Vec<Block>,
    /// Groups of categorical nodes, each of the batch length.
    pub cat_groups: Vec<Block>,
    /// Weight tensor of shape `[batch, states(cat_1), ..., states(cat_N)]`,
    /// flattened in row-major order.
    pub weights: Vec<f32>,
}

impl DiscreteFactor {
    /// A batch of bias terms: `E_i = -biases[i] * s_i`.
    pub fn bias(nodes: impl Into<Block>, biases: Vec<f32>) -> Self {
        Self {
            spin_groups: vec![nodes.into()],
            cat_groups: Vec::new(),
            weights: biases,
        }
    }

    /// A batch of pairwise couplings: `E_i = -weights[i] * s_ai * s_bi`.
    pub fn coupling(a: impl Into<Block>, b: impl Into<Block>, weights: Vec<f32>) -> Self {
        Self {
            spin_groups: vec![a.into(), b.into()],
            cat_groups: Vec::new(),
            weights,
        }
    }

    /// A batch of categorical bias terms: `E_i = -weights[i, c_i]`.
    pub fn cat_bias(nodes: impl Into<Block>, weights: Vec<f32>) -> Self {
        Self {
            spin_groups: Vec::new(),
            cat_groups: vec![nodes.into()],
            weights,
        }
    }

    /// A batch of pairwise categorical couplings: `E_i = -weights[i, c_ai, c_bi]`.
    pub fn cat_coupling(a: impl Into<Block>, b: impl Into<Block>, weights: Vec<f32>) -> Self {
        Self {
            spin_groups: Vec::new(),
            cat_groups: vec![a.into(), b.into()],
            weights,
        }
    }

    /// Number of instances in the batch.
    pub fn batch_len(&self) -> usize {
        self.spin_groups
            .first()
            .or_else(|| self.cat_groups.first())
            .map_or(0, |block| block.len())
    }

    /// Evaluate the total energy of this factor batch for the
    /// given dense per-node state.
    ///
    /// `values` holds one value per graph node: 0/1 for spins,
    /// the category index for categorical nodes.
    pub fn energy(&self, graph: &Graph, values: &[u32]) -> f64 {
        let mut cat_strides = vec![0usize; self.cat_groups.len()];
        let mut instance_stride = 1usize;
        for (stride, block) in cat_strides.iter_mut().zip(self.cat_groups.iter()).rev() {
            *stride = instance_stride;
            let states = match graph.node_kind(block.nodes[0]) {
                NodeKind::Categorical { states } => states,
                NodeKind::Spin => panic!("spin node in a categorical group"),
            };
            instance_stride *= states as usize;
        }

        let mut total = 0.0;
        for i in 0..self.batch_len() {
            let mut product = 1.0f64;
            for block in self.spin_groups.iter() {
                let value = values[block.nodes[i].index()];
                product *= if value != 0 { 1.0 } else { -1.0 };
            }
            let mut index = i * instance_stride;
            for (block, &stride) in self.cat_groups.iter().zip(cat_strides.iter()) {
                index += values[block.nodes[i].index()] as usize * stride;
            }
            total -= self.weights[index] as f64 * product;
        }
        total
    }
}

/// Evaluate the total energy of a model given by a list of factors.
pub fn total_energy(factors: &[DiscreteFactor], graph: &Graph, values: &[u32]) -> f64 {
    factors
        .iter()
        .map(|factor| factor.energy(graph, values))
        .sum()
}
