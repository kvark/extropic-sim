/// Kind of a random variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    /// A spin variable taking values in {-1, +1}.
    ///
    /// The state is stored as 0 (down, -1) or 1 (up, +1).
    Spin,
    /// A categorical variable taking integer values in `[0, states)`.
    Categorical {
        /// Number of distinct states.
        states: u32,
    },
}

impl NodeKind {
    /// Number of distinct values a node of this kind can take.
    pub fn state_count(&self) -> u32 {
        match *self {
            Self::Spin => 2,
            Self::Categorical { states } => states,
        }
    }
}

/// Handle of a node within a [`Graph`].
///
/// Nodes are cheap copyable identifiers. They are only meaningful
/// together with the graph that created them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Node(pub(crate) u32);

impl Node {
    /// Index of the node in the graph.
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

/// A collection of random variables that make up a probabilistic graphical model.
///
/// The graph only tracks the variables themselves. Dependencies between
/// them are introduced by factors when compiling a [`Program`](crate::Program).
#[derive(Clone, Debug, Default)]
pub struct Graph {
    kinds: Vec<NodeKind>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, kind: NodeKind) -> Node {
        let index = self.kinds.len() as u32;
        self.kinds.push(kind);
        Node(index)
    }

    /// Add a single spin node.
    pub fn add_spin(&mut self) -> Node {
        self.add(NodeKind::Spin)
    }

    /// Add `count` spin nodes.
    pub fn add_spins(&mut self, count: usize) -> Vec<Node> {
        (0..count).map(|_| self.add_spin()).collect()
    }

    /// Add a single categorical node with the given number of states.
    pub fn add_categorical(&mut self, states: u32) -> Node {
        self.add(NodeKind::Categorical { states })
    }

    /// Add `count` categorical nodes with the given number of states.
    pub fn add_categoricals(&mut self, count: usize, states: u32) -> Vec<Node> {
        (0..count).map(|_| self.add_categorical(states)).collect()
    }

    /// Kind of the given node.
    pub fn node_kind(&self, node: Node) -> NodeKind {
        self.kinds[node.index()]
    }

    /// Total number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.kinds.len()
    }

    /// Iterate over all nodes in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = Node> + '_ {
        (0..self.kinds.len() as u32).map(Node)
    }
}

/// An ordered set of nodes of the same kind.
///
/// Blocks are the unit of parallel updates in block Gibbs sampling:
/// all nodes of a block are updated simultaneously, conditioned on the
/// rest of the model. Nodes of a single block must not interact with
/// each other, which is validated at program compile time.
#[derive(Clone, Debug)]
pub struct Block {
    pub(crate) nodes: Vec<Node>,
}

impl Block {
    /// Create a block from a list of nodes.
    pub fn new(nodes: impl Into<Vec<Node>>) -> Self {
        Self {
            nodes: nodes.into(),
        }
    }

    /// Nodes of the block, in order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Number of nodes in the block.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the block contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl From<Vec<Node>> for Block {
    fn from(nodes: Vec<Node>) -> Self {
        Self::new(nodes)
    }
}

impl From<&[Node]> for Block {
    fn from(nodes: &[Node]) -> Self {
        Self::new(nodes)
    }
}

impl<const N: usize> From<[Node; N]> for Block {
    fn from(nodes: [Node; N]) -> Self {
        Self::new(nodes)
    }
}
