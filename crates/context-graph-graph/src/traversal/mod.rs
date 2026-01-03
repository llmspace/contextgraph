//! Graph traversal algorithms.
//!
//! This module provides graph traversal algorithms for navigating the
//! knowledge graph with edge type filtering and NT weight modulation.
//!
//! # Algorithms
//!
//! - **BFS**: Breadth-first search for shortest paths and level-order exploration
//! - **DFS**: Depth-first search (iterative, NOT recursive) for deep exploration
//! - **A***: A* search with hyperbolic distance heuristic for optimal pathfinding
//!
//! # Edge Filtering
//!
//! All traversals support filtering by:
//! - Edge types (Semantic, Temporal, Causal, Hierarchical)
//! - Minimum weight threshold
//! - Domain-specific modulation via NT weights
//!
//! # Components
//!
//! - BFS traversal (TODO: M04-T16)
//! - DFS traversal (TODO: M04-T17)
//! - A* traversal with hyperbolic heuristic (TODO: M04-T17a)
//! - Traversal utilities (TODO: M04-T22)
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights.formula: w_eff = base * (1 + excitatory - inhibitory + 0.5*modulatory)

// TODO: M04-T16 - Implement BFS traversal
// pub struct BfsIterator<'a> { ... }
// impl<'a> BfsIterator<'a> {
//     pub fn new(storage: &'a GraphStorage, start: NodeId, filter: EdgeFilter) -> Self
// }
// impl<'a> Iterator for BfsIterator<'a> {
//     type Item = (NodeId, usize); // (node, depth)
// }

// TODO: M04-T17 - Implement DFS traversal
// pub struct DfsIterator<'a> { ... }
// impl<'a> DfsIterator<'a> {
//     pub fn new(storage: &'a GraphStorage, start: NodeId, filter: EdgeFilter) -> Self
// }
// Note: MUST be iterative, NOT recursive (to avoid stack overflow)

// TODO: M04-T17a - Implement A* traversal
// pub fn astar(
//     storage: &GraphStorage,
//     start: NodeId,
//     goal: NodeId,
//     heuristic: impl Fn(&PoincarePoint, &PoincarePoint) -> f32,
// ) -> GraphResult<Vec<NodeId>>

// TODO: M04-T22 - Implement traversal utilities
// pub struct EdgeFilter {
//     pub edge_types: Option<Vec<EdgeType>>,
//     pub min_weight: f32,
//     pub domain: Option<Domain>,
// }
// pub fn get_modulated_weight(edge: &GraphEdge, domain: Domain) -> f32
