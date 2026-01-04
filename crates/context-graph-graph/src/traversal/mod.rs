//! Graph traversal algorithms.
//!
//! This module provides graph traversal algorithms for navigating the
//! knowledge graph with edge type filtering and NT weight modulation.
//!
//! # Algorithms
//!
//! - **BFS**: Breadth-first search for shortest paths and level-order exploration (M04-T16 ✓)
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
//! - BFS traversal (M04-T16 ✓)
//! - DFS traversal (TODO: M04-T17)
//! - A* traversal with hyperbolic heuristic (TODO: M04-T17a)
//! - Traversal utilities (TODO: M04-T22)
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights.formula: w_eff = base * (1 + excitatory - inhibitory + 0.5*modulatory)
//!
//! # Examples
//!
//! ## Basic BFS Traversal
//!
//! ```rust,ignore
//! use context_graph_graph::traversal::{bfs_traverse, BfsParams, Domain};
//!
//! let params = BfsParams::default()
//!     .max_depth(3)
//!     .domain(Domain::Code)
//!     .min_weight(0.3);
//!
//! let result = bfs_traverse(&storage, start_node, params)?;
//! println!("Found {} nodes", result.node_count());
//! ```
//!
//! ## Shortest Path
//!
//! ```rust,ignore
//! use context_graph_graph::traversal::bfs_shortest_path;
//!
//! if let Some(path) = bfs_shortest_path(&storage, start, target, 10)? {
//!     println!("Path: {:?}", path);
//! }
//! ```

// M04-T16: BFS traversal with domain modulation
pub mod bfs;

// Re-export BFS public API
pub use bfs::{
    bfs_domain_neighborhood, bfs_neighborhood, bfs_shortest_path, bfs_traverse, BfsParams,
    BfsResult, Domain, EdgeType, NodeId,
};

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
