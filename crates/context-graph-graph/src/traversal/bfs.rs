//! BFS (Breadth-First Search) graph traversal with Marblestone domain modulation.
//!
//! Explores the graph level by level, applying edge type filtering and
//! NT weight modulation based on query domain.
//!
//! # Performance
//!
//! Target: <100ms for depth=6 on 10M node graph.
//! Uses VecDeque for O(1) frontier operations.
//! Uses HashSet for O(1) visited lookup.
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights.formula: Canonical modulation formula
//! - AP-009: NaN/Infinity clamped to valid range

use std::collections::{HashMap, HashSet, VecDeque};

use uuid::Uuid;

use crate::error::GraphResult;
use crate::storage::{GraphEdge, GraphStorage};

// Re-export edge types for convenience
pub use crate::storage::edges::{Domain, EdgeType};

/// Node ID type for BFS (i64 for storage compatibility).
pub type NodeId = i64;

/// Parameters for BFS traversal.
///
/// Controls depth limits, node limits, and filtering behavior.
#[derive(Debug, Clone)]
pub struct BfsParams {
    /// Maximum depth to traverse (default: 6).
    /// Depth 0 is the start node.
    pub max_depth: usize,

    /// Maximum number of nodes to visit (default: 10000).
    /// Prevents runaway traversal on dense graphs.
    pub max_nodes: usize,

    /// Filter to specific edge types (None = all types).
    pub edge_types: Option<Vec<EdgeType>>,

    /// Domain filter for edge weighting (None = no domain preference).
    /// When set, uses `get_modulated_weight(domain)` instead of base weight.
    pub domain_filter: Option<Domain>,

    /// Minimum edge weight threshold (after modulation).
    /// Edges below this weight are not traversed.
    pub min_weight: f32,

    /// Whether to include edge data in results.
    pub include_edges: bool,

    /// Whether to record traversal on edges (updates steering_reward).
    /// Only set true if you will persist the updated edges.
    pub record_traversal: bool,
}

impl Default for BfsParams {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_nodes: 10_000,
            edge_types: None,
            domain_filter: None,
            min_weight: 0.0,
            include_edges: true,
            record_traversal: false,
        }
    }
}

impl BfsParams {
    /// Create params with specific max depth.
    #[must_use]
    pub fn with_depth(max_depth: usize) -> Self {
        Self {
            max_depth,
            ..Default::default()
        }
    }

    /// Create params for specific domain.
    #[must_use]
    pub fn for_domain(domain: Domain) -> Self {
        Self {
            domain_filter: Some(domain),
            ..Default::default()
        }
    }

    /// Builder: set max depth.
    #[must_use]
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Builder: set max nodes.
    #[must_use]
    pub fn max_nodes(mut self, nodes: usize) -> Self {
        self.max_nodes = nodes;
        self
    }

    /// Builder: set edge types filter.
    #[must_use]
    pub fn edge_types(mut self, types: Vec<EdgeType>) -> Self {
        self.edge_types = Some(types);
        self
    }

    /// Builder: set domain filter.
    #[must_use]
    pub fn domain(mut self, domain: Domain) -> Self {
        self.domain_filter = Some(domain);
        self
    }

    /// Builder: set minimum weight threshold.
    #[must_use]
    pub fn min_weight(mut self, weight: f32) -> Self {
        self.min_weight = weight;
        self
    }

    /// Builder: set whether to include edges in results.
    #[must_use]
    pub fn include_edges(mut self, include: bool) -> Self {
        self.include_edges = include;
        self
    }
}

/// Result of BFS traversal.
#[derive(Debug, Clone)]
pub struct BfsResult {
    /// Visited node IDs in BFS order (i64).
    pub nodes: Vec<NodeId>,

    /// Traversed edges (if include_edges was true).
    pub edges: Vec<GraphEdge>,

    /// Number of nodes found at each depth level.
    pub depth_counts: HashMap<usize, usize>,

    /// Starting node ID.
    pub start_node: NodeId,

    /// Actual maximum depth reached.
    pub max_depth_reached: usize,

    /// Whether traversal was limited by max_nodes.
    pub truncated: bool,
}

impl BfsResult {
    /// Check if no nodes were found (shouldn't happen - start node always included).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get total node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get total edge count.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get count of nodes at specific depth.
    #[must_use]
    pub fn nodes_at_depth(&self, depth: usize) -> usize {
        *self.depth_counts.get(&depth).unwrap_or(&0)
    }
}

/// Convert UUID to i64 for storage key operations.
///
/// This reverses `Uuid::from_u64_pair(id as u64, 0)` used in storage.
/// from_u64_pair stores values in big-endian order in the UUID bytes.
#[inline]
fn uuid_to_i64(uuid: &Uuid) -> i64 {
    let bytes = uuid.as_bytes();
    // from_u64_pair uses big-endian byte order
    i64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Perform BFS traversal from a starting node.
///
/// # Arguments
/// * `storage` - Graph storage backend
/// * `start` - Starting node ID (i64)
/// * `params` - Traversal parameters
///
/// # Returns
/// * `Ok(BfsResult)` - Traversal results
/// * `Err(GraphError::Storage*)` - Storage access failed
///
/// # Performance
/// Target: <100ms for depth=6 on 10M node graph
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_graph::traversal::bfs::{bfs_traverse, BfsParams, Domain};
///
/// let params = BfsParams::default()
///     .max_depth(3)
///     .domain(Domain::Code)
///     .min_weight(0.3);
///
/// let result = bfs_traverse(&storage, start_node, params)?;
/// println!("Found {} nodes at {} depth levels",
///     result.node_count(), result.max_depth_reached);
///
/// // Access modulated weights
/// for edge in &result.edges {
///     let w = edge.get_modulated_weight(Domain::Code);
///     println!("Edge {} -> {} weight: {}", edge.source, edge.target, w);
/// }
/// ```
pub fn bfs_traverse(
    storage: &GraphStorage,
    start: NodeId,
    params: BfsParams,
) -> GraphResult<BfsResult> {
    // Pre-allocate with reasonable capacity
    let mut visited: HashSet<NodeId> = HashSet::with_capacity(params.max_nodes.min(10000));
    let mut frontier: VecDeque<(NodeId, usize)> = VecDeque::with_capacity(1000);
    let mut result_nodes: Vec<NodeId> = Vec::with_capacity(params.max_nodes.min(10000));
    let mut result_edges: Vec<GraphEdge> = if params.include_edges {
        Vec::with_capacity(params.max_nodes.min(10000))
    } else {
        Vec::new()
    };
    let mut depth_counts: HashMap<usize, usize> = HashMap::new();
    let mut max_depth_reached: usize = 0;
    let mut truncated = false;

    // Initialize with start node
    frontier.push_back((start, 0));
    visited.insert(start);

    while let Some((current_node, depth)) = frontier.pop_front() {
        // Check node limit BEFORE processing
        if result_nodes.len() >= params.max_nodes {
            truncated = true;
            log::debug!(
                "BFS truncated at {} nodes (limit: {})",
                result_nodes.len(),
                params.max_nodes
            );
            break;
        }

        // Add current node to results
        result_nodes.push(current_node);
        *depth_counts.entry(depth).or_insert(0) += 1;
        max_depth_reached = max_depth_reached.max(depth);

        // Don't expand if at max depth
        if depth >= params.max_depth {
            continue;
        }

        // Get full edges with Marblestone fields using get_outgoing_edges
        let edges = storage.get_outgoing_edges(current_node)?;

        for edge in edges {
            // Filter by edge type if specified
            if let Some(ref allowed_types) = params.edge_types {
                if !allowed_types.contains(&edge.edge_type) {
                    continue;
                }
            }

            // Get effective weight (with domain modulation if specified)
            let effective_weight = if let Some(domain) = params.domain_filter {
                edge.get_modulated_weight(domain)
            } else {
                edge.weight
            };

            // Filter by minimum weight
            if effective_weight < params.min_weight {
                continue;
            }

            // Convert UUID target to i64 for visited tracking
            let target_i64 = uuid_to_i64(&edge.target);

            // Skip if already visited
            if visited.contains(&target_i64) {
                continue;
            }

            // Add to visited and frontier
            visited.insert(target_i64);
            frontier.push_back((target_i64, depth + 1));

            // Collect edge if requested
            if params.include_edges {
                result_edges.push(edge);
            }
        }
    }

    log::debug!(
        "BFS complete: {} nodes, {} edges, max_depth={}",
        result_nodes.len(),
        result_edges.len(),
        max_depth_reached
    );

    Ok(BfsResult {
        nodes: result_nodes,
        edges: result_edges,
        depth_counts,
        start_node: start,
        max_depth_reached,
        truncated,
    })
}

/// Find shortest path between two nodes using BFS.
///
/// # Arguments
/// * `storage` - Graph storage backend
/// * `start` - Starting node ID
/// * `target` - Target node ID
/// * `max_depth` - Maximum search depth
///
/// # Returns
/// * `Ok(Some(path))` - Path from start to target (inclusive)
/// * `Ok(None)` - No path found within max_depth
pub fn bfs_shortest_path(
    storage: &GraphStorage,
    start: NodeId,
    target: NodeId,
    max_depth: usize,
) -> GraphResult<Option<Vec<NodeId>>> {
    if start == target {
        return Ok(Some(vec![start]));
    }

    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut frontier: VecDeque<(NodeId, usize)> = VecDeque::new();
    let mut parent: HashMap<NodeId, NodeId> = HashMap::new();

    frontier.push_back((start, 0));
    visited.insert(start);

    while let Some((current_node, depth)) = frontier.pop_front() {
        if depth >= max_depth {
            continue;
        }

        let edges = storage.get_outgoing_edges(current_node)?;

        for edge in edges {
            let target_i64 = uuid_to_i64(&edge.target);

            if visited.contains(&target_i64) {
                continue;
            }

            parent.insert(target_i64, current_node);
            visited.insert(target_i64);

            if target_i64 == target {
                // Reconstruct path
                let mut path = vec![target];
                let mut current = target;

                while let Some(&prev) = parent.get(&current) {
                    path.push(prev);
                    current = prev;
                }

                path.reverse();
                return Ok(Some(path));
            }

            frontier.push_back((target_i64, depth + 1));
        }
    }

    Ok(None)
}

/// Get all nodes within a given distance from start.
///
/// Convenience wrapper around bfs_traverse.
pub fn bfs_neighborhood(
    storage: &GraphStorage,
    start: NodeId,
    max_distance: usize,
) -> GraphResult<Vec<NodeId>> {
    let params = BfsParams::with_depth(max_distance).include_edges(false);
    let result = bfs_traverse(storage, start, params)?;
    Ok(result.nodes)
}

/// Get nodes within distance, filtered by domain.
///
/// Returns only nodes reachable via edges with weight >= min_weight
/// after domain modulation.
pub fn bfs_domain_neighborhood(
    storage: &GraphStorage,
    start: NodeId,
    max_distance: usize,
    domain: Domain,
    min_weight: f32,
) -> GraphResult<Vec<NodeId>> {
    let params = BfsParams::with_depth(max_distance)
        .domain(domain)
        .min_weight(min_weight)
        .include_edges(false);
    let result = bfs_traverse(storage, start, params)?;
    Ok(result.nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Create test graph and return (storage, start_node_id, tempdir).
    /// TempDir must be kept alive for the duration of the test.
    fn setup_test_graph() -> (GraphStorage, NodeId, tempfile::TempDir) {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        // Create a simple tree structure:
        //     1
        //    / \
        //   2   3
        //  /|   |\
        // 4 5   6 7

        // Use Uuid::from_u64_pair to create consistent UUIDs from i64 node IDs
        let uuid = |id: i64| Uuid::from_u64_pair(id as u64, 0);

        let edges = vec![
            // From node 1
            GraphEdge::new(1, uuid(1), uuid(2), EdgeType::Semantic, 0.8, Domain::General),
            GraphEdge::new(2, uuid(1), uuid(3), EdgeType::Semantic, 0.8, Domain::General),
            // From node 2
            GraphEdge::new(3, uuid(2), uuid(4), EdgeType::Semantic, 0.7, Domain::General),
            GraphEdge::new(4, uuid(2), uuid(5), EdgeType::Semantic, 0.7, Domain::General),
            // From node 3
            GraphEdge::new(5, uuid(3), uuid(6), EdgeType::Hierarchical, 0.7, Domain::Code),
            GraphEdge::new(6, uuid(3), uuid(7), EdgeType::Hierarchical, 0.7, Domain::Code),
        ];

        storage.put_edges(&edges).expect("put_edges failed");

        (storage, 1, dir)
    }

    #[test]
    fn test_bfs_basic_traversal() {
        let (storage, start, _dir) = setup_test_graph();

        let result = bfs_traverse(&storage, start, BfsParams::default())
            .expect("BFS failed");

        // Should find all 7 nodes
        assert_eq!(result.node_count(), 7, "Expected 7 nodes, got {}", result.node_count());
        assert_eq!(result.nodes[0], 1, "Start node should be first");

        // Verify depth counts
        assert_eq!(result.nodes_at_depth(0), 1, "Depth 0: 1 node");
        assert_eq!(result.nodes_at_depth(1), 2, "Depth 1: 2 nodes");
        assert_eq!(result.nodes_at_depth(2), 4, "Depth 2: 4 nodes");

        assert!(!result.truncated);
        assert_eq!(result.max_depth_reached, 2);
    }

    #[test]
    fn test_bfs_max_depth_limit() {
        let (storage, start, _dir) = setup_test_graph();

        let result = bfs_traverse(
            &storage,
            start,
            BfsParams::default().max_depth(1),
        ).expect("BFS failed");

        // Should find only depth 0 and 1: nodes 1, 2, 3
        assert_eq!(result.node_count(), 3);
        assert_eq!(result.max_depth_reached, 1);
    }

    #[test]
    fn test_bfs_max_nodes_limit() {
        let (storage, start, _dir) = setup_test_graph();

        let result = bfs_traverse(
            &storage,
            start,
            BfsParams::default().max_nodes(3),
        ).expect("BFS failed");

        assert_eq!(result.node_count(), 3);
        assert!(result.truncated);
    }

    #[test]
    fn test_bfs_edge_type_filter() {
        let (storage, start, _dir) = setup_test_graph();

        // Only follow Semantic edges (not Hierarchical)
        let result = bfs_traverse(
            &storage,
            start,
            BfsParams::default().edge_types(vec![EdgeType::Semantic]),
        ).expect("BFS failed");

        // Nodes 6 and 7 are only reachable via Hierarchical edges
        // So we should find: 1, 2, 3, 4, 5 = 5 nodes
        assert_eq!(result.node_count(), 5);
    }

    #[test]
    fn test_bfs_domain_modulation() {
        let (storage, start, _dir) = setup_test_graph();

        // With Code domain, edges from node 3 (Domain::Code) get bonus
        let result = bfs_traverse(
            &storage,
            start,
            BfsParams::default().domain(Domain::Code).min_weight(0.5),
        ).expect("BFS failed");

        // All edges should pass 0.5 threshold
        assert!(result.node_count() >= 1);

        // Verify modulated weights are accessible
        for edge in &result.edges {
            let w = edge.get_modulated_weight(Domain::Code);
            assert!(w >= 0.5, "Edge weight {} should be >= 0.5", w);
        }
    }

    #[test]
    fn test_bfs_shortest_path_found() {
        let (storage, _, _dir) = setup_test_graph();

        let path = bfs_shortest_path(&storage, 1, 7, 10)
            .expect("BFS failed");

        assert!(path.is_some());
        let path = path.unwrap();

        assert_eq!(path[0], 1, "Path should start at node 1");
        assert_eq!(*path.last().unwrap(), 7, "Path should end at node 7");
        assert_eq!(path.len(), 3, "Path should be: 1 -> 3 -> 7");
    }

    #[test]
    fn test_bfs_shortest_path_not_found() {
        let (storage, _, _dir) = setup_test_graph();

        // Node 99 doesn't exist
        let path = bfs_shortest_path(&storage, 1, 99, 10)
            .expect("BFS failed");

        assert!(path.is_none());
    }

    #[test]
    fn test_bfs_neighborhood() {
        let (storage, start, _dir) = setup_test_graph();

        let neighbors = bfs_neighborhood(&storage, start, 1)
            .expect("BFS failed");

        // Distance 1: nodes 1, 2, 3
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_bfs_empty_graph() {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        let result = bfs_traverse(&storage, 1, BfsParams::default())
            .expect("BFS failed");

        // Should return just the start node (even if it has no edges)
        assert_eq!(result.node_count(), 1);
        assert_eq!(result.nodes[0], 1);
        assert_eq!(result.edge_count(), 0);
    }

    #[test]
    fn test_bfs_min_weight_filter() {
        let (storage, start, _dir) = setup_test_graph();

        // Set high min_weight to filter most edges
        let result = bfs_traverse(
            &storage,
            start,
            BfsParams::default().min_weight(0.9),
        ).expect("BFS failed");

        // Most edges have weight 0.7-0.8, should be filtered
        // Only start node should be returned
        assert!(result.node_count() < 7);
    }

    #[test]
    fn test_bfs_cyclic_graph() {
        let dir = tempdir().expect("Failed to create temp dir");
        let storage = GraphStorage::open_default(dir.path())
            .expect("Failed to open storage");

        // Create cycle: 1 -> 2 -> 3 -> 1
        let uuid = |id: i64| Uuid::from_u64_pair(id as u64, 0);
        let edges = vec![
            GraphEdge::new(1, uuid(1), uuid(2), EdgeType::Semantic, 0.8, Domain::General),
            GraphEdge::new(2, uuid(2), uuid(3), EdgeType::Semantic, 0.8, Domain::General),
            GraphEdge::new(3, uuid(3), uuid(1), EdgeType::Semantic, 0.8, Domain::General),
        ];
        storage.put_edges(&edges).expect("put_edges failed");

        let result = bfs_traverse(&storage, 1, BfsParams::default())
            .expect("BFS failed");

        // Should visit each node exactly once, no infinite loop
        assert_eq!(result.node_count(), 3);
        assert!(result.nodes.contains(&1));
        assert!(result.nodes.contains(&2));
        assert!(result.nodes.contains(&3));
    }
}
