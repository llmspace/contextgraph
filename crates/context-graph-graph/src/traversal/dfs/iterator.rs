//! DFS Iterator for lazy traversal.
//!
//! Provides an iterator-based interface to DFS traversal,
//! yielding nodes one at a time.

use std::collections::HashSet;

use crate::error::GraphResult;
use crate::storage::GraphStorage;
use crate::traversal::astar::helpers::uuid_to_i64;

use super::types::{DfsParams, NodeId};

/// DFS Iterator for lazy traversal.
///
/// Yields nodes one at a time without building full result.
/// Useful for early termination or memory-constrained scenarios.
pub struct DfsIterator<'a> {
    storage: &'a GraphStorage,
    stack: Vec<(NodeId, usize)>,
    visited: HashSet<NodeId>,
    params: DfsParams,
}

impl<'a> DfsIterator<'a> {
    /// Create a new DFS iterator.
    pub fn new(storage: &'a GraphStorage, start: NodeId, params: DfsParams) -> Self {
        Self {
            storage,
            stack: vec![(start, 0)],
            visited: HashSet::new(),
            params,
        }
    }
}

impl<'a> Iterator for DfsIterator<'a> {
    type Item = GraphResult<(NodeId, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (current, depth) = self.stack.pop()?;

            // Skip if already visited
            if self.visited.contains(&current) {
                continue;
            }

            // Check max nodes limit
            if let Some(max) = self.params.max_nodes {
                if self.visited.len() >= max {
                    return None;
                }
            }

            // Mark as visited
            self.visited.insert(current);

            // Expand children if not at max depth
            if self.params.max_depth.map_or(true, |max| depth < max) {
                // Get outgoing edges
                let edges = match self.storage.get_outgoing_edges(current) {
                    Ok(e) => e,
                    Err(err) => return Some(Err(err)),
                };

                let mut neighbors: Vec<NodeId> = Vec::new();

                for edge in edges {
                    // Filter by edge type
                    if let Some(ref allowed_types) = self.params.edge_types {
                        if !allowed_types.contains(&edge.edge_type) {
                            continue;
                        }
                    }

                    // Check weight threshold
                    let effective_weight = edge.get_modulated_weight(self.params.domain);
                    if effective_weight < self.params.min_weight {
                        continue;
                    }

                    let neighbor_id = uuid_to_i64(&edge.target);

                    if !self.visited.contains(&neighbor_id) {
                        neighbors.push(neighbor_id);
                    }
                }

                // Push in reverse for correct order
                for neighbor_id in neighbors.into_iter().rev() {
                    self.stack.push((neighbor_id, depth + 1));
                }
            }

            return Some(Ok((current, depth)));
        }
    }
}
