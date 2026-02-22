//! Bidirectional A* search algorithm.
//!
//! Searches from both start and goal simultaneously for improved performance.
//! Forward search follows outgoing edges; backward search follows incoming edges.

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::config::HyperbolicConfig;
use crate::error::{GraphError, GraphResult};
use crate::hyperbolic::poincare::PoincarePoint;
use crate::hyperbolic::PoincareBall;
use crate::storage::GraphStorage;

use super::helpers::{edge_cost, to_hyperbolic_point, uuid_to_i64};
use super::node::AstarNode;
use super::types::{AstarParams, AstarResult, NodeId};

/// Expand one direction of the bidirectional search.
///
/// For forward search: follows outgoing edges, neighbor = edge.target
/// For backward search: follows incoming edges, neighbor = edge.source
#[allow(clippy::too_many_arguments)]
fn expand_direction(
    storage: &GraphStorage,
    ball: &PoincareBall,
    params: &AstarParams,
    current_id: NodeId,
    current_g: f32,
    is_forward: bool,
    g_scores: &mut HashMap<NodeId, f32>,
    parent_map: &mut HashMap<NodeId, NodeId>,
    closed_set: &HashSet<NodeId>,
    open_set: &mut BinaryHeap<AstarNode>,
    other_g: &HashMap<NodeId, f32>,
    target_point: &PoincarePoint,
    best_path_cost: &mut f32,
    meeting_node: &mut Option<NodeId>,
) -> GraphResult<()> {
    // GRAPH-H1 FIX: Forward follows outgoing edges, backward follows incoming edges
    let edges = if is_forward {
        storage.get_outgoing_edges(current_id)?
    } else {
        storage.get_incoming_edges(current_id)?
    };

    for edge in edges {
        // Filter by edge type
        if let Some(ref allowed_types) = params.edge_types {
            if !allowed_types.contains(&edge.edge_type) {
                continue;
            }
        }

        // Get modulated weight
        let effective_weight = edge.get_modulated_weight(params.domain);

        // Filter by minimum weight
        if effective_weight < params.min_weight {
            continue;
        }

        // GRAPH-H1 FIX: Forward neighbor = target, backward neighbor = source
        let neighbor_id = if is_forward {
            uuid_to_i64(&edge.target)
        } else {
            uuid_to_i64(&edge.source)
        };

        if closed_set.contains(&neighbor_id) {
            continue;
        }

        let tentative_g = current_g + edge_cost(effective_weight);
        let neighbor_g = *g_scores.get(&neighbor_id).unwrap_or(&f32::INFINITY);

        if tentative_g >= neighbor_g {
            continue;
        }

        parent_map.insert(neighbor_id, current_id);
        g_scores.insert(neighbor_id, tentative_g);

        // Get heuristic
        let neighbor_point = to_hyperbolic_point(
            storage
                .get_hyperbolic(neighbor_id)?
                .ok_or(GraphError::MissingHyperbolicData(neighbor_id))?,
        );

        let h = params.heuristic_scale * ball.distance(&neighbor_point, target_point);
        open_set.push(AstarNode::new(neighbor_id, tentative_g, h));

        // Check if meets other search
        if let Some(&other_cost) = other_g.get(&neighbor_id) {
            let path_cost = tentative_g + other_cost;
            if path_cost < *best_path_cost {
                *best_path_cost = path_cost;
                *meeting_node = Some(neighbor_id);
            }
        }
    }

    Ok(())
}

/// A* with bidirectional search optimization.
///
/// Searches from both start and goal simultaneously, meeting in the middle.
/// Forward search follows outgoing edges (source→target).
/// Backward search follows incoming edges (target→source).
///
/// # Arguments
/// * `storage` - Graph storage backend
/// * `start` - Starting node ID
/// * `goal` - Goal node ID
/// * `params` - A* parameters
///
/// # Returns
/// Same as `astar_search`
pub fn astar_bidirectional(
    storage: &GraphStorage,
    start: NodeId,
    goal: NodeId,
    params: AstarParams,
) -> GraphResult<AstarResult> {
    // Handle trivial case
    if start == goal {
        return Ok(AstarResult::found(vec![start], 0.0, 1, 0));
    }

    // Initialize Poincare ball
    let config = HyperbolicConfig::default();
    let ball = PoincareBall::new(config);

    // Get hyperbolic embeddings (NO FALLBACK)
    let start_point = to_hyperbolic_point(
        storage
            .get_hyperbolic(start)?
            .ok_or(GraphError::MissingHyperbolicData(start))?,
    );
    let goal_point = to_hyperbolic_point(
        storage
            .get_hyperbolic(goal)?
            .ok_or(GraphError::MissingHyperbolicData(goal))?,
    );

    // Forward search state
    let estimated_nodes = params.max_nodes / 2;
    let mut forward_open: BinaryHeap<AstarNode> = BinaryHeap::new();
    let mut forward_g: HashMap<NodeId, f32> = HashMap::with_capacity(estimated_nodes);
    let mut forward_parent: HashMap<NodeId, NodeId> = HashMap::with_capacity(estimated_nodes);
    let mut forward_closed: HashSet<NodeId> = HashSet::new();

    let h_start = params.heuristic_scale * ball.distance(&start_point, &goal_point);
    forward_open.push(AstarNode::new(start, 0.0, h_start));
    forward_g.insert(start, 0.0);

    // Backward search state
    let mut backward_open: BinaryHeap<AstarNode> = BinaryHeap::new();
    let mut backward_g: HashMap<NodeId, f32> = HashMap::with_capacity(estimated_nodes);
    let mut backward_parent: HashMap<NodeId, NodeId> = HashMap::with_capacity(estimated_nodes);
    let mut backward_closed: HashSet<NodeId> = HashSet::new();

    let h_goal = params.heuristic_scale * ball.distance(&goal_point, &start_point);
    backward_open.push(AstarNode::new(goal, 0.0, h_goal));
    backward_g.insert(goal, 0.0);

    let mut best_path_cost = f32::INFINITY;
    let mut meeting_node: Option<NodeId> = None;
    let mut nodes_explored = 0;

    // Alternate between forward and backward search
    let mut forward_turn = true;

    loop {
        if forward_open.is_empty() && backward_open.is_empty() {
            break;
        }

        if nodes_explored >= params.max_nodes {
            log::debug!("Bidirectional A* limit reached: {} nodes", nodes_explored);
            break;
        }

        // Determine which direction to expand
        let is_forward = if forward_turn && !forward_open.is_empty() {
            true
        } else if !backward_open.is_empty() {
            false
        } else if !forward_open.is_empty() {
            true
        } else {
            break;
        };

        forward_turn = !forward_turn;

        // Select the correct state for this direction
        let (open_set, g_scores, parent_map, closed_set, other_closed, other_g, target_point) =
            if is_forward {
                (
                    &mut forward_open,
                    &mut forward_g,
                    &mut forward_parent,
                    &mut forward_closed,
                    &backward_closed,
                    &backward_g,
                    &goal_point,
                )
            } else {
                (
                    &mut backward_open,
                    &mut backward_g,
                    &mut backward_parent,
                    &mut backward_closed,
                    &forward_closed,
                    &forward_g,
                    &start_point,
                )
            };

        // Pop from open set
        let Some(current) = open_set.pop() else {
            continue;
        };

        let current_id = current.node_id;

        // Skip if explored
        if closed_set.contains(&current_id) {
            continue;
        }

        closed_set.insert(current_id);
        nodes_explored += 1;

        // Check if other search has reached this node
        if other_closed.contains(&current_id) {
            let this_cost = *g_scores.get(&current_id).unwrap_or(&f32::INFINITY);
            let other_cost = *other_g.get(&current_id).unwrap_or(&f32::INFINITY);
            let path_cost = this_cost + other_cost;

            if path_cost < best_path_cost {
                best_path_cost = path_cost;
                meeting_node = Some(current_id);
            }
        }

        // Early termination: if best f-score > best path, we're done
        if current.f_score >= best_path_cost {
            break;
        }

        let current_g = *g_scores.get(&current_id).unwrap_or(&f32::INFINITY);

        // GRAPH-H1 FIX: Expand using direction-appropriate edge traversal
        expand_direction(
            storage,
            &ball,
            &params,
            current_id,
            current_g,
            is_forward,
            g_scores,
            parent_map,
            closed_set,
            open_set,
            other_g,
            target_point,
            &mut best_path_cost,
            &mut meeting_node,
        )?;
    }

    // Reconstruct path if found
    if let Some(meet) = meeting_node {
        let mut forward_path = vec![meet];
        let mut node = meet;
        while let Some(&parent) = forward_parent.get(&node) {
            forward_path.push(parent);
            node = parent;
        }
        forward_path.reverse();

        let mut backward_path = Vec::new();
        node = meet;
        while let Some(&parent) = backward_parent.get(&node) {
            backward_path.push(parent);
            node = parent;
        }

        // Combine paths (forward_path ends with meet, backward_path starts after meet)
        forward_path.extend(backward_path);

        return Ok(AstarResult::found(
            forward_path,
            best_path_cost,
            nodes_explored,
            forward_open.len() + backward_open.len(),
        ));
    }

    Ok(AstarResult::no_path(nodes_explored))
}
