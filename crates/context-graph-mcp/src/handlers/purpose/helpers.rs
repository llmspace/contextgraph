//! Helper functions for purpose handlers.
//!
//! Provides JSON conversion and statistics computation utilities
//! shared across purpose and goal handler modules.

use serde_json::json;

use context_graph_core::purpose::{GoalHierarchy, GoalLevel, GoalNode};

/// Convert a GoalNode to JSON representation.
pub(super) fn goal_to_json(goal: &GoalNode) -> serde_json::Value {
    json!({
        "id": goal.id.to_string(),
        "description": goal.description,
        "level": format!("{:?}", goal.level),
        "level_depth": goal.level.depth(),
        "parent_id": goal.parent_id.map(|p| p.to_string()),
        "discovery": {
            "method": format!("{:?}", goal.discovery.method),
            "confidence": goal.discovery.confidence,
            "cluster_size": goal.discovery.cluster_size,
            "coherence": goal.discovery.coherence
        },
        "propagation_weight": goal.level.propagation_weight(),
        "child_count": goal.child_ids.len(),
        "is_north_star": goal.is_north_star()
    })
}

/// Compute hierarchy statistics.
pub(super) fn compute_hierarchy_stats(hierarchy: &GoalHierarchy) -> serde_json::Value {
    let north_star_count = hierarchy.at_level(GoalLevel::NorthStar).len();
    let strategic_count = hierarchy.at_level(GoalLevel::Strategic).len();
    let tactical_count = hierarchy.at_level(GoalLevel::Tactical).len();
    let immediate_count = hierarchy.at_level(GoalLevel::Immediate).len();

    json!({
        "total_goals": hierarchy.len(),
        "has_north_star": hierarchy.has_north_star(),
        "level_counts": {
            "north_star": north_star_count,
            "strategic": strategic_count,
            "tactical": tactical_count,
            "immediate": immediate_count
        },
        "is_valid": hierarchy.validate().is_ok()
    })
}
