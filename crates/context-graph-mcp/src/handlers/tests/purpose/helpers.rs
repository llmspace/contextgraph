//! Helper Functions for Purpose Tests
//!
//! TASK-CORE-005: UUID-based goal ID extraction utilities.

use serde_json::json;

use crate::handlers::Handlers;
use crate::protocol::JsonRpcId;

use super::super::make_request;

// =============================================================================
// Helper Functions for UUID-based Goal Tests (TASK-CORE-005)
// =============================================================================

/// Extract goal IDs from the hierarchy via get_all query.
/// TASK-P0-001: Updated for 3-level hierarchy - now returns (first_strategic_id, all_strategic_ids, tactical_ids, immediate_ids).
/// The first return value is the first Strategic goal (top-level) for backwards compatibility.
pub(crate) async fn get_goal_ids_from_hierarchy(
    handlers: &Handlers,
) -> (String, Vec<String>, Vec<String>, Vec<String>) {
    let query_params = json!({ "operation": "get_all" });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(999)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;
    let result = response.result.expect("get_all should succeed");
    let goals = result
        .get("goals")
        .and_then(|v| v.as_array())
        .expect("Should have goals");

    let mut strategic_ids = Vec::new();
    let mut tactical_ids = Vec::new();
    let mut immediate_ids = Vec::new();

    for goal in goals {
        let id = goal
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let level = goal.get("level").and_then(|v| v.as_str()).unwrap_or("");
        match level {
            "Strategic" => strategic_ids.push(id),
            "Tactical" => tactical_ids.push(id),
            "Immediate" => immediate_ids.push(id),
            _ => {}
        }
    }

    // TASK-P0-001: Return first Strategic as the top-level goal
    let first_strategic = strategic_ids.first().cloned().unwrap_or_default();
    (first_strategic, strategic_ids, tactical_ids, immediate_ids)
}
