//! goal/hierarchy_query Tests
//!
//! Tests for the goal/hierarchy_query MCP endpoint.
//! Supports operations: get_all, get_goal, get_children, get_ancestors, get_subtree.

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::super::{create_test_handlers, make_request};
use super::helpers::get_goal_ids_from_hierarchy;

/// Test goal/hierarchy_query get_all operation.
/// TASK-P0-001: Updated field names (has_top_level_goals)
#[tokio::test]
async fn test_goal_hierarchy_get_all() {
    let handlers = create_test_handlers();

    let query_params = json!({
        "operation": "get_all"
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_none(),
        "goal/hierarchy_query get_all should succeed"
    );
    let result = response.result.expect("Should have result");

    let goals = result.get("goals").and_then(|v| v.as_array());
    assert!(goals.is_some(), "Should have goals array");
    assert!(
        !goals.unwrap().is_empty(),
        "Should have at least one goal (Strategic)"
    );

    let stats = result
        .get("hierarchy_stats")
        .expect("Should have hierarchy_stats");
    assert!(
        stats.get("total_goals").is_some(),
        "Should have total_goals"
    );
    // TASK-P0-001: has_top_level_goals field
    assert!(
        stats.get("has_top_level_goals").is_some(),
        "Should have has_top_level_goals"
    );
    assert!(
        stats.get("level_counts").is_some(),
        "Should have level_counts"
    );
}

/// Test goal/hierarchy_query get_goal operation.
/// TASK-CORE-005: Updated to use UUID-based goal IDs instead of hardcoded strings.
/// TASK-P0-001: Updated for 3-level hierarchy - uses Strategic level.
#[tokio::test]
async fn test_goal_hierarchy_get_goal() {
    let handlers = create_test_handlers();

    // First, get the actual Strategic ID from the hierarchy
    let (strategic_id, _, _, _) = get_goal_ids_from_hierarchy(&handlers).await;
    assert!(!strategic_id.is_empty(), "Should have Strategic goal");

    let query_params = json!({
        "operation": "get_goal",
        "goal_id": strategic_id
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_none(),
        "goal/hierarchy_query get_goal should succeed"
    );
    let result = response.result.expect("Should have result");

    let goal = result.get("goal").expect("Should have goal");
    assert_eq!(
        goal.get("id").and_then(|v| v.as_str()),
        Some(strategic_id.as_str()),
        "Should return correct goal"
    );
    // TASK-P0-001: Strategic level
    assert_eq!(
        goal.get("level").and_then(|v| v.as_str()),
        Some("Strategic"),
        "Should be Strategic level"
    );
    // TASK-P0-001: Check is_top_level
    assert_eq!(
        goal.get("is_top_level").and_then(|v| v.as_bool()),
        Some(true),
        "Should be marked as top-level"
    );
}

/// Test goal/hierarchy_query get_children operation.
/// TASK-CORE-005: Updated to use UUID-based goal IDs instead of hardcoded strings.
/// TASK-P0-001: Updated for 3-level hierarchy - finds Strategic with children.
#[tokio::test]
async fn test_goal_hierarchy_get_children() {
    let handlers = create_test_handlers();

    // First, get all goal IDs from the hierarchy
    let (_, strategic_ids, _, _) = get_goal_ids_from_hierarchy(&handlers).await;
    assert!(!strategic_ids.is_empty(), "Should have Strategic goals");

    // Find a Strategic goal that has children by querying each one
    let mut strategic_with_children: Option<(String, usize)> = None;
    for strategic_id in &strategic_ids {
        let query_params = json!({
            "operation": "get_children",
            "goal_id": strategic_id
        });
        let query_request = make_request(
            "goal/hierarchy_query",
            Some(JsonRpcId::Number(1)),
            Some(query_params),
        );
        let response = handlers.dispatch(query_request).await;
        if response.error.is_none() {
            let result = response.result.as_ref().expect("Should have result");
            let children = result
                .get("children")
                .and_then(|v| v.as_array())
                .map(|c| c.len())
                .unwrap_or(0);
            if children > 0 {
                strategic_with_children = Some((strategic_id.clone(), children));
                break;
            }
        }
    }

    // TASK-P0-001: The test hierarchy has Strategic 1 with 1 Tactical child
    let (strategic_id, child_count) =
        strategic_with_children.expect("At least one Strategic should have children");

    let query_params = json!({
        "operation": "get_children",
        "goal_id": strategic_id
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(2)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_none(),
        "goal/hierarchy_query get_children should succeed"
    );
    let result = response.result.expect("Should have result");

    assert_eq!(
        result.get("parent_goal_id").and_then(|v| v.as_str()),
        Some(strategic_id.as_str()),
        "Should return parent_goal_id"
    );

    let children = result.get("children").and_then(|v| v.as_array());
    assert!(children.is_some(), "Should have children array");

    // Verify the found Strategic has children (should be 1 Tactical)
    let children = children.unwrap();
    assert_eq!(
        children.len(),
        child_count,
        "Strategic with children should have expected child count"
    );
}

/// Test goal/hierarchy_query get_ancestors operation.
/// TASK-CORE-005: Updated to use UUID-based goal IDs instead of hardcoded strings.
#[tokio::test]
async fn test_goal_hierarchy_get_ancestors() {
    let handlers = create_test_handlers();

    // First, get the actual immediate goal ID from the hierarchy
    let (_, _, _, immediate_ids) = get_goal_ids_from_hierarchy(&handlers).await;
    assert!(
        !immediate_ids.is_empty(),
        "Should have at least one Immediate goal"
    );
    let immediate_id = &immediate_ids[0];

    // Get ancestors of immediate goal
    let query_params = json!({
        "operation": "get_ancestors",
        "goal_id": immediate_id
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_none(),
        "goal/hierarchy_query get_ancestors should succeed"
    );
    let result = response.result.expect("Should have result");

    let ancestors = result.get("ancestors").and_then(|v| v.as_array());
    assert!(ancestors.is_some(), "Should have ancestors array");

    // Path should be: Immediate -> Tactical -> Strategic (3-level hierarchy)
    let ancestors = ancestors.unwrap();
    assert!(
        ancestors.len() >= 3,
        "Should have at least 3 ancestors (including self)"
    );
}

/// Test goal/hierarchy_query get_subtree operation.
/// TASK-CORE-005: Updated to use UUID-based goal IDs instead of hardcoded strings.
#[tokio::test]
async fn test_goal_hierarchy_get_subtree() {
    let handlers = create_test_handlers();

    // First, get the actual strategic goal ID from the hierarchy
    let (_, strategic_ids, _, _) = get_goal_ids_from_hierarchy(&handlers).await;
    assert!(
        !strategic_ids.is_empty(),
        "Should have at least one Strategic goal"
    );
    let strategic_id = &strategic_ids[0];

    // Get subtree rooted at strategic goal
    let query_params = json!({
        "operation": "get_subtree",
        "goal_id": strategic_id
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_none(),
        "goal/hierarchy_query get_subtree should succeed"
    );
    let result = response.result.expect("Should have result");

    assert_eq!(
        result.get("root_goal_id").and_then(|v| v.as_str()),
        Some(strategic_id.as_str()),
        "Should return root_goal_id"
    );

    let subtree = result.get("subtree").and_then(|v| v.as_array());
    assert!(subtree.is_some(), "Should have subtree array");

    // Subtree includes the root node and any descendants - at minimum just the root
    let subtree = subtree.unwrap();
    assert!(
        !subtree.is_empty(),
        "Subtree should have at least the root node"
    );
}

/// Test goal/hierarchy_query fails with missing operation.
#[tokio::test]
async fn test_goal_hierarchy_missing_operation_fails() {
    let handlers = create_test_handlers();

    // Use any goal_id - test checks that 'operation' is required
    let query_params = json!({
        "goal_id": "00000000-0000-0000-0000-000000000000"
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_some(),
        "goal/hierarchy_query must fail without operation"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("operation"),
        "Error should mention missing operation"
    );
}

/// Test goal/hierarchy_query fails with unknown operation.
#[tokio::test]
async fn test_goal_hierarchy_unknown_operation_fails() {
    let handlers = create_test_handlers();

    let query_params = json!({
        "operation": "invalid_op"
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_some(),
        "goal/hierarchy_query must fail with unknown operation"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
}

/// Test goal/hierarchy_query fails with non-existent goal.
/// TASK-CORE-005: Updated to use valid UUID format for non-existent goal.
#[tokio::test]
async fn test_goal_hierarchy_goal_not_found_fails() {
    let handlers = create_test_handlers();

    // Use a valid UUID format that doesn't exist in the hierarchy
    let query_params = json!({
        "operation": "get_goal",
        "goal_id": "00000000-0000-0000-0000-000000000000"
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    assert!(
        response.error.is_some(),
        "goal/hierarchy_query must fail with non-existent goal"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32020,
        "Should return GOAL_NOT_FOUND error code"
    );
}
