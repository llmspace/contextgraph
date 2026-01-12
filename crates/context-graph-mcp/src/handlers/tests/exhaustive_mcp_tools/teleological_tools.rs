//! Teleological MCP tools tests (5 tools):
//! - search_teleological
//! - compute_teleological_vector
//! - fuse_embeddings
//! - update_synergy_matrix
//! - manage_teleological_profile

use serde_json::json;
use uuid::Uuid;

use crate::handlers::tests::create_test_handlers;
use super::helpers::{make_tool_call, assert_success, assert_tool_error, get_tool_data};

// -------------------------------------------------------------------------
// search_teleological
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_search_teleological_with_content() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "search_teleological",
        json!({
            "query_content": "test query for teleological search"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "search_teleological");
}

#[tokio::test]
async fn test_search_teleological_with_strategy() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "search_teleological",
        json!({
            "query_content": "test",
            "strategy": "cosine"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "search_teleological");
}

#[tokio::test]
async fn test_search_teleological_synergy_weighted() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "search_teleological",
        json!({
            "query_content": "test",
            "strategy": "synergy_weighted"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "search_teleological");
}

#[tokio::test]
async fn test_search_teleological_with_scope() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "search_teleological",
        json!({
            "query_content": "test",
            "scope": "purpose_vector_only"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "search_teleological");
}

// -------------------------------------------------------------------------
// compute_teleological_vector
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_compute_teleological_vector_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "compute_teleological_vector",
        json!({
            "content": "Test content for teleological vector computation"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "compute_teleological_vector");

    let data = get_tool_data(&response);
    // Response contains nested structure: { "vector": { "purpose_vector": [...], ... }, ... }
    let vector = data.get("vector").expect("Must have vector field");
    assert!(
        vector.get("purpose_vector").is_some(),
        "Must have purpose_vector inside vector"
    );
}

#[tokio::test]
async fn test_compute_teleological_vector_with_tucker() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "compute_teleological_vector",
        json!({
            "content": "Test content",
            "compute_tucker": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "compute_teleological_vector");
}

#[tokio::test]
async fn test_compute_teleological_vector_missing_content() {
    let handlers = create_test_handlers();
    let request = make_tool_call("compute_teleological_vector", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "compute_teleological_vector");
}

// -------------------------------------------------------------------------
// fuse_embeddings
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_fuse_embeddings_basic() {
    let handlers = create_test_handlers();
    let memory_id = Uuid::new_v4().to_string();

    let request = make_tool_call(
        "fuse_embeddings",
        json!({
            "memory_id": memory_id
        }),
    );

    let response = handlers.dispatch(request).await;
    // May fail if memory doesn't exist
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_fuse_embeddings_with_method() {
    let handlers = create_test_handlers();
    let memory_id = Uuid::new_v4().to_string();

    let request = make_tool_call(
        "fuse_embeddings",
        json!({
            "memory_id": memory_id,
            "fusion_method": "attention"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_fuse_embeddings_missing_memory_id() {
    let handlers = create_test_handlers();
    let request = make_tool_call("fuse_embeddings", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "fuse_embeddings");
}

// -------------------------------------------------------------------------
// update_synergy_matrix
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_update_synergy_matrix_relevant() {
    let handlers = create_test_handlers();
    let query_id = Uuid::new_v4().to_string();
    let result_id = Uuid::new_v4().to_string();

    let request = make_tool_call(
        "update_synergy_matrix",
        json!({
            "query_vector_id": query_id,
            "result_vector_id": result_id,
            "feedback": "relevant"
        }),
    );

    let response = handlers.dispatch(request).await;
    // May fail if vectors don't exist
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_update_synergy_matrix_not_relevant() {
    let handlers = create_test_handlers();
    let query_id = Uuid::new_v4().to_string();
    let result_id = Uuid::new_v4().to_string();

    let request = make_tool_call(
        "update_synergy_matrix",
        json!({
            "query_vector_id": query_id,
            "result_vector_id": result_id,
            "feedback": "not_relevant"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_update_synergy_matrix_missing_params() {
    let handlers = create_test_handlers();
    let request = make_tool_call("update_synergy_matrix", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "update_synergy_matrix");
}

// -------------------------------------------------------------------------
// manage_teleological_profile
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_manage_teleological_profile_list() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "manage_teleological_profile",
        json!({
            "action": "list"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "manage_teleological_profile");
}

#[tokio::test]
async fn test_manage_teleological_profile_create() {
    let handlers = create_test_handlers();
    // Create action requires profile_id and weights [f32; 13]
    let request = make_tool_call(
        "manage_teleological_profile",
        json!({
            "action": "create",
            "profile_id": "test-profile-001",
            "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "manage_teleological_profile");
}

#[tokio::test]
async fn test_manage_teleological_profile_missing_action() {
    let handlers = create_test_handlers();
    let request = make_tool_call("manage_teleological_profile", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "manage_teleological_profile");
}
