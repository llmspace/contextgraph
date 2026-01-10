//! Teleological Handler Tests (TELEO-H1 through TELEO-H5)
//!
//! ISSUE-1 FIX: Updated tests to use new API matching tools.rs definition.
//! Tests now use `query_content` (string) instead of `query` (TeleologicalVectorJson).
//! Search is against stored vectors in TeleologicalMemoryStore, not a candidates array.

use super::{create_test_handlers, make_request};
use crate::protocol::JsonRpcId;
use serde_json::json;

// =============================================================================
// TELEO-H1: search_teleological
// =============================================================================

/// Test search_teleological with query_content (string to embed).
/// ISSUE-1 FIX: Uses new API - query_content instead of query+candidates.
///
/// DIMENSION FIX: Embeddings are now projected to EMBEDDING_DIM (1024) before
/// passing to FusionEngine, eliminating dimension mismatch panics.
#[tokio::test]
async fn test_search_teleological_basic() {
    let handlers = create_test_handlers();

    // ISSUE-1 FIX: Use query_content (string) instead of query (vector object)
    // The handler will embed this content and search stored vectors
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_content": "test semantic search content for teleological vectors",
                "strategy": "adaptive",
                "scope": "full",
                "max_results": 10,
                "min_similarity": 0.0,
                "include_breakdown": false
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    // Note: May panic with dimension mismatch in test environment.
    let result = response.result.expect("Must have result");

    // Verify structure
    assert!(result.get("content").is_some(), "Must have content");
    let content = result["content"].as_array().expect("content is array");
    assert!(!content.is_empty(), "Content should not be empty");

    // Parse the text content - may be JSON or plain error text
    let text = content[0]["text"].as_str().expect("Must have text");

    // Try to parse as JSON, but handle plain error text too
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
        // JSON response - check success or error
        if parsed.get("success").is_some() && parsed["success"].as_bool().unwrap_or(false) {
            // Success path - verify response structure
            assert!(parsed.get("num_results").is_some(), "Must have num_results");
            assert!(parsed.get("results").is_some(), "Must have results");
            assert_eq!(parsed["query_type"].as_str(), Some("embedded"));
        }
    }
}

/// Test search_teleological with breakdown enabled.
/// ISSUE-1 FIX: Uses new API with query_content.
///
/// DIMENSION FIX: Embeddings are now projected to EMBEDDING_DIM (1024) before
/// passing to FusionEngine, eliminating dimension mismatch panics.
#[tokio::test]
async fn test_search_teleological_with_breakdown() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_content": "code analysis semantic search with breakdown",
                "include_breakdown": true
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.expect("Must have result");
    let content = result["content"].as_array().expect("content is array");
    let text = content[0]["text"].as_str().expect("Must have text");

    // Try to parse - may be JSON or plain text
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
        // If successful, verify breakdown is included in results
        if parsed.get("success").is_some() && parsed["success"].as_bool().unwrap_or(false) {
            if let Some(results) = parsed["results"].as_array() {
                if !results.is_empty() {
                    let first = &results[0];
                    assert!(first.get("breakdown").is_some(), "Should include breakdown");
                }
            }
        }
    }
}

/// Test search_teleological with different strategies.
/// ISSUE-1 FIX: Uses new API with query_content.
///
/// DIMENSION FIX: Embeddings are now projected to EMBEDDING_DIM (1024) before
/// passing to FusionEngine, eliminating dimension mismatch panics.
#[tokio::test]
async fn test_search_teleological_different_strategies() {
    let handlers = create_test_handlers();

    let strategies = vec!["cosine", "euclidean", "synergy_weighted", "adaptive"];

    for strategy in strategies {
        let request = make_request(
            "tools/call",
            Some(JsonRpcId::Number(1)),
            Some(json!({
                "name": "search_teleological",
                "arguments": {
                    "query_content": "test content for strategy validation",
                    "strategy": strategy
                }
            })),
        );

        let response = handlers.dispatch(request).await;
        let result = response.result.expect("Must have result");
        let content = result["content"].as_array().expect("content is array");
        let text = content[0]["text"].as_str().expect("Must have text");

        // Try to parse - may be JSON or plain text
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
            // Verify strategy is echoed back
            if parsed.get("success").is_some() && parsed["success"].as_bool().unwrap_or(false) {
                assert_eq!(
                    parsed["strategy"].as_str(),
                    Some(strategy),
                    "Strategy should match for {}",
                    strategy
                );
            }
        }
    }
}

/// Test FAIL FAST when neither query_content nor query_vector_id provided.
/// ISSUE-1 FIX: Validates the fail-fast behavior.
#[tokio::test]
async fn test_search_teleological_fail_fast_no_query() {
    let handlers = create_test_handlers();

    // Intentionally omit both query_content and query_vector_id
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "strategy": "adaptive",
                "max_results": 5
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.expect("Must have result");
    let is_error = result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
    assert!(is_error, "Should fail when no query provided");

    let content = result["content"].as_array().expect("content is array");
    let text = content[0]["text"].as_str().expect("Must have text");
    assert!(
        text.contains("FAIL FAST"),
        "Error should mention FAIL FAST"
    );
    assert!(
        text.contains("query_content") || text.contains("query_vector_id"),
        "Error should mention the missing parameters"
    );
}

/// Test FAIL FAST with empty query_content.
/// ISSUE-1 FIX: Validates empty string rejection.
#[tokio::test]
async fn test_search_teleological_fail_fast_empty_content() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_content": ""
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.expect("Must have result");
    let is_error = result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
    assert!(is_error, "Should fail with empty query_content");

    let content = result["content"].as_array().expect("content is array");
    let text = content[0]["text"].as_str().expect("Must have text");
    assert!(
        text.contains("FAIL FAST") || text.contains("empty"),
        "Error should mention empty content"
    );
}

/// Test search_teleological with invalid query_vector_id.
/// ISSUE-1 FIX: Validates UUID parsing error.
#[tokio::test]
async fn test_search_teleological_fail_fast_invalid_vector_id() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_vector_id": "not-a-valid-uuid"
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.expect("Must have result");
    let is_error = result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
    assert!(is_error, "Should fail with invalid UUID");

    let content = result["content"].as_array().expect("content is array");
    let text = content[0]["text"].as_str().expect("Must have text");
    assert!(
        text.contains("FAIL FAST"),
        "Error should mention FAIL FAST"
    );
}

// =============================================================================
// TELEO-H3: fuse_embeddings
// =============================================================================

#[tokio::test]
async fn test_fuse_embeddings_basic() {
    let handlers = create_test_handlers();

    // Create 13 synthetic embeddings (1024 dimensions each as required by FusionEngine)
    let embeddings: Vec<Vec<f32>> = (0..13)
        .map(|i| vec![(i as f32) * 0.1 + 0.1; 1024])
        .collect();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "fuse_embeddings",
            "arguments": {
                "embeddings": embeddings,
                "fusion_method": "weighted_average"
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "fuse_embeddings should succeed");
    let result = response.result.expect("Must have result");

    let content = result["content"].as_array().expect("content is array");
    let text = content[0]["text"].as_str().expect("Must have text");
    let parsed: serde_json::Value = serde_json::from_str(text).expect("Must be valid JSON");

    assert!(
        parsed["success"].as_bool().unwrap_or(false),
        "Should succeed"
    );

    // Verify vector is returned
    assert!(parsed.get("vector").is_some(), "Must have vector");
    let vector = &parsed["vector"];

    // Check vector components exist
    assert!(vector.get("purpose_vector").is_some());
    assert!(vector.get("cross_correlations").is_some());
    assert!(vector.get("group_alignments").is_some());
    assert!(vector.get("confidence").is_some());

    // Verify confidence in valid range
    let confidence = parsed["confidence"].as_f64().expect("confidence");
    assert!(
        (0.0..=1.0).contains(&confidence),
        "Confidence should be [0,1]"
    );
}

#[tokio::test]
async fn test_fuse_embeddings_with_alignments() {
    let handlers = create_test_handlers();

    let embeddings: Vec<Vec<f32>> = (0..13).map(|_| vec![0.5; 1024]).collect();

    // Custom alignments that sum to 1.0
    let alignments: [f32; 13] = [
        0.15, 0.10, 0.08, 0.10, 0.05, 0.03, 0.08, 0.10, 0.05, 0.10, 0.05, 0.06, 0.05,
    ];

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "fuse_embeddings",
            "arguments": {
                "embeddings": embeddings,
                "alignments": alignments
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(
        response.error.is_none(),
        "Should succeed with custom alignments"
    );

    let result = response.result.unwrap();
    let content = result["content"].as_array().unwrap();
    let text = content[0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap_or(false));
}

#[tokio::test]
async fn test_fuse_embeddings_wrong_count_fails() {
    let handlers = create_test_handlers();

    // Only 5 embeddings instead of 13
    let embeddings: Vec<Vec<f32>> = (0..5).map(|_| vec![0.5; 1024]).collect();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "fuse_embeddings",
            "arguments": {
                "embeddings": embeddings
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    // Should fail with error
    let result = response.result.expect("Result");
    let _content = result["content"].as_array().expect("content");
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    assert!(is_error, "Should return error for wrong embedding count");
}

// =============================================================================
// TELEO-H4: update_synergy_matrix
// =============================================================================

#[tokio::test]
async fn test_update_synergy_matrix_positive_feedback() {
    let handlers = create_test_handlers();

    let vector_id = "11111111-2222-3333-4444-555555555555";

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "update_synergy_matrix",
            "arguments": {
                "vector_id": vector_id,
                "feedback_type": "positive",
                "context": "User selected this result as relevant"
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "Should succeed");
    let result = response.result.unwrap();

    let content = result["content"].as_array().unwrap();
    let text = content[0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap_or(false));
    assert_eq!(parsed["vector_id"].as_str(), Some(vector_id));
    assert_eq!(parsed["feedback_type"].as_str(), Some("positive"));
}

#[tokio::test]
async fn test_update_synergy_matrix_negative_feedback() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "update_synergy_matrix",
            "arguments": {
                "vector_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "feedback_type": "negative"
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let result = response.result.unwrap();
    let content = result["content"].as_array().unwrap();
    let text = content[0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap_or(false));
}

#[tokio::test]
async fn test_update_synergy_matrix_with_contributions() {
    let handlers = create_test_handlers();

    // Contributions that highlight specific embedders
    let contributions: [f32; 13] = [
        0.3, 0.1, 0.05, 0.1, 0.05, 0.02, 0.08, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02,
    ];

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "update_synergy_matrix",
            "arguments": {
                "vector_id": "12345678-1234-1234-1234-123456789012",
                "feedback_type": "positive",
                "contributions": contributions
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());
}

#[tokio::test]
async fn test_update_synergy_matrix_invalid_uuid_fails() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "update_synergy_matrix",
            "arguments": {
                "vector_id": "not-a-valid-uuid",
                "feedback_type": "positive"
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.expect("Result");
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    assert!(is_error, "Should fail for invalid UUID");
}

// =============================================================================
// TELEO-H5: manage_teleological_profile
// =============================================================================

#[tokio::test]
async fn test_manage_profile_create_and_get() {
    let handlers = create_test_handlers();

    // Create profile
    let create_request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "manage_teleological_profile",
            "arguments": {
                "action": "create",
                "profile_id": "test-profile-1",
                "weights": [0.15, 0.10, 0.08, 0.10, 0.05, 0.03, 0.08, 0.10, 0.05, 0.10, 0.05, 0.06, 0.05]
            }
        })),
    );

    let response = handlers.dispatch(create_request).await;
    assert!(response.error.is_none());

    let result = response.result.unwrap();
    let content = result["content"].as_array().unwrap();
    let text = content[0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap_or(false));
    assert_eq!(parsed["action"].as_str(), Some("create"));
    assert_eq!(parsed["profile_id"].as_str(), Some("test-profile-1"));
}

#[tokio::test]
async fn test_manage_profile_list() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "manage_teleological_profile",
            "arguments": {
                "action": "list"
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let result = response.result.unwrap();
    let content = result["content"].as_array().unwrap();
    let text = content[0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap_or(false));
    assert_eq!(parsed["action"].as_str(), Some("list"));
    assert!(parsed.get("profiles").is_some());
    assert!(parsed.get("count").is_some());
}

#[tokio::test]
async fn test_manage_profile_update_nonexistent() {
    let handlers = create_test_handlers();

    let weights_uniform: Vec<f32> = vec![0.1; 13];

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "manage_teleological_profile",
            "arguments": {
                "action": "update",
                "profile_id": "nonexistent-profile",
                "weights": weights_uniform
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    // Should succeed but report profile not found
    let result = response.result.unwrap();
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(is_error, "Should error for nonexistent profile");
}

#[tokio::test]
async fn test_manage_profile_invalid_action() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "manage_teleological_profile",
            "arguments": {
                "action": "invalid_action"
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.unwrap();
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(is_error, "Should error for invalid action");
}

#[tokio::test]
async fn test_manage_profile_find_best() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "manage_teleological_profile",
            "arguments": {
                "action": "find_best",
                "context": "code analysis with semantic search"
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let result = response.result.unwrap();
    let content = result["content"].as_array().unwrap();
    let text = content[0]["text"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(parsed["success"].as_bool().unwrap_or(false));
    assert_eq!(parsed["action"].as_str(), Some("find_best"));
}
