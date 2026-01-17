//! Edge Case Tests
//!
//! Tests for error handling, boundary conditions, and edge cases.

use super::infrastructure::*;
use context_graph_core::traits::TeleologicalMemoryStore;

/// EDGE CASE 1: Empty content should fail with INVALID_PARAMS.
#[tokio::test]
async fn test_edge_case_empty_content() {
    let ctx = TestContext::new();
    let before_count = ctx.store.count().await.unwrap();

    let request = make_request("memory/store", 1, json!({"content": "", "importance": 0.5}));
    let response = ctx.handlers.dispatch(request).await;

    assert!(response.error.is_some(), "Empty content MUST return error");
    assert_eq!(response.error.unwrap().code, error_codes::INVALID_PARAMS);
    assert_eq!(ctx.store.count().await.unwrap(), before_count);
}

/// EDGE CASE 2: Invalid UUID format should fail.
#[tokio::test]
async fn test_edge_case_invalid_uuid() {
    let ctx = TestContext::new();
    let request = make_request("memory/retrieve", 1, json!({"fingerprintId": "not-valid"}));
    let response = ctx.handlers.dispatch(request).await;

    assert!(response.error.is_some(), "Invalid UUID MUST return error");
    assert_eq!(response.error.unwrap().code, error_codes::INVALID_PARAMS);
}

/// EDGE CASE 3: 12-element weight array (must be 13).
#[tokio::test]
async fn test_edge_case_12_weights_instead_of_13() {
    let ctx = TestContext::new();
    let invalid_weights: Vec<f64> = vec![0.083; 12];

    let request = make_request(
        "search/multi",
        1,
        json!({
            "query": "test", "query_type": "custom", "weights": invalid_weights, "minSimilarity": 0.0
        }),
    );
    let response = ctx.handlers.dispatch(request).await;

    assert!(response.error.is_some(), "12-element weights MUST fail");
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::INVALID_PARAMS);
    assert!(error.message.contains("13") || error.message.contains("weight"));
}

/// EDGE CASE 4: space_index 13 (valid range is 0-12).
#[tokio::test]
async fn test_edge_case_space_index_13() {
    let ctx = TestContext::new();
    let request = make_request(
        "search/single_space",
        1,
        json!({
            "query": "test", "space_index": 13, "minSimilarity": 0.0
        }),
    );
    let response = ctx.handlers.dispatch(request).await;

    assert!(response.error.is_some(), "space_index=13 MUST fail");
    assert_eq!(response.error.unwrap().code, error_codes::INVALID_PARAMS);
}

/// EDGE CASE 5: Store & Alignment work autonomously without North Star.
#[tokio::test]
async fn test_edge_case_alignment_autonomous_operation() {
    let ctx = TestContext::new_without_north_star();
    assert!(
        !ctx.hierarchy.read().has_top_level_goals(),
        "MUST NOT have North Star"
    );

    // Store should succeed without North Star (AUTONOMOUS OPERATION)
    let store_request = make_request(
        "memory/store",
        1,
        json!({
            "content": "Test content for autonomous operation", "importance": 0.5
        }),
    );
    let store_response = ctx.handlers.dispatch(store_request).await;
    assert!(
        store_response.error.is_none(),
        "Store MUST succeed without North Star"
    );
    assert!(store_response
        .result
        .expect("Should have result")
        .get("fingerprintId")
        .is_some());

    // Deprecated method returns METHOD_NOT_FOUND
    let align_request = make_request(
        "purpose/north_star_alignment",
        2,
        json!({
            "fingerprint_id": "00000000-0000-0000-0000-000000000001"
        }),
    );
    let response = ctx.handlers.dispatch(align_request).await;
    assert!(
        response.error.is_some(),
        "Deprecated method must return error"
    );
    assert_eq!(
        response.error.unwrap().code,
        -32601,
        "Must return METHOD_NOT_FOUND"
    );
}

/// EDGE CASE 6: Fingerprint not found.
#[tokio::test]
async fn test_edge_case_fingerprint_not_found() {
    let ctx = TestContext::new();
    let request = make_request(
        "memory/retrieve",
        1,
        json!({
            "fingerprintId": Uuid::new_v4().to_string()
        }),
    );
    let response = ctx.handlers.dispatch(request).await;

    assert!(
        response.error.is_some(),
        "Non-existent fingerprint MUST fail"
    );
    assert_eq!(
        response.error.unwrap().code,
        error_codes::FINGERPRINT_NOT_FOUND
    );
}

/// EDGE CASE 7: Invalid Johari embedder index 13.
#[tokio::test]
async fn test_edge_case_johari_embedder_index_13() {
    let ctx = TestContext::new();
    let fp = create_fingerprint_with_johari([JohariQuadrant::Unknown; NUM_EMBEDDERS]);
    let memory_id = ctx.store.store(fp).await.expect("Store should succeed");

    let request = make_request(
        "johari/transition",
        1,
        json!({
            "memory_id": memory_id.to_string(),
            "embedder_index": 13, "to_quadrant": "open", "trigger": "dream_consolidation"
        }),
    );
    let response = ctx.handlers.dispatch(request).await;

    assert!(response.error.is_some(), "embedder_index=13 MUST fail");
    assert_eq!(
        response.error.unwrap().code,
        error_codes::JOHARI_INVALID_EMBEDDER_INDEX
    );
}

/// EDGE CASE 8: Meta-UTL insufficient training data.
#[tokio::test]
async fn test_edge_case_meta_utl_insufficient_data() {
    let ctx = TestContext::new();
    assert_eq!(ctx.meta_utl_tracker.read().validation_count, 0);

    let request = make_request_no_params("meta_utl/optimized_weights", 1);
    let response = ctx.handlers.dispatch(request).await;

    assert!(response.error.is_some(), "0 validations MUST fail");
    assert_eq!(
        response.error.unwrap().code,
        error_codes::META_UTL_INSUFFICIENT_DATA
    );
}

/// EDGE CASE 9: Validate non-existent prediction.
#[tokio::test]
async fn test_edge_case_validate_unknown_prediction() {
    let ctx = TestContext::new();
    let request = make_request(
        "meta_utl/validate_prediction",
        1,
        json!({
            "prediction_id": Uuid::new_v4().to_string(),
            "actual_outcome": {"coherence_delta": 0.02, "alignment_delta": 0.05}
        }),
    );
    let response = ctx.handlers.dispatch(request).await;

    assert!(response.error.is_some(), "Unknown prediction MUST fail");
    assert_eq!(
        response.error.unwrap().code,
        error_codes::META_UTL_PREDICTION_NOT_FOUND
    );
}

/// EDGE CASE 10: Goal not found in hierarchy.
#[tokio::test]
async fn test_edge_case_goal_not_found() {
    let ctx = TestContext::new();
    let request = make_request(
        "goal/hierarchy_query",
        1,
        json!({
            "operation": "get_goal", "goal_id": "00000000-0000-0000-0000-000000000000"
        }),
    );
    let response = ctx.handlers.dispatch(request).await;

    assert!(response.error.is_some(), "Non-existent goal MUST fail");
    assert_eq!(response.error.unwrap().code, error_codes::GOAL_NOT_FOUND);
}
