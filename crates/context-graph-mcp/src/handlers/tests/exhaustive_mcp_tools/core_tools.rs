//! Core MCP tools tests (6 tools):
//! - inject_context
//! - store_memory
//! - get_memetic_status
//! - get_graph_manifest
//! - search_graph
//! - utl_status

use serde_json::json;

use super::helpers::{assert_success, assert_tool_error, get_tool_data, make_tool_call};
use super::synthetic_data;
use crate::handlers::tests::create_test_handlers;

// -------------------------------------------------------------------------
// inject_context
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_inject_context_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "content": synthetic_data::content::SIMPLE_TEXT,
            "rationale": "Testing basic injection"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "inject_context");

    let data = get_tool_data(&response);
    assert!(
        data.get("fingerprintId").is_some(),
        "Must return fingerprintId"
    );
    assert!(data.get("utl").is_some(), "Must return UTL metrics");
}

#[tokio::test]
async fn test_inject_context_with_all_params() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "content": synthetic_data::content::TECHNICAL_CODE,
            "rationale": "Testing code injection",
            "modality": "code",
            "importance": synthetic_data::importance::HIGH
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "inject_context");

    let data = get_tool_data(&response);
    assert!(data.get("fingerprintId").is_some());
}

#[tokio::test]
async fn test_inject_context_unicode() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "content": synthetic_data::content::UNICODE,
            "rationale": "Testing unicode handling"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "inject_context");
}

#[tokio::test]
async fn test_inject_context_missing_content() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "rationale": "Missing content field"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "inject_context");
}

#[tokio::test]
async fn test_inject_context_missing_rationale() {
    // NOTE: The handler currently treats rationale as optional (defaults to empty string).
    // This test verifies the current behavior - rationale is NOT required.
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "content": "Content without rationale"
        }),
    );

    let response = handlers.dispatch(request).await;
    // Handler accepts missing rationale - uses empty string default
    assert_success(&response, "inject_context");
}

#[tokio::test]
async fn test_inject_context_utl_values_in_range() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "content": "Test content for UTL verification",
            "rationale": "Verifying UTL values"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "inject_context");

    let data = get_tool_data(&response);
    let utl = data.get("utl").expect("Must have utl");

    let learning_score = utl["learningScore"].as_f64().unwrap_or(0.0);
    let entropy = utl["entropy"].as_f64().unwrap_or(0.0);
    let coherence = utl["coherence"].as_f64().unwrap_or(0.0);

    assert!(
        (0.0..=1.0).contains(&learning_score),
        "learningScore must be in [0,1]"
    );
    assert!((0.0..=1.0).contains(&entropy), "entropy must be in [0,1]");
    assert!(
        (0.0..=1.0).contains(&coherence),
        "coherence must be in [0,1]"
    );
}

// -------------------------------------------------------------------------
// store_memory
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_store_memory_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "store_memory",
        json!({
            "content": synthetic_data::content::SIMPLE_TEXT
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "store_memory");

    let data = get_tool_data(&response);
    assert!(
        data.get("fingerprintId").is_some(),
        "Must return fingerprintId"
    );
}

#[tokio::test]
async fn test_store_memory_with_importance() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "store_memory",
        json!({
            "content": synthetic_data::content::SIMPLE_TEXT,
            "importance": synthetic_data::importance::HIGH
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "store_memory");
}

#[tokio::test]
async fn test_store_memory_with_tags() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "store_memory",
        json!({
            "content": "Tagged content",
            "tags": ["test", "synthetic", "validation"]
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "store_memory");
}

#[tokio::test]
async fn test_store_memory_missing_content() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "store_memory",
        json!({
            "importance": 0.5
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "store_memory");
}

#[tokio::test]
async fn test_store_memory_boundary_importance_min() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "store_memory",
        json!({
            "content": "Minimum importance test",
            "importance": synthetic_data::importance::MIN
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "store_memory");
}

#[tokio::test]
async fn test_store_memory_boundary_importance_max() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "store_memory",
        json!({
            "content": "Maximum importance test",
            "importance": synthetic_data::importance::MAX
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "store_memory");
}

// -------------------------------------------------------------------------
// get_memetic_status (covered in tools_call.rs but adding edge cases)
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_memetic_status_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("get_memetic_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_memetic_status");

    let data = get_tool_data(&response);
    assert!(data.get("phase").is_some(), "Must have phase");
    assert!(
        data.get("fingerprintCount").is_some(),
        "Must have fingerprintCount"
    );
    assert!(data.get("utl").is_some(), "Must have utl");
    assert!(data.get("layers").is_some(), "Must have layers");
}

#[tokio::test]
async fn test_get_memetic_status_layers_structure() {
    let handlers = create_test_handlers();
    let request = make_tool_call("get_memetic_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_memetic_status");

    let data = get_tool_data(&response);
    let layers = data.get("layers").expect("Must have layers");

    // 5-layer bio-nervous architecture
    assert!(
        layers.is_array() || layers.is_object(),
        "layers must be array or object"
    );
}

// -------------------------------------------------------------------------
// get_graph_manifest
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_graph_manifest_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("get_graph_manifest", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_graph_manifest");

    let data = get_tool_data(&response);
    // Should describe 5-layer architecture
    assert!(
        data.get("layers").is_some() || data.get("architecture").is_some(),
        "Must describe architecture"
    );
}

// -------------------------------------------------------------------------
// search_graph
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_search_graph_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "search_graph",
        json!({
            "query": "test search query"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "search_graph");

    let data = get_tool_data(&response);
    assert!(data.get("results").is_some(), "Must have results");
    assert!(data.get("count").is_some(), "Must have count");
}

#[tokio::test]
async fn test_search_graph_with_top_k() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "search_graph",
        json!({
            "query": "test",
            "topK": 5
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "search_graph");
}

#[tokio::test]
async fn test_search_graph_with_min_similarity() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "search_graph",
        json!({
            "query": "test",
            "minSimilarity": 0.5
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "search_graph");
}

#[tokio::test]
async fn test_search_graph_missing_query() {
    let handlers = create_test_handlers();
    let request = make_tool_call("search_graph", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "search_graph");
}

#[tokio::test]
async fn test_search_graph_boundary_top_k_max() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "search_graph",
        json!({
            "query": "test",
            "topK": 100
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "search_graph");
}

// -------------------------------------------------------------------------
// utl_status
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_utl_status_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("utl_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "utl_status");

    let data = get_tool_data(&response);
    assert!(
        data.get("lifecycle_phase").is_some(),
        "Must have lifecycle_phase"
    );
    assert!(
        data.get("interaction_count").is_some(),
        "Must have interaction_count"
    );
    assert!(data.get("entropy").is_some(), "Must have entropy");
    assert!(data.get("coherence").is_some(), "Must have coherence");
    assert!(
        data.get("learning_score").is_some(),
        "Must have learning_score"
    );
}

#[tokio::test]
async fn test_utl_status_consolidation_phase_valid() {
    let handlers = create_test_handlers();
    let request = make_tool_call("utl_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "utl_status");

    let data = get_tool_data(&response);
    let phase = data["consolidation_phase"]
        .as_str()
        .expect("consolidation_phase must be string");

    assert!(
        synthetic_data::consolidation::VALID_PHASES.contains(&phase),
        "Consolidation phase '{}' must be one of {:?}",
        phase,
        synthetic_data::consolidation::VALID_PHASES
    );
}

#[tokio::test]
async fn test_utl_status_thresholds_present() {
    let handlers = create_test_handlers();
    let request = make_tool_call("utl_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "utl_status");

    let data = get_tool_data(&response);
    let thresholds = data.get("thresholds").expect("Must have thresholds");

    assert!(thresholds.get("entropy_trigger").is_some());
    assert!(thresholds.get("coherence_trigger").is_some());
    assert!(thresholds.get("min_importance_store").is_some());
    assert!(thresholds.get("consolidation_threshold").is_some());
}
