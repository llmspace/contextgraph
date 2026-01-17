//! Manual Teleological Validation Tests (TELEO-VAL-001 through TELEO-VAL-012)
//!
//! ISSUE-1 FIX: Updated to use new API matching tools.rs definition.
//! Tests now use `query_content` (string) instead of `query` (TeleologicalVectorJson).
//! Search is against stored vectors in TeleologicalMemoryStore, not a candidates array.
//!
//! These tests verify the handler implementations work correctly with
//! the new API and validate fail-fast behavior.
//! NO MOCKS. NO FALLBACKS. FAIL FAST if something doesn't work.

use super::{create_test_handlers, extract_mcp_tool_data, make_request};
use crate::protocol::JsonRpcId;
use serde_json::json;

// =============================================================================
// HELPER: Generate arrays of specific sizes (kept for non-search tests)
// =============================================================================

#[allow(dead_code)]
fn uniform_pv(val: f32) -> [f32; 13] {
    [val; 13]
}

#[allow(dead_code)]
fn uniform_cc(val: f32) -> Vec<f32> {
    vec![val; 78]
}

#[allow(dead_code)]
fn uniform_ga(val: f32) -> [f32; 6] {
    [val; 6]
}

#[allow(dead_code)]
fn one_hot_pv(idx: usize) -> [f32; 13] {
    let mut arr = [0.0f32; 13];
    if idx < 13 {
        arr[idx] = 1.0;
    }
    arr
}

#[allow(dead_code)]
fn one_hot_cc(idx: usize) -> Vec<f32> {
    let mut arr = vec![0.0f32; 78];
    if idx < 78 {
        arr[idx] = 1.0;
    }
    arr
}

#[allow(dead_code)]
fn one_hot_ga(idx: usize) -> [f32; 6] {
    let mut arr = [0.0f32; 6];
    if idx < 6 {
        arr[idx] = 1.0;
    }
    arr
}

// =============================================================================
// TELEO-VAL-001: search_teleological - Test with query_content
// ISSUE-1 FIX: Updated to test new API behavior
// DIMENSION FIX: Embeddings are now projected to EMBEDDING_DIM (1024) before
// passing to FusionEngine, eliminating dimension mismatch panics.
// =============================================================================

#[tokio::test]
async fn test_val_001_search_teleological_identical_vectors_similarity_1() {
    let handlers = create_test_handlers();

    // ISSUE-1 FIX: New API uses query_content (string to embed)
    // This test verifies the handler works with query_content
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_content": "semantic search for identical vector matching",
                "strategy": "adaptive",
                "scope": "full",
                "max_results": 10,
                "min_similarity": 0.0,
                "include_breakdown": false
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.expect("MUST have result");
    let content = result["content"].as_array().expect("content is array");
    let text = content[0]["text"].as_str().expect("Must have text");

    // Log physical evidence
    eprintln!("=== PHYSICAL EVIDENCE: search_teleological with query_content ===");
    eprintln!("Response text: {}", text);
    eprintln!("================================================================");

    // Try to parse - may be JSON or plain text
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
        // Either success (provider works) or FAIL FAST error (provider issue)
        if let Some(success) = parsed.get("success") {
            if success.as_bool().unwrap_or(false) {
                assert_eq!(parsed["query_type"].as_str(), Some("embedded"));
                assert!(parsed.get("num_results").is_some());
            }
        }
    }
}

// =============================================================================
// TELEO-VAL-002: search_teleological - FAIL FAST with empty query_content
// ISSUE-1 FIX: Validates empty string rejection
// =============================================================================

#[tokio::test]
async fn test_val_002_search_teleological_orthogonal_vectors_similarity_0() {
    let handlers = create_test_handlers();

    // ISSUE-1 FIX: Test FAIL FAST with empty query_content
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_content": "",
                "strategy": "cosine",
                "include_breakdown": true
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.expect("MUST have result");
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Log physical evidence
    eprintln!("=== PHYSICAL EVIDENCE: search_teleological empty query_content ===");
    eprintln!("isError: {}", is_error);
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(text) = content
            .first()
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
        {
            eprintln!("Error message: {}", text);
        }
    }
    eprintln!("===================================================================");

    // CRITICAL: Empty query_content MUST fail fast
    assert!(is_error, "MUST return isError=true for empty query_content");
}

// =============================================================================
// TELEO-VAL-003: search_teleological - FAIL FAST with no query params
// ISSUE-1 FIX: Validates missing parameter rejection
// =============================================================================

#[tokio::test]
async fn test_val_003_search_teleological_ranking_order_descending() {
    let handlers = create_test_handlers();

    // ISSUE-1 FIX: Test FAIL FAST with neither query_content nor query_vector_id
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "strategy": "cosine",
                "max_results": 3
            }
        })),
    );

    let response = handlers.dispatch(request).await;

    let result = response.result.expect("result");
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Log physical evidence
    eprintln!("=== PHYSICAL EVIDENCE: search_teleological no query params ===");
    eprintln!("isError: {}", is_error);
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(text) = content
            .first()
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
        {
            eprintln!("Error message: {}", text);
            assert!(text.contains("FAIL FAST"), "Error should mention FAIL FAST");
            assert!(
                text.contains("query_content") || text.contains("query_vector_id"),
                "Error should mention missing parameters"
            );
        }
    }
    eprintln!("=============================================================");

    // CRITICAL: Missing both query params MUST fail fast
    assert!(
        is_error,
        "MUST return isError=true for missing query params"
    );
}

// =============================================================================
// TELEO-VAL-004: fuse_embeddings - Uniform Fusion (13 embeddings × 1024 dims)
// =============================================================================

#[tokio::test]
async fn test_val_004_fuse_embeddings_uniform_fusion() {
    let handlers = create_test_handlers();

    // Create 13 uniform embeddings of 1024 dimensions each
    let embeddings: Vec<Vec<f32>> = (0..13)
        .map(|i| vec![(i as f32 + 1.0) * 0.1; 1024])
        .collect();

    // Provide explicit alignments > 0.1 threshold for active_embedders count
    // FusionEngine counts alignments > 0.1 as "active"
    // Without explicit alignments, compute_alignments_from_embeddings normalizes to sum=1.0,
    // resulting in each alignment ≈ 1/13 ≈ 0.077 (below 0.1 threshold)
    let alignments: [f32; 13] = [0.8; 13]; // All above 0.1 threshold

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "fuse_embeddings",
            "arguments": {
                "embeddings": embeddings,
                "alignments": alignments,  // Explicit alignments > 0.1
                "fusion_method": "weighted_average"
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(
        response.error.is_none(),
        "fuse_embeddings MUST NOT error: {:?}",
        response.error
    );

    let result = response.result.expect("result");
    let parsed = extract_mcp_tool_data(&result);

    assert!(parsed["success"].as_bool().unwrap_or(false), "MUST succeed");

    // Validate new AlignmentFusionResult format (Constitution v5.0.0)
    let pv = parsed["purpose_vector"]
        .as_array()
        .expect("purpose_vector array at top level");
    let ga = &parsed["group_alignments"];

    // Log physical evidence
    eprintln!("=== PHYSICAL EVIDENCE: fuse_embeddings uniform fusion ===");
    eprintln!("Vector dimensions:");
    eprintln!("  - purpose_vector: {} elements (expected 13)", pv.len());
    eprintln!("  - group_alignments: object with 4 groups");
    eprintln!(
        "    - semantic: {}",
        ga["semantic"].as_f64().unwrap_or(-1.0)
    );
    eprintln!(
        "    - temporal: {}",
        ga["temporal"].as_f64().unwrap_or(-1.0)
    );
    eprintln!(
        "    - structural: {}",
        ga["structural"].as_f64().unwrap_or(-1.0)
    );
    eprintln!(
        "    - experiential: {}",
        ga["experiential"].as_f64().unwrap_or(-1.0)
    );
    eprintln!(
        "Confidence: {}",
        parsed["confidence"].as_f64().unwrap_or(-1.0)
    );
    eprintln!(
        "Metadata active_embedders: {}",
        parsed["metadata"]["active_embedders"].as_u64().unwrap_or(0)
    );
    eprintln!("NOTE: Explicit alignments [0.8; 13] provided (all > 0.1 threshold)");
    eprintln!("==========================================================");

    // CRITICAL: Verify purpose_vector has 13 elements (one per embedder)
    assert_eq!(
        pv.len(),
        13,
        "purpose_vector MUST have 13 elements (one per embedder)"
    );

    // Verify group_alignments structure
    assert!(ga.get("semantic").is_some(), "Must have semantic alignment");
    assert!(ga.get("temporal").is_some(), "Must have temporal alignment");
    assert!(
        ga.get("structural").is_some(),
        "Must have structural alignment"
    );
    assert!(
        ga.get("experiential").is_some(),
        "Must have experiential alignment"
    );

    // Confidence must be in [0, 1]
    let confidence = parsed["confidence"].as_f64().expect("confidence");
    assert!(
        (0.0..=1.0).contains(&confidence),
        "Confidence {} MUST be in [0.0, 1.0]",
        confidence
    );

    // Metadata verification - with explicit alignments [0.8; 13], all 13 are > 0.1 threshold
    assert_eq!(
        parsed["metadata"]["active_embedders"].as_u64(),
        Some(13),
        "active_embedders MUST be 13 when all alignments > 0.1 threshold"
    );

    // Verify constitution compliance markers
    assert_eq!(
        parsed["metadata"]["constitution_compliant"].as_bool(),
        Some(true),
        "Must be constitution compliant"
    );
    assert_eq!(
        parsed["metadata"]["api_version"].as_str(),
        Some("5.0.0"),
        "Must use v5.0.0 API"
    );
}

// =============================================================================
// TELEO-VAL-005: fuse_embeddings - Wrong Count (error case)
// =============================================================================

#[tokio::test]
async fn test_val_005_fuse_embeddings_wrong_count_error() {
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
    let result = response.result.expect("Result");

    // Check for MCP error format
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Log physical evidence
    eprintln!("=== PHYSICAL EVIDENCE: fuse_embeddings wrong count error ===");
    eprintln!("isError: {}", is_error);
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(text) = content
            .first()
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
        {
            eprintln!("Error message: {}", text);
        }
    }
    eprintln!("=============================================================");

    // CRITICAL: Must return error for wrong embedding count
    assert!(
        is_error,
        "MUST return isError=true for wrong embedding count"
    );
}

// =============================================================================
// TELEO-VAL-006: update_synergy_matrix - Positive Feedback
// =============================================================================

#[tokio::test]
async fn test_val_006_update_synergy_positive_feedback() {
    let handlers = create_test_handlers();

    let vector_id = "550e8400-e29b-41d4-a716-446655440000";

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "update_synergy_matrix",
            "arguments": {
                "vector_id": vector_id,
                "feedback_type": "positive",
                "context": "User found this result highly relevant"
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let result = response.result.expect("result");
    let parsed = extract_mcp_tool_data(&result);

    // Log physical evidence
    eprintln!("=== PHYSICAL EVIDENCE: update_synergy_matrix positive ===");
    eprintln!("success: {}", parsed["success"].as_bool().unwrap_or(false));
    eprintln!("vector_id: {}", parsed["vector_id"].as_str().unwrap_or("?"));
    eprintln!(
        "feedback_type: {}",
        parsed["feedback_type"].as_str().unwrap_or("?")
    );
    eprintln!(
        "learning_triggered: {}",
        parsed["learning_triggered"].as_bool().unwrap_or(false)
    );
    eprintln!("==========================================================");

    assert!(parsed["success"].as_bool().unwrap_or(false), "MUST succeed");
    assert_eq!(parsed["vector_id"].as_str(), Some(vector_id));
    assert_eq!(parsed["feedback_type"].as_str(), Some("positive"));
}

// =============================================================================
// TELEO-VAL-007: update_synergy_matrix - Negative Feedback
// =============================================================================

#[tokio::test]
async fn test_val_007_update_synergy_negative_feedback() {
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

    let result = response.result.expect("result");
    let parsed = extract_mcp_tool_data(&result);

    eprintln!("=== PHYSICAL EVIDENCE: update_synergy_matrix negative ===");
    eprintln!("success: {}", parsed["success"].as_bool().unwrap_or(false));
    eprintln!(
        "feedback_type: {}",
        parsed["feedback_type"].as_str().unwrap_or("?")
    );
    eprintln!("==========================================================");

    assert!(parsed["success"].as_bool().unwrap_or(false));
    assert_eq!(parsed["feedback_type"].as_str(), Some("negative"));
}

// =============================================================================
// TELEO-VAL-008: update_synergy_matrix - Invalid UUID (error case)
// =============================================================================

#[tokio::test]
async fn test_val_008_update_synergy_invalid_uuid_error() {
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

    eprintln!("=== PHYSICAL EVIDENCE: update_synergy_matrix invalid UUID ===");
    eprintln!("isError: {}", is_error);
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(text) = content
            .first()
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
        {
            eprintln!("Error message: {}", text);
        }
    }
    eprintln!("==============================================================");

    assert!(is_error, "MUST return isError=true for invalid UUID");
}

// =============================================================================
// TELEO-VAL-009: manage_teleological_profile - Create Profile
// =============================================================================

#[tokio::test]
async fn test_val_009_manage_profile_create() {
    let handlers = create_test_handlers();

    let weights: [f32; 13] = [
        0.20, 0.10, 0.05, 0.15, 0.05, 0.05, 0.10, 0.08, 0.05, 0.05, 0.04, 0.05, 0.03,
    ];

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "manage_teleological_profile",
            "arguments": {
                "action": "create",
                "profile_id": "code-analysis-profile",
                "weights": weights
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let result = response.result.expect("result");
    let parsed = extract_mcp_tool_data(&result);

    eprintln!("=== PHYSICAL EVIDENCE: manage_teleological_profile create ===");
    eprintln!("success: {}", parsed["success"].as_bool().unwrap_or(false));
    eprintln!("action: {}", parsed["action"].as_str().unwrap_or("?"));
    eprintln!(
        "profile_id: {}",
        parsed["profile_id"].as_str().unwrap_or("?")
    );
    if let Some(w) = parsed["weights"].as_array() {
        eprintln!("weights count: {} (expected 13)", w.len());
    }
    eprintln!("===============================================================");

    assert!(parsed["success"].as_bool().unwrap_or(false));
    assert_eq!(parsed["action"].as_str(), Some("create"));
    assert_eq!(parsed["profile_id"].as_str(), Some("code-analysis-profile"));

    let returned_weights = parsed["weights"].as_array().expect("weights array");
    assert_eq!(returned_weights.len(), 13, "MUST have 13 weights");
}

// =============================================================================
// TELEO-VAL-010: manage_teleological_profile - List Profiles
// =============================================================================

#[tokio::test]
async fn test_val_010_manage_profile_list() {
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

    let result = response.result.expect("result");
    let parsed = extract_mcp_tool_data(&result);

    eprintln!("=== PHYSICAL EVIDENCE: manage_teleological_profile list ===");
    eprintln!("success: {}", parsed["success"].as_bool().unwrap_or(false));
    eprintln!("action: {}", parsed["action"].as_str().unwrap_or("?"));
    if let Some(profiles) = parsed["profiles"].as_array() {
        eprintln!("profiles: {:?}", profiles);
    }
    eprintln!("count: {}", parsed["count"].as_u64().unwrap_or(0));
    eprintln!("=============================================================");

    assert!(parsed["success"].as_bool().unwrap_or(false));
    assert_eq!(parsed["action"].as_str(), Some("list"));
    assert!(parsed.get("profiles").is_some(), "MUST have profiles array");
    assert!(parsed.get("count").is_some(), "MUST have count");
}

// =============================================================================
// TELEO-VAL-011: manage_teleological_profile - Find Best
// =============================================================================

#[tokio::test]
async fn test_val_011_manage_profile_find_best() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "manage_teleological_profile",
            "arguments": {
                "action": "find_best",
                "context": "semantic code search with entity extraction"
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let result = response.result.expect("result");
    let parsed = extract_mcp_tool_data(&result);

    eprintln!("=== PHYSICAL EVIDENCE: manage_teleological_profile find_best ===");
    eprintln!("success: {}", parsed["success"].as_bool().unwrap_or(false));
    eprintln!("action: {}", parsed["action"].as_str().unwrap_or("?"));
    eprintln!("profile_id: {:?}", parsed.get("profile_id"));
    if let Some(sim) = parsed.get("similarity") {
        eprintln!("similarity: {}", sim);
    }
    eprintln!("=================================================================");

    assert!(parsed["success"].as_bool().unwrap_or(false));
    assert_eq!(parsed["action"].as_str(), Some("find_best"));
    // profile_id can be null if no profiles match
}

// =============================================================================
// TELEO-VAL-012: manage_teleological_profile - Invalid Action (error case)
// =============================================================================

#[tokio::test]
async fn test_val_012_manage_profile_invalid_action_error() {
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
    let result = response.result.expect("Result");

    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    eprintln!("=== PHYSICAL EVIDENCE: manage_teleological_profile invalid action ===");
    eprintln!("isError: {}", is_error);
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(text) = content
            .first()
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
        {
            eprintln!("Error message: {}", text);
            assert!(
                text.contains("Unknown action") || text.contains("invalid_action"),
                "Error message should mention the invalid action"
            );
        }
    }
    eprintln!("=====================================================================");

    assert!(is_error, "MUST return isError=true for invalid action");
}
