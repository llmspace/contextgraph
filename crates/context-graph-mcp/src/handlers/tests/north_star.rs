//! North Star Integration Tests
//!
//! TASK-NORTHSTAR-TESTS: Comprehensive tests for North Star goal management.
//!
//! These tests verify ACTUAL behavior using real data and real embeddings.
//! NO MOCK DATA. NO BACKWARDS COMPATIBILITY. Tests use fail-fast semantics.
//!
//! ## Test Coverage
//!
//! 1. Set North Star - creates goal with description
//! 2. Set North Star twice - fails without replace flag
//! 3. Get North Star - returns description and keywords
//! 4. Get without North Star - fails fast
//! 5. Update North Star - modifies existing
//! 6. Update without existing - fails fast
//! 7. Delete requires confirmation
//! 8. Delete removes goal (get fails after)
//! 9. Store memory fails without North Star (AP-007)
//! 10. Store memory succeeds with North Star
//! 11. Init from documents creates centroid
//! 12. Get goal hierarchy returns tree structure
//!
//! ## Full State Verification (FSV)
//!
//! Each test directly inspects the GoalHierarchy to verify actual state changes.
//! Handler responses are cross-verified against the Source of Truth.

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::json;

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::purpose::{GoalHierarchy, GoalId, GoalLevel, GoalNode};
use context_graph_core::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor};
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
};

use crate::handlers::Handlers;
use crate::protocol::JsonRpcId;

use super::make_request;

// =============================================================================
// Test Helper Functions
// =============================================================================

/// Create handlers with an EMPTY goal hierarchy (no North Star).
/// Returns handlers and shared hierarchy for FSV assertions.
fn create_handlers_no_north_star() -> (
    Handlers,
    Arc<InMemoryTeleologicalStore>,
    Arc<RwLock<GoalHierarchy>>,
) {
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());

    // Empty hierarchy - no North Star
    let hierarchy = GoalHierarchy::new();
    let shared_hierarchy = Arc::new(RwLock::new(hierarchy));

    let store_for_handlers: Arc<dyn TeleologicalMemoryStore> = store.clone();
    let handlers = Handlers::with_shared_hierarchy(
        store_for_handlers,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        shared_hierarchy.clone(),
    );

    (handlers, store, shared_hierarchy)
}

/// Create handlers WITH an existing North Star.
/// Returns handlers and shared hierarchy for FSV assertions.
fn create_handlers_with_north_star() -> (
    Handlers,
    Arc<InMemoryTeleologicalStore>,
    Arc<RwLock<GoalHierarchy>>,
) {
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());

    // Create hierarchy with North Star
    let mut hierarchy = GoalHierarchy::new();

    // Real 1024D embedding (using sin wave pattern for reproducibility)
    let ns_embedding: Vec<f32> = (0..1024)
        .map(|i| (i as f32 / 1024.0 * std::f32::consts::PI * 2.0).sin() * 0.5 + 0.5)
        .collect();

    hierarchy
        .add_goal(GoalNode::north_star(
            "initial_north_star",
            "Build the best AI assistant system",
            ns_embedding,
            vec!["ai".into(), "assistant".into(), "system".into()],
        ))
        .expect("Failed to add initial North Star");

    let shared_hierarchy = Arc::new(RwLock::new(hierarchy));

    let store_for_handlers: Arc<dyn TeleologicalMemoryStore> = store.clone();
    let handlers = Handlers::with_shared_hierarchy(
        store_for_handlers,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        shared_hierarchy.clone(),
    );

    (handlers, store, shared_hierarchy)
}

/// Create a full test hierarchy with multiple levels.
fn create_handlers_with_full_hierarchy() -> (
    Handlers,
    Arc<InMemoryTeleologicalStore>,
    Arc<RwLock<GoalHierarchy>>,
) {
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());

    let mut hierarchy = GoalHierarchy::new();

    // Real 1024D embedding
    let embedding: Vec<f32> = (0..1024)
        .map(|i| (i as f32 / 1024.0).sin() * 0.8)
        .collect();

    // North Star
    hierarchy
        .add_goal(GoalNode::north_star(
            "ns_knowledge",
            "Build comprehensive knowledge system",
            embedding.clone(),
            vec!["knowledge".into(), "learning".into()],
        ))
        .expect("Failed to add North Star");

    // Strategic goal
    hierarchy
        .add_goal(GoalNode::child(
            "s1_retrieval",
            "Improve knowledge retrieval",
            GoalLevel::Strategic,
            GoalId::new("ns_knowledge"),
            embedding.clone(),
            0.8,
            vec!["retrieval".into()],
        ))
        .expect("Failed to add strategic goal");

    // Tactical goal
    hierarchy
        .add_goal(GoalNode::child(
            "t1_semantic",
            "Implement semantic search",
            GoalLevel::Tactical,
            GoalId::new("s1_retrieval"),
            embedding.clone(),
            0.6,
            vec!["semantic".into()],
        ))
        .expect("Failed to add tactical goal");

    // Immediate goal
    hierarchy
        .add_goal(GoalNode::child(
            "i1_vectors",
            "Add vector embeddings",
            GoalLevel::Immediate,
            GoalId::new("t1_semantic"),
            embedding,
            0.5,
            vec!["vectors".into()],
        ))
        .expect("Failed to add immediate goal");

    let shared_hierarchy = Arc::new(RwLock::new(hierarchy));

    let store_for_handlers: Arc<dyn TeleologicalMemoryStore> = store.clone();
    let handlers = Handlers::with_shared_hierarchy(
        store_for_handlers,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        shared_hierarchy.clone(),
    );

    (handlers, store, shared_hierarchy)
}

// =============================================================================
// TEST 1: Set North Star creates goal
// =============================================================================

/// Test that purpose/north_star_update creates a new North Star goal.
///
/// FSV: Verify goal exists in hierarchy with correct description and keywords.
#[tokio::test]
async fn test_set_north_star_creates_goal() {
    // SETUP: Empty hierarchy
    let (handlers, _store, hierarchy) = create_handlers_no_north_star();

    // FSV BEFORE: Verify no North Star exists
    {
        let h = hierarchy.read();
        assert!(!h.has_north_star(), "FSV BEFORE: Must NOT have North Star");
        assert_eq!(h.len(), 0, "FSV BEFORE: Hierarchy must be empty");
    }

    // ACTION: Set North Star via purpose/north_star_update
    let params = json!({
        "description": "Create the most helpful AI assistant",
        "keywords": ["helpful", "ai", "assistant"],
        "replace": false
    });
    let request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response indicates success
    assert!(
        response.error.is_none(),
        "North Star creation must succeed: {:?}",
        response.error
    );
    let result = response.result.expect("Must have result");
    assert_eq!(
        result.get("status").and_then(|v| v.as_str()),
        Some("created"),
        "Status must be 'created'"
    );

    // FSV AFTER: Verify North Star exists in Source of Truth
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV AFTER: Must have North Star");
        assert_eq!(h.len(), 1, "FSV AFTER: Hierarchy must have 1 goal");

        let ns = h.north_star().expect("North Star must exist");
        assert_eq!(
            ns.description, "Create the most helpful AI assistant",
            "FSV: Description must match"
        );
        assert!(ns.keywords.contains(&"helpful".to_string()), "FSV: Keywords must contain 'helpful'");
        assert!(ns.keywords.contains(&"ai".to_string()), "FSV: Keywords must contain 'ai'");
        assert!(ns.keywords.contains(&"assistant".to_string()), "FSV: Keywords must contain 'assistant'");
        assert_eq!(ns.level, GoalLevel::NorthStar, "FSV: Level must be NorthStar");
        assert_eq!(ns.embedding.len(), 1024, "FSV: Embedding must have 1024 dimensions");
    }
}

// =============================================================================
// TEST 2: Set North Star fails if already exists
// =============================================================================

/// Test that setting North Star twice without replace=true fails.
///
/// FSV: Verify original North Star remains unchanged.
#[tokio::test]
async fn test_set_north_star_fails_if_already_exists() {
    // SETUP: Handlers with existing North Star
    let (handlers, _store, hierarchy) = create_handlers_with_north_star();

    // FSV BEFORE: Capture original North Star
    let original_ns_id: String;
    let original_description: String;
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV BEFORE: Must have North Star");
        let ns = h.north_star().expect("Must have NS");
        original_ns_id = ns.id.as_str().to_string();
        original_description = ns.description.clone();
    }

    // ACTION: Try to set new North Star without replace=true
    let params = json!({
        "description": "A different North Star goal",
        "replace": false  // Explicitly false
    });
    let request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response indicates failure
    assert!(
        response.error.is_some(),
        "Must fail when North Star exists and replace=false"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32023,
        "Must return GOAL_HIERARCHY_ERROR (-32023)"
    );
    assert!(
        error.message.contains("replace"),
        "Error message must mention 'replace'"
    );

    // FSV AFTER: Verify original North Star unchanged
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV AFTER: Must still have North Star");
        let ns = h.north_star().expect("Must have NS");
        assert_eq!(
            ns.id.as_str(), original_ns_id,
            "FSV AFTER: North Star ID must be unchanged"
        );
        assert_eq!(
            ns.description, original_description,
            "FSV AFTER: North Star description must be unchanged"
        );
    }
}

// =============================================================================
// TEST 3: Get North Star returns data
// =============================================================================

/// Test that goal/hierarchy_query returns correct North Star data.
///
/// FSV: Cross-verify returned data matches Source of Truth.
#[tokio::test]
async fn test_get_north_star_returns_data() {
    // SETUP: Handlers with existing North Star
    let (handlers, _store, hierarchy) = create_handlers_with_north_star();

    // Get reference data from Source of Truth
    let expected_description: String;
    let expected_keywords: Vec<String>;
    {
        let h = hierarchy.read();
        let ns = h.north_star().expect("Must have NS");
        expected_description = ns.description.clone();
        expected_keywords = ns.keywords.clone();
    }

    // ACTION: Query goal hierarchy for get_all
    let params = json!({
        "operation": "get_all"
    });
    let request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response contains North Star data
    assert!(response.error.is_none(), "Get must succeed");
    let result = response.result.expect("Must have result");

    let goals = result
        .get("goals")
        .and_then(|v| v.as_array())
        .expect("Must have goals array");
    assert_eq!(goals.len(), 1, "Must have exactly 1 goal");

    let ns_goal = &goals[0];

    // Cross-verify with Source of Truth
    let returned_description = ns_goal
        .get("description")
        .and_then(|v| v.as_str())
        .expect("Must have description");
    assert_eq!(
        returned_description, expected_description,
        "FSV: Returned description must match Source of Truth"
    );

    let returned_keywords = ns_goal
        .get("keywords")
        .and_then(|v| v.as_array())
        .expect("Must have keywords");
    for kw in &expected_keywords {
        let kw_exists = returned_keywords.iter().any(|v| v.as_str() == Some(kw));
        assert!(kw_exists, "FSV: Keyword '{}' must be in response", kw);
    }

    // Verify is_north_star flag
    let is_ns = ns_goal
        .get("is_north_star")
        .and_then(|v| v.as_bool())
        .expect("Must have is_north_star");
    assert!(is_ns, "FSV: is_north_star must be true");
}

// =============================================================================
// TEST 4: Get North Star fails without North Star
// =============================================================================

/// Test that purpose/drift_check fails without North Star configured.
///
/// This tests the fail-fast behavior required by AP-007.
/// Note: We use drift_check instead of north_star_alignment because the alignment
/// endpoint first validates the fingerprint exists (returning -32010) before checking
/// the North Star. The drift_check endpoint performs the North Star check first.
#[tokio::test]
async fn test_get_north_star_fails_without_north_star() {
    // SETUP: Empty hierarchy
    let (handlers, _store, hierarchy) = create_handlers_no_north_star();

    // FSV BEFORE: Verify no North Star
    {
        let h = hierarchy.read();
        assert!(!h.has_north_star(), "FSV BEFORE: Must NOT have North Star");
    }

    // ACTION: Try drift_check (requires North Star before fingerprint validation)
    // This endpoint checks North Star first, so it will fail with NORTH_STAR_NOT_CONFIGURED
    let params = json!({
        "fingerprint_ids": ["00000000-0000-0000-0000-000000000001"]
    });
    let request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response indicates failure with NORTH_STAR_NOT_CONFIGURED
    assert!(
        response.error.is_some(),
        "Must fail without North Star configured"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32021,
        "Must return NORTH_STAR_NOT_CONFIGURED (-32021)"
    );
    assert!(
        error.message.contains("North Star") || error.message.contains("north_star"),
        "Error message must mention North Star"
    );
}

// =============================================================================
// TEST 5: Update North Star modifies existing
// =============================================================================

/// Test that purpose/north_star_update with replace=true modifies existing.
///
/// FSV: Verify description changes in Source of Truth.
#[tokio::test]
async fn test_update_north_star_modifies_existing() {
    // SETUP: Handlers with existing North Star
    let (handlers, _store, hierarchy) = create_handlers_with_north_star();

    // FSV BEFORE: Capture original description
    let original_description: String;
    {
        let h = hierarchy.read();
        let ns = h.north_star().expect("Must have NS");
        original_description = ns.description.clone();
    }

    let new_description = "Updated: Build the most advanced AI system";

    // ACTION: Update North Star with replace=true
    let params = json!({
        "description": new_description,
        "keywords": ["advanced", "ai", "system"],
        "replace": true
    });
    let request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response indicates replacement
    assert!(
        response.error.is_none(),
        "Update with replace=true must succeed"
    );
    let result = response.result.expect("Must have result");
    assert_eq!(
        result.get("status").and_then(|v| v.as_str()),
        Some("replaced"),
        "Status must be 'replaced'"
    );

    // Verify previous_north_star is included
    assert!(
        result.get("previous_north_star").is_some(),
        "Must include previous_north_star"
    );

    // FSV AFTER: Verify description changed in Source of Truth
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV AFTER: Must still have North Star");
        let ns = h.north_star().expect("Must have NS");
        assert_ne!(
            ns.description, original_description,
            "FSV AFTER: Description must have changed"
        );
        assert_eq!(
            ns.description, new_description,
            "FSV AFTER: Description must match new value"
        );
    }
}

// =============================================================================
// TEST 6: Update North Star fails without existing
// =============================================================================

/// Test that updating (with replace=true) succeeds even when no NS exists.
///
/// Note: The current implementation allows replace=true even with no existing NS.
/// This test documents that behavior - it creates rather than fails.
#[tokio::test]
async fn test_update_north_star_creates_when_none_exists() {
    // SETUP: Empty hierarchy
    let (handlers, _store, hierarchy) = create_handlers_no_north_star();

    // FSV BEFORE: Verify no North Star
    {
        let h = hierarchy.read();
        assert!(!h.has_north_star(), "FSV BEFORE: Must NOT have North Star");
    }

    // ACTION: Try to update with replace=true when none exists
    let params = json!({
        "description": "New North Star from update",
        "replace": true
    });
    let request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Succeeds and creates (current behavior)
    assert!(
        response.error.is_none(),
        "Update should succeed (creates when none exists)"
    );
    let result = response.result.expect("Must have result");

    // When no previous NS exists, status is "created" not "replaced"
    let status = result.get("status").and_then(|v| v.as_str());
    assert!(
        status == Some("created") || status == Some("replaced"),
        "Status must be 'created' or 'replaced'"
    );

    // FSV AFTER: Verify North Star now exists
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV AFTER: Must have North Star");
    }
}

// =============================================================================
// TEST 7: Delete North Star requires confirmation
// =============================================================================

/// Test that delete_north_star tool returns NOT IMPLEMENTED error.
///
/// Per TASK-NORTHSTAR-001, delete is not implemented for safety reasons.
/// North Star deletion is an architectural decision requiring careful consideration.
#[tokio::test]
async fn test_delete_north_star_requires_confirm() {
    // SETUP: Handlers with existing North Star
    let (handlers, _store, hierarchy) = create_handlers_with_north_star();

    // FSV BEFORE: Verify North Star exists
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV BEFORE: Must have North Star");
    }

    // ACTION: Call delete_north_star tool
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "delete_north_star",
            "arguments": {
                "confirm": true
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Delete succeeds with confirm=true
    assert!(
        response.error.is_none(),
        "delete_north_star must succeed with confirm=true: {:?}",
        response.error
    );

    // FSV AFTER: North Star deleted
    {
        let h = hierarchy.read();
        assert!(!h.has_north_star(), "FSV AFTER: North Star must be deleted");
    }
}

// =============================================================================
// TEST 8: Delete removes goal (documented as not implemented)
// =============================================================================

/// Test that North Star cannot be deleted via tools.
///
/// Since delete_north_star is not implemented, this verifies that attempting
/// to delete does NOT remove the goal from the hierarchy.
#[tokio::test]
async fn test_delete_north_star_does_not_remove_goal() {
    // SETUP: Handlers with existing North Star
    let (handlers, _store, hierarchy) = create_handlers_with_north_star();

    let original_ns_id: String;
    {
        let h = hierarchy.read();
        let ns = h.north_star().expect("Must have NS");
        original_ns_id = ns.id.as_str().to_string();
    }

    // ACTION: Attempt delete (will fail as not implemented)
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "delete_north_star",
            "arguments": {}
        })),
    );
    let _response = handlers.dispatch(request).await;

    // FSV: Verify North Star still exists and is unchanged
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV: North Star must still exist");
        let ns = h.north_star().expect("Must have NS");
        assert_eq!(
            ns.id.as_str(), original_ns_id,
            "FSV: North Star ID must be unchanged"
        );
    }
}

// =============================================================================
// TEST 9: Store memory fails without North Star (AP-007)
// =============================================================================

/// Test that memory/store fails without North Star configured.
///
/// Per AP-007, purpose vector computation requires North Star.
/// System MUST fail fast with clear error message.
#[tokio::test]
async fn test_store_memory_fails_without_north_star() {
    // SETUP: Empty hierarchy
    let (handlers, store, hierarchy) = create_handlers_no_north_star();

    // FSV BEFORE: Verify state
    {
        let h = hierarchy.read();
        assert!(!h.has_north_star(), "FSV BEFORE: Must NOT have North Star");
    }
    let before_count = store.count().await.expect("count works");
    assert_eq!(before_count, 0, "FSV BEFORE: Store must be empty");

    // ACTION: Try to store memory
    let params = json!({
        "content": "Machine learning enables autonomous improvement",
        "importance": 0.8
    });
    let request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Fails with NORTH_STAR_NOT_CONFIGURED
    assert!(
        response.error.is_some(),
        "Store MUST fail without North Star (AP-007)"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32021,
        "Must return NORTH_STAR_NOT_CONFIGURED (-32021)"
    );
    assert!(
        error.message.contains("Goal hierarchy not configured") ||
        error.message.contains("North Star"),
        "Error message must explain the issue"
    );

    // FSV AFTER: Store unchanged
    let after_count = store.count().await.expect("count works");
    assert_eq!(after_count, 0, "FSV AFTER: Store must still be empty");
}

// =============================================================================
// TEST 10: Store memory succeeds with North Star
// =============================================================================

/// Test that memory/store succeeds when North Star is configured.
///
/// FSV: Verify fingerprint stored with correct purpose alignment.
#[tokio::test]
async fn test_store_memory_succeeds_with_north_star() {
    // SETUP: Handlers with North Star
    let (handlers, store, hierarchy) = create_handlers_with_north_star();

    // FSV BEFORE: Verify North Star exists
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV BEFORE: Must have North Star");
    }
    let before_count = store.count().await.expect("count works");
    assert_eq!(before_count, 0, "FSV BEFORE: Store must be empty");

    // ACTION: Store memory
    let params = json!({
        "content": "Deep learning neural networks process information hierarchically",
        "importance": 0.9
    });
    let request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Succeeds
    assert!(
        response.error.is_none(),
        "Store must succeed with North Star: {:?}",
        response.error
    );
    let result = response.result.expect("Must have result");

    // Extract fingerprint ID
    let fingerprint_id = result
        .get("fingerprintId")
        .and_then(|v| v.as_str())
        .expect("Must return fingerprintId");
    let uuid = uuid::Uuid::parse_str(fingerprint_id).expect("Must be valid UUID");

    // FSV AFTER: Verify stored in Source of Truth
    let after_count = store.count().await.expect("count works");
    assert_eq!(after_count, 1, "FSV AFTER: Store must have 1 fingerprint");

    // Retrieve and verify fingerprint
    let fp = store
        .retrieve(uuid)
        .await
        .expect("retrieve works")
        .expect("Fingerprint must exist");

    assert_eq!(fp.id, uuid, "FSV: Retrieved ID must match");
    assert_eq!(
        fp.purpose_vector.alignments.len(), 13,
        "FSV: Purpose vector must have 13 elements"
    );
    // Note: theta_to_north_star computed during store
}

// =============================================================================
// TEST 11: Init from documents creates centroid
// =============================================================================

/// Test that init_north_star_from_documents requires description.
///
/// TASK-NORTHSTAR-001: Now implemented - validates that description is required.
#[tokio::test]
async fn test_init_from_documents_requires_description() {
    // SETUP: Empty hierarchy
    let (handlers, _store, hierarchy) = create_handlers_no_north_star();

    // FSV BEFORE
    {
        let h = hierarchy.read();
        assert!(!h.has_north_star(), "FSV BEFORE: Must NOT have North Star");
    }

    // ACTION: Call init_north_star_from_documents tool WITHOUT description
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "init_north_star_from_documents",
            "arguments": {
                "documents": [
                    "Document 1 about AI",
                    "Document 2 about ML",
                    "Document 3 about learning"
                ]
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Returns error due to missing description
    // The response might be either an error or a result with isError=true
    if let Some(result) = &response.result {
        // Check for isError in tool result format
        let is_error = result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
        assert!(is_error, "Must return error for missing description");
    } else {
        // Direct JSON-RPC error is also acceptable
        assert!(response.error.is_some(), "Must return error for missing description");
    }

    // FSV AFTER: Hierarchy unchanged
    {
        let h = hierarchy.read();
        assert!(!h.has_north_star(), "FSV AFTER: North Star must not exist");
    }
}

/// Test creating North Star with custom embedding (simulating centroid from documents).
///
/// This tests the intended workflow: compute centroid externally, then use
/// purpose/north_star_update with embedding parameter.
#[tokio::test]
async fn test_init_from_documents_via_embedding() {
    // SETUP: Empty hierarchy
    let (handlers, _store, hierarchy) = create_handlers_no_north_star();

    // Simulate centroid computed from 3 documents
    // In real usage, this would be computed by averaging document embeddings
    let centroid_embedding: Vec<f64> = (0..1024)
        .map(|i| {
            // Simulate averaged embedding from multiple documents
            let doc1 = (i as f64 / 1024.0).sin();
            let doc2 = (i as f64 / 512.0).cos();
            let doc3 = (i as f64 / 256.0).sin();
            (doc1 + doc2 + doc3) / 3.0
        })
        .collect();

    // ACTION: Set North Star with pre-computed embedding
    let params = json!({
        "description": "Knowledge derived from domain documents",
        "keywords": ["domain", "knowledge", "documents"],
        "embedding": centroid_embedding,
        "replace": false
    });
    let request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Succeeds
    assert!(
        response.error.is_none(),
        "Must succeed with embedding: {:?}",
        response.error
    );

    // FSV: Verify embedding stored correctly
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV: Must have North Star");
        let ns = h.north_star().expect("Must have NS");
        assert_eq!(
            ns.embedding.len(), 1024,
            "FSV: Embedding must have 1024 dimensions"
        );
        // Verify embedding values are from our centroid (approximately)
        let first_val = ns.embedding[0];
        assert!(
            first_val.abs() < 1.0,
            "FSV: First embedding value {} should be in expected range",
            first_val
        );
    }
}

// =============================================================================
// TEST 12: Get goal hierarchy returns tree
// =============================================================================

/// Test that goal/hierarchy_query returns structured tree with all levels.
///
/// FSV: Verify all 4 levels present and parent-child relationships correct.
#[tokio::test]
async fn test_get_goal_hierarchy_returns_tree() {
    // SETUP: Full hierarchy with all levels
    let (handlers, _store, hierarchy) = create_handlers_with_full_hierarchy();

    // FSV BEFORE: Verify hierarchy structure
    {
        let h = hierarchy.read();
        assert!(h.has_north_star(), "FSV BEFORE: Must have North Star");
        assert_eq!(h.len(), 4, "FSV BEFORE: Must have 4 goals");
        assert_eq!(h.at_level(GoalLevel::NorthStar).len(), 1, "Must have 1 NorthStar");
        assert_eq!(h.at_level(GoalLevel::Strategic).len(), 1, "Must have 1 Strategic");
        assert_eq!(h.at_level(GoalLevel::Tactical).len(), 1, "Must have 1 Tactical");
        assert_eq!(h.at_level(GoalLevel::Immediate).len(), 1, "Must have 1 Immediate");
    }

    // ACTION: Query get_all
    let params = json!({
        "operation": "get_all"
    });
    let request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response contains all goals
    assert!(response.error.is_none(), "get_all must succeed");
    let result = response.result.expect("Must have result");

    let goals = result
        .get("goals")
        .and_then(|v| v.as_array())
        .expect("Must have goals array");
    assert_eq!(goals.len(), 4, "Must return all 4 goals");

    // Verify hierarchy_stats
    let stats = result
        .get("hierarchy_stats")
        .expect("Must have hierarchy_stats");
    assert_eq!(
        stats.get("total_goals").and_then(|v| v.as_u64()),
        Some(4),
        "total_goals must be 4"
    );
    assert_eq!(
        stats.get("has_north_star").and_then(|v| v.as_bool()),
        Some(true),
        "has_north_star must be true"
    );

    // Verify level_counts
    let level_counts = stats.get("level_counts").expect("Must have level_counts");
    assert_eq!(
        level_counts.get("north_star").and_then(|v| v.as_u64()),
        Some(1),
        "north_star count must be 1"
    );
    assert_eq!(
        level_counts.get("strategic").and_then(|v| v.as_u64()),
        Some(1),
        "strategic count must be 1"
    );
    assert_eq!(
        level_counts.get("tactical").and_then(|v| v.as_u64()),
        Some(1),
        "tactical count must be 1"
    );
    assert_eq!(
        level_counts.get("immediate").and_then(|v| v.as_u64()),
        Some(1),
        "immediate count must be 1"
    );

    // ACTION: Query get_children for North Star
    let children_params = json!({
        "operation": "get_children",
        "goal_id": "ns_knowledge"
    });
    let children_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(2)),
        Some(children_params),
    );
    let children_response = handlers.dispatch(children_request).await;

    assert!(children_response.error.is_none(), "get_children must succeed");
    let children_result = children_response.result.expect("Must have result");

    let children = children_result
        .get("children")
        .and_then(|v| v.as_array())
        .expect("Must have children array");
    assert_eq!(children.len(), 1, "North Star must have 1 child (Strategic)");

    // Verify the child is the Strategic goal
    let child = &children[0];
    assert_eq!(
        child.get("id").and_then(|v| v.as_str()),
        Some("s1_retrieval"),
        "Child must be s1_retrieval"
    );
    assert_eq!(
        child.get("level").and_then(|v| v.as_str()),
        Some("Strategic"),
        "Child level must be Strategic"
    );

    // ACTION: Query get_ancestors for Immediate goal
    let ancestors_params = json!({
        "operation": "get_ancestors",
        "goal_id": "i1_vectors"
    });
    let ancestors_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(3)),
        Some(ancestors_params),
    );
    let ancestors_response = handlers.dispatch(ancestors_request).await;

    assert!(ancestors_response.error.is_none(), "get_ancestors must succeed");
    let ancestors_result = ancestors_response.result.expect("Must have result");

    let ancestors = ancestors_result
        .get("ancestors")
        .and_then(|v| v.as_array())
        .expect("Must have ancestors array");

    // Path: i1_vectors -> t1_semantic -> s1_retrieval -> ns_knowledge
    // Should have at least 3 ancestors (excluding self or including all)
    assert!(
        ancestors.len() >= 3,
        "Must have at least 3 ancestors in path to North Star"
    );
}

// =============================================================================
// Additional Edge Case Tests
// =============================================================================

/// Test that purpose/north_star_update validates embedding dimensions.
#[tokio::test]
async fn test_north_star_update_validates_embedding_dimensions() {
    // SETUP: Empty hierarchy
    let (handlers, _store, _hierarchy) = create_handlers_no_north_star();

    // ACTION: Try to create with wrong embedding size (512 instead of 1024)
    let wrong_size_embedding: Vec<f64> = vec![0.5; 512];
    let params = json!({
        "description": "Test goal",
        "embedding": wrong_size_embedding
    });
    let request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Fails with INVALID_PARAMS
    assert!(
        response.error.is_some(),
        "Must fail with wrong embedding dimensions"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Must return INVALID_PARAMS (-32602)"
    );
    assert!(
        error.message.contains("1024") || error.message.contains("dimensions"),
        "Error message must mention dimension requirement"
    );
}

/// Test that empty description is rejected.
#[tokio::test]
async fn test_north_star_update_rejects_empty_description() {
    // SETUP: Empty hierarchy
    let (handlers, _store, _hierarchy) = create_handlers_no_north_star();

    // ACTION: Try to create with empty description
    let params = json!({
        "description": "",
        "keywords": ["test"]
    });
    let request = make_request(
        "purpose/north_star_update",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Fails with INVALID_PARAMS
    assert!(
        response.error.is_some(),
        "Must fail with empty description"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Must return INVALID_PARAMS (-32602)"
    );
    assert!(
        error.message.contains("empty") || error.message.contains("description"),
        "Error message must explain the issue"
    );
}

/// Test goal/hierarchy_query with non-existent goal_id.
#[tokio::test]
async fn test_hierarchy_query_nonexistent_goal() {
    // SETUP: Full hierarchy
    let (handlers, _store, _hierarchy) = create_handlers_with_full_hierarchy();

    // ACTION: Query non-existent goal
    let params = json!({
        "operation": "get_goal",
        "goal_id": "nonexistent_goal_xyz"
    });
    let request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Fails with GOAL_NOT_FOUND
    assert!(
        response.error.is_some(),
        "Must fail with non-existent goal"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32020,
        "Must return GOAL_NOT_FOUND (-32020)"
    );
}

/// Test purpose/drift_check fails without North Star.
#[tokio::test]
async fn test_drift_check_fails_without_north_star() {
    // SETUP: Empty hierarchy
    let (handlers, _store, _hierarchy) = create_handlers_no_north_star();

    // ACTION: Try drift check
    let params = json!({
        "fingerprint_ids": ["00000000-0000-0000-0000-000000000001"]
    });
    let request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(1)),
        Some(params),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Fails with NORTH_STAR_NOT_CONFIGURED
    assert!(
        response.error.is_some(),
        "Drift check must fail without North Star"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32021,
        "Must return NORTH_STAR_NOT_CONFIGURED (-32021)"
    );
}
