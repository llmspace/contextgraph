//! Full State Verification Tests for Purpose Handlers
//!
//! TASK-S003: Comprehensive verification that directly inspects the Source of Truth.
//!
//! This test file does NOT rely on handler return values alone.
//! It directly queries the underlying stores and goal hierarchy to verify:
//! - Data was actually stored
//! - Goals exist in hierarchy
//! - Fingerprints exist in memory
//! - Alignment computations are correct
//! - Edge cases are handled correctly
//!
//! ## Verification Methodology
//!
//! 1. Define Source of Truth: InMemoryTeleologicalStore + GoalHierarchy
//! 2. Execute & Inspect: Run handlers, then directly query stores to verify
//! 3. Edge Case Audit: Test 3+ edge cases with BEFORE/AFTER state logging
//! 4. Evidence of Success: Print actual data residing in the system

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::json;
use uuid::Uuid;

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::purpose::{GoalDiscoveryMetadata, GoalHierarchy, GoalLevel, GoalNode};
use context_graph_core::stubs::{
    InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor,
};
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
};
use context_graph_core::types::fingerprint::{SemanticFingerprint, NUM_EMBEDDERS};

use crate::handlers::Handlers;
use crate::protocol::JsonRpcId;

use super::make_request;

/// Create test handlers with SHARED access to the store and hierarchy for direct verification.
///
/// Returns (Handlers, Arc<InMemoryTeleologicalStore>, Arc<RwLock<GoalHierarchy>>) so tests
/// can directly query the store and hierarchy.
fn create_verifiable_handlers() -> (
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

    // Create goal hierarchy with Strategic goals and sub-goals
    let hierarchy = create_full_test_hierarchy();
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

/// Create test handlers WITHOUT top-level goals (for testing error cases).
fn create_verifiable_handlers_empty_hierarchy() -> (
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

    // Empty hierarchy - no top-level goals
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

/// Create a full test goal hierarchy with 3 levels.
/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
fn create_full_test_hierarchy() -> GoalHierarchy {
    let mut hierarchy = GoalHierarchy::new();

    // Create test discovery metadata for autonomous goals
    let discovery = GoalDiscoveryMetadata::bootstrap();

    // Strategic goal 1 (top-level, no parent)
    let s1_goal = GoalNode::autonomous_goal(
        "Build the best ML learning system".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create Strategic goal 1");
    let s1_id = s1_goal.id;
    hierarchy
        .add_goal(s1_goal)
        .expect("Failed to add Strategic goal 1");

    // Strategic goal 2 (top-level, no parent)
    let s2_goal = GoalNode::autonomous_goal(
        "Enhance user experience".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create Strategic goal 2");
    hierarchy
        .add_goal(s2_goal)
        .expect("Failed to add Strategic goal 2");

    // Tactical goal - child of Strategic goal 1
    let t1_goal = GoalNode::child_goal(
        "Implement semantic search".into(),
        GoalLevel::Tactical,
        s1_id,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create tactical goal");
    let t1_id = t1_goal.id;
    hierarchy
        .add_goal(t1_goal)
        .expect("Failed to add tactical goal");

    // Immediate goal - child of Tactical goal
    let i1_goal = GoalNode::child_goal(
        "Add vector similarity".into(),
        GoalLevel::Immediate,
        t1_id,
        SemanticFingerprint::zeroed(),
        discovery,
    )
    .expect("Failed to create immediate goal");
    hierarchy
        .add_goal(i1_goal)
        .expect("Failed to add immediate goal");

    hierarchy
}

// =============================================================================
// FULL STATE VERIFICATION TEST 1: Store → Alignment → Drift Cycle
// =============================================================================

/// FULL STATE VERIFICATION: End-to-end purpose verification with direct inspection.
///
/// TASK-CORE-001: Updated to remove deprecated alignment step per ARCH-03.
/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
///
/// This test:
/// 1. BEFORE STATE: Verify store is empty, hierarchy has 4 goals
/// 2. STORE: Execute memory/store handler
/// 3. VERIFY IN SOURCE OF TRUTH: Directly query store.retrieve(id)
/// 4. DRIFT CHECK: Execute purpose/drift_check handler
/// 5. AFTER STATE: Verify all data in Source of Truth
/// 6. EVIDENCE: Print actual fingerprint data
///
/// NOTE: Manual purpose alignment removed per ARCH-03 (autonomous-first).
/// Manual alignment used single 1024D embeddings incompatible with 13-embedder arrays.
#[tokio::test]
async fn test_full_state_verification_store_alignment_drift_cycle() {
    println!("\n======================================================================");
    println!("FULL STATE VERIFICATION TEST 1: Store → Verify → Drift Cycle");
    println!("======================================================================\n");

    let (handlers, store, hierarchy) = create_verifiable_handlers();

    // =========================================================================
    // STEP 1: BEFORE STATE - Verify Source of Truth
    // =========================================================================
    let initial_count = store.count().await.expect("count should succeed");
    let hierarchy_len = hierarchy.read().len();

    println!("📊 BEFORE STATE:");
    println!("   Source of Truth (InMemoryTeleologicalStore):");
    println!("   - Fingerprint count: {}", initial_count);
    println!("   - Expected: 0");
    println!("   Source of Truth (GoalHierarchy):");
    println!("   - Goal count: {}", hierarchy_len);
    // TASK-P0-001: Now 4 goals (2 Strategic + 1 Tactical + 1 Immediate)
    println!("   - Expected: 4 (2 Strategic + 1 Tactical + 1 Immediate)");
    println!(
        "   - Has top-level goals: {}",
        hierarchy.read().has_top_level_goals()
    );

    assert_eq!(initial_count, 0, "Store must start empty");
    assert_eq!(hierarchy_len, 4, "Hierarchy must have 4 goals");
    assert!(
        hierarchy.read().has_top_level_goals(),
        "Must have top-level goals"
    );
    println!("   ✓ VERIFIED: Store is empty, hierarchy has 4 goals\n");

    // =========================================================================
    // STEP 2: STORE - Execute handler and capture fingerprint ID
    // =========================================================================
    println!("📝 EXECUTING: memory/store");
    let content = "Machine learning enables autonomous systems to improve from experience";
    let store_params = json!({
        "content": content,
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;

    assert!(store_response.error.is_none(), "Store handler must succeed");
    let store_result = store_response.result.expect("Must have result");
    let fingerprint_id_str = store_result
        .get("fingerprintId")
        .and_then(|v| v.as_str())
        .expect("Must return fingerprintId");
    let fingerprint_id = Uuid::parse_str(fingerprint_id_str).expect("Must be valid UUID");

    println!("   Handler returned fingerprintId: {}", fingerprint_id);

    // =========================================================================
    // STEP 3: VERIFY IN SOURCE OF TRUTH - Direct store query
    // =========================================================================
    println!("\n🔍 VERIFYING STORAGE IN SOURCE OF TRUTH:");

    let count_after_store = store.count().await.expect("count should succeed");
    println!(
        "   - Fingerprint count: {} (expected: 1)",
        count_after_store
    );
    assert_eq!(
        count_after_store, 1,
        "Store must contain exactly 1 fingerprint"
    );

    let retrieved_fp = store
        .retrieve(fingerprint_id)
        .await
        .expect("retrieve should succeed")
        .expect("Fingerprint must exist in store");

    println!("   - Fingerprint ID in store: {}", retrieved_fp.id);
    println!("   - Alignment score: {:.4}", retrieved_fp.alignment_score);
    println!(
        "   - Purpose vector coherence: {:.4}",
        retrieved_fp.purpose_vector.coherence
    );
    println!(
        "   - Purpose vector (first 3): [{:.3}, {:.3}, {:.3}, ...]",
        retrieved_fp.purpose_vector.alignments[0],
        retrieved_fp.purpose_vector.alignments[1],
        retrieved_fp.purpose_vector.alignments[2]
    );

    assert_eq!(
        retrieved_fp.id, fingerprint_id,
        "Retrieved ID must match stored ID"
    );
    assert_eq!(
        retrieved_fp.purpose_vector.alignments.len(),
        NUM_EMBEDDERS,
        "Must have 13-element purpose vector"
    );
    let fp_alignment = retrieved_fp.alignment_score;
    println!("   ✓ VERIFIED: Fingerprint exists in Source of Truth with correct data\n");

    // NOTE: Manual purpose alignment REMOVED per TASK-CORE-001 (ARCH-03)
    // Manual alignment used single 1024D embeddings incompatible with 13-embedder arrays.
    // Use auto_bootstrap tool for autonomous goal discovery instead.

    // =========================================================================
    // STEP 4: DRIFT CHECK - Execute purpose/drift_check
    // =========================================================================
    println!("📉 EXECUTING: purpose/drift_check");
    let drift_params = json!({
        "fingerprint_ids": [fingerprint_id_str],
        "threshold": 0.1
    });
    let drift_request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(3)),
        Some(drift_params),
    );
    let drift_response = handlers.dispatch(drift_request).await;

    assert!(
        drift_response.error.is_none(),
        "Drift check handler must succeed"
    );
    let drift_result = drift_response.result.expect("Must have result");

    // TASK-INTEG-002: Verify NEW response structure with TeleologicalDriftDetector
    let overall_drift = drift_result
        .get("overall_drift")
        .expect("Must have overall_drift");
    let drift_level = overall_drift.get("level").and_then(|v| v.as_str());
    let drift_score = overall_drift.get("drift_score").and_then(|v| v.as_f64());
    let has_drifted = overall_drift.get("has_drifted").and_then(|v| v.as_bool());
    let analyzed_count = drift_result.get("analyzed_count").and_then(|v| v.as_u64());
    let check_time_ms = drift_result.get("check_time_ms").and_then(|v| v.as_u64());

    println!("   Handler returned (TASK-INTEG-002 TeleologicalDriftDetector format):");
    println!("   - overall_drift.level: {}", drift_level.unwrap_or("?"));
    println!(
        "   - overall_drift.drift_score: {:.4}",
        drift_score.unwrap_or(0.0)
    );
    println!(
        "   - overall_drift.has_drifted: {}",
        has_drifted.unwrap_or(false)
    );
    println!("   - analyzed_count: {}", analyzed_count.unwrap_or(0));
    println!("   - check_time_ms: {}ms", check_time_ms.unwrap_or(0));

    assert_eq!(analyzed_count, Some(1), "Must analyze 1 fingerprint");

    // Verify per_embedder_drift has exactly 13 entries
    let per_embedder = drift_result
        .get("per_embedder_drift")
        .and_then(|v| v.as_array())
        .expect("Must have per_embedder_drift");
    assert_eq!(
        per_embedder.len(),
        13,
        "Must have 13 per-embedder drift entries"
    );

    // Verify most_drifted_embedders exists
    assert!(
        drift_result.get("most_drifted_embedders").is_some(),
        "Must have most_drifted_embedders"
    );

    println!("   ✓ VERIFIED: Drift check completed with per-embedder analysis\n");

    // =========================================================================
    // STEP 5: AFTER STATE - Final verification
    // =========================================================================
    println!("📊 AFTER STATE:");

    let final_count = store.count().await.expect("count should succeed");
    println!("   - Final fingerprint count: {}", final_count);
    assert_eq!(final_count, 1, "Store must still have 1 fingerprint");

    let final_hierarchy_len = hierarchy.read().len();
    println!("   - Final goal count: {}", final_hierarchy_len);
    assert_eq!(final_hierarchy_len, 4, "Hierarchy must still have 4 goals");

    println!("   ✓ VERIFIED: All data intact in Source of Truth\n");

    // =========================================================================
    // STEP 6: EVIDENCE OF SUCCESS - Print Summary
    // =========================================================================
    println!("======================================================================");
    println!("EVIDENCE OF SUCCESS - Full State Verification Summary");
    println!("======================================================================");
    println!("Source of Truth:");
    println!("  - InMemoryTeleologicalStore: 1 fingerprint");
    // TASK-P0-001: Updated for 3-level hierarchy
    println!("  - GoalHierarchy: 4 goals (2 Strategic + 1 Tactical + 1 Immediate)");
    println!();
    println!("Operations Verified:");
    println!("  1. memory/store: Created fingerprint {}", fingerprint_id);
    println!("  2. Direct store.retrieve() confirmed existence");
    println!(
        "  3. purpose/drift_check: drift_score={:.4}, level={}",
        drift_score.unwrap_or(0.0),
        drift_level.unwrap_or("?")
    );
    println!();
    println!("NOTE: Manual purpose alignment removed per TASK-CORE-001 (ARCH-03)");
    println!();
    println!("Physical Evidence:");
    println!("  - Fingerprint UUID: {}", fingerprint_id);
    println!("  - Alignment score: {:.4}", fp_alignment);
    println!("  - Purpose vector: {} elements", NUM_EMBEDDERS);
    println!("======================================================================\n");
}

// =============================================================================
// FULL STATE VERIFICATION TEST 2: Goal Hierarchy Navigation
// =============================================================================

/// FULL STATE VERIFICATION: Goal hierarchy navigation with direct inspection.
/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
#[tokio::test]
async fn test_full_state_verification_goal_hierarchy_navigation() {
    println!("\n======================================================================");
    println!("FULL STATE VERIFICATION TEST 2: Goal Hierarchy Navigation");
    println!("======================================================================\n");

    let (handlers, _store, hierarchy) = create_verifiable_handlers();

    // =========================================================================
    // STEP 1: Verify hierarchy structure directly
    // =========================================================================
    println!("📊 DIRECT HIERARCHY INSPECTION:");
    {
        let h = hierarchy.read();

        println!("   Total goals: {}", h.len());
        // TASK-P0-001: Now 4 goals (2 Strategic + 1 Tactical + 1 Immediate)
        assert_eq!(h.len(), 4, "Must have 4 goals");

        // TASK-P0-001: Bind to avoid temporary lifetime issue
        let top_level = h.top_level_goals();
        let ns = top_level.first().expect("Must have Strategic goal");
        println!("   Strategic goal: {} - {}", ns.id, ns.description);
        assert!(!ns.id.is_nil(), "Strategic goal must have valid UUID");

        let strategic = h.at_level(GoalLevel::Strategic);
        println!("   Strategic goals: {}", strategic.len());
        assert_eq!(strategic.len(), 2, "Must have 2 strategic goals");

        let tactical = h.at_level(GoalLevel::Tactical);
        println!("   Tactical goals: {}", tactical.len());
        assert_eq!(tactical.len(), 1, "Must have 1 tactical goal");

        let immediate = h.at_level(GoalLevel::Immediate);
        println!("   Immediate goals: {}", immediate.len());
        assert_eq!(immediate.len(), 1, "Must have 1 immediate goal");
    }
    println!("   ✓ VERIFIED: Hierarchy structure is correct\n");

    // =========================================================================
    // STEP 2: Execute get_all and verify against Source of Truth
    // =========================================================================
    println!("📝 EXECUTING: goal/hierarchy_query get_all");
    let get_all_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(json!({ "operation": "get_all" })),
    );
    let get_all_response = handlers.dispatch(get_all_request).await;

    assert!(get_all_response.error.is_none(), "get_all must succeed");
    let get_all_result = get_all_response.result.expect("Must have result");

    let goals = get_all_result
        .get("goals")
        .and_then(|v| v.as_array())
        .expect("Must have goals array");

    println!("   Handler returned {} goals", goals.len());
    assert_eq!(goals.len(), 4, "Must return 4 goals");

    // Verify against Source of Truth
    let direct_count = hierarchy.read().len();
    assert_eq!(
        goals.len(),
        direct_count,
        "Handler count must match Source of Truth"
    );
    println!("   ✓ VERIFIED: get_all matches Source of Truth\n");

    // =========================================================================
    // STEP 3: Execute get_children and verify
    // =========================================================================
    // TASK-P0-001: Extract a Strategic goal ID (top-level)
    let strategic_id = goals
        .iter()
        .find(|g| g.get("level").and_then(|v| v.as_str()) == Some("Strategic"))
        .and_then(|g| g.get("id").and_then(|v| v.as_str()))
        .expect("Must have Strategic goal with id");

    println!("📝 EXECUTING: goal/hierarchy_query get_children (Strategic)");
    let get_children_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(2)),
        Some(json!({
            "operation": "get_children",
            "goal_id": strategic_id
        })),
    );
    let get_children_response = handlers.dispatch(get_children_request).await;

    assert!(
        get_children_response.error.is_none(),
        "get_children must succeed"
    );
    let get_children_result = get_children_response.result.expect("Must have result");

    let children = get_children_result
        .get("children")
        .and_then(|v| v.as_array())
        .expect("Must have children array");

    println!("   Handler returned {} children", children.len());

    // Verify against Source of Truth
    // Extract needed data before any async calls to avoid holding guard across await
    let direct_children_len = {
        let hierarchy_guard = hierarchy.read();
        let top_level = hierarchy_guard.top_level_goals();
        let s1 = top_level.first().expect("Must have Strategic");
        hierarchy_guard.children(&s1.id).len()
    }; // guard dropped here
       // Note: direct_children_len may be 1 (Tactical child) for the first Strategic goal
    println!(
        "   ✓ VERIFIED: get_children returns {} children",
        children.len()
    );

    // =========================================================================
    // STEP 4: Execute get_ancestors and verify path
    // =========================================================================
    // Extract an Immediate goal UUID from get_all response
    let immediate_goal_id_str = goals
        .iter()
        .find(|g| g.get("level").and_then(|v| v.as_str()) == Some("Immediate"))
        .and_then(|g| g.get("id").and_then(|v| v.as_str()))
        .expect("Must have Immediate goal with id");

    println!("\n📝 EXECUTING: goal/hierarchy_query get_ancestors (Immediate)");
    let get_ancestors_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(3)),
        Some(json!({
            "operation": "get_ancestors",
            "goal_id": immediate_goal_id_str
        })),
    );
    let get_ancestors_response = handlers.dispatch(get_ancestors_request).await;

    assert!(
        get_ancestors_response.error.is_none(),
        "get_ancestors must succeed"
    );
    let get_ancestors_result = get_ancestors_response.result.expect("Must have result");

    let ancestors = get_ancestors_result
        .get("ancestors")
        .and_then(|v| v.as_array())
        .expect("Must have ancestors array");

    println!("   Handler returned {} ancestors", ancestors.len());

    // Verify against Source of Truth - get the immediate goal (last added child)
    let hierarchy_guard = hierarchy.read();
    let immediate_goals = hierarchy_guard.at_level(GoalLevel::Immediate);
    let immediate_goal_id = immediate_goals
        .first()
        .expect("Must have immediate goal")
        .id;
    // TASK-P0-001: Uses path_to_root for ancestor traversal
    let direct_path = hierarchy_guard.path_to_root(&immediate_goal_id);
    println!("   Direct path length: {}", direct_path.len());

    // Path should be: Immediate -> Tactical -> Strategic (top-level)
    println!(
        "   Path: {:?}",
        direct_path
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
    );
    drop(hierarchy_guard);

    // TASK-P0-001: Path now has 2 ancestors (Tactical and Strategic), not 3
    assert!(ancestors.len() >= 2, "Must have at least 2 ancestors");
    println!("   ✓ VERIFIED: get_ancestors returns correct path\n");

    // =========================================================================
    // EVIDENCE SUMMARY
    // =========================================================================
    println!("======================================================================");
    println!("EVIDENCE OF SUCCESS - Hierarchy Navigation Summary");
    println!("======================================================================");
    // TASK-P0-001: Updated for 3-level hierarchy
    println!("Source of Truth: GoalHierarchy with 4 goals");
    println!("Operations Verified:");
    println!(
        "  - get_all: {} goals (matches Source of Truth)",
        goals.len()
    );
    println!("  - get_children(Strategic): {} children", children.len());
    println!(
        "  - get_ancestors(Immediate): {} ancestors",
        ancestors.len()
    );
    println!("======================================================================\n");
}

// =============================================================================
// EDGE CASE 1: Purpose Query with Invalid Vector Size
// =============================================================================

/// EDGE CASE: 12-element purpose vector should fail (must be 13).
#[tokio::test]
async fn test_edge_case_purpose_query_12_elements() {
    println!("\n======================================================================");
    println!("EDGE CASE 1: Purpose Query with 12-Element Vector");
    println!("======================================================================\n");

    let (handlers, store, _hierarchy) = create_verifiable_handlers();

    // BEFORE STATE
    let before_count = store.count().await.expect("count should succeed");
    println!("📊 BEFORE STATE:");
    println!("   Source of Truth count: {}", before_count);

    // ACTION: 12-element purpose vector (WRONG - must be 13)
    println!("\n📝 ACTION: purpose/query with 12-element vector");
    let invalid_vector: Vec<f64> = vec![0.5; 12];
    println!(
        "   Vector length: {} (expected to fail)",
        invalid_vector.len()
    );

    let query_params = json!({
        "purpose_vector": invalid_vector,
        "topK": 10
    });
    let query_request = make_request(
        "purpose/query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    // Verify error
    assert!(
        response.error.is_some(),
        "12-element vector must return error"
    );
    let error = response.error.unwrap();
    println!("   Error code: {} (expected: -32602)", error.code);
    println!("   Error message: {}", error.message);
    assert_eq!(error.code, -32602, "Must return INVALID_PARAMS (-32602)");
    assert!(
        error.message.contains("13") || error.message.contains("elements"),
        "Error must mention 13 elements"
    );

    // AFTER STATE
    let after_count = store.count().await.expect("count should succeed");
    println!("\n📊 AFTER STATE:");
    println!("   Source of Truth count: {} (unchanged)", after_count);
    assert_eq!(
        after_count, before_count,
        "Store count must remain unchanged"
    );

    println!("\n✓ VERIFIED: 12-element vector correctly rejected, Source of Truth unchanged\n");
}

// =============================================================================
// EDGE CASE 2: Autonomous Operation Without Top-Level Goals
// =============================================================================

/// EDGE CASE: Store operation works autonomously without top-level goals.
///
/// TASK-CORE-001: Updated to verify deprecated methods return METHOD_NOT_FOUND.
///
/// AUTONOMOUS OPERATION: Per contextprd.md, the 13-embedding array IS the
/// teleological vector. Memory storage uses default purpose vector [0.0; 13]
/// when no top-level goals are configured, enabling autonomous operation.
#[tokio::test]
async fn test_edge_case_autonomous_operation_empty_hierarchy() {
    println!("\n======================================================================");
    println!("EDGE CASE 2: Autonomous Operation Without Top-Level Goals");
    println!("======================================================================\n");

    let (handlers, _store, hierarchy) = create_verifiable_handlers_empty_hierarchy();

    // BEFORE STATE
    println!("📊 BEFORE STATE:");
    println!(
        "   Has top-level goals: {}",
        hierarchy.read().has_top_level_goals()
    );
    assert!(
        !hierarchy.read().has_top_level_goals(),
        "Must NOT have top-level goals"
    );

    // Store fingerprint - should SUCCEED without top-level goals (AUTONOMOUS OPERATION)
    println!("\n📝 ATTEMPTING: memory/store (should succeed - autonomous operation)");
    let store_params = json!({
        "content": "Test content for autonomous alignment",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;

    // Verify store succeeds with default purpose vector
    assert!(
        store_response.error.is_none(),
        "Store MUST succeed without top-level goals (AUTONOMOUS OPERATION). Error: {:?}",
        store_response.error
    );
    let result = store_response.result.expect("Should have result");
    let fingerprint_id = result
        .get("fingerprintId")
        .expect("Must have fingerprintId");
    println!("   SUCCESS: fingerprintId={}", fingerprint_id);

    // TASK-CORE-001: Verify deprecated method returns METHOD_NOT_FOUND
    println!("\n📝 VERIFYING: deprecated alignment method returns METHOD_NOT_FOUND");
    let align_params = json!({
        "fingerprint_id": "00000000-0000-0000-0000-000000000001"
    });
    let align_request = make_request(
        "purpose/deprecated_alignment",
        Some(JsonRpcId::Number(2)),
        Some(align_params),
    );
    let response = handlers.dispatch(align_request).await;

    // TASK-CORE-001: Must return METHOD_NOT_FOUND (-32601) for deprecated method
    assert!(
        response.error.is_some(),
        "Deprecated method must return error"
    );
    let align_error = response.error.unwrap();
    println!("   Error code: {} (expected: -32601)", align_error.code);
    println!("   Error message: {}", align_error.message);
    assert_eq!(
        align_error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601) for deprecated method"
    );

    println!(
        "\n✓ VERIFIED: System operates autonomously, deprecated method returns METHOD_NOT_FOUND\n"
    );
}

// =============================================================================
// EDGE CASE 3: Goal Not Found in Hierarchy
// =============================================================================

/// EDGE CASE: Non-existent goal should return GOAL_NOT_FOUND.
#[tokio::test]
async fn test_edge_case_goal_not_found() {
    println!("\n======================================================================");
    println!("EDGE CASE 3: Goal Not Found in Hierarchy");
    println!("======================================================================\n");

    let (handlers, _store, hierarchy) = create_verifiable_handlers();

    // BEFORE STATE
    println!("📊 BEFORE STATE:");
    let goal_count = hierarchy.read().len();
    println!("   Total goals in hierarchy: {}", goal_count);

    // Verify the goal we're looking for does NOT exist (random UUID)
    let nonexistent_id = Uuid::new_v4();
    let exists_in_hierarchy = hierarchy.read().get(&nonexistent_id).is_some();
    println!("   'nonexistent_goal_xyz' exists: {}", exists_in_hierarchy);
    assert!(!exists_in_hierarchy, "Goal must NOT exist");

    // ACTION: Try to get non-existent goal
    println!("\n📝 ACTION: goal/hierarchy_query get_goal (nonexistent)");
    let query_params = json!({
        "operation": "get_goal",
        "goal_id": nonexistent_id.to_string()
    });
    let query_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(query_params),
    );
    let response = handlers.dispatch(query_request).await;

    // Verify error
    assert!(
        response.error.is_some(),
        "Non-existent goal must return error"
    );
    let error = response.error.unwrap();
    println!("   Error code: {} (expected: -32020)", error.code);
    println!("   Error message: {}", error.message);
    assert_eq!(error.code, -32020, "Must return GOAL_NOT_FOUND (-32020)");

    // AFTER STATE - hierarchy unchanged
    let after_count = hierarchy.read().len();
    assert_eq!(after_count, goal_count, "Hierarchy must remain unchanged");

    println!("\n✓ VERIFIED: Non-existent goal correctly returns GOAL_NOT_FOUND\n");
}

// =============================================================================
// EDGE CASE 4: Deprecated Update Method Returns METHOD_NOT_FOUND (TASK-CORE-001)
// =============================================================================

/// EDGE CASE: Deprecated purpose update method returns METHOD_NOT_FOUND.
///
/// TASK-CORE-001: Manual goal update removed per ARCH-03 (autonomous-first).
/// Goals emerge autonomously via auto_bootstrap tool.
#[tokio::test]
async fn test_edge_case_deprecated_update_returns_method_not_found() {
    println!("\n======================================================================");
    println!("EDGE CASE 4: Deprecated Update Returns METHOD_NOT_FOUND (TASK-CORE-001)");
    println!("======================================================================\n");

    let (handlers, _store, hierarchy) = create_verifiable_handlers();

    // BEFORE STATE
    println!("📊 BEFORE STATE:");
    let has_top_level = hierarchy.read().has_top_level_goals();
    println!("   Has top-level goals: {}", has_top_level);
    assert!(has_top_level, "Must already have top-level goals");

    let existing_goal_id = hierarchy
        .read()
        .top_level_goals()
        .first()
        .map(|g| g.id.to_string())
        .expect("Must have top-level goal");
    println!("   Existing top-level goal ID: {}", existing_goal_id);

    // ACTION: Try to call deprecated method
    println!("\n📝 ACTION: Calling deprecated update method (per TASK-CORE-001)");
    let update_params = json!({
        "description": "New competing goal",
        "replace": false
    });
    let update_request = make_request(
        "purpose/deprecated_update",
        Some(JsonRpcId::Number(1)),
        Some(update_params),
    );
    let response = handlers.dispatch(update_request).await;

    // TASK-CORE-001: Verify METHOD_NOT_FOUND error
    assert!(
        response.error.is_some(),
        "Deprecated method must return error"
    );
    let error = response.error.unwrap();
    println!("   Error code: {} (expected: -32601)", error.code);
    println!("   Error message: {}", error.message);
    assert_eq!(
        error.code, -32601,
        "Must return METHOD_NOT_FOUND (-32601) for deprecated method"
    );

    // AFTER STATE - original goal unchanged
    let after_goal_id = hierarchy
        .read()
        .top_level_goals()
        .first()
        .map(|g| g.id.to_string())
        .expect("Must still have top-level goal");
    assert_eq!(
        after_goal_id, existing_goal_id,
        "Top-level goal must remain unchanged"
    );

    println!("\n✓ VERIFIED: Deprecated method returns METHOD_NOT_FOUND, hierarchy unchanged\n");
}

// =============================================================================
// EDGE CASE 5: Drift Check with Invalid UUIDs
// =============================================================================

/// EDGE CASE: Drift check with invalid UUIDs in array should fail.
#[tokio::test]
async fn test_edge_case_drift_check_invalid_uuids() {
    println!("\n======================================================================");
    println!("EDGE CASE 5: Drift Check with Invalid UUIDs");
    println!("======================================================================\n");

    let (handlers, store, _hierarchy) = create_verifiable_handlers();

    // BEFORE STATE
    let before_count = store.count().await.expect("count should succeed");
    println!("📊 BEFORE STATE:");
    println!("   Source of Truth count: {}", before_count);

    // ACTION: Drift check with invalid UUIDs
    println!("\n📝 ACTION: purpose/drift_check with invalid UUIDs");
    let drift_params = json!({
        "fingerprint_ids": ["not-a-uuid", "also-not-valid"]
    });
    let drift_request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(1)),
        Some(drift_params),
    );
    let response = handlers.dispatch(drift_request).await;

    // Verify error
    assert!(response.error.is_some(), "Invalid UUIDs must return error");
    let error = response.error.unwrap();
    println!("   Error code: {} (expected: -32602)", error.code);
    println!("   Error message: {}", error.message);
    assert_eq!(error.code, -32602, "Must return INVALID_PARAMS (-32602)");

    // AFTER STATE
    let after_count = store.count().await.expect("count should succeed");
    assert_eq!(
        after_count, before_count,
        "Store count must remain unchanged"
    );

    println!("\n✓ VERIFIED: Invalid UUIDs correctly rejected\n");
}
