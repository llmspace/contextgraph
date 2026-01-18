//! Full State Verification Test
//!
//! Complete purpose workflow end-to-end test.

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::super::{create_test_handlers, make_request};

/// FULL STATE VERIFICATION: Complete purpose workflow test.
///
/// TASK-CORE-001: Updated to remove deprecated alignment step.
/// TASK-P0-001: Updated for 3-level hierarchy (has_top_level_goals).
///
/// Tests the full purpose lifecycle with real data:
/// 1. Create handlers with test hierarchy
/// 2. Verify hierarchy exists via goal/hierarchy_query get_all
/// 3. Store content and verify storage
/// 4. Query via purpose/query with 13D vector
/// 5. Find aligned memories via goal/aligned_memories
/// 6. Check drift via purpose/drift_check
///
/// NOTE: Purpose alignment removed per ARCH-03.
/// Uses real GoalHierarchy with STUB storage (InMemoryTeleologicalStore).
#[tokio::test]
async fn test_full_state_verification_purpose_workflow() {
    let handlers = create_test_handlers();

    // =========================================================================
    // STEP 1: Verify hierarchy exists via goal/hierarchy_query get_all
    // =========================================================================
    let hierarchy_request = make_request(
        "goal/hierarchy_query",
        Some(JsonRpcId::Number(1)),
        Some(json!({ "operation": "get_all" })),
    );
    let hierarchy_response = handlers.dispatch(hierarchy_request).await;

    assert!(
        hierarchy_response.error.is_none(),
        "goal/hierarchy_query get_all must succeed"
    );
    let hierarchy_result = hierarchy_response.result.expect("Must have result");

    let stats = hierarchy_result
        .get("hierarchy_stats")
        .expect("Must have stats");
    // TASK-P0-001: Check has_top_level_goals
    assert_eq!(
        stats.get("has_top_level_goals").and_then(|v| v.as_bool()),
        Some(true),
        "Must have top-level Strategic goals configured"
    );

    let goals = hierarchy_result
        .get("goals")
        .and_then(|v| v.as_array())
        .expect("Must have goals");
    assert!(!goals.is_empty(), "Must have at least one goal");

    println!(
        "[FSV] STEP 1 VERIFIED: Hierarchy has {} goals with top-level Strategic",
        goals.len()
    );

    // =========================================================================
    // STEP 2: Store multiple fingerprints with different content
    // =========================================================================
    let contents = [
        "Machine learning algorithms for predictive analytics",
        "Improving retrieval accuracy through semantic embeddings",
        "Knowledge graph construction for entity linking",
    ];

    let mut stored_ids: Vec<String> = Vec::new();
    for (i, content) in contents.iter().enumerate() {
        let store_params = json!({
            "content": content,
            "importance": 0.7 + (i as f64 * 0.1)
        });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(10 + i as i64)),
            Some(store_params),
        );
        let store_response = handlers.dispatch(store_request).await;
        assert!(store_response.error.is_none(), "Store {} must succeed", i);

        let fp_id = store_response
            .result
            .unwrap()
            .get("fingerprintId")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        stored_ids.push(fp_id);
    }

    assert_eq!(stored_ids.len(), 3, "Must have stored 3 fingerprints");
    println!(
        "[FSV] STEP 2 VERIFIED: Stored {} fingerprints",
        stored_ids.len()
    );

    // =========================================================================
    // STEP 3: Query via purpose/query with 13D vector
    // =========================================================================
    let purpose_vector: Vec<f64> = vec![
        0.8, 0.5, 0.3, 0.3, 0.6, 0.2, 0.7, 0.5, 0.4, 0.3, 0.5, 0.3, 0.2,
    ];

    let purpose_query_params = json!({
        "purpose_vector": purpose_vector,
        "topK": 10,
        "include_scores": true
    });
    let purpose_query_request = make_request(
        "purpose/query",
        Some(JsonRpcId::Number(20)),
        Some(purpose_query_params),
    );
    let purpose_query_response = handlers.dispatch(purpose_query_request).await;

    assert!(
        purpose_query_response.error.is_none(),
        "purpose/query must succeed"
    );
    let purpose_result = purpose_query_response.result.expect("Must have result");
    let purpose_results = purpose_result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Must have results");

    println!(
        "[FSV] STEP 3 VERIFIED: purpose/query returned {} results",
        purpose_results.len()
    );

    // NOTE: Legacy alignment step REMOVED per TASK-CORE-001 (ARCH-03)
    // Goals now emerge autonomously from topic clustering.

    // =========================================================================
    // STEP 4: Find aligned memories via goal/aligned_memories
    // =========================================================================
    // Extract a Strategic goal UUID from the hierarchy for aligned_memories query
    let strategic_goal_id = goals
        .iter()
        .find(|g| g.get("level").and_then(|v| v.as_str()) == Some("Strategic"))
        .and_then(|g| g.get("id").and_then(|v| v.as_str()))
        .expect("Must have Strategic goal with id");

    let aligned_params = json!({
        "goal_id": strategic_goal_id,
        "topK": 10,
        "minAlignment": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let aligned_request = make_request(
        "goal/aligned_memories",
        Some(JsonRpcId::Number(40)),
        Some(aligned_params),
    );
    let aligned_response = handlers.dispatch(aligned_request).await;

    assert!(
        aligned_response.error.is_none(),
        "goal/aligned_memories must succeed"
    );
    let aligned_result = aligned_response.result.expect("Must have result");
    let aligned_count = aligned_result
        .get("count")
        .and_then(|v| v.as_u64())
        .expect("Must have count");

    println!(
        "[FSV] STEP 4 VERIFIED: goal/aligned_memories found {} memories",
        aligned_count
    );

    // =========================================================================
    // STEP 5: Check drift via purpose/drift_check
    // =========================================================================
    let drift_params = json!({
        "fingerprint_ids": &stored_ids,
        "threshold": 0.1
    });
    let drift_request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(50)),
        Some(drift_params),
    );
    let drift_response = handlers.dispatch(drift_request).await;

    assert!(
        drift_response.error.is_none(),
        "purpose/drift_check must succeed"
    );
    let drift_result = drift_response.result.expect("Must have result");

    // Response has: overall_drift, per_embedder_drift, analyzed_count, etc.
    let analyzed_count = drift_result
        .get("analyzed_count")
        .and_then(|v| v.as_u64())
        .expect("Must have analyzed_count");

    let overall_drift = drift_result
        .get("overall_drift")
        .expect("Must have overall_drift");
    let has_drifted = overall_drift
        .get("has_drifted")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    assert_eq!(analyzed_count, 3, "Must check all 3 fingerprints");

    println!(
        "[FSV] STEP 5 VERIFIED: drift_check analyzed {} fingerprints, drifted: {}",
        analyzed_count, has_drifted
    );

    // =========================================================================
    // VERIFICATION COMPLETE
    // =========================================================================
    println!("\n======================================================================");
    println!("[FSV] FULL STATE VERIFICATION COMPLETE - All purpose handlers working");
    println!("======================================================================\n");
}
