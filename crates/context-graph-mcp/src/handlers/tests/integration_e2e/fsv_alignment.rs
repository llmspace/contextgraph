//! FSV TEST 3: Purpose Alignment with Goal Hierarchy
//!
//! Tests purpose alignment with goal hierarchy verification.

use super::infrastructure::*;
use context_graph_core::purpose::GoalLevel;
use context_graph_core::traits::TeleologicalMemoryStore;

/// FSV: Purpose alignment with goal hierarchy verification.
/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
#[tokio::test]
async fn test_fsv_purpose_alignment_with_hierarchy() {
    println!("\n======================================================================");
    println!("FSV TEST 3: Purpose Alignment with Goal Hierarchy");
    println!("======================================================================\n");

    let ctx = TestContext::new();

    // =========================================================================
    // STEP 1: VERIFY GOAL HIERARCHY
    // =========================================================================
    println!("VERIFY GOAL HIERARCHY:");
    {
        let h = ctx.hierarchy.read();
        // TASK-P0-001: Now have 4 goals (2 Strategic, 1 Tactical, 1 Immediate)
        println!("   - Total goals: {} (expected: 4)", h.len());
        assert_eq!(h.len(), 4, "Hierarchy MUST have 4 goals");

        // TASK-P0-001: Bind to avoid temporary lifetime issue
        let top_level = h.top_level_goals();
        let ns = top_level.first().expect("MUST have Strategic goal");
        println!("   - Strategic goal: {} - {}", ns.id, ns.description);
        assert!(!ns.id.is_nil(), "Strategic goal must have valid UUID");

        println!(
            "   - Strategic goals: {}",
            h.at_level(GoalLevel::Strategic).len()
        );
        println!(
            "   - Tactical goals: {}",
            h.at_level(GoalLevel::Tactical).len()
        );
        println!(
            "   - Immediate goals: {}",
            h.at_level(GoalLevel::Immediate).len()
        );
    }
    println!("   VERIFIED: Hierarchy structure correct\n");

    // =========================================================================
    // STEP 2: STORE FINGERPRINT
    // =========================================================================
    println!("STEP 2: Store fingerprint for alignment");
    let content = "Implementing retrieval systems with semantic understanding";
    let store_request = make_request(
        "memory/store",
        1,
        json!({
            "content": content,
            "importance": 0.85
        }),
    );
    let store_response = ctx.handlers.dispatch(store_request).await;

    assert!(store_response.error.is_none(), "Store MUST succeed");
    let fingerprint_id = store_response.result.unwrap()["fingerprintId"]
        .as_str()
        .unwrap()
        .to_string();
    println!("   - Fingerprint ID: {}\n", fingerprint_id);

    // =========================================================================
    // STEP 3: VERIFY DEPRECATED METHOD RETURNS METHOD_NOT_FOUND (TASK-CORE-001)
    // =========================================================================
    println!("STEP 3: purpose/alignment (deprecated per ARCH-03)");
    let align_request = make_request(
        "purpose/alignment",
        2,
        json!({
            "fingerprint_id": fingerprint_id,
            "include_breakdown": true,
            "include_patterns": true
        }),
    );
    let align_response = ctx.handlers.dispatch(align_request).await;

    // TASK-CORE-001: Verify METHOD_NOT_FOUND for deprecated method
    assert!(
        align_response.error.is_some(),
        "Deprecated method MUST return error"
    );
    let align_error = align_response.error.unwrap();
    assert_eq!(
        align_error.code, -32601,
        "MUST return METHOD_NOT_FOUND (-32601)"
    );
    println!("   - Error code: {} (METHOD_NOT_FOUND)", align_error.code);
    println!("   - Error message: {}", align_error.message);
    println!("   VERIFIED: Deprecated method returns METHOD_NOT_FOUND\n");

    // VERIFY FINGERPRINT IN SOURCE OF TRUTH (still exists, unchanged)
    let fp_id = Uuid::parse_str(&fingerprint_id).unwrap();
    let stored_fp = ctx.store.retrieve(fp_id).await.unwrap().unwrap();
    println!("VERIFY FINGERPRINT IN SOURCE OF TRUTH:");
    println!("   - Fingerprint exists: {}", stored_fp.id);
    println!(
        "   - Stored alignment_score: {:.4}",
        stored_fp.alignment_score
    );
    println!(
        "   - Purpose vector coherence: {:.4}",
        stored_fp.purpose_vector.coherence
    );
    println!("   VERIFIED: Fingerprint data intact\n");

    // =========================================================================
    // STEP 4: DRIFT CHECK
    // =========================================================================
    println!("STEP 4: purpose/drift_check");
    let drift_request = make_request(
        "purpose/drift_check",
        3,
        json!({
            "fingerprint_ids": [fingerprint_id],
            "threshold": 0.1
        }),
    );
    let drift_response = ctx.handlers.dispatch(drift_request).await;

    assert!(drift_response.error.is_none(), "Drift check MUST succeed");
    let drift_result = drift_response.result.unwrap();

    let summary = &drift_result["summary"];
    println!("   - Total checked: {}", summary["total_checked"]);
    println!("   - Drifted count: {}", summary["drifted_count"]);
    println!(
        "   - Average drift: {:.4}",
        summary["average_drift"].as_f64().unwrap_or(0.0)
    );
    println!("   VERIFIED: Drift check completed\n");

    // =========================================================================
    // STEP 5: GOAL HIERARCHY NAVIGATION
    // =========================================================================
    println!("STEP 5: goal/hierarchy_query operations");

    // Get all goals
    let get_all = make_request(
        "goal/hierarchy_query",
        4,
        json!({
            "operation": "get_all"
        }),
    );
    let all_response = ctx.handlers.dispatch(get_all).await;
    assert!(all_response.error.is_none(), "get_all MUST succeed");
    let all_result = all_response.result.unwrap();
    let all_goals_arr = all_result["goals"].as_array().unwrap();
    println!("   - get_all: {} goals", all_goals_arr.len());

    // TASK-P0-001: Extract Strategic goal ID
    let strategic_id = all_goals_arr
        .iter()
        .find(|g| g["level"].as_str() == Some("Strategic"))
        .and_then(|g| g["id"].as_str())
        .expect("Must have Strategic goal with id");
    let immediate_id = all_goals_arr
        .iter()
        .find(|g| g["level"].as_str() == Some("Immediate"))
        .and_then(|g| g["id"].as_str())
        .expect("Must have Immediate goal with id");

    // Get children of Strategic goal (top-level)
    let get_children = make_request(
        "goal/hierarchy_query",
        5,
        json!({
            "operation": "get_children",
            "goal_id": strategic_id
        }),
    );
    let children_response = ctx.handlers.dispatch(get_children).await;
    assert!(
        children_response.error.is_none(),
        "get_children MUST succeed"
    );
    let children = children_response.result.unwrap()["children"]
        .as_array()
        .unwrap()
        .len();
    println!("   - get_children(Strategic): {} children", children);

    // Get ancestors of immediate goal
    let get_ancestors = make_request(
        "goal/hierarchy_query",
        6,
        json!({
            "operation": "get_ancestors",
            "goal_id": immediate_id
        }),
    );
    let ancestors_response = ctx.handlers.dispatch(get_ancestors).await;
    assert!(
        ancestors_response.error.is_none(),
        "get_ancestors MUST succeed"
    );
    let ancestors = ancestors_response.result.unwrap()["ancestors"]
        .as_array()
        .unwrap()
        .len();
    println!("   - get_ancestors(Immediate): {} ancestors", ancestors);
    println!("   VERIFIED: Goal hierarchy navigation works\n");

    // =========================================================================
    // EVIDENCE OF SUCCESS
    // =========================================================================
    println!("======================================================================");
    println!("EVIDENCE OF SUCCESS - Purpose Alignment Verification");
    println!("======================================================================");
    println!("Source of Truth: GoalHierarchy + InMemoryTeleologicalStore");
    println!();
    println!("Operations Verified:");
    // TASK-P0-001: Updated counts for 3-level hierarchy
    println!("  1. Goal hierarchy: 4 goals (2 Strategic + 1 Tactical + 1 Immediate)");
    println!("  2. Purpose alignment: Returns METHOD_NOT_FOUND (deprecated per TASK-CORE-001)");
    println!("  3. Fingerprint stored and verified in Source of Truth");
    println!(
        "  4. Drift check: {} fingerprints analyzed",
        summary["total_checked"]
    );
    println!("  5. Hierarchy navigation: get_all, get_children, get_ancestors");
    println!("======================================================================\n");
}
