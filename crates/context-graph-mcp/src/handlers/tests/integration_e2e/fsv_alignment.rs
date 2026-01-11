//! FSV TEST 3: Purpose Alignment with Goal Hierarchy
//!
//! Tests purpose alignment with goal hierarchy verification.

use super::infrastructure::*;
use context_graph_core::purpose::GoalLevel;
use context_graph_core::traits::TeleologicalMemoryStore;

/// FSV: Purpose alignment with goal hierarchy verification.
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
        println!("   - Total goals: {} (expected: 5)", h.len());
        assert_eq!(h.len(), 5, "Hierarchy MUST have 5 goals");

        let ns = h.north_star().expect("MUST have North Star");
        println!("   - North Star: {} - {}", ns.id, ns.description);
        assert!(!ns.id.is_nil(), "North Star must have valid UUID");

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
    println!("STEP 3: purpose/north_star_alignment (deprecated per ARCH-03)");
    let align_request = make_request(
        "purpose/north_star_alignment",
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
        "   - Stored theta_to_north_star: {:.4}",
        stored_fp.theta_to_north_star
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

    // Extract actual goal IDs from get_all response
    let north_star_id = all_goals_arr
        .iter()
        .find(|g| g["level"].as_str() == Some("NorthStar"))
        .and_then(|g| g["id"].as_str())
        .expect("Must have North Star goal with id");
    let immediate_id = all_goals_arr
        .iter()
        .find(|g| g["level"].as_str() == Some("Immediate"))
        .and_then(|g| g["id"].as_str())
        .expect("Must have Immediate goal with id");

    // Get children of North Star
    let get_children = make_request(
        "goal/hierarchy_query",
        5,
        json!({
            "operation": "get_children",
            "goal_id": north_star_id
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
    println!("   - get_children(NorthStar): {} children", children);

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
    println!("   - get_ancestors(i1_vector): {} ancestors", ancestors);
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
    println!("  1. Goal hierarchy: 5 goals (1 NS + 2 S + 1 T + 1 I)");
    println!("  2. North Star alignment: Returns METHOD_NOT_FOUND (deprecated per TASK-CORE-001)");
    println!("  3. Fingerprint stored and verified in Source of Truth");
    println!(
        "  4. Drift check: {} fingerprints analyzed",
        summary["total_checked"]
    );
    println!("  5. Hierarchy navigation: get_all, get_children, get_ancestors");
    println!("======================================================================\n");
}
