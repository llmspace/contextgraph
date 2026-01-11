//! FSV TEST 4: Johari Quadrant Operations
//!
//! Tests Johari quadrant distribution, transitions, and batch operations.

use super::infrastructure::*;
use context_graph_core::traits::TeleologicalMemoryStore;

/// FSV: Johari quadrant distribution, transitions, and batch operations.
#[tokio::test]
async fn test_fsv_johari_quadrant_operations() {
    println!("\n======================================================================");
    println!("FSV TEST 4: Johari Quadrant Operations");
    println!("======================================================================\n");

    let ctx = TestContext::new();

    // =========================================================================
    // STEP 1: CREATE FINGERPRINT WITH SPECIFIC JOHARI CONFIGURATION
    // =========================================================================
    println!("STEP 1: Create fingerprint with specific Johari configuration");
    let quadrants = [
        JohariQuadrant::Open,    // E1
        JohariQuadrant::Open,    // E2
        JohariQuadrant::Hidden,  // E3
        JohariQuadrant::Hidden,  // E4
        JohariQuadrant::Blind,   // E5
        JohariQuadrant::Blind,   // E6
        JohariQuadrant::Unknown, // E7
        JohariQuadrant::Unknown, // E8
        JohariQuadrant::Unknown, // E9
        JohariQuadrant::Unknown, // E10
        JohariQuadrant::Unknown, // E11
        JohariQuadrant::Unknown, // E12
        JohariQuadrant::Unknown, // E13
    ];

    let fp = create_fingerprint_with_johari(quadrants);
    let memory_id = ctx.store.store(fp).await.expect("Store should succeed");
    println!("   - Memory ID: {}", memory_id);
    println!("   - Configuration: 2 Open, 2 Hidden, 2 Blind, 7 Unknown\n");

    // =========================================================================
    // STEP 2: GET DISTRIBUTION
    // =========================================================================
    println!("STEP 2: johari/get_distribution");
    let dist_request = make_request(
        "johari/get_distribution",
        1,
        json!({
            "memory_id": memory_id.to_string(),
            "include_confidence": true
        }),
    );
    let dist_response = ctx.handlers.dispatch(dist_request).await;

    assert!(
        dist_response.error.is_none(),
        "Distribution MUST succeed: {:?}",
        dist_response.error
    );
    let dist_result = dist_response.result.unwrap();

    let summary = &dist_result["summary"];
    let open_count = summary["open_count"].as_u64().unwrap();
    let hidden_count = summary["hidden_count"].as_u64().unwrap();
    let blind_count = summary["blind_count"].as_u64().unwrap();
    let unknown_count = summary["unknown_count"].as_u64().unwrap();

    println!(
        "   Distribution: {} Open, {} Hidden, {} Blind, {} Unknown",
        open_count, hidden_count, blind_count, unknown_count
    );

    assert_eq!(open_count, 2, "MUST have 2 Open");
    assert_eq!(hidden_count, 2, "MUST have 2 Hidden");
    assert_eq!(blind_count, 2, "MUST have 2 Blind");
    assert_eq!(unknown_count, 7, "MUST have 7 Unknown");

    // Verify 13 per-embedder quadrants returned
    let per_embedder = dist_result["per_embedder_quadrants"].as_array().unwrap();
    assert_eq!(
        per_embedder.len(),
        NUM_EMBEDDERS,
        "MUST return 13 embedder quadrants"
    );
    println!("   VERIFIED: Distribution matches configuration\n");

    // =========================================================================
    // STEP 3: SINGLE TRANSITION (Unknown -> Open)
    // =========================================================================
    println!("STEP 3: johari/transition (E7 Unknown -> Open)");

    // BEFORE STATE
    let before = ctx.store.retrieve(memory_id).await.unwrap().unwrap();
    println!("   BEFORE: E7 = {:?}", before.johari.dominant_quadrant(6));
    assert_eq!(before.johari.dominant_quadrant(6), JohariQuadrant::Unknown);

    let transition_request = make_request(
        "johari/transition",
        2,
        json!({
            "memory_id": memory_id.to_string(),
            "embedder_index": 6,
            "to_quadrant": "open",
            "trigger": "dream_consolidation"
        }),
    );
    let transition_response = ctx.handlers.dispatch(transition_request).await;

    assert!(
        transition_response.error.is_none(),
        "Transition MUST succeed: {:?}",
        transition_response.error
    );
    let trans_result = transition_response.result.unwrap();
    println!("   - from_quadrant: {}", trans_result["from_quadrant"]);
    println!("   - to_quadrant: {}", trans_result["to_quadrant"]);
    println!("   - success: {}", trans_result["success"]);

    // VERIFY IN SOURCE OF TRUTH
    let after = ctx.store.retrieve(memory_id).await.unwrap().unwrap();
    println!("   AFTER: E7 = {:?}", after.johari.dominant_quadrant(6));
    assert_eq!(
        after.johari.dominant_quadrant(6),
        JohariQuadrant::Open,
        "Transition MUST persist to store"
    );
    println!("   VERIFIED: Transition persisted to Source of Truth\n");

    // =========================================================================
    // STEP 4: BATCH TRANSITION
    // =========================================================================
    println!("STEP 4: johari/transition_batch (E8, E9 Unknown -> Hidden)");

    let batch_request = make_request(
        "johari/transition_batch",
        3,
        json!({
            "memory_id": memory_id.to_string(),
            "transitions": [
                { "embedder_index": 7, "to_quadrant": "hidden", "trigger": "dream_consolidation" },
                { "embedder_index": 8, "to_quadrant": "hidden", "trigger": "dream_consolidation" }
            ]
        }),
    );
    let batch_response = ctx.handlers.dispatch(batch_request).await;

    assert!(
        batch_response.error.is_none(),
        "Batch MUST succeed: {:?}",
        batch_response.error
    );
    let batch_result = batch_response.result.unwrap();
    println!(
        "   - transitions_applied: {}",
        batch_result["transitions_applied"]
    );

    // VERIFY IN SOURCE OF TRUTH
    let after_batch = ctx.store.retrieve(memory_id).await.unwrap().unwrap();
    assert_eq!(
        after_batch.johari.dominant_quadrant(7),
        JohariQuadrant::Hidden
    );
    assert_eq!(
        after_batch.johari.dominant_quadrant(8),
        JohariQuadrant::Hidden
    );
    println!("   VERIFIED: Batch transitions persisted\n");

    // =========================================================================
    // STEP 5: CROSS-SPACE ANALYSIS
    // =========================================================================
    println!("STEP 5: johari/cross_space_analysis");
    let analysis_request = make_request(
        "johari/cross_space_analysis",
        4,
        json!({
            "memory_ids": [memory_id.to_string()],
            "analysis_type": "blind_spots"
        }),
    );
    let analysis_response = ctx.handlers.dispatch(analysis_request).await;

    assert!(analysis_response.error.is_none(), "Analysis MUST succeed");
    let analysis_result = analysis_response.result.unwrap();

    let blind_spots = analysis_result["blind_spots"].as_array().unwrap().len();
    let opportunities = analysis_result["learning_opportunities"]
        .as_array()
        .unwrap()
        .len();
    println!("   - Blind spots: {}", blind_spots);
    println!("   - Learning opportunities: {}", opportunities);
    println!("   VERIFIED: Cross-space analysis completed\n");

    // =========================================================================
    // EVIDENCE OF SUCCESS
    // =========================================================================
    println!("======================================================================");
    println!("EVIDENCE OF SUCCESS - Johari Quadrant Verification");
    println!("======================================================================");
    println!("Source of Truth: InMemoryTeleologicalStore");
    println!();
    println!("Operations Verified:");
    println!("  1. get_distribution: 13 embedder quadrants returned");
    println!("  2. transition: E7 Unknown -> Open (persisted)");
    println!("  3. transition_batch: E8, E9 Unknown -> Hidden (persisted)");
    println!(
        "  4. cross_space_analysis: {} blind spots, {} opportunities",
        blind_spots, opportunities
    );
    println!();
    println!("Physical Evidence:");
    println!("  - Memory ID: {}", memory_id);
    println!("  - All transitions verified in store via retrieve()");
    println!("======================================================================\n");
}
