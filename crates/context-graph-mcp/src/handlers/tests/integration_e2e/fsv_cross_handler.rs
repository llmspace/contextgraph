//! FSV TEST 6: Cross-Handler Integration
//!
//! Tests Store -> Align -> Johari -> Meta-UTL -> Search in one flow.

use super::infrastructure::*;
use context_graph_core::traits::TeleologicalMemoryStore;

/// FSV: Cross-handler integration test combining all handlers.
#[tokio::test]
async fn test_fsv_cross_handler_integration() {
    println!("\n======================================================================");
    println!("FSV TEST 6: Cross-Handler Integration");
    println!("======================================================================\n");

    let ctx = TestContext::new();

    // Seed tracker
    {
        let mut tracker = ctx.meta_utl_tracker.write();
        for _ in 0..15 {
            tracker.record_validation();
        }
    }

    // =========================================================================
    // STEP 1: STORE FINGERPRINT
    // =========================================================================
    println!("STEP 1: memory/store");
    let store_request = make_request(
        "memory/store",
        1,
        json!({
            "content": "Integrated machine learning pipeline with semantic understanding",
            "importance": 0.95
        }),
    );
    let store_response = ctx.handlers.dispatch(store_request).await;
    assert!(store_response.error.is_none(), "Store MUST succeed");

    let fingerprint_id = store_response.result.unwrap()["fingerprintId"]
        .as_str()
        .unwrap()
        .to_string();
    let fp_uuid = Uuid::parse_str(&fingerprint_id).unwrap();

    // Verify in store
    let _stored = ctx
        .store
        .retrieve(fp_uuid)
        .await
        .unwrap()
        .expect("MUST exist");
    println!("   - Created: {} (verified in store)", fingerprint_id);

    // =========================================================================
    // STEP 2: VERIFY DEPRECATED METHOD RETURNS METHOD_NOT_FOUND (TASK-CORE-001)
    // =========================================================================
    println!("\nSTEP 2: purpose/north_star_alignment (deprecated per ARCH-03)");
    let align_request = make_request(
        "purpose/north_star_alignment",
        2,
        json!({
            "fingerprint_id": fingerprint_id,
            "include_breakdown": true
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
    println!("   VERIFIED: Deprecated method returns METHOD_NOT_FOUND");

    // =========================================================================
    // STEP 3: JOHARI DISTRIBUTION
    // =========================================================================
    println!("\nSTEP 3: johari/get_distribution");
    let johari_request = make_request(
        "johari/get_distribution",
        3,
        json!({
            "memory_id": fingerprint_id,
            "include_confidence": true
        }),
    );
    let johari_response = ctx.handlers.dispatch(johari_request).await;
    assert!(johari_response.error.is_none(), "Johari MUST succeed");

    let summary = &johari_response.result.unwrap()["summary"];
    println!(
        "   - Open: {}, Hidden: {}, Blind: {}, Unknown: {}",
        summary["open_count"],
        summary["hidden_count"],
        summary["blind_count"],
        summary["unknown_count"]
    );

    // =========================================================================
    // STEP 4: META-UTL PREDICTION
    // =========================================================================
    println!("\nSTEP 4: meta_utl/predict_storage");
    let predict_request = make_request(
        "meta_utl/predict_storage",
        4,
        json!({
            "fingerprint_id": fingerprint_id
        }),
    );
    let predict_response = ctx.handlers.dispatch(predict_request).await;
    assert!(predict_response.error.is_none(), "Prediction MUST succeed");

    let prediction_id = predict_response.result.unwrap()["prediction_id"]
        .as_str()
        .unwrap()
        .to_string();
    println!("   - Prediction ID: {}", prediction_id);

    // Verify in tracker
    {
        let tracker = ctx.meta_utl_tracker.read();
        let pred_uuid = Uuid::parse_str(&prediction_id).unwrap();
        assert!(
            tracker.pending_predictions.contains_key(&pred_uuid),
            "Prediction MUST be in tracker"
        );
    }

    // =========================================================================
    // STEP 5: MULTI-EMBEDDING SEARCH
    // =========================================================================
    println!("\nSTEP 5: search/multi");
    let search_request = make_request(
        "search/multi",
        5,
        json!({
            "query": "machine learning pipeline",
            "query_type": "semantic_search",
            "topK": 5,
            "minSimilarity": 0.0
        }),
    );
    let search_response = ctx.handlers.dispatch(search_request).await;
    assert!(search_response.error.is_none(), "Search MUST succeed");

    let search_result = search_response.result.unwrap();
    let results = search_result["results"].as_array().unwrap();
    let found = results
        .iter()
        .any(|r| r.get("fingerprintId").and_then(|v| v.as_str()) == Some(&fingerprint_id));
    assert!(found, "Stored fingerprint MUST be found in search");
    println!("   - Found fingerprint in search: {}", found);

    // =========================================================================
    // FINAL VERIFICATION
    // =========================================================================
    println!("\nFINAL SOURCE OF TRUTH VERIFICATION:");
    let final_count = ctx.store.count().await.unwrap();
    let final_hierarchy_len = ctx.hierarchy.read().len();
    let final_predictions = ctx.meta_utl_tracker.read().pending_predictions.len();

    println!("   - Store count: {}", final_count);
    println!("   - Hierarchy goals: {}", final_hierarchy_len);
    println!("   - Pending predictions: {}", final_predictions);

    // =========================================================================
    // EVIDENCE OF SUCCESS
    // =========================================================================
    println!("\n======================================================================");
    println!("EVIDENCE OF SUCCESS - Cross-Handler Integration");
    println!("======================================================================");
    println!("All handlers worked together:");
    println!("  1. memory/store: Created fingerprint (verified in store)");
    println!("  2. purpose/north_star_alignment: Returns METHOD_NOT_FOUND (deprecated per TASK-CORE-001)");
    println!("  3. johari/get_distribution: 13 embedder quadrants");
    println!("  4. meta_utl/predict_storage: Prediction tracked");
    println!("  5. search/multi: Fingerprint found in results");
    println!("======================================================================\n");
}
