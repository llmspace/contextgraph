//! FSV TEST 5: Meta-UTL Prediction and Validation Cycle
//!
//! Tests Meta-UTL prediction creation, validation, and learning trajectory.

use super::infrastructure::*;

/// FSV: Meta-UTL prediction creation, validation, and learning trajectory.
#[tokio::test]
async fn test_fsv_meta_utl_prediction_validation_cycle() {
    println!("\n======================================================================");
    println!("FSV TEST 5: Meta-UTL Prediction and Validation Cycle");
    println!("======================================================================\n");

    let ctx = TestContext::new();

    // =========================================================================
    // STEP 1: SEED TRACKER WITH VALIDATIONS
    // =========================================================================
    println!("STEP 1: Seed MetaUtlTracker with validation history");
    {
        let mut tracker = ctx.meta_utl_tracker.write();
        for _ in 0..20 {
            tracker.record_validation();
            for i in 0..NUM_EMBEDDERS {
                tracker.record_accuracy(i, 0.82 + (i as f32 * 0.01));
            }
        }
    }

    // VERIFY IN SOURCE OF TRUTH
    {
        let tracker = ctx.meta_utl_tracker.read();
        println!("   - validation_count: {}", tracker.validation_count);
        println!(
            "   - current_weights sum: {:.4}",
            tracker.current_weights.iter().sum::<f32>()
        );
    }
    println!("   VERIFIED: Tracker seeded\n");

    // =========================================================================
    // STEP 2: STORE FINGERPRINT FOR PREDICTION
    // =========================================================================
    println!("STEP 2: Store fingerprint for prediction");
    let store_request = make_request(
        "memory/store",
        1,
        json!({
            "content": "Neural network training optimization techniques",
            "importance": 0.9
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
    // STEP 3: LEARNING TRAJECTORY
    // =========================================================================
    println!("STEP 3: meta_utl/learning_trajectory");
    let trajectory_request = make_request(
        "meta_utl/learning_trajectory",
        2,
        json!({
            "include_accuracy_trend": true
        }),
    );
    let trajectory_response = ctx.handlers.dispatch(trajectory_request).await;

    assert!(
        trajectory_response.error.is_none(),
        "Trajectory MUST succeed"
    );
    let trajectory_result = trajectory_response.result.unwrap();

    let trajectories = trajectory_result["trajectories"].as_array().unwrap();
    assert_eq!(
        trajectories.len(),
        NUM_EMBEDDERS,
        "MUST return 13 trajectories"
    );
    println!(
        "   - Trajectories: {} (all 13 embedders)",
        trajectories.len()
    );

    let summary = &trajectory_result["system_summary"];
    println!("   - Overall accuracy: {}", summary["overall_accuracy"]);
    println!("   - Best performer: {}", summary["best_performing_space"]);
    println!("   VERIFIED: Learning trajectory complete\n");

    // =========================================================================
    // STEP 4: PREDICT STORAGE
    // =========================================================================
    println!("STEP 4: meta_utl/predict_storage");
    let predict_request = make_request(
        "meta_utl/predict_storage",
        3,
        json!({
            "fingerprint_id": fingerprint_id,
            "include_confidence": true
        }),
    );
    let predict_response = ctx.handlers.dispatch(predict_request).await;

    assert!(
        predict_response.error.is_none(),
        "Prediction MUST succeed: {:?}",
        predict_response.error
    );
    let predict_result = predict_response.result.unwrap();

    let prediction_id = predict_result["prediction_id"]
        .as_str()
        .unwrap()
        .to_string();
    let confidence = predict_result["confidence"].as_f64().unwrap_or(0.0);
    println!("   - Prediction ID: {}", prediction_id);
    println!(
        "   - Coherence delta: {}",
        predict_result["predictions"]["coherence_delta"]
    );
    println!("   - Confidence: {:.4}", confidence);

    // VERIFY IN SOURCE OF TRUTH
    {
        let tracker = ctx.meta_utl_tracker.read();
        let pred_uuid = Uuid::parse_str(&prediction_id).unwrap();
        let exists = tracker.pending_predictions.contains_key(&pred_uuid);
        println!("   - Prediction in tracker: {}", exists);
        assert!(exists, "Prediction MUST be in MetaUtlTracker");
    }
    println!("   VERIFIED: Prediction stored in tracker\n");

    // =========================================================================
    // STEP 5: VALIDATE PREDICTION
    // =========================================================================
    println!("STEP 5: meta_utl/validate_prediction");
    let validation_count_before = ctx.meta_utl_tracker.read().validation_count;

    let validate_request = make_request(
        "meta_utl/validate_prediction",
        4,
        json!({
            "prediction_id": prediction_id,
            "actual_outcome": {
                "coherence_delta": 0.015,
                "alignment_delta": 0.045
            }
        }),
    );
    let validate_response = ctx.handlers.dispatch(validate_request).await;

    assert!(
        validate_response.error.is_none(),
        "Validation MUST succeed: {:?}",
        validate_response.error
    );
    let validate_result = validate_response.result.unwrap();

    let validation = &validate_result["validation"];
    println!("   - Prediction error: {}", validation["prediction_error"]);
    println!("   - Accuracy score: {}", validation["accuracy_score"]);

    // VERIFY IN SOURCE OF TRUTH
    {
        let tracker = ctx.meta_utl_tracker.read();
        let pred_uuid = Uuid::parse_str(&prediction_id).unwrap();
        let removed = !tracker.pending_predictions.contains_key(&pred_uuid);
        println!("   - Prediction removed from tracker: {}", removed);
        assert!(removed, "Prediction MUST be removed after validation");

        println!("   - validation_count before: {}", validation_count_before);
        println!("   - validation_count after: {}", tracker.validation_count);
        assert!(
            tracker.validation_count > validation_count_before,
            "validation_count MUST increase"
        );
    }
    println!("   VERIFIED: Validation processed correctly\n");

    // =========================================================================
    // STEP 6: HEALTH METRICS (Verifies fail-fast behavior)
    // =========================================================================
    // TASK-EMB-024: StubSystemMonitor intentionally fails with NotImplemented.
    // This is CORRECT behavior - no fake/simulated metrics allowed.
    println!("STEP 6: meta_utl/health_metrics (verify fail-fast)");
    let health_request = make_request(
        "meta_utl/health_metrics",
        5,
        json!({
            "include_targets": true,
            "include_recommendations": true
        }),
    );
    let health_response = ctx.handlers.dispatch(health_request).await;

    // VERIFY FAIL-FAST BEHAVIOR
    assert!(
        health_response.error.is_some(),
        "health_metrics MUST fail when using StubSystemMonitor (TASK-EMB-024)"
    );
    let health_error = health_response.error.as_ref().unwrap();
    assert_eq!(
        health_error.code,
        error_codes::SYSTEM_MONITOR_ERROR,
        "Should return SYSTEM_MONITOR_ERROR"
    );
    println!(
        "   - Error code: {} (SYSTEM_MONITOR_ERROR)",
        health_error.code
    );
    println!("   - Error message: {}", health_error.message);
    println!("   VERIFIED: Fail-fast behavior working correctly\n");

    // =========================================================================
    // EVIDENCE OF SUCCESS
    // =========================================================================
    println!("======================================================================");
    println!("EVIDENCE OF SUCCESS - Meta-UTL Verification");
    println!("======================================================================");
    println!("Source of Truth: MetaUtlTracker");
    println!();
    println!("Operations Verified:");
    println!("  1. learning_trajectory: 13 embedder trajectories");
    println!("  2. predict_storage: Prediction stored in tracker");
    println!("  3. validate_prediction: Prediction removed, count incremented");
    println!("  4. health_metrics: Correctly fails with StubSystemMonitor (TASK-EMB-024)");
    println!();
    println!("Physical Evidence:");
    println!("  - Prediction ID: {}", prediction_id);
    println!(
        "  - Validation count increased: {} -> {}",
        validation_count_before,
        ctx.meta_utl_tracker.read().validation_count
    );
    println!("======================================================================\n");
}
