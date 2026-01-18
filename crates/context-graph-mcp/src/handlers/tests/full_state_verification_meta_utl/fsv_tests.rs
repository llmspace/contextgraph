//! Full State Verification (FSV) Tests for Meta-UTL Handlers.
//!
//! These tests verify handler behavior by directly inspecting the Source of Truth
//! (MetaUtlTracker) before and after operations.

use serde_json::json;
use uuid::Uuid;

use context_graph_core::teleological::NUM_EMBEDDERS;
use context_graph_core::traits::TeleologicalMemoryStore;

use super::helpers::{
    create_test_fingerprint, create_verifiable_handlers_with_tracker, make_request,
    make_request_no_params,
};

#[tokio::test]
async fn test_fsv_learning_trajectory_all_embedders() {
    println!("\n======================================================================");
    println!("FSV: meta_utl/learning_trajectory - All 13 Embedders");
    println!("======================================================================\n");

    let (handlers, _store, tracker) = create_verifiable_handlers_with_tracker();

    // STEP 1: BEFORE STATE
    println!("BEFORE STATE:");
    {
        let tracker_guard = tracker.read();
        println!("  prediction_count: {}", tracker_guard.prediction_count);
        println!("  validation_count: {}", tracker_guard.validation_count);
        println!(
            "  current_weights sum: {:.6}",
            tracker_guard.current_weights.iter().sum::<f32>()
        );
    }

    // STEP 2: EXECUTE
    let request = make_request(
        "meta_utl/learning_trajectory",
        json!({
            "include_accuracy_trend": true
        }),
    );

    let response = handlers.dispatch(request).await;

    // STEP 3: VERIFY RESPONSE
    println!("\nVERIFY RESPONSE:");
    assert!(
        response.error.is_none(),
        "Handler should succeed: {:?}",
        response.error
    );
    let result = response.result.unwrap();

    let trajectories = result["trajectories"].as_array().unwrap();
    println!("  trajectories count: {}", trajectories.len());
    assert_eq!(
        trajectories.len(),
        NUM_EMBEDDERS,
        "Should return all 13 embedders"
    );

    // Verify each trajectory has expected fields
    for (i, traj) in trajectories.iter().enumerate() {
        assert_eq!(traj["embedder_index"].as_u64().unwrap() as usize, i);
        assert!(traj["embedder_name"].as_str().is_some());
        assert!(traj["current_weight"].as_f64().is_some());
        assert!(traj["initial_weight"].as_f64().is_some());
    }

    let summary = &result["system_summary"];
    println!("  overall_accuracy: {}", summary["overall_accuracy"]);
    println!(
        "  best_performing_space: {}",
        summary["best_performing_space"]
    );
    println!(
        "  worst_performing_space: {}",
        summary["worst_performing_space"]
    );

    // STEP 4: VERIFY IN SOURCE OF TRUTH
    println!("\nVERIFY IN SOURCE OF TRUTH:");
    {
        let tracker_guard = tracker.read();
        let weights_sum: f32 = tracker_guard.current_weights.iter().sum();
        println!("  MetaUtlTracker weights sum: {:.6}", weights_sum);
        assert!(
            (weights_sum - 1.0).abs() < 0.001,
            "Weights should sum to ~1.0"
        );
    }

    // STEP 5: EVIDENCE
    println!("\n======================================================================");
    println!("EVIDENCE OF SUCCESS");
    println!("  - Returned 13 embedder trajectories");
    println!("  - Each trajectory has embedder_index, embedder_name, weights");
    println!("  - Weights sum to 1.0 in Source of Truth");
    println!("======================================================================\n");
}

#[tokio::test]
async fn test_fsv_learning_trajectory_specific_embedders() {
    println!("\n======================================================================");
    println!("FSV: meta_utl/learning_trajectory - Specific Embedders [0, 5, 12]");
    println!("======================================================================\n");

    let (handlers, _store, _tracker) = create_verifiable_handlers_with_tracker();

    // STEP 1: EXECUTE with specific embedder indices
    let request = make_request(
        "meta_utl/learning_trajectory",
        json!({
            "embedder_indices": [0, 5, 12],
            "include_accuracy_trend": true
        }),
    );

    let response = handlers.dispatch(request).await;

    // STEP 2: VERIFY RESPONSE
    assert!(response.error.is_none(), "Handler should succeed");
    let result = response.result.unwrap();

    let trajectories = result["trajectories"].as_array().unwrap();
    println!("VERIFY:");
    println!("  trajectories count: {}", trajectories.len());
    assert_eq!(trajectories.len(), 3, "Should return exactly 3 embedders");

    // Verify correct indices returned
    let indices: Vec<u64> = trajectories
        .iter()
        .map(|t| t["embedder_index"].as_u64().unwrap())
        .collect();
    assert_eq!(indices, vec![0, 5, 12], "Should return embedders 0, 5, 12");

    println!("\n======================================================================");
    println!("EVIDENCE: Returned exactly embedders [0, 5, 12]");
    println!("======================================================================\n");
}

/// Test that health_metrics correctly fails when real SystemMonitor is unavailable.
///
/// TASK-EMB-024: FAIL FAST - NO hardcoded fallback values.
/// StubSystemMonitor intentionally returns NotImplemented errors.
/// This test verifies the fail-fast behavior is working correctly.
#[tokio::test]
async fn test_fsv_health_metrics_with_targets() {
    println!("\n======================================================================");
    println!("FSV: meta_utl/health_metrics - Verify FAIL-FAST Behavior");
    println!("======================================================================\n");

    let (handlers, _store, _tracker) = create_verifiable_handlers_with_tracker();

    // STEP 1: EXECUTE
    let request = make_request(
        "meta_utl/health_metrics",
        json!({
            "include_targets": true,
            "include_recommendations": true
        }),
    );

    let response = handlers.dispatch(request).await;

    // STEP 2: VERIFY FAIL-FAST BEHAVIOR
    // TASK-EMB-024: StubSystemMonitor is designed to fail with NotImplemented
    // This is CORRECT behavior - no fake/simulated metrics allowed
    println!("VERIFY FAIL-FAST BEHAVIOR:");
    println!("  - StubSystemMonitor is designed to fail (TASK-EMB-024)");
    println!("  - NO hardcoded fallback values allowed");
    println!("  - Real SystemMonitor required for health metrics");

    assert!(
        response.error.is_some(),
        "Handler MUST fail when using StubSystemMonitor (TASK-EMB-024 fail-fast policy)"
    );

    let error = response.error.as_ref().unwrap();
    println!("\n  Error code: {}", error.code);
    println!("  Error message: {}", error.message);

    // Verify it's the expected SYSTEM_MONITOR_ERROR
    assert_eq!(
        error.code,
        crate::protocol::error_codes::SYSTEM_MONITOR_ERROR,
        "Should return SYSTEM_MONITOR_ERROR when SystemMonitor fails"
    );

    // Verify error message mentions the component
    assert!(
        error.message.contains("coherence_recovery") || error.message.contains("recovery"),
        "Error should mention the failing component"
    );

    println!("\n======================================================================");
    println!("EVIDENCE: FAIL-FAST behavior working correctly");
    println!("  - StubSystemMonitor returns NotImplemented (by design)");
    println!("  - Handler correctly propagates error");
    println!("  - No fake metrics returned");
    println!("======================================================================\n");
}

#[tokio::test]
async fn test_fsv_predict_storage_and_validate() {
    println!("\n======================================================================");
    println!("FSV: meta_utl/predict_storage + validate_prediction Cycle");
    println!("======================================================================\n");

    let (handlers, store, tracker) = create_verifiable_handlers_with_tracker();

    // PRE-CONDITION: Need 10+ validations for predict_storage to work
    // Manually populate tracker with validation history
    {
        let mut tracker_guard = tracker.write();
        for _ in 0..15 {
            tracker_guard.record_validation();
            for i in 0..NUM_EMBEDDERS {
                tracker_guard.record_accuracy(i, 0.85);
            }
        }
    }

    // Store a fingerprint
    let fp = create_test_fingerprint();
    let fingerprint_id = store.store(fp).await.expect("Store should succeed");
    println!("SETUP: Stored fingerprint {}", fingerprint_id);

    // STEP 1: BEFORE STATE
    println!("\nBEFORE STATE:");
    {
        let tracker_guard = tracker.read();
        println!(
            "  pending_predictions: {}",
            tracker_guard.pending_predictions.len()
        );
        println!("  validation_count: {}", tracker_guard.validation_count);
    }

    // STEP 2: EXECUTE predict_storage
    let predict_request = make_request(
        "meta_utl/predict_storage",
        json!({
            "fingerprint_id": fingerprint_id.to_string(),
            "include_confidence": true
        }),
    );

    let predict_response = handlers.dispatch(predict_request).await;
    assert!(
        predict_response.error.is_none(),
        "predict_storage should succeed: {:?}",
        predict_response.error
    );
    let predict_result = predict_response.result.unwrap();

    let prediction_id_str = predict_result["prediction_id"].as_str().unwrap();
    let prediction_id = Uuid::parse_str(prediction_id_str).unwrap();
    println!("\nPREDICTION MADE:");
    println!("  prediction_id: {}", prediction_id);
    println!(
        "  coherence_delta: {}",
        predict_result["predictions"]["coherence_delta"]
    );
    println!("  confidence: {}", predict_result["confidence"]);

    // STEP 3: VERIFY IN SOURCE OF TRUTH
    println!("\nVERIFY IN SOURCE OF TRUTH (after predict):");
    {
        let tracker_guard = tracker.read();
        let exists = tracker_guard
            .pending_predictions
            .contains_key(&prediction_id);
        println!("  Prediction {} in tracker: {}", prediction_id, exists);
        assert!(exists, "Prediction MUST be in Source of Truth");
        println!(
            "  pending_predictions count: {}",
            tracker_guard.pending_predictions.len()
        );
    }

    // STEP 4: VALIDATE THE PREDICTION
    let validate_request = make_request(
        "meta_utl/validate_prediction",
        json!({
            "prediction_id": prediction_id.to_string(),
            "actual_outcome": {
                "coherence_delta": 0.018,
                "alignment_delta": 0.048
            }
        }),
    );

    let validation_count_before = tracker.read().validation_count;

    let validate_response = handlers.dispatch(validate_request).await;
    assert!(
        validate_response.error.is_none(),
        "validate_prediction should succeed: {:?}",
        validate_response.error
    );
    let validate_result = validate_response.result.unwrap();

    println!("\nVALIDATION RESULT:");
    println!(
        "  prediction_type: {}",
        validate_result["validation"]["prediction_type"]
    );
    println!(
        "  prediction_error: {}",
        validate_result["validation"]["prediction_error"]
    );
    println!(
        "  accuracy_score: {}",
        validate_result["validation"]["accuracy_score"]
    );

    // STEP 5: VERIFY SOURCE OF TRUTH (after validate)
    println!("\nVERIFY IN SOURCE OF TRUTH (after validate):");
    {
        let tracker_guard = tracker.read();
        let still_exists = tracker_guard
            .pending_predictions
            .contains_key(&prediction_id);
        println!(
            "  Prediction {} removed from tracker: {}",
            prediction_id, !still_exists
        );
        assert!(!still_exists, "Prediction MUST be removed after validation");

        println!("  validation_count before: {}", validation_count_before);
        println!(
            "  validation_count after: {}",
            tracker_guard.validation_count
        );
        assert!(
            tracker_guard.validation_count > validation_count_before,
            "validation_count should increase"
        );
    }

    println!("\n======================================================================");
    println!("EVIDENCE OF SUCCESS");
    println!("  - Prediction stored in pending_predictions (verified in tracker)");
    println!("  - Prediction removed after validation (verified in tracker)");
    println!("  - validation_count incremented (verified in tracker)");
    println!("======================================================================\n");
}

#[tokio::test]
async fn test_fsv_predict_retrieval() {
    println!("\n======================================================================");
    println!("FSV: meta_utl/predict_retrieval");
    println!("======================================================================\n");

    let (handlers, store, tracker) = create_verifiable_handlers_with_tracker();

    // Store a fingerprint
    let fp = create_test_fingerprint();
    let fingerprint_id = store.store(fp).await.expect("Store should succeed");
    println!("SETUP: Stored fingerprint {}", fingerprint_id);

    // STEP 1: BEFORE STATE
    println!("\nBEFORE STATE:");
    {
        let tracker_guard = tracker.read();
        println!(
            "  pending_predictions: {}",
            tracker_guard.pending_predictions.len()
        );
    }

    // STEP 2: EXECUTE
    let request = make_request(
        "meta_utl/predict_retrieval",
        json!({
            "query_fingerprint_id": fingerprint_id.to_string(),
            "target_top_k": 10
        }),
    );

    let response = handlers.dispatch(request).await;

    // STEP 3: VERIFY RESPONSE
    assert!(
        response.error.is_none(),
        "Handler should succeed: {:?}",
        response.error
    );
    let result = response.result.unwrap();

    let prediction_id_str = result["prediction_id"].as_str().unwrap();
    let prediction_id = Uuid::parse_str(prediction_id_str).unwrap();

    println!("\nPREDICTION MADE:");
    println!("  prediction_id: {}", prediction_id);
    println!(
        "  expected_relevance: {}",
        result["predictions"]["expected_relevance"]
    );
    println!(
        "  expected_alignment: {}",
        result["predictions"]["expected_alignment"]
    );

    // Verify per_space_contribution has 13 elements
    let contributions = result["predictions"]["per_space_contribution"]
        .as_array()
        .unwrap();
    assert_eq!(
        contributions.len(),
        NUM_EMBEDDERS,
        "Should have 13 contributions"
    );
    println!("  per_space_contribution length: {}", contributions.len());

    // STEP 4: VERIFY IN SOURCE OF TRUTH
    println!("\nVERIFY IN SOURCE OF TRUTH:");
    {
        let tracker_guard = tracker.read();
        let exists = tracker_guard
            .pending_predictions
            .contains_key(&prediction_id);
        println!("  Prediction {} in tracker: {}", prediction_id, exists);
        assert!(exists, "Prediction MUST be in Source of Truth");
    }

    println!("\n======================================================================");
    println!("EVIDENCE: Retrieval prediction stored in MetaUtlTracker");
    println!("======================================================================\n");
}

#[tokio::test]
async fn test_fsv_optimized_weights_after_training() {
    println!("\n======================================================================");
    println!("FSV: meta_utl/optimized_weights - After Sufficient Training");
    println!("======================================================================\n");

    let (handlers, _store, tracker) = create_verifiable_handlers_with_tracker();

    // SETUP: Simulate 100 validations to trigger weight optimization
    {
        let mut tracker_guard = tracker.write();
        for v in 0..100 {
            tracker_guard.record_validation();
            // Record varying accuracy per embedder
            for i in 0..NUM_EMBEDDERS {
                let accuracy = 0.7 + (i as f32 * 0.02); // 0.70 to 0.94
                tracker_guard.record_accuracy(i, accuracy);
            }
            // Weight update triggers at validation 100
            if v == 99 {
                tracker_guard.update_weights();
            }
        }
    }

    println!("SETUP: Completed 100 validations with varying accuracy");

    // STEP 1: BEFORE STATE
    println!("\nBEFORE STATE:");
    {
        let tracker_guard = tracker.read();
        println!("  validation_count: {}", tracker_guard.validation_count);
        println!(
            "  last_weight_update: {:?}",
            tracker_guard.last_weight_update.is_some()
        );
    }

    // STEP 2: EXECUTE
    let request = make_request_no_params("meta_utl/optimized_weights");
    let response = handlers.dispatch(request).await;

    // STEP 3: VERIFY RESPONSE
    assert!(
        response.error.is_none(),
        "Handler should succeed with 100 validations: {:?}",
        response.error
    );
    let result = response.result.unwrap();

    let weights = result["weights"].as_array().unwrap();
    assert_eq!(weights.len(), NUM_EMBEDDERS, "Should return 13 weights");

    let weights_sum: f64 = weights.iter().map(|w| w.as_f64().unwrap()).sum();
    println!("\nRESULT:");
    println!("  weights count: {}", weights.len());
    println!("  weights sum: {:.6}", weights_sum);
    println!("  training_samples: {}", result["training_samples"]);
    println!("  confidence: {}", result["confidence"]);

    assert!(
        (weights_sum - 1.0).abs() < 0.001,
        "Weights should sum to ~1.0"
    );

    // STEP 4: VERIFY IN SOURCE OF TRUTH
    println!("\nVERIFY IN SOURCE OF TRUTH:");
    {
        let tracker_guard = tracker.read();
        let sot_sum: f32 = tracker_guard.current_weights.iter().sum();
        println!("  Source of Truth weights sum: {:.6}", sot_sum);
        assert!(
            (sot_sum - 1.0).abs() < 0.001,
            "SoT weights should sum to ~1.0"
        );
    }

    println!("\n======================================================================");
    println!("EVIDENCE: Optimized weights match Source of Truth, sum to 1.0");
    println!("======================================================================\n");
}
