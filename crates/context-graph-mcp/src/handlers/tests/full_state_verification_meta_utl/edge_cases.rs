//! Edge Case Tests for Meta-UTL Handlers.
//!
//! Tests invalid inputs, missing data, and boundary conditions.

use serde_json::json;
use uuid::Uuid;

use context_graph_core::teleological::NUM_EMBEDDERS;
use context_graph_core::traits::TeleologicalMemoryStore;

use crate::protocol::error_codes;

use super::helpers::{
    create_test_fingerprint, create_verifiable_handlers_with_tracker, make_request,
    make_request_no_params,
};

#[tokio::test]
async fn test_edge_case_embedder_index_13() {
    println!("\n======================================================================");
    println!("EDGE CASE: Invalid embedder index >= 13");
    println!("======================================================================\n");

    let (handlers, _store, tracker) = create_verifiable_handlers_with_tracker();

    // BEFORE STATE
    println!("BEFORE STATE:");
    {
        let tracker_guard = tracker.read();
        println!("  validation_count: {}", tracker_guard.validation_count);
    }

    // ACTION: Request learning_trajectory with invalid index
    let request = make_request(
        "meta_utl/learning_trajectory",
        json!({
            "embedder_indices": [0, 5, 13]  // 13 is invalid (must be 0-12)
        }),
    );

    let response = handlers.dispatch(request).await;

    // VERIFY: Should return INVALID_PARAMS error
    assert!(response.error.is_some(), "Should return error for index 13");
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::INVALID_PARAMS);
    println!("ERROR RETURNED:");
    println!("  code: {}", error.code);
    println!("  message: {}", error.message);
    assert!(error.message.contains("13") && error.message.contains("must be 0-12"));

    // AFTER STATE: Unchanged
    println!("\nAFTER STATE:");
    {
        let tracker_guard = tracker.read();
        println!(
            "  validation_count (unchanged): {}",
            tracker_guard.validation_count
        );
    }

    println!("\n======================================================================");
    println!("EVIDENCE: INVALID_PARAMS (-32602) returned for index 13");
    println!("======================================================================\n");
}

#[tokio::test]
async fn test_edge_case_validate_unknown_prediction() {
    println!("\n======================================================================");
    println!("EDGE CASE: Validate non-existent prediction");
    println!("======================================================================\n");

    let (handlers, _store, tracker) = create_verifiable_handlers_with_tracker();

    // BEFORE STATE
    println!("BEFORE STATE:");
    let predictions_before;
    {
        let tracker_guard = tracker.read();
        predictions_before = tracker_guard.pending_predictions.len();
        println!("  pending_predictions: {}", predictions_before);
    }

    // ACTION: Try to validate a random UUID
    let fake_prediction_id = Uuid::new_v4();
    let request = make_request(
        "meta_utl/validate_prediction",
        json!({
            "prediction_id": fake_prediction_id.to_string(),
            "actual_outcome": {
                "coherence_delta": 0.02,
                "alignment_delta": 0.05
            }
        }),
    );

    let response = handlers.dispatch(request).await;

    // VERIFY: Should return META_UTL_PREDICTION_NOT_FOUND
    assert!(
        response.error.is_some(),
        "Should return error for unknown prediction"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::META_UTL_PREDICTION_NOT_FOUND);
    println!("ERROR RETURNED:");
    println!("  code: {} (META_UTL_PREDICTION_NOT_FOUND)", error.code);
    println!("  message: {}", error.message);

    // AFTER STATE: Unchanged
    println!("\nAFTER STATE:");
    {
        let tracker_guard = tracker.read();
        assert_eq!(tracker_guard.pending_predictions.len(), predictions_before);
        println!(
            "  pending_predictions (unchanged): {}",
            tracker_guard.pending_predictions.len()
        );
    }

    println!("\n======================================================================");
    println!("EVIDENCE: META_UTL_PREDICTION_NOT_FOUND (-32040) returned");
    println!("======================================================================\n");
}

#[tokio::test]
async fn test_edge_case_optimized_weights_no_training() {
    println!("\n======================================================================");
    println!("EDGE CASE: Optimized weights with no training data");
    println!("======================================================================\n");

    let (handlers, _store, tracker) = create_verifiable_handlers_with_tracker();

    // BEFORE STATE: Fresh tracker with 0 validations
    println!("BEFORE STATE:");
    {
        let tracker_guard = tracker.read();
        println!("  validation_count: {}", tracker_guard.validation_count);
        println!(
            "  last_weight_update: {:?}",
            tracker_guard.last_weight_update
        );
        assert_eq!(tracker_guard.validation_count, 0);
    }

    // ACTION: Request optimized weights
    let request = make_request_no_params("meta_utl/optimized_weights");
    let response = handlers.dispatch(request).await;

    // VERIFY: Should return META_UTL_INSUFFICIENT_DATA
    assert!(
        response.error.is_some(),
        "Should return error for 0 validations"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::META_UTL_INSUFFICIENT_DATA);
    println!("ERROR RETURNED:");
    println!("  code: {} (META_UTL_INSUFFICIENT_DATA)", error.code);
    println!("  message: {}", error.message);
    assert!(error.message.contains("need 50 validations"));

    // AFTER STATE: Weights unchanged (still uniform 1/13)
    println!("\nAFTER STATE:");
    {
        let tracker_guard = tracker.read();
        let expected_uniform = 1.0 / NUM_EMBEDDERS as f32;
        for i in 0..NUM_EMBEDDERS {
            assert!((tracker_guard.current_weights[i] - expected_uniform).abs() < 0.001);
        }
        println!("  current_weights still uniform (1/13 each): verified");
    }

    println!("\n======================================================================");
    println!("EVIDENCE: META_UTL_INSUFFICIENT_DATA (-32042) returned, weights unchanged");
    println!("======================================================================\n");
}

#[tokio::test]
async fn test_edge_case_predict_storage_fingerprint_not_found() {
    println!("\n======================================================================");
    println!("EDGE CASE: Predict storage for non-existent fingerprint");
    println!("======================================================================\n");

    let (handlers, _store, tracker) = create_verifiable_handlers_with_tracker();

    // SETUP: Add validations so we pass the minimum threshold
    {
        let mut tracker_guard = tracker.write();
        for _ in 0..15 {
            tracker_guard.record_validation();
        }
    }

    // BEFORE STATE
    println!("BEFORE STATE:");
    {
        let tracker_guard = tracker.read();
        println!("  validation_count: {}", tracker_guard.validation_count);
        println!(
            "  pending_predictions: {}",
            tracker_guard.pending_predictions.len()
        );
    }

    // ACTION: Request prediction for non-existent fingerprint
    let fake_fingerprint_id = Uuid::new_v4();
    let request = make_request(
        "meta_utl/predict_storage",
        json!({
            "fingerprint_id": fake_fingerprint_id.to_string()
        }),
    );

    let response = handlers.dispatch(request).await;

    // VERIFY: Should return FINGERPRINT_NOT_FOUND
    assert!(
        response.error.is_some(),
        "Should return error for unknown fingerprint"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::FINGERPRINT_NOT_FOUND);
    println!("ERROR RETURNED:");
    println!("  code: {} (FINGERPRINT_NOT_FOUND)", error.code);
    println!("  message: {}", error.message);

    // AFTER STATE: No prediction stored
    println!("\nAFTER STATE:");
    {
        let tracker_guard = tracker.read();
        println!(
            "  pending_predictions (unchanged): {}",
            tracker_guard.pending_predictions.len()
        );
        assert_eq!(tracker_guard.pending_predictions.len(), 0);
    }

    println!("\n======================================================================");
    println!("EVIDENCE: FINGERPRINT_NOT_FOUND (-32010) returned");
    println!("======================================================================\n");
}

#[tokio::test]
async fn test_edge_case_validate_missing_outcome_field() {
    println!("\n======================================================================");
    println!("EDGE CASE: Validate prediction with missing outcome field");
    println!("======================================================================\n");

    let (handlers, store, tracker) = create_verifiable_handlers_with_tracker();

    // SETUP: Create prediction first
    {
        let mut tracker_guard = tracker.write();
        for _ in 0..15 {
            tracker_guard.record_validation();
        }
    }

    let fp = create_test_fingerprint();
    let fingerprint_id = store.store(fp).await.expect("Store should succeed");

    let predict_request = make_request(
        "meta_utl/predict_storage",
        json!({
            "fingerprint_id": fingerprint_id.to_string()
        }),
    );

    let predict_response = handlers.dispatch(predict_request).await;
    assert!(predict_response.error.is_none());
    let prediction_id = predict_response.result.unwrap()["prediction_id"]
        .as_str()
        .unwrap()
        .to_string();
    println!("SETUP: Created prediction {}", prediction_id);

    // ACTION: Validate with missing field
    let request = make_request(
        "meta_utl/validate_prediction",
        json!({
            "prediction_id": prediction_id,
            "actual_outcome": {
                "coherence_delta": 0.02
                // missing alignment_delta
            }
        }),
    );

    let response = handlers.dispatch(request).await;

    // VERIFY: Should return META_UTL_INVALID_OUTCOME
    assert!(
        response.error.is_some(),
        "Should return error for missing field"
    );
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::META_UTL_INVALID_OUTCOME);
    println!("ERROR RETURNED:");
    println!("  code: {} (META_UTL_INVALID_OUTCOME)", error.code);
    println!("  message: {}", error.message);
    assert!(error.message.contains("alignment_delta"));

    println!("\n======================================================================");
    println!("EVIDENCE: META_UTL_INVALID_OUTCOME (-32043) returned for missing field");
    println!("======================================================================\n");
}
