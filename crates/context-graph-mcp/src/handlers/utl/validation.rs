//! Validation handlers for Meta-UTL.
//!
//! TASK-S005: Implements validate_prediction and optimized_weights handlers.

use serde_json::json;
use tracing::{debug, warn};
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::core::PredictionType;
use super::super::Handlers;

impl Handlers {
    /// Handle meta_utl/validate_prediction request.
    ///
    /// Validates a prediction against actual outcome.
    /// TASK-S005: Updates embedder_accuracy and triggers weight optimization.
    pub(crate) async fn handle_meta_utl_validate_prediction(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("meta_utl/validate_prediction: starting");

        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        // Parse prediction_id
        let prediction_id_str = match params.get("prediction_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'prediction_id' parameter",
                );
            }
        };

        let prediction_id = match Uuid::parse_str(prediction_id_str) {
            Ok(uuid) => uuid,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid UUID format: {}", prediction_id_str),
                );
            }
        };

        // Parse actual_outcome
        let actual_outcome = match params.get("actual_outcome") {
            Some(outcome) => outcome.clone(),
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'actual_outcome' parameter",
                );
            }
        };

        // Get stored prediction
        let stored_prediction = {
            let mut tracker = self.meta_utl_tracker.write();
            match tracker.remove_prediction(&prediction_id) {
                Some(p) => p,
                None => {
                    warn!(
                        "meta_utl/validate_prediction: prediction not found: {}",
                        prediction_id
                    );
                    return JsonRpcResponse::error(
                        id,
                        error_codes::META_UTL_PREDICTION_NOT_FOUND,
                        format!("Prediction not found: {}", prediction_id),
                    );
                }
            }
        };

        let prediction_type = match stored_prediction.prediction_type {
            PredictionType::Storage => "storage",
            PredictionType::Retrieval => "retrieval",
        };

        // Calculate prediction error based on type
        let prediction_error: f32;
        let accuracy_score: f32;

        match stored_prediction.prediction_type {
            PredictionType::Storage => {
                // Validate storage prediction
                let predicted_coherence = stored_prediction
                    .predicted_values
                    .get("coherence_delta")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                let predicted_alignment = stored_prediction
                    .predicted_values
                    .get("alignment_delta")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;

                let actual_coherence = match actual_outcome
                    .get("coherence_delta")
                    .and_then(|v| v.as_f64())
                {
                    Some(c) => c as f32,
                    None => {
                        return JsonRpcResponse::error(
                            id,
                            error_codes::META_UTL_INVALID_OUTCOME,
                            "Invalid outcome: missing field 'coherence_delta'",
                        );
                    }
                };
                let actual_alignment = match actual_outcome
                    .get("alignment_delta")
                    .and_then(|v| v.as_f64())
                {
                    Some(a) => a as f32,
                    None => {
                        return JsonRpcResponse::error(
                            id,
                            error_codes::META_UTL_INVALID_OUTCOME,
                            "Invalid outcome: missing field 'alignment_delta'",
                        );
                    }
                };

                let coherence_error = (predicted_coherence - actual_coherence).abs();
                let alignment_error = (predicted_alignment - actual_alignment).abs();
                prediction_error = (coherence_error + alignment_error) / 2.0;
                accuracy_score = 1.0 - prediction_error.min(1.0);
            }
            PredictionType::Retrieval => {
                // Validate retrieval prediction
                let predicted_relevance = stored_prediction
                    .predicted_values
                    .get("expected_relevance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;

                let actual_relevance = match actual_outcome
                    .get("actual_relevance")
                    .and_then(|v| v.as_f64())
                {
                    Some(r) => r as f32,
                    None => {
                        return JsonRpcResponse::error(
                            id,
                            error_codes::META_UTL_INVALID_OUTCOME,
                            "Invalid outcome: missing field 'actual_relevance'",
                        );
                    }
                };

                prediction_error = (predicted_relevance - actual_relevance).abs();
                accuracy_score = 1.0 - prediction_error.min(1.0);
            }
        }

        // Update embedder accuracy for all embedders (weighted by contribution)
        let mut tracker = self.meta_utl_tracker.write();
        // Copy weights first to avoid borrow conflict
        let weights = tracker.current_weights;
        for (i, &weight) in weights.iter().enumerate() {
            // Weight accuracy by the embedder's contribution
            let weighted_accuracy = accuracy_score * weight;
            tracker.record_accuracy(i, weighted_accuracy + (1.0 - weight));
        }

        // Record validation (triggers weight update every 100 validations)
        tracker.record_validation();

        // Get new accuracies
        let new_embedder_accuracy: Vec<Option<f32>> = (0..NUM_EMBEDDERS)
            .map(|i| tracker.get_embedder_accuracy(i))
            .collect();

        let accuracy_updated = tracker.validation_count.is_multiple_of(100);

        debug!(
            "meta_utl/validate_prediction: validated {} prediction {} with accuracy {}",
            prediction_type, prediction_id, accuracy_score
        );

        JsonRpcResponse::success(
            id,
            json!({
                "validation": {
                    "prediction_type": prediction_type,
                    "prediction_error": prediction_error,
                    "accuracy_score": accuracy_score,
                    "accuracy_updated": accuracy_updated,
                    "new_embedder_accuracy": new_embedder_accuracy,
                }
            }),
        )
    }

    /// Handle meta_utl/optimized_weights request.
    ///
    /// Returns current meta-learned optimized weights.
    /// TASK-S005: Requires sufficient validation data before returning.
    pub(crate) async fn handle_meta_utl_optimized_weights(
        &self,
        id: Option<JsonRpcId>,
        _params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("meta_utl/optimized_weights: starting");

        let tracker = self.meta_utl_tracker.read();

        // Check for sufficient data
        if tracker.validation_count < 50 {
            warn!(
                "meta_utl/optimized_weights: insufficient data, need 50 validations, have {}",
                tracker.validation_count
            );
            return JsonRpcResponse::error(
                id,
                error_codes::META_UTL_INSUFFICIENT_DATA,
                format!(
                    "Insufficient data: need 50 validations, have {}",
                    tracker.validation_count
                ),
            );
        }

        // Check if weights have been computed
        if tracker.last_weight_update.is_none() {
            warn!("meta_utl/optimized_weights: weights not yet computed");
            return JsonRpcResponse::error(
                id,
                error_codes::META_UTL_NOT_INITIALIZED,
                "Weights not computed yet: no weight optimization has occurred",
            );
        }

        // Calculate confidence based on training samples
        let confidence = if tracker.validation_count >= 500 {
            0.95
        } else if tracker.validation_count >= 200 {
            0.85
        } else if tracker.validation_count >= 100 {
            0.75
        } else {
            0.6 + (tracker.validation_count as f32 / 500.0)
        };

        // Format last_updated timestamp
        let last_updated = tracker
            .last_weight_update
            .map(|instant| {
                // Convert Instant to approximate ISO timestamp
                let elapsed = instant.elapsed();
                let now = chrono::Utc::now();
                let updated_time = now - chrono::Duration::from_std(elapsed).unwrap_or_default();
                updated_time.format("%Y-%m-%dT%H:%M:%SZ").to_string()
            })
            .unwrap_or_else(|| "unknown".to_string());

        debug!(
            "meta_utl/optimized_weights: returning weights with {} training samples",
            tracker.validation_count
        );

        JsonRpcResponse::success(
            id,
            json!({
                "weights": tracker.current_weights.to_vec(),
                "confidence": confidence,
                "training_samples": tracker.validation_count,
                "last_updated": last_updated,
            }),
        )
    }
}
