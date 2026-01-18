//! Retrieval prediction handler for Meta-UTL.
//!
//! TASK-S005: Implements predict_retrieval handler.

use std::time::Instant;

use serde_json::json;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::constants::NUM_EMBEDDERS;

use super::super::core::{PredictionType, StoredPrediction};
use super::super::Handlers;

impl Handlers {
    /// Handle meta_utl/predict_retrieval request.
    ///
    /// Predicts retrieval quality before querying.
    /// TASK-S005: Stores prediction in MetaUtlTracker for later validation.
    pub(crate) async fn handle_meta_utl_predict_retrieval(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("meta_utl/predict_retrieval: starting");

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

        // Parse query_fingerprint_id
        let query_fingerprint_id_str =
            match params.get("query_fingerprint_id").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => {
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        "Missing 'query_fingerprint_id' parameter",
                    );
                }
            };

        let query_fingerprint_id = match Uuid::parse_str(query_fingerprint_id_str) {
            Ok(uuid) => uuid,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid UUID format: {}", query_fingerprint_id_str),
                );
            }
        };

        let target_top_k = params
            .get("target_top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        // Verify fingerprint exists
        match self.teleological_store.retrieve(query_fingerprint_id).await {
            Ok(Some(_fingerprint)) => {
                // Fingerprint exists, proceed with prediction
            }
            Ok(None) => {
                warn!(
                    "meta_utl/predict_retrieval: fingerprint not found: {}",
                    query_fingerprint_id
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Fingerprint not found: {}", query_fingerprint_id),
                );
            }
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Failed to retrieve fingerprint: {}", e),
                );
            }
        }

        // Generate prediction
        let prediction_id = Uuid::new_v4();

        // Calculate per-space contributions from current weights
        let tracker = self.meta_utl_tracker.read();
        let per_space_contribution: Vec<f32> = tracker.current_weights.to_vec();

        // Expected metrics based on historical accuracy
        let expected_relevance: f32 = {
            let mut total = 0.0f32;
            let mut count = 0usize;
            for i in 0..NUM_EMBEDDERS {
                if let Some(acc) = tracker.get_embedder_accuracy(i) {
                    total += acc * tracker.current_weights[i];
                    count += 1;
                }
            }
            if count > 0 {
                (total / count as f32 * NUM_EMBEDDERS as f32).min(0.95)
            } else {
                0.7
            }
        };

        let expected_alignment: f32 = expected_relevance * 1.1; // Slightly higher for alignment
        let expected_result_count = (target_top_k as f32 * 0.8).ceil() as usize;

        // Calculate confidence
        let confidence = if tracker.validation_count >= 50 {
            let accuracy_sum: f32 = (0..NUM_EMBEDDERS)
                .filter_map(|i| tracker.get_embedder_accuracy(i))
                .sum();
            let accuracy_count = (0..NUM_EMBEDDERS)
                .filter(|&i| tracker.accuracy_counts[i] > 0)
                .count();
            if accuracy_count > 0 {
                (accuracy_sum / accuracy_count as f32).min(0.99)
            } else {
                0.5
            }
        } else {
            0.4 + (tracker.validation_count as f32 / 125.0)
        };
        drop(tracker);

        // Store prediction for later validation
        let predicted_values = json!({
            "expected_relevance": expected_relevance,
            "expected_alignment": expected_alignment,
            "expected_result_count": expected_result_count,
            "per_space_contribution": per_space_contribution,
            "target_top_k": target_top_k,
        });

        let stored_prediction = StoredPrediction {
            _created_at: Instant::now(),
            prediction_type: PredictionType::Retrieval,
            predicted_values: predicted_values.clone(),
            fingerprint_id: query_fingerprint_id,
        };

        {
            let mut tracker = self.meta_utl_tracker.write();
            tracker.store_prediction(prediction_id, stored_prediction);
        }

        debug!(
            "meta_utl/predict_retrieval: stored prediction {} for query {}",
            prediction_id, query_fingerprint_id
        );

        JsonRpcResponse::success(
            id,
            json!({
                "predictions": {
                    "expected_relevance": expected_relevance,
                    "expected_alignment": expected_alignment,
                    "expected_result_count": expected_result_count,
                    "per_space_contribution": per_space_contribution,
                },
                "confidence": confidence,
                "prediction_id": prediction_id.to_string(),
            }),
        )
    }
}
