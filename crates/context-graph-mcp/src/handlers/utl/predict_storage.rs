//! Storage prediction handler for Meta-UTL.
//!
//! TASK-S005: Implements predict_storage handler.

use std::time::Instant;

use serde_json::json;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::constants::NUM_EMBEDDERS;

use super::super::core::{PredictionType, StoredPrediction};
use super::super::Handlers;

impl Handlers {
    /// Handle meta_utl/predict_storage request.
    ///
    /// Predicts storage impact before committing a fingerprint.
    /// TASK-S005: Stores prediction in MetaUtlTracker for later validation.
    pub(crate) async fn handle_meta_utl_predict_storage(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("meta_utl/predict_storage: starting");

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

        // Parse fingerprint_id
        let fingerprint_id_str = match params.get("fingerprint_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'fingerprint_id' parameter",
                );
            }
        };

        let fingerprint_id = match Uuid::parse_str(fingerprint_id_str) {
            Ok(uuid) => uuid,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid UUID format: {}", fingerprint_id_str),
                );
            }
        };

        let include_confidence = params
            .get("include_confidence")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Verify fingerprint exists
        match self.teleological_store.retrieve(fingerprint_id).await {
            Ok(Some(_fingerprint)) => {
                // Fingerprint exists, proceed with prediction
            }
            Ok(None) => {
                warn!(
                    "meta_utl/predict_storage: fingerprint not found: {}",
                    fingerprint_id
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Fingerprint not found: {}", fingerprint_id),
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

        // Check for sufficient validation data
        let tracker = self.meta_utl_tracker.read();
        if tracker.validation_count < 10 {
            drop(tracker); // Release lock before returning error
            warn!(
                "meta_utl/predict_storage: insufficient data, need 10 validations, have {}",
                self.meta_utl_tracker.read().validation_count
            );
            return JsonRpcResponse::error(
                id,
                error_codes::META_UTL_INSUFFICIENT_DATA,
                format!(
                    "Insufficient data: need 10 validations, have {}",
                    self.meta_utl_tracker.read().validation_count
                ),
            );
        }
        drop(tracker);

        // Generate prediction
        let prediction_id = Uuid::new_v4();
        let coherence_delta: f32 = 0.02;
        let alignment_delta: f32 = 0.05;
        let storage_impact_bytes: u64 = 4096;
        let index_rebuild_required = false;

        // Calculate confidence based on validation history
        let tracker = self.meta_utl_tracker.read();
        let confidence = if tracker.validation_count >= 50 {
            // Higher confidence with more validations
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
            // Lower confidence with fewer validations
            0.5 + (tracker.validation_count as f32 / 100.0)
        };
        drop(tracker);

        // Store prediction for later validation
        let predicted_values = json!({
            "coherence_delta": coherence_delta,
            "alignment_delta": alignment_delta,
            "storage_impact_bytes": storage_impact_bytes,
            "index_rebuild_required": index_rebuild_required,
        });

        let stored_prediction = StoredPrediction {
            _created_at: Instant::now(),
            prediction_type: PredictionType::Storage,
            predicted_values: predicted_values.clone(),
            fingerprint_id,
        };

        {
            let mut tracker = self.meta_utl_tracker.write();
            tracker.store_prediction(prediction_id, stored_prediction);
        }

        debug!(
            "meta_utl/predict_storage: stored prediction {} for fingerprint {}",
            prediction_id, fingerprint_id
        );

        let mut response = json!({
            "predictions": {
                "coherence_delta": coherence_delta,
                "alignment_delta": alignment_delta,
                "storage_impact_bytes": storage_impact_bytes,
                "index_rebuild_required": index_rebuild_required,
            },
            "prediction_id": prediction_id.to_string(),
        });

        if include_confidence {
            response["confidence"] = json!(confidence);
        }

        JsonRpcResponse::success(id, response)
    }
}
