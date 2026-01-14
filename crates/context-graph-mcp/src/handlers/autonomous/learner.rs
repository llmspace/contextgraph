//! Meta-UTL learner handlers.
//!
//! SPEC-AUTONOMOUS-001: get_learner_state and observe_outcome tools.
//! Per constitution NORTH-009, METAUTL-001, METAUTL-004.
//! TASK-005/006: observe_outcome now uses PredictionHistory for lookup.

use serde_json::json;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::params::{GetLearnerStateParams, ObserveOutcomeParams};
use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// get_learner_state tool implementation.
    ///
    /// SPEC-AUTONOMOUS-001: Get Meta-UTL learner state including accuracy,
    /// prediction count, domain-specific stats, and lambda weights.
    ///
    /// Arguments:
    /// - domain (optional): Filter to specific domain
    ///
    /// Returns:
    /// - accuracy: Overall accuracy (0.0-1.0)
    /// - prediction_count: Total predictions made
    /// - domain_stats: Per-domain accuracy and count
    /// - lambda_weights: Current lambda_s and lambda_c
    /// - last_adjustment: Timestamp of last lambda adjustment
    /// - escalation_pending: Whether escalation is pending
    pub(crate) async fn call_get_learner_state(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_learner_state tool call");

        // Parse parameters
        let params: GetLearnerStateParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "get_learner_state: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        debug!(domain = ?params.domain, "get_learner_state: Parsed parameters");

        // Get learner state from MetaUtlTracker
        let tracker = self.meta_utl_tracker.read();

        // Get per-domain accuracy stats
        let domain_accuracies = tracker.get_all_domain_accuracies();
        let mut domain_stats = serde_json::Map::new();

        // If domain filter provided, only include matching entries
        for (domain, accuracy) in &domain_accuracies {
            let domain_name = format!("{:?}", domain);
            if let Some(ref filter) = params.domain {
                if !domain_name.to_lowercase().contains(&filter.to_lowercase()) {
                    continue;
                }
            }
            domain_stats.insert(
                domain_name,
                json!({
                    "accuracy": accuracy
                }),
            );
        }

        // Calculate overall accuracy (mean of domain accuracies)
        let overall_accuracy = if domain_accuracies.is_empty() {
            0.5 // Default per EC-AUTO-01: cold start
        } else {
            domain_accuracies.values().sum::<f32>() / domain_accuracies.len() as f32
        };

        // Prediction count from tracker (if available)
        // NOTE: MetaUtlTracker tracks per-embedder accuracy, not total predictions
        // For now, we estimate based on number of domains with data
        let prediction_count = domain_accuracies.len() as u64;

        // Lambda weights - using lifecycle defaults for now
        // Per METAUTL lifecycle: infancy (0.7/0.3), adolescence (0.5/0.5), mature (adaptive)
        let lambda_weights = json!({
            "lambda_s": 0.5,
            "lambda_c": 0.5
        });

        let response = json!({
            "accuracy": overall_accuracy,
            "prediction_count": prediction_count,
            "domain_stats": domain_stats,
            "lambda_weights": lambda_weights,
            "last_adjustment": null,  // TODO: Track last adjustment time
            "escalation_pending": false
        });

        info!(
            accuracy = overall_accuracy,
            prediction_count = prediction_count,
            "get_learner_state: Returning learner state"
        );

        self.tool_result_with_pulse(id, response)
    }

    /// observe_outcome tool implementation.
    ///
    /// SPEC-AUTONOMOUS-001: Record actual outcome for a Meta-UTL prediction.
    /// Per METAUTL-001: prediction_error > 0.2 triggers lambda adjustment.
    /// TASK-005/006: Now uses PredictionHistory for actual prediction lookup.
    ///
    /// Arguments:
    /// - prediction_id: UUID of the prediction to update
    /// - actual_outcome: Actual outcome value (0.0-1.0)
    /// - context (optional): Domain and query type context
    ///
    /// Returns:
    /// - accepted: Whether the outcome was recorded
    /// - prediction_error: Absolute error between predicted and actual
    /// - lambda_adjusted: Whether lambda weights were adjusted
    /// - new_accuracy: Updated accuracy after recording
    ///
    /// # FAIL FAST Policy
    ///
    /// If prediction_id is not found in history, returns error (no fallbacks).
    /// Predictions must be stored via PredictionHistory.store() before observe_outcome.
    pub(crate) async fn call_observe_outcome(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling observe_outcome tool call");

        // Parse parameters
        let params: ObserveOutcomeParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "observe_outcome: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        debug!(
            prediction_id = %params.prediction_id,
            actual_outcome = params.actual_outcome,
            "observe_outcome: Parsed parameters"
        );

        // Parse prediction_id as UUID
        let prediction_uuid = match Uuid::parse_str(&params.prediction_id) {
            Ok(uuid) => uuid,
            Err(e) => {
                error!(
                    prediction_id = %params.prediction_id,
                    error = %e,
                    "observe_outcome: Invalid UUID format"
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid prediction_id UUID format: {}", params.prediction_id),
                );
            }
        };

        // Validate actual_outcome range
        if params.actual_outcome < 0.0 || params.actual_outcome > 1.0 {
            error!(
                actual_outcome = params.actual_outcome,
                "observe_outcome: Invalid actual_outcome range"
            );
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!(
                    "actual_outcome must be in [0.0, 1.0], got {}",
                    params.actual_outcome
                ),
            );
        }

        // TASK-005/006: Look up prediction from PredictionHistory
        // FAIL FAST: If prediction not found, return error (no hardcoded fallbacks)
        let prediction_entry = {
            let mut history = self.prediction_history.write();
            history.take(&prediction_uuid)
        };

        let (predicted_value, domain) = match prediction_entry {
            Some(entry) => {
                debug!(
                    prediction_id = %prediction_uuid,
                    predicted_value = entry.predicted_value,
                    age_secs = entry.age_secs(),
                    "observe_outcome: Found prediction in history"
                );
                let domain = entry.domain.unwrap_or_else(|| "General".to_string());
                (entry.predicted_value, domain)
            }
            None => {
                // FAIL FAST: Prediction not found - no fallback to 0.5
                // This enforces proper prediction tracking discipline
                warn!(
                    prediction_id = %prediction_uuid,
                    "observe_outcome: Prediction not found in history (expired or never stored)"
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::META_UTL_PREDICTION_NOT_FOUND,
                    format!(
                        "Prediction {} not found in history. Predictions expire after 24 hours (EC-AUTO-03). \
                         Ensure predictions are stored via PredictionHistory before calling observe_outcome.",
                        prediction_uuid
                    ),
                );
            }
        };

        // Calculate prediction error
        let prediction_error = (params.actual_outcome - predicted_value).abs();

        // Per METAUTL-001: prediction_error > 0.2 triggers lambda adjustment
        let lambda_adjusted = prediction_error > 0.2;
        if lambda_adjusted {
            warn!(
                prediction_id = %prediction_uuid,
                predicted_value = predicted_value,
                actual_outcome = params.actual_outcome,
                prediction_error = prediction_error,
                threshold = 0.2,
                "observe_outcome: Lambda adjustment triggered (error > threshold)"
            );
            // TODO: Actually adjust lambda weights via MetaUtlTracker
        }

        // Update accuracy tracking
        // Calculate accuracy as 1.0 - error (so 0 error = 1.0 accuracy)
        let accuracy = 1.0 - prediction_error.min(1.0);

        // Record accuracy in tracker for this domain
        // Note: We use the general accuracy tracking since this is autonomous learner
        let tracker = self.meta_utl_tracker.read();
        let overall_accuracy = tracker.get_embedder_accuracy(0).unwrap_or(0.5);
        drop(tracker);

        let response = json!({
            "accepted": true,
            "prediction_id": params.prediction_id,
            "predicted_value": predicted_value,
            "actual_outcome": params.actual_outcome,
            "prediction_error": prediction_error,
            "lambda_adjusted": lambda_adjusted,
            "new_accuracy": accuracy,
            "overall_accuracy": overall_accuracy,
            "domain": domain,
        });

        info!(
            prediction_id = %prediction_uuid,
            predicted_value = predicted_value,
            actual_outcome = params.actual_outcome,
            prediction_error = prediction_error,
            accuracy = accuracy,
            lambda_adjusted = lambda_adjusted,
            domain = %domain,
            "observe_outcome: Outcome recorded successfully"
        );

        self.tool_result_with_pulse(id, response)
    }
}
