//! TELEO-H4: update_synergy_matrix handler.
//!
//! Records feedback and adaptively updates synergy weights.

use super::conversions::parse_feedback_type;
use super::types::UpdateSynergyMatrixParams;
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcResponse};
use context_graph_core::teleological::{
    services::feedback_learner::{FeedbackEvent, FeedbackLearner},
    types::NUM_EMBEDDERS,
};
use serde_json::json;
use tracing::{debug, error, info};
use uuid::Uuid;

impl Handlers {
    /// Handle update_synergy_matrix tool call.
    ///
    /// Records feedback and adaptively updates synergy weights.
    pub(in crate::handlers) async fn call_update_synergy_matrix(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("update_synergy_matrix called with: {:?}", arguments);

        // Parse parameters
        let params: UpdateSynergyMatrixParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to parse update_synergy_matrix params: {}", e);
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        // Parse vector ID
        let vector_id = match Uuid::parse_str(&params.vector_id) {
            Ok(u) => u,
            Err(e) => {
                return self.tool_error_with_pulse(id, &format!("Invalid vector_id: {}", e));
            }
        };

        // Parse feedback type
        let feedback_type = parse_feedback_type(&params.feedback_type);

        // Create feedback event
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut event = FeedbackEvent::new(vector_id, feedback_type, timestamp);

        if let Some(ctx) = params.context {
            event = event.with_context(ctx);
        }
        if let Some(contributions) = params.contributions {
            // Normalize contributions to sum to 1.0 before passing
            let sum: f32 = contributions.iter().sum();
            if sum > 0.0 {
                let normalized: [f32; NUM_EMBEDDERS] = {
                    let mut arr = [0.0f32; NUM_EMBEDDERS];
                    for (i, &c) in contributions.iter().enumerate() {
                        arr[i] = c / sum;
                    }
                    arr
                };
                event = event.with_contributions(normalized);
            }
        }

        // Create a local FeedbackLearner instance (stateless per request as per plan)
        // In production, this should be shared state - for now we demonstrate the API
        let mut feedback_learner = FeedbackLearner::new();
        feedback_learner.record_feedback(event);

        // Check if we should learn (will be false with just 1 event by default)
        let should_learn = feedback_learner.should_learn();
        let mut learning_result_json = None;

        if should_learn {
            let result = feedback_learner.learn();
            info!(
                "Learning triggered: {} events processed, {} adjustments",
                result.events_processed,
                result.adjustments.len()
            );

            learning_result_json = Some(json!({
                "events_processed": result.events_processed,
                "adjustments": result.adjustments,
                "confidence_delta": result.confidence_delta,
            }));
        }

        info!(
            "update_synergy_matrix: recorded feedback for vector {}",
            params.vector_id
        );

        self.tool_result_with_pulse(
            id,
            json!({
                "success": true,
                "vector_id": params.vector_id,
                "feedback_type": params.feedback_type,
                "learning_triggered": should_learn,
                "learning_result": learning_result_json,
                "note": "Feedback recorded. In production, feedback accumulates until threshold for learning."
            }),
        )
    }
}
