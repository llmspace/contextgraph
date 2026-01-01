//! UTL computation handlers.

use serde_json::json;

use context_graph_core::types::{CognitivePulse, SuggestedAction, UtlContext};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::Handlers;

impl Handlers {
    /// Handle utl/compute request.
    pub(super) async fn handle_utl_compute(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
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

        let input = match params.get("input").and_then(|v| v.as_str()) {
            Some(i) => i,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'input' parameter",
                );
            }
        };

        let context = UtlContext::default();

        match self
            .utl_processor
            .compute_learning_score(input, &context)
            .await
        {
            Ok(score) => {
                let action = if score > 0.7 {
                    SuggestedAction::Consolidate
                } else if score > 0.4 {
                    SuggestedAction::Continue
                } else {
                    SuggestedAction::Explore
                };

                let pulse = CognitivePulse::new(
                    context.prior_entropy,
                    context.current_coherence,
                    0.0,
                    1.0,
                    action,
                    None,
                );
                JsonRpcResponse::success(id, json!({ "learningScore": score })).with_pulse(pulse)
            }
            Err(e) => JsonRpcResponse::error(id, error_codes::INTERNAL_ERROR, e.to_string()),
        }
    }

    /// Handle utl/metrics request.
    pub(super) async fn handle_utl_metrics(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
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

        let input = match params.get("input").and_then(|v| v.as_str()) {
            Some(i) => i,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'input' parameter",
                );
            }
        };

        let context = UtlContext::default();

        match self.utl_processor.compute_metrics(input, &context).await {
            Ok(metrics) => JsonRpcResponse::success(
                id,
                json!({
                    "entropy": metrics.entropy,
                    "coherence": metrics.coherence,
                    "learningScore": metrics.learning_score,
                    "surprise": metrics.surprise,
                    "coherenceChange": metrics.coherence_change,
                    "emotionalWeight": metrics.emotional_weight,
                    "alignment": metrics.alignment,
                }),
            ),
            Err(e) => JsonRpcResponse::error(id, error_codes::INTERNAL_ERROR, e.to_string()),
        }
    }
}
