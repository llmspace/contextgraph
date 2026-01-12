//! Johari batch transition handler.
//!
//! Handler for executing multiple transitions atomically (all-or-nothing).

use serde_json::json;
use tracing::{debug, error, instrument};
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::helpers::{parse_quadrant, parse_trigger};
use super::types::TransitionBatchParams;

impl Handlers {
    /// Handle johari/transition_batch request.
    ///
    /// Execute multiple transitions atomically (all-or-nothing).
    #[instrument(skip(self, params), fields(method = "johari/transition_batch"))]
    pub(in crate::handlers) async fn handle_johari_transition_batch(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Parse parameters - FAIL FAST
        let params: TransitionBatchParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(parsed) => parsed,
                Err(e) => {
                    error!("johari/transition_batch: Invalid parameters: {}", e);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!("Invalid parameters: {}", e),
                    );
                }
            },
            None => {
                error!("johari/transition_batch: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - memory_id, transitions required",
                );
            }
        };

        // Parse UUID - FAIL FAST
        let uuid = match Uuid::parse_str(&params.memory_id) {
            Ok(u) => u,
            Err(e) => {
                error!("johari/transition_batch: Invalid UUID: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid memory_id UUID: {}", e),
                );
            }
        };

        // Validate and convert transitions - FAIL FAST on any invalid
        let mut transitions = Vec::with_capacity(params.transitions.len());

        for (idx, item) in params.transitions.iter().enumerate() {
            // Validate embedder index
            if item.embedder_index >= NUM_EMBEDDERS {
                error!(
                    "johari/transition_batch: Invalid embedder index at {}: {}",
                    idx, item.embedder_index
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::JOHARI_BATCH_ERROR,
                    format!(
                        "Invalid embedder index at index {}: {} (must be 0-12)",
                        idx, item.embedder_index
                    ),
                );
            }

            // Parse quadrant
            let quadrant = match parse_quadrant(&item.to_quadrant) {
                Some(q) => q,
                None => {
                    error!(
                        "johari/transition_batch: Invalid quadrant at {}: {}",
                        idx, item.to_quadrant
                    );
                    return JsonRpcResponse::error(
                        id,
                        error_codes::JOHARI_BATCH_ERROR,
                        format!("Invalid to_quadrant at index {}: {}", idx, item.to_quadrant),
                    );
                }
            };

            // Parse trigger
            let trigger = match parse_trigger(&item.trigger) {
                Some(t) => t,
                None => {
                    error!(
                        "johari/transition_batch: Invalid trigger at {}: {}",
                        idx, item.trigger
                    );
                    return JsonRpcResponse::error(
                        id,
                        error_codes::JOHARI_BATCH_ERROR,
                        format!("Invalid trigger at index {}: {}", idx, item.trigger),
                    );
                }
            };

            transitions.push((item.embedder_index, quadrant, trigger));
        }

        debug!(
            "Executing batch of {} transitions for memory {}",
            transitions.len(),
            uuid
        );

        // Execute batch - FAIL FAST, all-or-nothing
        let updated_johari = match self
            .johari_manager
            .transition_batch(uuid, transitions)
            .await
        {
            Ok(j) => j,
            Err(e) => {
                error!("johari/transition_batch: Batch error: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::JOHARI_BATCH_ERROR,
                    format!("Batch transition error: {}", e),
                );
            }
        };

        let response = json!({
            "memory_id": params.memory_id,
            "success": true,
            "transitions_applied": params.transitions.len(),
            "updated_johari": {
                "quadrants": updated_johari.quadrants,
                "confidence": updated_johari.confidence
            }
        });

        debug!(
            "Batch transition successful: {} transitions applied",
            params.transitions.len()
        );

        JsonRpcResponse::success(id, response)
    }
}
