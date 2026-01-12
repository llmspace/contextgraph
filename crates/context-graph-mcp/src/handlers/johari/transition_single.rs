//! Johari single transition handler.
//!
//! Handler for executing a single validated Johari transition.

use serde_json::json;
use tracing::{debug, error, instrument};
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::helpers::{parse_quadrant, parse_trigger, quadrant_to_string};
use super::types::TransitionParams;

impl Handlers {
    /// Handle johari/transition request.
    ///
    /// Execute a single validated Johari transition.
    #[instrument(skip(self, params), fields(method = "johari/transition"))]
    pub(in crate::handlers) async fn handle_johari_transition(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Parse parameters - FAIL FAST
        let params: TransitionParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(parsed) => parsed,
                Err(e) => {
                    error!("johari/transition: Invalid parameters: {}", e);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!("Invalid parameters: {}", e),
                    );
                }
            },
            None => {
                error!("johari/transition: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - memory_id, embedder_index, to_quadrant, trigger required",
                );
            }
        };

        // Parse UUID - FAIL FAST
        let uuid = match Uuid::parse_str(&params.memory_id) {
            Ok(u) => u,
            Err(e) => {
                error!("johari/transition: Invalid UUID: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid memory_id UUID: {}", e),
                );
            }
        };

        // Validate embedder index - FAIL FAST
        if params.embedder_index >= NUM_EMBEDDERS {
            error!(
                "johari/transition: Invalid embedder index: {}",
                params.embedder_index
            );
            return JsonRpcResponse::error(
                id,
                error_codes::JOHARI_INVALID_EMBEDDER_INDEX,
                format!(
                    "Invalid embedder index: {} (must be 0-12)",
                    params.embedder_index
                ),
            );
        }

        // Parse quadrant - FAIL FAST
        let to_quadrant = match parse_quadrant(&params.to_quadrant) {
            Some(q) => q,
            None => {
                error!(
                    "johari/transition: Invalid quadrant: {}",
                    params.to_quadrant
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::JOHARI_INVALID_QUADRANT,
                    format!(
                        "Invalid to_quadrant: {} (must be open/hidden/blind/unknown)",
                        params.to_quadrant
                    ),
                );
            }
        };

        // Parse trigger - FAIL FAST
        let trigger = match parse_trigger(&params.trigger) {
            Some(t) => t,
            None => {
                error!("johari/transition: Invalid trigger: {}", params.trigger);
                return JsonRpcResponse::error(
                    id,
                    error_codes::JOHARI_TRANSITION_ERROR,
                    format!(
                        "Invalid trigger: {} (must be explicit_share/self_recognition/pattern_discovery/privatize/external_observation/dream_consolidation)",
                        params.trigger
                    ),
                );
            }
        };

        // Get current quadrant for response
        let current_fingerprint = match self.teleological_store.retrieve(uuid).await {
            Ok(Some(fp)) => fp,
            Ok(None) => {
                error!("johari/transition: Memory not found: {}", uuid);
                return JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Memory not found: {}", uuid),
                );
            }
            Err(e) => {
                error!("johari/transition: Storage error: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Storage error: {}", e),
                );
            }
        };

        let from_quadrant = current_fingerprint
            .johari
            .dominant_quadrant(params.embedder_index);

        debug!(
            "Executing transition E{}: {:?} -> {:?} via {:?}",
            params.embedder_index + 1,
            from_quadrant,
            to_quadrant,
            trigger
        );

        // Execute transition - FAIL FAST on invalid transition
        let updated_johari = match self
            .johari_manager
            .transition(uuid, params.embedder_index, to_quadrant, trigger)
            .await
        {
            Ok(j) => j,
            Err(e) => {
                error!("johari/transition: Transition error: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::JOHARI_TRANSITION_ERROR,
                    format!("Transition error: {}", e),
                );
            }
        };

        let response = json!({
            "memory_id": params.memory_id,
            "embedder_index": params.embedder_index,
            "from_quadrant": quadrant_to_string(from_quadrant),
            "to_quadrant": quadrant_to_string(to_quadrant),
            "trigger": params.trigger.to_lowercase(),
            "success": true,
            "updated_johari": {
                "quadrants": updated_johari.quadrants,
                "confidence": updated_johari.confidence
            }
        });

        debug!(
            "Transition successful: E{} {:?} -> {:?}",
            params.embedder_index + 1,
            from_quadrant,
            to_quadrant
        );

        JsonRpcResponse::success(id, response)
    }
}
