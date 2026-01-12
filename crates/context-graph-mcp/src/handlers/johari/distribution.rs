//! Johari distribution handlers.
//!
//! Handlers for get_distribution and find_by_quadrant operations.

use serde_json::json;
use tracing::{debug, error, instrument};
use uuid::Uuid;

use context_graph_core::johari::{QuadrantPattern, NUM_EMBEDDERS};
use context_graph_core::types::JohariQuadrant;

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::helpers::{parse_quadrant, quadrant_to_string};
use super::types::{
    DistributionSummary, EmbedderQuadrantInfo, FindByQuadrantParams, GetDistributionParams,
    SoftClassification, EMBEDDER_NAMES,
};

impl Handlers {
    /// Handle johari/get_distribution request.
    ///
    /// Returns per-embedder quadrant distribution for a memory.
    #[instrument(skip(self, params), fields(method = "johari/get_distribution"))]
    pub(in crate::handlers) async fn handle_johari_get_distribution(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Parse parameters - FAIL FAST on missing
        let params: GetDistributionParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(parsed) => parsed,
                Err(e) => {
                    error!("johari/get_distribution: Invalid parameters: {}", e);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!("Invalid parameters: {}", e),
                    );
                }
            },
            None => {
                error!("johari/get_distribution: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - memory_id required",
                );
            }
        };

        // Parse UUID - FAIL FAST on invalid
        let uuid = match Uuid::parse_str(&params.memory_id) {
            Ok(u) => u,
            Err(e) => {
                error!("johari/get_distribution: Invalid UUID: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid memory_id UUID: {}", e),
                );
            }
        };

        debug!("Getting Johari distribution for memory {}", uuid);

        // Retrieve fingerprint from store - FAIL FAST on not found
        let fingerprint = match self.teleological_store.retrieve(uuid).await {
            Ok(Some(fp)) => fp,
            Ok(None) => {
                error!("johari/get_distribution: Memory not found: {}", uuid);
                return JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Memory not found: {}", uuid),
                );
            }
            Err(e) => {
                error!("johari/get_distribution: Storage error: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Storage error: {}", e),
                );
            }
        };

        let johari = &fingerprint.johari;

        // Build per-embedder quadrant info
        let mut per_embedder_quadrants = Vec::with_capacity(NUM_EMBEDDERS);
        let mut open_count = 0;
        let mut hidden_count = 0;
        let mut blind_count = 0;
        let mut unknown_count = 0;
        let mut confidence_sum = 0.0f32;

        for (idx, &name) in EMBEDDER_NAMES.iter().enumerate().take(NUM_EMBEDDERS) {
            let quadrant = johari.dominant_quadrant(idx);
            let weights = johari.quadrants[idx];
            let confidence = johari.confidence[idx];
            let _ = name; // Used below in embedder_name field

            // Count quadrants
            match quadrant {
                JohariQuadrant::Open => open_count += 1,
                JohariQuadrant::Hidden => hidden_count += 1,
                JohariQuadrant::Blind => blind_count += 1,
                JohariQuadrant::Unknown => unknown_count += 1,
            }
            confidence_sum += confidence;

            // Build info
            let mut info = EmbedderQuadrantInfo {
                embedder_index: idx,
                embedder_name: EMBEDDER_NAMES[idx],
                quadrant: quadrant_to_string(quadrant),
                soft_classification: SoftClassification {
                    open: weights[0],
                    hidden: weights[1],
                    blind: weights[2],
                    unknown: weights[3],
                },
                confidence: None,
                predicted_next_quadrant: None,
            };

            if params.include_confidence {
                info.confidence = Some(confidence);
            }

            if params.include_transition_predictions {
                // Get predicted next quadrant from transition matrix
                let predicted = johari.predict_transition(idx, quadrant);
                info.predicted_next_quadrant = Some(quadrant_to_string(predicted));
            }

            per_embedder_quadrants.push(info);
        }

        let average_confidence = confidence_sum / NUM_EMBEDDERS as f32;

        let response = json!({
            "memory_id": params.memory_id,
            "per_embedder_quadrants": per_embedder_quadrants,
            "summary": DistributionSummary {
                open_count,
                hidden_count,
                blind_count,
                unknown_count,
                average_confidence,
            }
        });

        debug!(
            "Johari distribution retrieved: {} open, {} hidden, {} blind, {} unknown",
            open_count, hidden_count, blind_count, unknown_count
        );

        JsonRpcResponse::success(id, response)
    }

    /// Handle johari/find_by_quadrant request.
    ///
    /// Find memories where a specific embedder is in a target quadrant.
    #[instrument(skip(self, params), fields(method = "johari/find_by_quadrant"))]
    pub(in crate::handlers) async fn handle_johari_find_by_quadrant(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Parse parameters - FAIL FAST
        let params: FindByQuadrantParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(parsed) => parsed,
                Err(e) => {
                    error!("johari/find_by_quadrant: Invalid parameters: {}", e);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!("Invalid parameters: {}", e),
                    );
                }
            },
            None => {
                error!("johari/find_by_quadrant: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - embedder_index, quadrant required",
                );
            }
        };

        // Validate embedder index - FAIL FAST
        if params.embedder_index >= NUM_EMBEDDERS {
            error!(
                "johari/find_by_quadrant: Invalid embedder index: {}",
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
        let quadrant = match parse_quadrant(&params.quadrant) {
            Some(q) => q,
            None => {
                error!(
                    "johari/find_by_quadrant: Invalid quadrant: {}",
                    params.quadrant
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::JOHARI_INVALID_QUADRANT,
                    format!(
                        "Invalid quadrant: {} (must be open/hidden/blind/unknown)",
                        params.quadrant
                    ),
                );
            }
        };

        debug!(
            "Finding memories with E{} in {:?} quadrant",
            params.embedder_index + 1,
            quadrant
        );

        // Create pattern for specific embedder in specific quadrant
        let pattern = QuadrantPattern::AtLeast { quadrant, count: 1 };

        // Search using JohariTransitionManager
        let results = match self
            .johari_manager
            .find_by_quadrant(pattern, params.top_k * 2)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!("johari/find_by_quadrant: Search error: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Search error: {}", e),
                );
            }
        };

        // Filter by specific embedder and confidence threshold
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|(_, johari)| {
                johari.dominant_quadrant(params.embedder_index) == quadrant
                    && johari.confidence[params.embedder_index] >= params.min_confidence
            })
            .take(params.top_k)
            .map(|(memory_id, johari)| {
                let weights = johari.quadrants[params.embedder_index];
                json!({
                    "id": memory_id.to_string(),
                    "confidence": johari.confidence[params.embedder_index],
                    "soft_classification": [weights[0], weights[1], weights[2], weights[3]]
                })
            })
            .collect();

        let total_count = filtered.len();

        let response = json!({
            "embedder_index": params.embedder_index,
            "embedder_name": EMBEDDER_NAMES[params.embedder_index],
            "quadrant": params.quadrant.to_lowercase(),
            "memories": filtered,
            "total_count": total_count
        });

        debug!(
            "Found {} memories with E{} in {:?}",
            total_count,
            params.embedder_index + 1,
            quadrant
        );

        JsonRpcResponse::success(id, response)
    }
}
