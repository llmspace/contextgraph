//! Johari analysis handlers.
//!
//! Handlers for cross-space analysis and transition probabilities.

use serde_json::json;
use tracing::{debug, error, instrument};
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;
use context_graph_core::types::JohariQuadrant;

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::types::{CrossSpaceAnalysisParams, TransitionProbabilitiesParams, EMBEDDER_NAMES};

impl Handlers {
    /// Handle johari/cross_space_analysis request.
    ///
    /// Analyze cross-space patterns (blind spots, learning opportunities).
    #[instrument(skip(self, params), fields(method = "johari/cross_space_analysis"))]
    pub(in crate::handlers) async fn handle_johari_cross_space_analysis(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Parse parameters - FAIL FAST
        let params: CrossSpaceAnalysisParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(parsed) => parsed,
                Err(e) => {
                    error!("johari/cross_space_analysis: Invalid parameters: {}", e);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!("Invalid parameters: {}", e),
                    );
                }
            },
            None => {
                error!("johari/cross_space_analysis: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - memory_ids required",
                );
            }
        };

        let mut blind_spots = Vec::new();
        let mut learning_opportunities = Vec::new();

        for memory_id_str in &params.memory_ids {
            // Parse UUID - FAIL FAST
            let uuid = match Uuid::parse_str(memory_id_str) {
                Ok(u) => u,
                Err(e) => {
                    error!("johari/cross_space_analysis: Invalid UUID: {}", e);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!("Invalid memory_id UUID: {}", e),
                    );
                }
            };

            // Retrieve fingerprint - FAIL FAST
            let fingerprint = match self.teleological_store.retrieve(uuid).await {
                Ok(Some(fp)) => fp,
                Ok(None) => {
                    error!("johari/cross_space_analysis: Memory not found: {}", uuid);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::FINGERPRINT_NOT_FOUND,
                        format!("Memory not found: {}", uuid),
                    );
                }
                Err(e) => {
                    error!("johari/cross_space_analysis: Storage error: {}", e);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::STORAGE_ERROR,
                        format!("Storage error: {}", e),
                    );
                }
            };

            let johari = &fingerprint.johari;

            // Find blind spots: High Blind weight while E1 (semantic) has high Open weight
            // Returns (embedder_idx, severity) pairs sorted by severity descending
            let spots = johari.find_blind_spots();
            for (blind_embedder_idx, severity) in spots {
                // E1 (semantic) is always the "aware" space in this analysis
                let aware_space = 0_usize; // E1 semantic
                blind_spots.push(json!({
                    "memory_id": memory_id_str,
                    "aware_space": aware_space,
                    "aware_space_name": EMBEDDER_NAMES[aware_space],
                    "blind_space": blind_embedder_idx,
                    "blind_space_name": EMBEDDER_NAMES[blind_embedder_idx],
                    "severity": severity,
                    "description": format!(
                        "Semantic understanding (E1) without {} insight",
                        EMBEDDER_NAMES[blind_embedder_idx].split('_').nth(1).unwrap_or("other")
                    ),
                    "learning_suggestion": format!(
                        "Explore {} relationships via dream consolidation",
                        EMBEDDER_NAMES[blind_embedder_idx].split('_').nth(1).unwrap_or("related")
                    )
                }));
            }

            // Find learning opportunities: memories with many Unknown embedders
            let unknown_spaces: Vec<usize> = (0..NUM_EMBEDDERS)
                .filter(|&i| johari.dominant_quadrant(i) == JohariQuadrant::Unknown)
                .collect();

            if unknown_spaces.len() >= 5 {
                learning_opportunities.push(json!({
                    "memory_id": memory_id_str,
                    "unknown_spaces": unknown_spaces,
                    "potential": if unknown_spaces.len() >= 8 { "high" } else { "medium" }
                }));
            }
        }

        let response = json!({
            "blind_spots": blind_spots,
            "learning_opportunities": learning_opportunities,
            "quadrant_correlation": {} // Placeholder - full impl requires multi-memory scan
        });

        debug!(
            "Cross-space analysis complete: {} blind spots, {} learning opportunities",
            blind_spots.len(),
            learning_opportunities.len()
        );

        JsonRpcResponse::success(id, response)
    }

    /// Handle johari/transition_probabilities request.
    ///
    /// Get transition probability matrix for an embedder.
    #[instrument(skip(self, params), fields(method = "johari/transition_probabilities"))]
    pub(in crate::handlers) async fn handle_johari_transition_probabilities(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Parse parameters - FAIL FAST
        let params: TransitionProbabilitiesParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(parsed) => parsed,
                Err(e) => {
                    error!("johari/transition_probabilities: Invalid parameters: {}", e);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!("Invalid parameters: {}", e),
                    );
                }
            },
            None => {
                error!("johari/transition_probabilities: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - embedder_index, memory_id required",
                );
            }
        };

        // Validate embedder index - FAIL FAST
        if params.embedder_index >= NUM_EMBEDDERS {
            error!(
                "johari/transition_probabilities: Invalid embedder index: {}",
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

        // Parse UUID - FAIL FAST
        let uuid = match Uuid::parse_str(&params.memory_id) {
            Ok(u) => u,
            Err(e) => {
                error!("johari/transition_probabilities: Invalid UUID: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid memory_id UUID: {}", e),
                );
            }
        };

        // Retrieve fingerprint - FAIL FAST
        let fingerprint = match self.teleological_store.retrieve(uuid).await {
            Ok(Some(fp)) => fp,
            Ok(None) => {
                error!(
                    "johari/transition_probabilities: Memory not found: {}",
                    uuid
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Memory not found: {}", uuid),
                );
            }
            Err(e) => {
                error!("johari/transition_probabilities: Storage error: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Storage error: {}", e),
                );
            }
        };

        let johari = &fingerprint.johari;
        let trans_probs = johari.transition_probs[params.embedder_index];

        let response = json!({
            "embedder_index": params.embedder_index,
            "embedder_name": EMBEDDER_NAMES[params.embedder_index],
            "transition_matrix": {
                "from_open": {
                    "to_open": trans_probs[0][0],
                    "to_hidden": trans_probs[0][1],
                    "to_blind": trans_probs[0][2],
                    "to_unknown": trans_probs[0][3]
                },
                "from_hidden": {
                    "to_open": trans_probs[1][0],
                    "to_hidden": trans_probs[1][1],
                    "to_blind": trans_probs[1][2],
                    "to_unknown": trans_probs[1][3]
                },
                "from_blind": {
                    "to_open": trans_probs[2][0],
                    "to_hidden": trans_probs[2][1],
                    "to_blind": trans_probs[2][2],
                    "to_unknown": trans_probs[2][3]
                },
                "from_unknown": {
                    "to_open": trans_probs[3][0],
                    "to_hidden": trans_probs[3][1],
                    "to_blind": trans_probs[3][2],
                    "to_unknown": trans_probs[3][3]
                }
            },
            "sample_size": 150 // Placeholder - actual impl would track this
        });

        debug!(
            "Retrieved transition probabilities for E{}",
            params.embedder_index + 1
        );

        JsonRpcResponse::success(id, response)
    }
}
