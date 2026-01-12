//! Goal aligned memories handler.
//!
//! Handles the `goal/aligned_memories` MCP method for finding
//! memories aligned to a specific goal.

use serde_json::json;
use tracing::{debug, error, instrument};
use uuid::Uuid;

use context_graph_core::purpose::GoalNode;
use context_graph_core::traits::TeleologicalSearchOptions;
use context_graph_core::types::fingerprint::{AlignmentThreshold, PurposeVector, NUM_EMBEDDERS};

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::helpers::goal_to_json;

impl Handlers {
    /// Handle goal/aligned_memories request.
    ///
    /// Find memories aligned to a specific goal.
    ///
    /// # Request Parameters
    /// - `goal_id` (required): Goal ID to find aligned memories for
    /// - `min_alignment` (optional): Minimum alignment threshold, default 0.55 (Warning threshold)
    /// - `top_k` (optional): Maximum results, default 10
    ///
    /// # Response
    /// - `results`: Array of memories with alignment scores to the goal
    /// - `goal`: The goal being queried
    ///
    /// # Error Codes
    /// - GOAL_NOT_FOUND (-32020): Goal ID not found
    /// - PURPOSE_SEARCH_ERROR (-32016): Search failed
    #[instrument(skip(self, params), fields(method = "goal/aligned_memories"))]
    pub(in crate::handlers) async fn handle_goal_aligned_memories(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                error!("goal/aligned_memories: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - goal_id required",
                );
            }
        };

        // Extract goal_id (required)
        let goal_id = match params.get("goal_id").and_then(|v| v.as_str()) {
            Some(gid) => match Uuid::parse_str(gid) {
                Ok(uuid) => uuid,
                Err(_) => {
                    error!("goal/hierarchy_query: Invalid goal_id UUID format: {}", gid);
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!("Invalid goal_id UUID format: {}", gid),
                    );
                }
            },
            None => {
                error!("goal/aligned_memories: Missing 'goal_id' parameter");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing required 'goal_id' parameter",
                );
            }
        };

        // Get the goal from hierarchy (scoped block ensures lock is released before await)
        let goal: GoalNode = {
            let hierarchy = self.goal_hierarchy.read();
            match hierarchy.get(&goal_id) {
                Some(g) => g.clone(),
                None => {
                    error!(goal_id = %goal_id, "goal/aligned_memories: Goal not found");
                    return JsonRpcResponse::error(
                        id,
                        error_codes::GOAL_NOT_FOUND,
                        format!("Goal not found: {}", goal_id),
                    );
                }
            }
        };

        // top_k has a sensible default (pagination parameter)
        const DEFAULT_TOP_K: usize = 10;
        let top_k = params
            .get("topK")
            .or_else(|| params.get("top_k"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_TOP_K);

        // FAIL-FAST: min_alignment MUST be explicitly provided.
        // Per constitution AP-007: No silent fallbacks that mask user intent.
        // Using 0.55 (Warning threshold) as default would silently filter results
        // without user awareness. Client MUST specify their desired threshold.
        let min_alignment = match params
            .get("minAlignment")
            .or_else(|| params.get("min_alignment"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
        {
            Some(alignment) => {
                // Validate range [0.0, 1.0]
                if !(0.0..=1.0).contains(&alignment) {
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!(
                            "minAlignment must be between 0.0 and 1.0, got: {}. \
                             Reference thresholds: 0.75 (Perfect), 0.70 (Strong), \
                             0.55 (Warning), below 0.55 (Misaligned)",
                            alignment
                        ),
                    );
                }
                alignment
            }
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing required parameter 'minAlignment' (or 'min_alignment'). \
                     You must explicitly specify the alignment threshold for filtering results. \
                     Reference thresholds: 0.75 (Perfect), 0.70 (Strong), 0.55 (Warning), \
                     below 0.55 (Misaligned). Example: \"minAlignment\": 0.55"
                        .to_string(),
                );
            }
        };

        // Create purpose vector from goal's embedding
        // We use the goal's propagation weight to scale alignments
        let propagation_weight = goal.level.propagation_weight();

        // Create a purpose vector that emphasizes the goal's embedding space
        // For simplicity, we use equal alignments scaled by propagation weight
        let alignments = [propagation_weight; NUM_EMBEDDERS];
        let purpose_vector = PurposeVector {
            alignments,
            dominant_embedder: 0, // E1 semantic as dominant
            coherence: 1.0,
            stability: 1.0,
        };

        let search_start = std::time::Instant::now();

        // Build search options
        let options = TeleologicalSearchOptions::quick(top_k).with_min_alignment(min_alignment);

        // Execute purpose search
        match self
            .teleological_store
            .search_purpose(&purpose_vector, options)
            .await
        {
            Ok(results) => {
                let search_latency_ms = search_start.elapsed().as_millis();

                let results_json: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "id": r.fingerprint.id.to_string(),
                            "goal_alignment": r.purpose_alignment * propagation_weight,
                            "raw_alignment": r.purpose_alignment,
                            "theta_to_north_star": r.fingerprint.theta_to_north_star,
                            "threshold": format!("{:?}", AlignmentThreshold::classify(r.purpose_alignment))
                        })
                    })
                    .collect();

                debug!(
                    goal_id = %goal_id,
                    count = results.len(),
                    latency_ms = search_latency_ms,
                    "goal/aligned_memories: Completed"
                );

                JsonRpcResponse::success(
                    id,
                    json!({
                        "goal": goal_to_json(&goal),
                        "results": results_json,
                        "count": results.len(),
                        "min_alignment_filter": min_alignment,
                        "search_time_ms": search_latency_ms
                    }),
                )
            }
            Err(e) => {
                error!(error = %e, goal_id = %goal_id, "goal/aligned_memories: FAILED");
                JsonRpcResponse::error(
                    id,
                    error_codes::PURPOSE_SEARCH_ERROR,
                    format!("Aligned memories search failed: {}", e),
                )
            }
        }
    }
}
