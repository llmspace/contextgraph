//! Learning trajectory handler for Meta-UTL.
//!
//! TASK-S005: Implements learning_trajectory handler.

use serde_json::json;
use tracing::{debug, warn};

use context_graph_core::johari::NUM_EMBEDDERS;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;
use super::constants::{EMBEDDER_NAMES, LEARNING_SCORE_TARGET};

impl Handlers {
    /// Handle meta_utl/learning_trajectory request.
    ///
    /// Returns per-embedder learning trajectories with accuracy trends.
    /// TASK-S005: Exposes MetaUtlTracker accuracy data for monitoring.
    pub(crate) async fn handle_meta_utl_learning_trajectory(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("meta_utl/learning_trajectory: starting");

        // Parse optional parameters
        let params = params.unwrap_or(json!({}));

        // Parse embedder_indices - validate all are < 13
        let embedder_indices: Vec<usize> = match params.get("embedder_indices") {
            Some(indices) => {
                let indices = match indices.as_array() {
                    Some(arr) => arr,
                    None => {
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INVALID_PARAMS,
                            "embedder_indices must be an array",
                        );
                    }
                };
                let mut result = Vec::with_capacity(indices.len());
                for idx in indices {
                    let idx = match idx.as_u64() {
                        Some(n) => n as usize,
                        None => {
                            return JsonRpcResponse::error(
                                id,
                                error_codes::INVALID_PARAMS,
                                "embedder_indices must contain integers",
                            );
                        }
                    };
                    if idx >= NUM_EMBEDDERS {
                        warn!(
                            "meta_utl/learning_trajectory: invalid embedder index {}",
                            idx
                        );
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INVALID_PARAMS,
                            format!("Invalid embedder index {}: must be 0-12", idx),
                        );
                    }
                    result.push(idx);
                }
                result
            }
            None => (0..NUM_EMBEDDERS).collect(), // All 13 embedders
        };

        let _history_window = params
            .get("history_window")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;

        let include_accuracy_trend = params
            .get("include_accuracy_trend")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Get tracker data
        let tracker = self.meta_utl_tracker.read();

        // Build trajectories
        let initial_weight = 1.0 / NUM_EMBEDDERS as f32;
        let mut trajectories = Vec::with_capacity(embedder_indices.len());
        let mut total_accuracy = 0.0f32;
        let mut accuracy_count = 0usize;
        let mut best_space = 0usize;
        let mut best_accuracy = 0.0f32;
        let mut worst_space = 0usize;
        let mut worst_accuracy = 1.0f32;
        let mut spaces_above_target = 0usize;

        for &idx in &embedder_indices {
            let current_weight = tracker.current_weights[idx];
            let recent_accuracy = tracker.get_embedder_accuracy(idx);

            // Build accuracy history (last few samples from rolling window)
            let count = tracker.accuracy_counts[idx];
            let history_len = count.min(4);
            let mut accuracy_history = Vec::with_capacity(history_len);
            if count > 0 {
                let start_idx = count.saturating_sub(history_len);
                for i in start_idx..count {
                    accuracy_history.push(tracker.embedder_accuracy[idx][i]);
                }
            }

            let accuracy_trend = if include_accuracy_trend {
                tracker.get_accuracy_trend(idx)
            } else {
                None
            };

            let acc = recent_accuracy.unwrap_or(0.0);
            if acc > best_accuracy {
                best_accuracy = acc;
                best_space = idx;
            }
            if acc < worst_accuracy {
                worst_accuracy = acc;
                worst_space = idx;
            }
            if acc >= LEARNING_SCORE_TARGET {
                spaces_above_target += 1;
            }
            if recent_accuracy.is_some() {
                total_accuracy += acc;
                accuracy_count += 1;
            }

            trajectories.push(json!({
                "embedder_index": idx,
                "embedder_name": EMBEDDER_NAMES[idx],
                "current_weight": current_weight,
                "initial_weight": initial_weight,
                "weight_delta": current_weight - initial_weight,
                "recent_accuracy": recent_accuracy,
                "prediction_count": tracker.prediction_count,
                "accuracy_trend": accuracy_trend,
                "accuracy_history": accuracy_history,
            }));
        }

        let spaces_below_target = embedder_indices.len() - spaces_above_target;

        let overall_accuracy = if accuracy_count > 0 {
            total_accuracy / accuracy_count as f32
        } else {
            0.0
        };

        debug!(
            "meta_utl/learning_trajectory: returning {} trajectories",
            trajectories.len()
        );

        JsonRpcResponse::success(
            id,
            json!({
                "trajectories": trajectories,
                "system_summary": {
                    "overall_accuracy": overall_accuracy,
                    "best_performing_space": best_space,
                    "worst_performing_space": worst_space,
                    "spaces_above_target": spaces_above_target,
                    "spaces_below_target": spaces_below_target,
                }
            }),
        )
    }
}
