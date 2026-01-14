//! TELEO-H2: compute_teleological_vector handler.
//!
//! Computes a full TeleologicalVector from content using all 13 embedders.

use super::types::{ComputeTeleologicalVectorParams, TeleologicalVectorJson};
use super::utils::{compute_alignments_from_embeddings, extract_embeddings_from_fingerprint};
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcResponse};
use context_graph_core::teleological::services::FusionEngine;
use serde_json::json;
use tracing::{debug, error, info};

impl Handlers {
    /// Handle compute_teleological_vector tool call.
    ///
    /// Computes a full TeleologicalVector from content using all 13 embedders.
    pub(in crate::handlers) async fn call_compute_teleological_vector(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("compute_teleological_vector called with: {:?}", arguments);

        // Parse parameters
        let params: ComputeTeleologicalVectorParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to parse compute_teleological_vector params: {}", e);
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        if params.content.is_empty() {
            return self.tool_error_with_pulse(id, "Content cannot be empty");
        }

        // Use multi_array_provider to get all 13 embeddings
        let start = std::time::Instant::now();
        let embedding_result = match self.multi_array_provider.embed_all(&params.content).await {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to compute embeddings: {}", e);
                return self
                    .tool_error_with_pulse(id, &format!("Embedding computation failed: {}", e));
            }
        };
        let embed_duration = start.elapsed();

        // CONSTITUTION COMPLIANT: Extract embeddings using helper
        let embeddings = match extract_embeddings_from_fingerprint(&embedding_result.fingerprint) {
            Ok(e) => e,
            Err(e) => {
                let err_msg = e.to_error_string();
                error!("{}", err_msg);
                return self.tool_error_with_pulse(id, &err_msg);
            }
        };

        // Compute alignments and fuse using constitution-compliant method
        let alignments = compute_alignments_from_embeddings(&embeddings);
        let fusion_engine = FusionEngine::new();
        let fusion_result = fusion_engine.fuse_from_alignments(&alignments);

        info!(
            "compute_teleological_vector completed in {:?} (tucker={})",
            embed_duration, params.compute_tucker
        );

        // Return alignment-based result (AP-03/AP-05 compliant - no raw embeddings)
        self.tool_result_with_pulse(
            id,
            json!({
                "success": true,
                "purpose_vector": fusion_result.purpose_vector.to_vec(),
                "group_alignments": {
                    "semantic": fusion_result.group_alignments[0],
                    "temporal": fusion_result.group_alignments[1],
                    "structural": fusion_result.group_alignments[2],
                    "experiential": fusion_result.group_alignments[3],
                },
                "embedding_latency_ms": embed_duration.as_millis(),
                "fusion_confidence": fusion_result.confidence,
                "metadata": {
                    "active_embedders": fusion_result.active_embedders,
                    "dominant_group": fusion_result.dominant_group.map(|g| match g {
                        0 => "Semantic",
                        1 => "Temporal",
                        2 => "Structural",
                        3 => "Experiential",
                        _ => "Unknown",
                    }),
                },
            }),
        )
    }
}
