//! TELEO-H3: fuse_embeddings handler.
//!
//! Fuses multiple embeddings using synergy matrix and optional profile.

use super::types::{FuseEmbeddingsParams, TeleologicalVectorJson};
use super::utils::compute_alignments_from_embeddings;
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcResponse};
use context_graph_core::teleological::{services::FusionEngine, types::NUM_EMBEDDERS};
use serde_json::json;
use tracing::{debug, error, info};

impl Handlers {
    /// Handle fuse_embeddings tool call.
    ///
    /// Fuses multiple embeddings using synergy matrix and optional profile.
    pub(in crate::handlers) async fn call_fuse_embeddings(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("fuse_embeddings called with: {:?}", arguments);

        // Parse parameters
        let params: FuseEmbeddingsParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to parse fuse_embeddings params: {}", e);
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        // Validate embeddings count
        if params.embeddings.len() != NUM_EMBEDDERS {
            return self.tool_error_with_pulse(
                id,
                &format!(
                    "Expected {} embeddings, got {}",
                    NUM_EMBEDDERS,
                    params.embeddings.len()
                ),
            );
        }

        // Compute or use provided alignments
        let alignments = params
            .alignments
            .unwrap_or_else(|| compute_alignments_from_embeddings(&params.embeddings));

        // Create fusion engine
        let fusion_engine = FusionEngine::new();

        // Perform fusion
        let fusion_result = fusion_engine.fuse(&params.embeddings, &alignments);

        info!(
            "fuse_embeddings completed with confidence {}",
            fusion_result.confidence
        );

        self.tool_result_with_pulse(
            id,
            json!({
                "success": true,
                "vector": TeleologicalVectorJson::from_core(&fusion_result.vector, None),
                "confidence": fusion_result.confidence,
                "component_scores": {
                    "purpose_vector": fusion_result.component_scores.purpose_vector,
                    "correlations": fusion_result.component_scores.correlations,
                    "groups": fusion_result.component_scores.groups,
                },
                "metadata": {
                    "fusion_method": params.fusion_method,
                    "profile_applied": params.profile_id.is_some(),
                    "active_embedders": fusion_result.metadata.active_embedders,
                },
            }),
        )
    }
}
