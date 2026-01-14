//! TELEO-H3: fuse_embeddings handler.
//!
//! Fuses multiple embeddings using synergy matrix and optional profile.
//!
//! # Constitution Compliance
//!
//! Per AP-03, AP-05: Uses alignment-based fusion via `fuse_from_alignments()`.

use super::types::FuseEmbeddingsParams;
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
    ///
    /// # Constitution Compliance
    ///
    /// Per AP-03, AP-05: Uses `fuse_from_alignments()` for alignment-based fusion.
    /// Returns `AlignmentFusionResult` with purpose_vector and group_alignments.
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

        // Convert to fixed-size array (FAIL FAST on wrong size)
        let alignments_array: [f32; NUM_EMBEDDERS] = alignments
            .try_into()
            .expect("FAIL FAST: alignments must have exactly 13 elements");

        // Create fusion engine
        let fusion_engine = FusionEngine::new();

        // Perform alignment-based fusion (AP-03, AP-05 compliant)
        // NOTE: Using fuse_from_alignments() instead of deprecated fuse()
        let fusion_result = fusion_engine.fuse_from_alignments(&alignments_array);

        info!(
            target: "mcp_fusion",
            confidence = %fusion_result.confidence,
            active_embedders = %fusion_result.active_embedders,
            "fuse_embeddings completed (alignment-based)"
        );

        // Map dominant_group index to human-readable string
        let dominant_group_str = match fusion_result.dominant_group {
            Some(0) => "Semantic",
            Some(1) => "Temporal",
            Some(2) => "Structural",
            Some(3) => "Experiential",
            _ => "None",
        };

        // Build response with alignment-based result structure
        // NOTE: Response structure changed from deprecated fuse() format
        self.tool_result_with_pulse(
            id,
            json!({
                "success": true,
                "purpose_vector": fusion_result.purpose_vector,
                "group_alignments": {
                    "semantic": fusion_result.group_alignments[0],
                    "temporal": fusion_result.group_alignments[1],
                    "structural": fusion_result.group_alignments[2],
                    "experiential": fusion_result.group_alignments[3],
                },
                "confidence": fusion_result.confidence,
                "metadata": {
                    "fusion_method": params.fusion_method,
                    "profile_applied": params.profile_id.is_some(),
                    "active_embedders": fusion_result.active_embedders,
                    "dominant_group": dominant_group_str,
                    "constitution_compliant": true,
                    "api_version": "5.0.0",
                },
            }),
        )
    }
}
