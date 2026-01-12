//! Purpose query handler.
//!
//! Handles the `purpose/query` MCP method for querying memories
//! by 13D purpose vector similarity.

use serde_json::json;
use tracing::{debug, error, instrument};

use context_graph_core::traits::TeleologicalSearchOptions;
use context_graph_core::types::fingerprint::{PurposeVector, NUM_EMBEDDERS};

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// Handle purpose/query request.
    ///
    /// Query memories by 13D purpose vector similarity.
    ///
    /// # Request Parameters
    /// - `purpose_vector` (optional): 13-element alignment vector [0.0-1.0]
    /// - `min_alignment` (optional): Minimum alignment threshold
    /// - `top_k` (optional): Maximum results, default 10
    /// - `include_scores` (optional): Include per-embedder breakdown, default true
    ///
    /// # Response
    /// - `results`: Array of matching memories with purpose alignment scores
    /// - `query_metadata`: Purpose vector used, timing
    ///
    /// # Error Codes
    /// - INVALID_PARAMS (-32602): Invalid purpose vector format
    /// - PURPOSE_SEARCH_ERROR (-32016): Purpose search failed
    #[instrument(skip(self, params), fields(method = "purpose/query"))]
    pub(in crate::handlers) async fn handle_purpose_query(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = params.unwrap_or(json!({}));

        // Parse purpose vector (required for purpose query)
        let purpose_vector = match params.get("purpose_vector").and_then(|v| v.as_array()) {
            Some(arr) => {
                if arr.len() != NUM_EMBEDDERS {
                    error!(
                        count = arr.len(),
                        "purpose/query: Purpose vector must have 13 elements"
                    );
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INVALID_PARAMS,
                        format!(
                            "purpose_vector must have {} elements, got {}",
                            NUM_EMBEDDERS,
                            arr.len()
                        ),
                    );
                }

                let mut alignments = [0.0f32; NUM_EMBEDDERS];
                for (i, v) in arr.iter().enumerate() {
                    let value = v.as_f64().unwrap_or(0.0) as f32;
                    if !(0.0..=1.0).contains(&value) {
                        error!(
                            index = i,
                            value = value,
                            "purpose/query: Purpose vector values must be in [0.0, 1.0]"
                        );
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INVALID_PARAMS,
                            format!(
                                "purpose_vector[{}] = {} is out of range [0.0, 1.0]",
                                i, value
                            ),
                        );
                    }
                    alignments[i] = value;
                }

                // Find dominant embedder (highest alignment)
                let dominant_embedder = alignments
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0);

                // Compute coherence (inverse of standard deviation)
                let mean: f32 = alignments.iter().sum::<f32>() / NUM_EMBEDDERS as f32;
                let variance: f32 = alignments.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                    / NUM_EMBEDDERS as f32;
                let coherence = 1.0 / (1.0 + variance.sqrt());

                PurposeVector {
                    alignments,
                    dominant_embedder,
                    coherence,
                    stability: 1.0,
                }
            }
            None => {
                error!("purpose/query: Missing 'purpose_vector' parameter");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing required 'purpose_vector' parameter (array of 13 floats in [0.0, 1.0])",
                );
            }
        };

        let top_k = params
            .get("topK")
            .or_else(|| params.get("top_k"))
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let min_alignment = params
            .get("minAlignment")
            .or_else(|| params.get("min_alignment"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        let include_scores = params
            .get("include_scores")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let search_start = std::time::Instant::now();

        // Build search options
        let mut options = TeleologicalSearchOptions::quick(top_k);
        if let Some(align) = min_alignment {
            options = options.with_min_alignment(align);
        }

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
                        let mut result = json!({
                            "id": r.fingerprint.id.to_string(),
                            "purpose_alignment": r.purpose_alignment,
                            "theta_to_north_star": r.fingerprint.theta_to_north_star,
                        });

                        if include_scores {
                            result["purpose_vector"] =
                                json!(r.fingerprint.purpose_vector.alignments.to_vec());
                            result["dominant_embedder"] =
                                json!(r.fingerprint.purpose_vector.dominant_embedder);
                            result["coherence"] = json!(r.fingerprint.purpose_vector.coherence);
                        }

                        result["johari_quadrant"] =
                            json!(format!("{:?}", r.fingerprint.johari.dominant_quadrant(0)));

                        result
                    })
                    .collect();

                debug!(
                    count = results.len(),
                    latency_ms = search_latency_ms,
                    "purpose/query: Completed"
                );

                JsonRpcResponse::success(
                    id,
                    json!({
                        "results": results_json,
                        "count": results.len(),
                        "query_metadata": {
                            "purpose_vector_used": purpose_vector.alignments.to_vec(),
                            "min_alignment_filter": min_alignment,
                            "dominant_embedder": purpose_vector.dominant_embedder,
                            "query_coherence": purpose_vector.coherence,
                            "search_time_ms": search_latency_ms
                        }
                    }),
                )
            }
            Err(e) => {
                error!(error = %e, "purpose/query: FAILED");
                JsonRpcResponse::error(
                    id,
                    error_codes::PURPOSE_SEARCH_ERROR,
                    format!("Purpose query failed: {}", e),
                )
            }
        }
    }

    // NOTE: handle_north_star_alignment REMOVED per TASK-CORE-001 (ARCH-03)
    // Manual North Star alignment creates single 1024D embeddings incompatible with 13-embedder arrays.
    // Calls to purpose/north_star_alignment now return METHOD_NOT_FOUND (-32601).
    // Use auto_bootstrap_north_star tool for autonomous goal discovery instead.
}
