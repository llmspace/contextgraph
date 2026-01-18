//! TELEO-H1: search_teleological handler.
//!
//! Performs cross-correlation search across all 13 embedders using
//! configurable strategies and scopes.

use super::types::SearchTeleologicalParams;
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcResponse};
use context_graph_core::teleological::TeleologicalVector;
use context_graph_core::traits::TeleologicalSearchOptions;
use context_graph_core::types::fingerprint::SemanticFingerprint;
use serde_json::json;
use tracing::{debug, error, info, warn};

impl Handlers {
    /// Handle search_teleological tool call.
    ///
    /// ISSUE-1 FIX: Rewritten to match tool definition in tools.rs (lines 558-644).
    /// Accepts `query_content` (string to embed) OR `query_vector_id` (existing vector ID).
    /// Searches against stored vectors in TeleologicalMemoryStore, NOT a candidates array.
    pub(in crate::handlers) async fn call_search_teleological(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("search_teleological called with: {:?}", arguments);

        // Parse parameters
        let params: SearchTeleologicalParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to parse search_teleological params: {}", e);
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        // =====================================================================
        // FAIL FAST: Require exactly one of query_content or query_vector_id
        // Track both TeleologicalVector and SemanticFingerprint for embedder scores
        // =====================================================================
        let (query_vector, semantic_query): (TeleologicalVector, Option<SemanticFingerprint>) =
            match (&params.query_content, &params.query_vector_id) {
                (None, None) => {
                    error!(
                        "search_teleological: Neither query_content nor query_vector_id provided"
                    );
                    return self.tool_error_with_pulse(
                    id,
                    "FAIL FAST: Must provide either 'query_content' (string to embed) or 'query_vector_id' (existing vector ID). Neither was provided.",
                );
                }
                (Some(_), Some(_)) => {
                    warn!("search_teleological: Both query_content and query_vector_id provided, using query_content");
                    match self
                        .compute_query_vector_from_content(params.query_content.as_ref().unwrap())
                        .await
                    {
                        Ok((v, sem)) => (v, Some(sem)),
                        Err(e) => return self.tool_error_with_pulse(id, &e),
                    }
                }
                (Some(content), None) => {
                    if content.is_empty() {
                        return self.tool_error_with_pulse(
                            id,
                            "FAIL FAST: query_content cannot be empty string",
                        );
                    }
                    match self.compute_query_vector_from_content(content).await {
                        Ok((v, sem)) => (v, Some(sem)),
                        Err(e) => return self.tool_error_with_pulse(id, &e),
                    }
                }
                (None, Some(vector_id)) => match self.retrieve_vector_from_store(vector_id).await {
                    Ok(result) => result,
                    Err(e) => return self.tool_error_with_pulse(id, &e),
                },
            };

        // =====================================================================
        // Search stored vectors using TeleologicalMemoryStore.search_purpose()
        // =====================================================================
        let search_options = TeleologicalSearchOptions {
            top_k: params.max_results,
            min_similarity: params.min_similarity,
            include_deleted: false,
            min_alignment: Some(params.min_similarity),
            embedder_indices: Vec::new(),
            semantic_query,
            include_content: params.include_content,
        };

        let search_start = std::time::Instant::now();
        let search_results = match self
            .teleological_store
            .search_purpose(&query_vector.purpose_vector, search_options)
            .await
        {
            Ok(results) => results,
            Err(e) => {
                error!("search_purpose failed: {}", e);
                return self.tool_error_with_pulse(
                    id,
                    &format!("Search failed: {}. Store may be empty or unavailable.", e),
                );
            }
        };
        let search_duration = search_start.elapsed();

        // =====================================================================
        // TASK-CONTENT-011: Hydrate content if requested
        // =====================================================================
        let mut search_results = search_results;
        if params.include_content && !search_results.is_empty() {
            let ids: Vec<uuid::Uuid> = search_results.iter().map(|r| r.fingerprint.id).collect();
            match self.teleological_store.get_content_batch(&ids).await {
                Ok(contents) => {
                    for (result, content) in search_results.iter_mut().zip(contents.into_iter()) {
                        result.content = content;
                    }
                    debug!(
                        result_count = search_results.len(),
                        with_content = search_results
                            .iter()
                            .filter(|r| r.content.is_some())
                            .count(),
                        "Content hydrated for search results"
                    );
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "Failed to hydrate content for search results (results still valid without content)"
                    );
                }
            }
        }

        // =====================================================================
        // Build response with optional breakdown and content
        // =====================================================================
        let results_json: Vec<serde_json::Value> = search_results
            .iter()
            .enumerate()
            .map(|(rank, result)| {
                let mut entry = json!({
                    "rank": rank,
                    "id": result.fingerprint.id.to_string(),
                    "similarity": result.similarity,
                    "purpose_alignment": result.purpose_alignment,
                    "dominant_embedder": result.dominant_embedder(),
                });

                if params.include_breakdown {
                    entry["breakdown"] = json!({
                        "similarity": result.similarity,
                        "purpose_alignment": result.purpose_alignment,
                        "embedder_scores": result.embedder_scores,
                        "stage_scores": result.stage_scores,
                    });
                }

                if params.include_content {
                    entry["content"] = match &result.content {
                        Some(c) => json!(c),
                        None => serde_json::Value::Null,
                    };
                }

                entry
            })
            .collect();

        info!(
            "search_teleological found {} results in {:?} (max_results={}, min_similarity={})",
            results_json.len(),
            search_duration,
            params.max_results,
            params.min_similarity
        );

        self.tool_result_with_pulse(
            id,
            json!({
                "success": true,
                "strategy": params.strategy,
                "scope": params.scope,
                "query_type": if params.query_content.is_some() { "embedded" } else { "existing_vector" },
                "search_latency_ms": search_duration.as_millis(),
                "num_results": results_json.len(),
                "results": results_json,
            }),
        )
    }
}
