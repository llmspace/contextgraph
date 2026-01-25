//! Temporal tool implementations (search_recent).
//!
//! Per Constitution v6.5 and ARCH-25:
//! - E2 (V_freshness) finds recency patterns
//! - Temporal boost is POST-RETRIEVAL only, NOT in similarity fusion
//!
//! The search_recent tool provides a dedicated interface for temporal search,
//! automatically applying E2 freshness boost with configurable decay functions.

use chrono::Utc;
use tracing::{debug, error};

use context_graph_core::traits::{DecayFunction, SearchStrategy, TeleologicalSearchOptions};

use super::temporal_dtos::{
    compute_recency_score, format_age, SearchRecentParams, SearchRecentResponse,
    TemporalConfigSummary, TemporalSearchResultEntry,
};
use crate::handlers::core::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// search_recent tool implementation.
    ///
    /// Searches with E2 temporal boost applied POST-RETRIEVAL per ARCH-25.
    /// Returns results sorted by boosted score (semantic * temporal boost).
    pub(crate) async fn call_search_recent(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse parameters
        let params: SearchRecentParams = match serde_json::from_value(args) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "search_recent: Parameter parsing FAILED");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        if params.query.is_empty() {
            return self.tool_error_with_pulse(id, "Query cannot be empty");
        }

        // Clamp temporal weight
        let temporal_weight = params.temporal_weight.clamp(0.1, 1.0);
        let decay_function: DecayFunction = params.decay_function.into();

        debug!(
            query_preview = %params.query.chars().take(50).collect::<String>(),
            top_k = params.top_k,
            temporal_weight = temporal_weight,
            decay_function = ?decay_function,
            temporal_scale = ?params.temporal_scale,
            "search_recent: Starting temporal search"
        );

        // Generate query embedding
        let query_embedding = match self.multi_array_provider.embed_all(&params.query).await {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "search_recent: Query embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Query embedding failed: {}", e));
            }
        };

        // Build search options - use E1Only strategy for base search
        let mut options = TeleologicalSearchOptions::default();
        options.top_k = params.top_k * 2; // Over-fetch for temporal reranking
        options.min_similarity = params.min_similarity;
        options.strategy = SearchStrategy::E1Only;
        options.include_content = params.include_content;

        // Run base semantic search
        let results = match self
            .teleological_store
            .search_semantic(&query_embedding, options)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "search_recent: Search FAILED");
                return self.tool_error_with_pulse(id, &format!("Search failed: {}", e));
            }
        };

        if results.is_empty() {
            return self.tool_result_with_pulse(
                id,
                serde_json::to_value(SearchRecentResponse {
                    query: params.query,
                    results: vec![],
                    count: 0,
                    temporal_config: TemporalConfigSummary {
                        temporal_weight,
                        decay_function: format!("{:?}", decay_function),
                        temporal_scale: format!("{:?}", params.temporal_scale),
                    },
                })
                .unwrap(),
            );
        }

        // Apply temporal boost POST-RETRIEVAL
        let now_ms = Utc::now().timestamp_millis();
        let horizon_secs = params.temporal_scale.horizon_seconds();

        let mut boosted_results: Vec<TemporalSearchResultEntry> = results
            .into_iter()
            .map(|r| {
                let memory_ts = r.fingerprint.created_at.timestamp_millis();
                let recency_score =
                    compute_recency_score(memory_ts, now_ms, decay_function, horizon_secs);

                // Per ARCH-25: Temporal boost is multiplicative POST-retrieval
                // boosted_score = semantic_score * (1 + temporal_weight * (recency_score - 0.5))
                let boost_factor = (1.0 + temporal_weight * (recency_score - 0.5)).clamp(0.8, 1.2);
                let final_score = r.similarity * boost_factor;

                let age_description = format_age(memory_ts, now_ms);
                let created_at = r.fingerprint.created_at.to_rfc3339();

                TemporalSearchResultEntry {
                    id: r.fingerprint.id.to_string(),
                    semantic_score: r.similarity,
                    recency_score,
                    final_score,
                    age_description,
                    content: r.content,
                    created_at,
                }
            })
            .collect();

        // Sort by final score descending
        boosted_results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to requested top_k
        boosted_results.truncate(params.top_k);
        let count = boosted_results.len();

        debug!(
            result_count = count,
            "search_recent: Temporal search complete"
        );

        // Build response
        let response = SearchRecentResponse {
            query: params.query,
            results: boosted_results,
            count,
            temporal_config: TemporalConfigSummary {
                temporal_weight,
                decay_function: format!("{:?}", decay_function),
                temporal_scale: format!("{:?}", params.temporal_scale),
            },
        };

        match serde_json::to_value(&response) {
            Ok(json) => self.tool_result_with_pulse(id, json),
            Err(e) => {
                error!(error = %e, "search_recent: Response serialization FAILED");
                self.tool_error_with_pulse(id, &format!("Response serialization failed: {}", e))
            }
        }
    }
}
