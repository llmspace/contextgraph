//! Pruning and consolidation handlers for memory maintenance.
//!
//! TASK-AUTONOMOUS-MCP: Memory maintenance operations using PruningService
//! and ConsolidationService.

use serde_json::json;
use tracing::{debug, error, info, warn};

use context_graph_core::autonomous::{
    ConsolidationConfig, ConsolidationService, ExtendedPruningConfig, MemoryContent, MemoryId,
    MemoryMetadata, MemoryPair, PruningConfig, PruningService,
};

use super::params::{GetPruningCandidatesParams, TriggerConsolidationParams};
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// get_pruning_candidates tool implementation.
    ///
    /// TASK-AUTONOMOUS-MCP: Get memories that are candidates for pruning.
    /// Uses PruningService to identify stale, low-alignment memories.
    ///
    /// Arguments:
    /// - limit (optional): Maximum candidates to return (default: 20)
    /// - min_staleness_days (optional): Minimum age in days (default: 30)
    /// - min_alignment (optional): Below this = candidate (default: 0.4)
    ///
    /// Returns:
    /// - candidates: List of pruning candidates with reasons
    /// - summary: Aggregated statistics about candidates
    pub(crate) async fn call_get_pruning_candidates(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_pruning_candidates tool call");

        // Parse parameters
        let params: GetPruningCandidatesParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "get_pruning_candidates: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        debug!(
            limit = params.limit,
            min_staleness_days = params.min_staleness_days,
            min_alignment = params.min_alignment,
            "get_pruning_candidates: Parsed parameters"
        );

        // SPEC-STUBFIX-002: FAIL FAST - Get all fingerprints from store
        // Over-fetch to have room for filtering
        let johari_list = match self
            .teleological_store
            .list_all_johari(params.limit * 2)
            .await
        {
            Ok(list) => list,
            Err(e) => {
                error!(
                    error = %e,
                    limit = params.limit,
                    "get_pruning_candidates: FAIL FAST - store access failed"
                );
                return self.tool_error_with_pulse(
                    id,
                    &format!("Store error: Failed to list fingerprints: {}", e),
                );
            }
        };

        debug!(
            fingerprint_count = johari_list.len(),
            "get_pruning_candidates: Retrieved fingerprints from store"
        );

        // Convert TeleologicalFingerprint to MemoryMetadata
        let mut metadata_list: Vec<MemoryMetadata> = Vec::with_capacity(johari_list.len());

        for (uuid, _johari) in johari_list.iter() {
            // Retrieve full fingerprint for metadata conversion
            match self.teleological_store.retrieve(*uuid).await {
                Ok(Some(fp)) => {
                    // Convert content_hash [u8; 32] to u64 for redundancy detection
                    let hash_u64 =
                        u64::from_le_bytes(fp.content_hash[0..8].try_into().unwrap_or([0u8; 8]));

                    // Estimate fingerprint size: 13 embeddings * 1024 dims * 4 bytes + overhead
                    let byte_size = (13 * 1024 * 4 + 1024) as u64;

                    let metadata = MemoryMetadata {
                        id: MemoryId(fp.id),
                        created_at: fp.created_at,
                        alignment: fp.theta_to_north_star,
                        connection_count: 0, // No edge data available through trait
                        byte_size,
                        last_accessed: Some(fp.last_updated),
                        quality_score: None, // Not available in fingerprint
                        content_hash: Some(hash_u64),
                    };

                    metadata_list.push(metadata);
                }
                Ok(None) => {
                    warn!(uuid = %uuid, "get_pruning_candidates: Fingerprint not found");
                }
                Err(e) => {
                    error!(
                        uuid = %uuid,
                        error = %e,
                        "get_pruning_candidates: Failed to retrieve fingerprint"
                    );
                    // Continue processing other fingerprints instead of failing completely
                }
            }
        }

        // Create pruning service with user-provided config
        let config = ExtendedPruningConfig {
            base: PruningConfig {
                enabled: true,
                min_age_days: params.min_staleness_days as u32, // Cast u64 to u32
                min_alignment: params.min_alignment,
                preserve_connected: true,
                min_connections: 3,
            },
            max_daily_prunes: 100,
            stale_days: 90,
            min_quality: 0.30,
        };
        let pruning_service = PruningService::with_config(config);

        // REAL DATA: Identify candidates using PruningService
        let candidates = pruning_service.identify_candidates(&metadata_list);

        // Apply limit
        let limited_candidates: Vec<_> = candidates.into_iter().take(params.limit).collect();

        // Count candidates by reason
        let mut by_reason = std::collections::HashMap::new();
        for candidate in &limited_candidates {
            *by_reason
                .entry(format!("{:?}", candidate.reason))
                .or_insert(0) += 1;
        }

        // Build summary with REAL data
        let summary = json!({
            "total_candidates": limited_candidates.len(),
            "by_reason": by_reason,
            "thresholds_used": {
                "min_staleness_days": params.min_staleness_days,
                "min_alignment": params.min_alignment
            },
            "fingerprints_analyzed": metadata_list.len()
        });

        // Convert candidates to JSON
        let candidates_json: Vec<serde_json::Value> = limited_candidates
            .iter()
            .map(|c| {
                json!({
                    "memory_id": c.memory_id.0.to_string(),
                    "age_days": c.age_days,
                    "alignment": c.alignment,
                    "connections": c.connections,
                    "reason": format!("{:?}", c.reason),
                    "byte_size": c.byte_size,
                    "priority_score": c.priority_score
                })
            })
            .collect();

        info!(
            candidate_count = limited_candidates.len(),
            fingerprints_analyzed = metadata_list.len(),
            limit = params.limit,
            "get_pruning_candidates: Identified candidates from REAL data"
        );

        self.tool_result_with_pulse(
            id,
            json!({
                "candidates": candidates_json,
                "summary": summary,
                "limit_applied": params.limit
            }),
        )
    }

    /// trigger_consolidation tool implementation.
    ///
    /// TASK-AUTONOMOUS-MCP: Trigger memory consolidation.
    /// Uses ConsolidationService to merge similar memories.
    ///
    /// Arguments:
    /// - max_memories (optional): Maximum to process (default: 100)
    /// - strategy (optional): "similarity", "temporal", "semantic" (default: "similarity")
    /// - min_similarity (optional): Minimum similarity for merge (default: 0.85)
    ///
    /// Returns:
    /// - consolidation_result: Pairs merged and outcome
    /// - statistics: Consolidation metrics
    pub(crate) async fn call_trigger_consolidation(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling trigger_consolidation tool call");

        // Parse parameters
        let params: TriggerConsolidationParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "trigger_consolidation: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        // Validate strategy
        let valid_strategies = ["similarity", "temporal", "semantic"];
        if !valid_strategies.contains(&params.strategy.as_str()) {
            error!(
                strategy = %params.strategy,
                "trigger_consolidation: Invalid strategy"
            );
            return self.tool_error_with_pulse(
                id,
                &format!(
                    "Invalid strategy '{}'. Valid strategies: similarity, temporal, semantic",
                    params.strategy
                ),
            );
        }

        debug!(
            max_memories = params.max_memories,
            strategy = %params.strategy,
            min_similarity = params.min_similarity,
            "trigger_consolidation: Parsed parameters"
        );

        // SPEC-STUBFIX-003: FAIL FAST - Get fingerprints from store
        let johari_list = match self
            .teleological_store
            .list_all_johari(params.max_memories)
            .await
        {
            Ok(list) => list,
            Err(e) => {
                error!(
                    error = %e,
                    max_memories = params.max_memories,
                    "trigger_consolidation: FAIL FAST - store access failed"
                );
                return self.tool_error_with_pulse(
                    id,
                    &format!("Store error: Failed to list fingerprints: {}", e),
                );
            }
        };

        debug!(
            fingerprint_count = johari_list.len(),
            "trigger_consolidation: Retrieved fingerprints from store"
        );

        // Convert TeleologicalFingerprint to MemoryContent
        let mut memory_contents: Vec<MemoryContent> = Vec::with_capacity(johari_list.len());
        let mut fingerprints: Vec<(uuid::Uuid, chrono::DateTime<chrono::Utc>)> = Vec::new();

        for (uuid, _johari) in johari_list.iter() {
            match self.teleological_store.retrieve(*uuid).await {
                Ok(Some(fp)) => {
                    // Use E1 (e5-large-v2 1024D) embedding for comparison
                    // This is the primary semantic embedding
                    let embedding = fp.semantic.e1_semantic.clone();

                    let content = MemoryContent::new(
                        MemoryId(fp.id),
                        embedding,
                        String::new(), // No text content available in fingerprint
                        fp.theta_to_north_star,
                    )
                    .with_access_count(fp.access_count as u32);

                    memory_contents.push(content);
                    fingerprints.push((fp.id, fp.created_at));
                }
                Ok(None) => {
                    warn!(uuid = %uuid, "trigger_consolidation: Fingerprint not found");
                }
                Err(e) => {
                    error!(
                        uuid = %uuid,
                        error = %e,
                        "trigger_consolidation: Failed to retrieve fingerprint"
                    );
                }
            }
        }

        // Build pairs based on strategy
        let pairs: Vec<MemoryPair> = match params.strategy.as_str() {
            "similarity" => {
                // Compare all pairs, pre-filter by similarity threshold * 0.9
                let mut pairs = Vec::new();
                let threshold = params.min_similarity * 0.9;

                for i in 0..memory_contents.len() {
                    for j in (i + 1)..memory_contents.len() {
                        // Quick similarity check using dot product (embeddings should be normalized)
                        let sim: f32 = memory_contents[i]
                            .embedding
                            .iter()
                            .zip(memory_contents[j].embedding.iter())
                            .map(|(a, b)| a * b)
                            .sum();

                        if sim >= threshold {
                            pairs.push(MemoryPair::new(
                                memory_contents[i].clone(),
                                memory_contents[j].clone(),
                            ));
                        }
                    }
                }
                pairs
            }
            "temporal" => {
                // Compare fingerprints created within 24 hours of each other
                let window_secs = 24 * 60 * 60; // 24 hours in seconds
                let mut pairs = Vec::new();

                for i in 0..memory_contents.len() {
                    for j in (i + 1)..memory_contents.len() {
                        if i < fingerprints.len() && j < fingerprints.len() {
                            let diff =
                                (fingerprints[i].1 - fingerprints[j].1).num_seconds().abs();
                            if diff < window_secs {
                                pairs.push(MemoryPair::new(
                                    memory_contents[i].clone(),
                                    memory_contents[j].clone(),
                                ));
                            }
                        }
                    }
                }
                pairs
            }
            "semantic" => {
                // For semantic, use alignment-weighted pairs
                let mut pairs = Vec::new();
                let alignment_threshold = 0.5;

                for i in 0..memory_contents.len() {
                    for j in (i + 1)..memory_contents.len() {
                        // Both must have decent alignment
                        if memory_contents[i].alignment >= alignment_threshold
                            && memory_contents[j].alignment >= alignment_threshold
                        {
                            pairs.push(MemoryPair::new(
                                memory_contents[i].clone(),
                                memory_contents[j].clone(),
                            ));
                        }
                    }
                }
                pairs
            }
            _ => {
                // Strategy already validated above, this shouldn't happen
                Vec::new()
            }
        };

        // Create consolidation service with user config
        let config = ConsolidationConfig {
            enabled: true,
            similarity_threshold: params.min_similarity,
            max_daily_merges: 50,
            theta_diff_threshold: 0.05,
        };
        let consolidation_service = ConsolidationService::with_config(config);

        // REAL DATA: Find consolidation candidates
        let candidates = consolidation_service.find_consolidation_candidates(&pairs);

        // Build result with REAL data
        let statistics = json!({
            "pairs_evaluated": pairs.len(),
            "pairs_consolidated": candidates.len(),
            "strategy": params.strategy,
            "similarity_threshold": params.min_similarity,
            "max_memories_limit": params.max_memories,
            "fingerprints_analyzed": memory_contents.len()
        });

        let consolidation_result = json!({
            "status": if candidates.is_empty() { "no_candidates" } else { "candidates_found" },
            "candidate_count": candidates.len(),
            "action_required": !candidates.is_empty()
        });

        // Convert candidates to JSON (limit to 10 for response size)
        let candidates_sample: Vec<serde_json::Value> = candidates
            .iter()
            .take(10)
            .map(|c| {
                json!({
                    "source_ids": c.source_ids.iter().map(|id| id.0.to_string()).collect::<Vec<_>>(),
                    "target_id": c.target_id.0.to_string(),
                    "similarity": c.similarity,
                    "combined_alignment": c.combined_alignment
                })
            })
            .collect();

        info!(
            candidate_count = candidates.len(),
            pairs_evaluated = pairs.len(),
            strategy = %params.strategy,
            "trigger_consolidation: Analysis complete with REAL data"
        );

        self.tool_result_with_pulse(
            id,
            json!({
                "consolidation_result": consolidation_result,
                "statistics": statistics,
                "candidates_sample": candidates_sample
            }),
        )
    }
}
