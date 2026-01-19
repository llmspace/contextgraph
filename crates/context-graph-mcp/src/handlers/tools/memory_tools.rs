//! Memory operation tool implementations (inject_context, store_memory, search_graph).

use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::{debug, error, warn};

use context_graph_core::purpose::{
    DefaultPurposeComputer, PurposeComputeConfig, PurposeVectorComputer,
};
use context_graph_core::traits::TeleologicalSearchOptions;
use context_graph_core::types::fingerprint::{
    PurposeVector, TeleologicalFingerprint, NUM_EMBEDDERS,
};
use context_graph_core::teleological::matrix_search::embedder_names;
use context_graph_core::types::UtlContext;

use crate::protocol::JsonRpcId;
use crate::protocol::JsonRpcResponse;

use super::super::Handlers;

impl Handlers {
    /// inject_context tool implementation.
    ///
    /// TASK-S001: Updated to use TeleologicalMemoryStore with 13-embedding fingerprint.
    ///
    /// Injects context into the memory graph with UTL metrics computation.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(crate) async fn call_inject_context(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c.to_string(),
            Some(_) => return self.tool_error_with_pulse(id, "Content cannot be empty"),
            None => return self.tool_error_with_pulse(id, "Missing 'content' parameter"),
        };

        let rationale = args.get("rationale").and_then(|v| v.as_str()).unwrap_or("");
        let _importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        // Get goal_vector from top-level strategic goal for UTL alignment computation
        // Per constitution.yaml: alignment = cos(content_embedding, goal_vector)
        // Without goal_vector, alignment always returns 1.0 (useless)
        // Per TASK-CORE-005: Use E1 semantic embedding from TeleologicalArray for UTL alignment
        let goal_vector = {
            let hierarchy = self.goal_hierarchy.read();
            hierarchy
                .top_level_goals()
                .first()
                .map(|ns| ns.teleological_array.e1_semantic.clone())
        };

        // Compute UTL metrics for the content
        let context = UtlContext {
            goal_vector,
            ..Default::default()
        };
        let metrics = match self.utl_processor.compute_metrics(&content, &context).await {
            Ok(m) => m,
            Err(e) => {
                error!(error = %e, "inject_context: UTL processing FAILED");
                return self.tool_error_with_pulse(id, &format!("UTL processing failed: {}", e));
            }
        };

        // Generate all 13 embeddings using MultiArrayEmbeddingProvider
        let embedding_output = match self.multi_array_provider.embed_all(&content).await {
            Ok(output) => output,
            Err(e) => {
                error!(error = %e, "inject_context: Multi-array embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Embedding failed: {}", e));
            }
        };

        // Compute content hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let content_hash: [u8; 32] = hasher.finalize().into();

        // AUTONOMOUS OPERATION: Compute purpose vector if top-level goal exists,
        // otherwise use default (neutral) alignment.
        //
        // From contextprd.md: "The array [of 13 embeddings] IS the teleological vector"
        // Purpose alignment is SECONDARY metadata - the 13-embedding fingerprint is primary.
        // This allows autonomous operation without manual goal configuration.
        //
        // When top-level goal exists: PV = [A(E1,V), A(E2,V), ..., A(E13,V)]
        // where A(Ei, V) = cos(theta) between embedder i and goal V
        //
        // When no top-level goal: PV = [0.0; 13] (neutral alignment)
        // Memories can be stored immediately; purpose can be recomputed later.
        let purpose_vector = {
            let hierarchy = self.goal_hierarchy.read().clone();

            // If no top-level goal is defined, use default purpose vector
            // This enables AUTONOMOUS operation - no manual configuration required
            if hierarchy.top_level_goals().is_empty() {
                debug!(
                    "inject_context: No top-level goal configured. Using default purpose vector. \
                     Memory will be stored with neutral alignment (can be recomputed later)."
                );
                PurposeVector::default()
            } else {
                // Compute purpose vector using DefaultPurposeComputer
                // This computes alignment for each of 13 embedding spaces
                let config = PurposeComputeConfig::with_hierarchy(hierarchy);

                match DefaultPurposeComputer::new()
                    .compute_purpose(&embedding_output.fingerprint, &config)
                    .await
                {
                    Ok(pv) => {
                        debug!(
                            aggregate_alignment = pv.aggregate_alignment(),
                            dominant_embedder = pv.dominant_embedder,
                            coherence = pv.coherence,
                            "inject_context: Purpose vector computed for semantic fingerprint"
                        );
                        pv
                    }
                    Err(e) => {
                        // If top-level goal exists but computation fails, THAT is an error
                        error!(
                            error = %e,
                            "inject_context: Failed to compute purpose vector. \
                             Cannot store memory without alignment metadata."
                        );
                        return self.tool_error_with_pulse(
                            id,
                            &format!("Purpose vector computation failed: {}", e),
                        );
                    }
                }
            }
        };

        // TASK-FIX-CLUSTERING: Compute cluster array BEFORE fingerprint is consumed
        // This must be done before TeleologicalFingerprint::new() moves the semantic fingerprint.
        let cluster_array = embedding_output.fingerprint.to_cluster_array();

        // Create TeleologicalFingerprint with REAL computed purpose vector
        let fingerprint =
            TeleologicalFingerprint::new(embedding_output.fingerprint, purpose_vector, content_hash);
        let fingerprint_id = fingerprint.id;

        // Store in TeleologicalMemoryStore
        if let Err(e) = self.teleological_store.store(fingerprint).await {
            error!(error = %e, "inject_context: Storage FAILED");
            return self.tool_error_with_pulse(id, &format!("Storage failed: {}", e));
        }

        // TASK-FIX-CLUSTERING: Insert into cluster_manager for topic detection
        // This enables MultiSpaceClusterManager to track this memory for HDBSCAN/BIRCH clustering.
        // Per PRD Section 5: Topics emerge from multi-space clustering with weighted_agreement >= 2.5.
        {
            let mut cluster_mgr = self.cluster_manager.write();
            if let Err(e) = cluster_mgr.insert(fingerprint_id, &cluster_array) {
                // Non-fatal: fingerprint is stored, clustering can be retried via detect_topics
                warn!(
                    fingerprint_id = %fingerprint_id,
                    error = %e,
                    "inject_context: Failed to insert into cluster_manager. \
                     Topic detection may not include this memory until next recluster."
                );
            } else {
                debug!(
                    fingerprint_id = %fingerprint_id,
                    "inject_context: Inserted into cluster_manager for topic detection"
                );
            }
        }

        // TASK-CONTENT-001: Store content text alongside fingerprint
        // Content storage failure is non-fatal - fingerprint is primary data
        // Pattern matches store_memory implementation for API consistency
        if let Err(e) = self
            .teleological_store
            .store_content(fingerprint_id, &content)
            .await
        {
            warn!(
                fingerprint_id = %fingerprint_id,
                error = %e,
                content_size = content.len(),
                "inject_context: Failed to store content text (fingerprint saved successfully). \
                 Content retrieval will return None for this fingerprint."
            );
        } else {
            debug!(
                fingerprint_id = %fingerprint_id,
                content_size = content.len(),
                "inject_context: Content text stored successfully"
            );
        }

        self.tool_result_with_pulse(
            id,
            json!({
                "fingerprintId": fingerprint_id.to_string(),
                "rationale": rationale,
                "embedderCount": NUM_EMBEDDERS,
                "embeddingLatencyMs": embedding_output.total_latency.as_millis(),
                "utl": {
                    "learningScore": metrics.learning_score,
                    "entropy": metrics.entropy,
                    "coherence": metrics.coherence,
                    "surprise": metrics.surprise
                }
            }),
        )
    }

    /// store_memory tool implementation.
    ///
    /// TASK-S001: Updated to use TeleologicalMemoryStore with 13-embedding fingerprint.
    ///
    /// Stores content in the memory graph.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(crate) async fn call_store_memory(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c.to_string(),
            Some(_) => return self.tool_error_with_pulse(id, "Content cannot be empty"),
            None => return self.tool_error_with_pulse(id, "Missing 'content' parameter"),
        };

        let _importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        // Generate all 13 embeddings using MultiArrayEmbeddingProvider
        let embedding_output = match self.multi_array_provider.embed_all(&content).await {
            Ok(output) => output,
            Err(e) => {
                error!(error = %e, "store_memory: Multi-array embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Embedding failed: {}", e));
            }
        };

        // Compute content hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let content_hash: [u8; 32] = hasher.finalize().into();

        // ARCH-03: AUTONOMOUS OPERATION - Compute purpose vector if strategic goal exists,
        // otherwise use default (neutral) alignment for autonomous seeding.
        //
        // From contextprd.md: "The array [of 13 embeddings] IS the teleological vector"
        // Purpose alignment is SECONDARY metadata - the 13-embedding fingerprint is primary.
        // This allows autonomous operation without manual strategic goal configuration.
        //
        // When strategic goal exists: PV = [A(E1,V), A(E2,V), ..., A(E13,V)]
        // where A(Ei, V) = cos(theta) between embedder i and strategic goal V
        //
        // When no strategic goal: PV = [0.0; 13] (neutral alignment)
        // Memories can be stored immediately; purpose can be recomputed later via
        // topic clustering once enough fingerprints exist.
        let purpose_vector = {
            let hierarchy = self.goal_hierarchy.read().clone();

            // If no strategic goal is defined, use default purpose vector
            // This enables AUTONOMOUS operation per ARCH-03 - no manual configuration required
            if hierarchy.top_level_goals().is_empty() {
                debug!(
                    "store_memory: No strategic goal configured. Using default purpose vector. \
                     Memory will be stored with neutral alignment (can be recomputed later \
                     via topic clustering)."
                );
                PurposeVector::default()
            } else {
                // Compute purpose vector using DefaultPurposeComputer
                // This computes alignment for each of 13 embedding spaces
                let config = PurposeComputeConfig::with_hierarchy(hierarchy);

                match DefaultPurposeComputer::new()
                    .compute_purpose(&embedding_output.fingerprint, &config)
                    .await
                {
                    Ok(pv) => {
                        debug!(
                            aggregate_alignment = pv.aggregate_alignment(),
                            dominant_embedder = pv.dominant_embedder,
                            coherence = pv.coherence,
                            "store_memory: Purpose vector computed for semantic fingerprint"
                        );
                        pv
                    }
                    Err(e) => {
                        // If strategic goal exists but computation fails, THAT is an error
                        error!(
                            error = %e,
                            "store_memory: Failed to compute purpose vector. \
                             Cannot store memory without alignment metadata."
                        );
                        return self.tool_error_with_pulse(
                            id,
                            &format!("Purpose vector computation failed: {}", e),
                        );
                    }
                }
            }
        };

        // TASK-FIX-CLUSTERING: Compute cluster array BEFORE fingerprint is consumed
        // This must be done before TeleologicalFingerprint::new() moves the semantic fingerprint.
        let cluster_array = embedding_output.fingerprint.to_cluster_array();

        // Create TeleologicalFingerprint with REAL computed purpose vector
        let fingerprint =
            TeleologicalFingerprint::new(embedding_output.fingerprint, purpose_vector, content_hash);
        let fingerprint_id = fingerprint.id;

        match self.teleological_store.store(fingerprint).await {
            Ok(_) => {
                // TASK-FIX-CLUSTERING: Insert into cluster_manager for topic detection
                // This enables MultiSpaceClusterManager to track this memory for HDBSCAN/BIRCH clustering.
                // Per PRD Section 5: Topics emerge from multi-space clustering with weighted_agreement >= 2.5.
                {
                    let mut cluster_mgr = self.cluster_manager.write();
                    if let Err(e) = cluster_mgr.insert(fingerprint_id, &cluster_array) {
                        // Non-fatal: fingerprint is stored, clustering can be retried via detect_topics
                        warn!(
                            fingerprint_id = %fingerprint_id,
                            error = %e,
                            "store_memory: Failed to insert into cluster_manager. \
                             Topic detection may not include this memory until next recluster."
                        );
                    } else {
                        debug!(
                            fingerprint_id = %fingerprint_id,
                            "store_memory: Inserted into cluster_manager for topic detection"
                        );
                    }
                }

                // TASK-CONTENT-010: Store content text alongside fingerprint
                // Content storage failure is non-fatal - fingerprint is primary data
                if let Err(e) = self
                    .teleological_store
                    .store_content(fingerprint_id, &content)
                    .await
                {
                    warn!(
                        fingerprint_id = %fingerprint_id,
                        error = %e,
                        content_size = content.len(),
                        "store_memory: Failed to store content text (fingerprint saved successfully). \
                         Content retrieval will return None for this fingerprint."
                    );
                } else {
                    debug!(
                        fingerprint_id = %fingerprint_id,
                        content_size = content.len(),
                        "store_memory: Content text stored successfully"
                    );
                }

                self.tool_result_with_pulse(
                    id,
                    json!({
                        "fingerprintId": fingerprint_id.to_string(),
                        "embedderCount": NUM_EMBEDDERS,
                        "embeddingLatencyMs": embedding_output.total_latency.as_millis()
                    }),
                )
            }
            Err(e) => {
                error!(error = %e, "store_memory: Storage FAILED");
                self.tool_error_with_pulse(id, &format!("Storage failed: {}", e))
            }
        }
    }

    /// search_graph tool implementation.
    ///
    /// TASK-S001: Updated to use TeleologicalMemoryStore search_semantic.
    ///
    /// Searches the memory graph for matching content.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(crate) async fn call_search_graph(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            Some(_) => return self.tool_error_with_pulse(id, "Query cannot be empty"),
            None => return self.tool_error_with_pulse(id, "Missing 'query' parameter"),
        };

        let top_k = args.get("topK").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        // TASK-CONTENT-002: Parse includeContent parameter (default: false for backward compatibility)
        let include_content = args
            .get("includeContent")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let options = TeleologicalSearchOptions::quick(top_k);

        // Generate query embedding
        let query_embedding = match self.multi_array_provider.embed_all(query).await {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "search_graph: Query embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Query embedding failed: {}", e));
            }
        };

        match self
            .teleological_store
            .search_semantic(&query_embedding, options)
            .await
        {
            Ok(results) => {
                // TASK-CONTENT-003: Hydrate content if requested
                // Batch retrieve content for all results to minimize I/O
                let contents: Vec<Option<String>> = if include_content && !results.is_empty() {
                    let ids: Vec<uuid::Uuid> = results.iter().map(|r| r.fingerprint.id).collect();
                    match self.teleological_store.get_content_batch(&ids).await {
                        Ok(c) => c,
                        Err(e) => {
                            warn!(
                                error = %e,
                                result_count = results.len(),
                                "search_graph: Content hydration failed. Results will not include content."
                            );
                            // Return None for all - graceful degradation
                            vec![None; ids.len()]
                        }
                    }
                } else {
                    // Not requested or no results - empty vec
                    vec![]
                };

                let results_json: Vec<_> = results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        // Convert embedder index to human-readable name (E1_Semantic, etc.)
                        let dominant_idx = r.dominant_embedder();
                        let dominant_name = embedder_names::name(dominant_idx);

                        let mut entry = json!({
                            "fingerprintId": r.fingerprint.id.to_string(),
                            "similarity": r.similarity,
                            "purposeAlignment": r.purpose_alignment,
                            "dominantEmbedder": dominant_name,
                            "alignmentScore": r.fingerprint.alignment_score
                        });
                        // Only include content field when includeContent=true
                        if include_content {
                            entry["content"] = match contents.get(i).and_then(|c| c.as_ref()) {
                                Some(c) => json!(c),
                                None => serde_json::Value::Null,
                            };
                        }
                        entry
                    })
                    .collect();

                self.tool_result_with_pulse(
                    id,
                    json!({ "results": results_json, "count": results_json.len() }),
                )
            }
            Err(e) => {
                error!(error = %e, "search_graph: Search FAILED");
                self.tool_error_with_pulse(id, &format!("Search failed: {}", e))
            }
        }
    }
}
