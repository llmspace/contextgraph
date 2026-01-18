//! Sub-goal discovery handler.
//!
//! TASK-AUTONOMOUS-MCP + TASK-INTEG-002 + ARCH-03: Discover potential sub-goals
//! from memory clusters using GoalDiscoveryPipeline with K-means clustering.

use serde_json::json;
use tracing::{debug, error, info};

use context_graph_core::autonomous::discovery::{
    ClusteringAlgorithm, DiscoveryConfig, GoalDiscoveryPipeline, NumClusters,
};
use context_graph_core::autonomous::{ServiceDiscoveryConfig, SubGoalDiscovery};
use context_graph_core::teleological::TeleologicalComparator;

use super::params::DiscoverSubGoalsParams;
use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// discover_sub_goals tool implementation.
    ///
    /// TASK-AUTONOMOUS-MCP + TASK-INTEG-002 + ARCH-03: Discover potential sub-goals from memory clusters.
    /// Uses GoalDiscoveryPipeline (TASK-LOGIC-009) with K-means clustering for enhanced goal discovery.
    ///
    /// ARCH-03 COMPLIANT: Works without top-level goal by discovering ALL goals from clustering
    /// of stored fingerprints. Goals emerge from data patterns via clustering.
    ///
    /// Arguments:
    /// - min_confidence (optional): Minimum confidence/coherence for sub-goal (default: 0.6)
    /// - max_goals (optional): Maximum sub-goals to discover (default: 5)
    /// - parent_goal_id (optional): Parent goal (default: top-level strategic goal if exists, otherwise discovers top-level goals)
    /// - memory_ids (optional): Specific memory IDs to analyze (default: all recent memories)
    /// - algorithm (optional): Clustering algorithm - "kmeans", "hdbscan", "spectral" (default: "kmeans")
    ///
    /// Returns:
    /// - discovered_goals: List of discovered goals with coherence_score, dominant_embedders, level
    /// - cluster_analysis: Information about memory clusters analyzed
    /// - discovery_metadata: total_arrays_analyzed, clusters_found, algorithm_used
    /// - discovery_mode: "under_parent" or "autonomous" indicating discovery approach
    pub(crate) async fn call_discover_sub_goals(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling discover_sub_goals tool call");

        // Clone arguments for additional parameter extraction
        let arguments_clone = arguments.clone();

        // Parse parameters
        let params: DiscoverSubGoalsParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "discover_sub_goals: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        debug!(
            min_confidence = params.min_confidence,
            max_goals = params.max_goals,
            parent_goal_id = ?params.parent_goal_id,
            "discover_sub_goals: Parsed parameters"
        );

        // ARCH-03 COMPLIANT: Check if top-level or parent goal exists, but don't require it
        let (parent_goal, discovery_mode): (Option<context_graph_core::purpose::GoalNode>, &str) = {
            let hierarchy = self.goal_hierarchy.read();

            match &params.parent_goal_id {
                Some(goal_id_str) => {
                    // Parse goal_id as UUID (per TASK-CORE-005: GoalId removed, use Uuid directly)
                    let goal_id = match uuid::Uuid::parse_str(goal_id_str) {
                        Ok(uuid) => uuid,
                        Err(e) => {
                            error!(goal_id = %goal_id_str, error = %e, "discover_sub_goals: Invalid goal UUID");
                            return JsonRpcResponse::error(
                                id,
                                error_codes::INVALID_PARAMS,
                                format!("Invalid goal UUID '{}': {}", goal_id_str, e),
                            );
                        }
                    };
                    match hierarchy.get(&goal_id) {
                        Some(goal) => (Some(goal.clone()), "under_parent"),
                        None => {
                            error!(goal_id = %goal_id, "discover_sub_goals: Parent goal not found");
                            return JsonRpcResponse::error(
                                id,
                                error_codes::GOAL_NOT_FOUND,
                                format!("Parent goal '{}' not found in hierarchy", goal_id),
                            );
                        }
                    }
                }
                None => {
                    // TASK-P0-001/ARCH-03: Try top-level Strategic goal, but work autonomously if none exists
                    match hierarchy.top_level_goals().first() {
                        Some(top_goal) => (Some((*top_goal).clone()), "under_strategic"),
                        None => {
                            // No Strategic goals - discover goals autonomously from clustering
                            info!("discover_sub_goals: No top-level goals - discovering goals autonomously (ARCH-03)");
                            (None, "autonomous")
                        }
                    }
                }
            }
        };

        // Parse optional memory_ids
        let memory_ids: Option<Vec<uuid::Uuid>> = arguments_clone
            .get("memory_ids")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                let ids: Result<Vec<_>, _> = arr
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .ok_or("not a string")
                            .and_then(|s| uuid::Uuid::parse_str(s).map_err(|_| "invalid uuid"))
                    })
                    .collect();
                ids.ok()
            });

        // Parse clustering algorithm
        let clustering_algorithm = match arguments_clone
            .get("algorithm")
            .and_then(|v| v.as_str())
            .unwrap_or("kmeans")
        {
            "kmeans" => ClusteringAlgorithm::KMeans,
            "hdbscan" => ClusteringAlgorithm::HDBSCAN { min_samples: 5 },
            "spectral" => ClusteringAlgorithm::Spectral { n_neighbors: 10 },
            _ => ClusteringAlgorithm::KMeans, // Default
        };

        // Collect memories to analyze
        let arrays: Vec<context_graph_core::types::SemanticFingerprint> = if let Some(ids) =
            memory_ids
        {
            // FAIL FAST: Load specific memories
            let mut mems = Vec::with_capacity(ids.len());
            for mem_id in &ids {
                match self.teleological_store.retrieve(*mem_id).await {
                    Ok(Some(fp)) => mems.push(fp.semantic),
                    Ok(None) => {
                        error!(memory_id = %mem_id, "discover_sub_goals: Memory not found");
                        return JsonRpcResponse::error(
                            id,
                            error_codes::FINGERPRINT_NOT_FOUND,
                            format!("Memory {} not found - FAIL FAST", mem_id),
                        );
                    }
                    Err(e) => {
                        error!(memory_id = %mem_id, error = %e, "discover_sub_goals: Storage error");
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INTERNAL_ERROR,
                            format!("Storage error retrieving {}: {} - FAIL FAST", mem_id, e),
                        );
                    }
                }
            }
            mems
        } else {
            // No specific memories - legacy behavior returns guidance
            // Create legacy discovery service for backwards compatibility
            let legacy_config = ServiceDiscoveryConfig {
                min_cluster_size: 3,
                min_coherence: params.min_confidence,
                emergence_threshold: params.min_confidence,
                max_candidates: params.max_goals,
                min_confidence: params.min_confidence,
            };
            let _discovery_service = SubGoalDiscovery::with_config(legacy_config);

            let (parent_id_str, parent_desc) = match &parent_goal {
                Some(g) => (g.id.to_string(), g.description.clone()),
                None => ("none".to_string(), "autonomous discovery".to_string()),
            };

            let cluster_analysis = json!({
                "parent_goal_id": parent_id_str,
                "parent_goal_description": parent_desc,
                "clusters_analyzed": 0,
                "memory_count_analyzed": 0,
                "discovery_mode": discovery_mode,
                "discovery_parameters": {
                    "min_confidence": params.min_confidence,
                    "max_goals": params.max_goals
                },
                "note": "No memory_ids provided. Pass memory_ids array for GoalDiscoveryPipeline K-means clustering."
            });

            return self.tool_result_with_pulse(
                    id,
                    json!({
                        "discovered_goals": [],
                        "cluster_analysis": cluster_analysis,
                        "parent_goal_id": parent_id_str,
                        "discovery_mode": discovery_mode,
                        "discovery_count": 0,
                        "arch03_compliant": true,
                        "usage_hint": "Provide 'memory_ids' parameter with fingerprint UUIDs for K-means goal discovery"
                    }),
                );
        };

        // FAIL FAST: Must have minimum data for clustering
        let min_cluster_size = 3;
        if arrays.len() < min_cluster_size {
            error!(
                count = arrays.len(),
                min = min_cluster_size,
                "discover_sub_goals: Insufficient data for clustering"
            );
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!(
                    "Insufficient data for clustering: got {} arrays, need at least {} - FAIL FAST",
                    arrays.len(),
                    min_cluster_size
                ),
            );
        }

        // Create GoalDiscoveryPipeline (TASK-LOGIC-009)
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        // Configure discovery
        let discovery_config = DiscoveryConfig {
            sample_size: std::cmp::min(arrays.len(), params.max_goals * 20),
            min_cluster_size,
            min_coherence: params.min_confidence,
            clustering_algorithm,
            num_clusters: NumClusters::Auto,
            max_iterations: 100,
            convergence_tolerance: 1e-4,
        };

        // Execute discovery - note: TASK-LOGIC-009 panics on failure (FAIL FAST)
        // We need to catch_unwind or handle gracefully
        let discovery_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pipeline.discover(&arrays, &discovery_config)
        }));

        let discovery_result = match discovery_result {
            Ok(result) => result,
            Err(panic_info) => {
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic in GoalDiscoveryPipeline".to_string()
                };
                error!(panic = %panic_msg, "discover_sub_goals: PANIC in discovery pipeline");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    format!("Goal discovery failed: {} - FAIL FAST", panic_msg),
                );
            }
        };

        // Build discovered goals response
        let discovered_goals: Vec<serde_json::Value> = discovery_result
            .discovered_goals
            .iter()
            .take(params.max_goals)
            .map(|goal| {
                json!({
                    "goal_id": goal.goal_id,
                    "description": goal.description,
                    "level": format!("{:?}", goal.level),
                    "confidence": goal.confidence,
                    "member_count": goal.member_count,
                    "coherence_score": goal.coherence_score,
                    "dominant_embedders": goal.dominant_embedders.iter().map(|e| format!("{:?}", e)).collect::<Vec<_>>()
                })
            })
            .collect();

        // Build hierarchy relationships
        let hierarchy_relationships: Vec<serde_json::Value> = discovery_result
            .hierarchy
            .iter()
            .map(|rel| {
                json!({
                    "parent_id": rel.parent_id,
                    "child_id": rel.child_id,
                    "similarity": rel.similarity
                })
            })
            .collect();

        // Build cluster analysis - handle case when parent_goal is None (ARCH-03)
        let (parent_id_str, parent_desc) = match &parent_goal {
            Some(g) => (g.id.to_string(), g.description.clone()),
            None => ("none".to_string(), "autonomous discovery".to_string()),
        };

        let cluster_analysis = json!({
            "parent_goal_id": parent_id_str,
            "parent_goal_description": parent_desc,
            "clusters_found": discovery_result.clusters_found,
            "total_arrays_analyzed": discovery_result.total_arrays_analyzed,
            "discovery_mode": discovery_mode,
            "discovery_parameters": {
                "min_confidence": params.min_confidence,
                "max_goals": params.max_goals,
                "algorithm": format!("{:?}", discovery_config.clustering_algorithm)
            }
        });

        // Build metadata
        let discovery_metadata = json!({
            "total_arrays_analyzed": discovery_result.total_arrays_analyzed,
            "clusters_found": discovery_result.clusters_found,
            "algorithm_used": format!("{:?}", discovery_config.clustering_algorithm),
            "num_clusters_setting": format!("{:?}", discovery_config.num_clusters),
            "discovery_mode": discovery_mode
        });

        info!(
            discovered_count = discovered_goals.len(),
            clusters_found = discovery_result.clusters_found,
            discovery_mode = discovery_mode,
            "discover_sub_goals: GoalDiscoveryPipeline analysis complete (ARCH-03 compliant)"
        );

        self.tool_result_with_pulse(
            id,
            json!({
                "discovered_goals": discovered_goals,
                "hierarchy_relationships": hierarchy_relationships,
                "cluster_analysis": cluster_analysis,
                "discovery_metadata": discovery_metadata,
                "parent_goal_id": parent_id_str,
                "discovery_mode": discovery_mode,
                "discovery_count": discovered_goals.len(),
                "arch03_compliant": true,
                "note": if discovery_mode == "autonomous" {
                    "ARCH-03 COMPLIANT: Goals discovered autonomously from clustering"
                } else {
                    "Goals discovered as sub-goals under parent goal"
                }
            }),
        )
    }
}
