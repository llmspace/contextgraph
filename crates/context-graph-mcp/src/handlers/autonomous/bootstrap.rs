//! Bootstrap handler for autonomous North Star discovery.
//!
//! TASK-AUTONOMOUS-MCP + ARCH-03: Bootstrap autonomous system by DISCOVERING purpose
//! from stored teleological fingerprints. NO MANUAL GOAL SETTING REQUIRED.

use serde_json::json;
use tracing::{debug, error, info, warn};

use context_graph_core::autonomous::discovery::{
    ClusteringAlgorithm, DiscoveryConfig, GoalDiscoveryPipeline, NumClusters,
};
use context_graph_core::teleological::TeleologicalComparator;

use super::error_codes::autonomous_error_codes;
use super::params::AutoBootstrapParams;
use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// auto_bootstrap_north_star tool implementation.
    ///
    /// TASK-AUTONOMOUS-MCP + ARCH-03: Bootstrap autonomous system by DISCOVERING purpose
    /// from stored teleological fingerprints. NO MANUAL GOAL SETTING REQUIRED.
    ///
    /// Per constitution ARCH-03: "System MUST operate autonomously without manual goal setting.
    /// Goals emerge from data patterns via clustering."
    ///
    /// Arguments:
    /// - confidence_threshold (optional): Minimum confidence for bootstrap (default: 0.7)
    /// - max_candidates (optional): Maximum candidates to evaluate (default: 10)
    ///
    /// Returns:
    /// - bootstrap_result: Discovered North Star from clustering stored fingerprints
    /// - initialized_services: List of services now active
    /// - recommendations: Suggested next actions
    pub(crate) async fn call_auto_bootstrap_north_star(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling auto_bootstrap_north_star tool call");

        // Parse parameters
        let params: AutoBootstrapParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "auto_bootstrap_north_star: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        debug!(
            confidence_threshold = params.confidence_threshold,
            max_candidates = params.max_candidates,
            "auto_bootstrap_north_star: Parsed parameters"
        );

        // Check if North Star already exists - if so, just report it
        {
            let hierarchy = self.goal_hierarchy.read();
            if let Some(ns) = hierarchy.north_star() {
                info!(
                    goal_id = %ns.id,
                    "auto_bootstrap_north_star: North Star already exists"
                );

                let initialized_services = vec![
                    "DriftDetector",
                    "DriftCorrector",
                    "PruningService",
                    "ConsolidationService",
                    "SubGoalDiscovery",
                    "ThresholdLearner",
                ];

                return self.tool_result_with_pulse(
                    id,
                    json!({
                        "bootstrap_result": {
                            "goal_id": ns.id.to_string(),
                            "goal_text": ns.description,
                            "confidence": 1.0,
                            "source": "existing_north_star"
                        },
                        "initialized_services": initialized_services,
                        "recommendations": [
                            "Monitor alignment drift regularly with get_alignment_drift",
                            "Run get_pruning_candidates weekly to identify stale memories",
                            "Use discover_sub_goals after significant content accumulation",
                            "Check get_autonomous_status for system health"
                        ],
                        "status": "already_bootstrapped",
                        "note": format!("North Star '{}' already exists. {} service(s) ready.",
                            ns.id, initialized_services.len())
                    }),
                );
            }
        }

        // ARCH-03: DISCOVER North Star from stored fingerprints via clustering
        // First, retrieve all stored fingerprints
        let fingerprints = match self.teleological_store.list_all_johari(1000).await {
            Ok(fps) => fps,
            Err(e) => {
                error!(error = %e, "auto_bootstrap_north_star: Failed to list stored fingerprints");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    format!("Failed to list stored fingerprints: {} - FAIL FAST", e),
                );
            }
        };

        // FAIL FAST: Need fingerprints to discover purpose from
        if fingerprints.is_empty() {
            warn!("auto_bootstrap_north_star: No teleological fingerprints stored yet");
            return JsonRpcResponse::error(
                id,
                autonomous_error_codes::BOOTSTRAP_ERROR,
                "No teleological fingerprints stored. Store memories first using store_memory or compute_teleological_vector, then bootstrap will discover emergent purpose patterns from the stored fingerprints. - FAIL FAST",
            );
        }

        // Retrieve full fingerprints to get semantic arrays
        let fp_ids: Vec<uuid::Uuid> = fingerprints.iter().map(|(id, _)| *id).collect();
        let mut semantic_arrays: Vec<context_graph_core::types::SemanticFingerprint> =
            Vec::with_capacity(fp_ids.len());

        for fp_id in &fp_ids {
            match self.teleological_store.retrieve(*fp_id).await {
                Ok(Some(fp)) => semantic_arrays.push(fp.semantic),
                Ok(None) => {
                    warn!(memory_id = %fp_id, "auto_bootstrap_north_star: Fingerprint listed but not found");
                }
                Err(e) => {
                    error!(memory_id = %fp_id, error = %e, "auto_bootstrap_north_star: Storage error");
                    return JsonRpcResponse::error(
                        id,
                        error_codes::INTERNAL_ERROR,
                        format!("Storage error retrieving {}: {} - FAIL FAST", fp_id, e),
                    );
                }
            }
        }

        // FAIL FAST: Need minimum data for clustering
        let min_cluster_size = 3;
        if semantic_arrays.len() < min_cluster_size {
            warn!(
                count = semantic_arrays.len(),
                min = min_cluster_size,
                "auto_bootstrap_north_star: Insufficient data for goal discovery"
            );
            return JsonRpcResponse::error(
                id,
                autonomous_error_codes::BOOTSTRAP_ERROR,
                format!(
                    "Insufficient data for goal discovery: got {} fingerprints, need at least {}. Store more memories first. - FAIL FAST",
                    semantic_arrays.len(),
                    min_cluster_size
                ),
            );
        }

        // Use GoalDiscoveryPipeline to discover the emergent North Star
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let discovery_config = DiscoveryConfig {
            sample_size: std::cmp::min(semantic_arrays.len(), 500),
            min_cluster_size,
            min_coherence: params.confidence_threshold,
            clustering_algorithm: ClusteringAlgorithm::KMeans,
            num_clusters: NumClusters::Auto,
            max_iterations: 100,
            convergence_tolerance: 1e-4,
        };

        // Execute discovery - catch panics as GoalDiscoveryPipeline fails fast
        let discovery_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pipeline.discover(&semantic_arrays, &discovery_config)
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
                error!(panic = %panic_msg, "auto_bootstrap_north_star: PANIC in discovery pipeline");
                return JsonRpcResponse::error(
                    id,
                    autonomous_error_codes::BOOTSTRAP_ERROR,
                    format!("Goal discovery failed: {} - FAIL FAST", panic_msg),
                );
            }
        };

        // FAIL FAST: Must have discovered at least one goal
        if discovery_result.discovered_goals.is_empty() {
            error!("auto_bootstrap_north_star: No goals discovered from clustering");
            return JsonRpcResponse::error(
                id,
                autonomous_error_codes::BOOTSTRAP_ERROR,
                format!(
                    "No emergent goals discovered. Clustering {} fingerprints found no coherent patterns above confidence threshold {}. Try storing more diverse memories or lowering confidence_threshold. - FAIL FAST",
                    semantic_arrays.len(),
                    params.confidence_threshold
                ),
            );
        }

        // The highest-confidence discovered goal becomes the North Star
        let north_star_candidate = discovery_result
            .discovered_goals
            .iter()
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("At least one goal exists");

        // Create the North Star goal in the hierarchy
        let north_star_description = format!(
            "Emergent purpose: {} (discovered from {} fingerprints, coherence: {:.2})",
            north_star_candidate.description,
            semantic_arrays.len(),
            north_star_candidate.coherence_score
        );

        // Store in goal hierarchy
        let north_star_id = {
            let mut hierarchy = self.goal_hierarchy.write();
            use context_graph_core::purpose::{
                DiscoveryMethod, GoalDiscoveryMetadata, GoalLevel, GoalNode,
            };

            // Create discovery metadata for this bootstrapped goal
            let discovery_metadata = GoalDiscoveryMetadata::new(
                DiscoveryMethod::Clustering,
                north_star_candidate.confidence,
                north_star_candidate.member_count,
                north_star_candidate.coherence_score,
            )
            .expect("Valid discovery metadata");

            // Create the GoalNode using the proper constructor
            let goal = GoalNode::autonomous_goal(
                north_star_description.clone(),
                GoalLevel::NorthStar,
                north_star_candidate.centroid.clone(),
                discovery_metadata,
            )
            .expect("Valid goal node");

            let goal_id = goal.id;

            // Add the goal to the hierarchy
            hierarchy
                .add_goal(goal)
                .expect("North Star should be addable");

            goal_id
        };

        let initialized_services = vec![
            "DriftDetector",
            "DriftCorrector",
            "PruningService",
            "ConsolidationService",
            "SubGoalDiscovery",
            "ThresholdLearner",
        ];

        let recommendations = vec![
            "Monitor alignment drift regularly with get_alignment_drift",
            "Run get_pruning_candidates weekly to identify stale memories",
            "Use discover_sub_goals to find subordinate goals under the North Star",
            "Check get_autonomous_status for system health",
            "The North Star was discovered autonomously - it may evolve as more memories are stored",
        ];

        info!(
            goal_id = %north_star_id,
            confidence = north_star_candidate.confidence,
            coherence = north_star_candidate.coherence_score,
            fingerprints_analyzed = semantic_arrays.len(),
            clusters_found = discovery_result.clusters_found,
            "auto_bootstrap_north_star: North Star DISCOVERED from stored fingerprints (ARCH-03 compliant)"
        );

        self.tool_result_with_pulse(
            id,
            json!({
                "bootstrap_result": {
                    "goal_id": north_star_id.to_string(),
                    "goal_text": north_star_description,
                    "confidence": north_star_candidate.confidence,
                    "coherence_score": north_star_candidate.coherence_score,
                    "source": "discovered_from_clustering",
                    "dominant_embedders": north_star_candidate.dominant_embedders.iter().map(|e| format!("{:?}", e)).collect::<Vec<_>>(),
                    "member_count": north_star_candidate.member_count
                },
                "discovery_metadata": {
                    "fingerprints_analyzed": semantic_arrays.len(),
                    "clusters_found": discovery_result.clusters_found,
                    "goals_discovered": discovery_result.discovered_goals.len(),
                    "algorithm": "KMeans"
                },
                "initialized_services": initialized_services,
                "recommendations": recommendations,
                "status": "bootstrapped",
                "note": format!(
                    "ARCH-03 COMPLIANT: North Star '{}' DISCOVERED autonomously from {} stored fingerprints via K-means clustering. {} service(s) ready.",
                    north_star_id,
                    semantic_arrays.len(),
                    initialized_services.len()
                )
            }),
        )
    }
}
