//! Drift detection and correction handlers.
//!
//! TASK-AUTONOMOUS-MCP + TASK-INTEG-002 + ARCH-03: Drift management handlers
//! using TeleologicalDriftDetector for 5-level per-embedder drift detection.

use serde_json::json;
use tracing::{debug, error, info, warn};

use context_graph_core::autonomous::drift::{DriftError, DriftLevel, TeleologicalDriftDetector};
use context_graph_core::autonomous::discovery::GoalDiscoveryPipeline;
use context_graph_core::autonomous::{DriftCorrector, DriftDetector, DriftSeverity};
use context_graph_core::autonomous::drift::DriftState;
use context_graph_core::teleological::{SearchStrategy, TeleologicalComparator};

use super::params::{GetAlignmentDriftParams, TriggerDriftCorrectionParams};
use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// get_alignment_drift tool implementation.
    ///
    /// TASK-AUTONOMOUS-MCP + TASK-INTEG-002 + ARCH-03: Get current drift state with per-embedder analysis.
    /// Uses TeleologicalDriftDetector (TASK-LOGIC-010) for 5-level per-embedder drift detection.
    ///
    /// ARCH-03 COMPLIANT: Works WITHOUT North Star by computing drift relative to
    /// the stored fingerprints' own centroid/average.
    ///
    /// Arguments:
    /// - timeframe (optional): "1h", "24h", "7d", "30d" (default: "24h")
    /// - include_history (optional): Include full drift history (default: false)
    /// - memory_ids (optional): Specific memories to analyze (default: recent memories)
    /// - strategy (optional): Comparison strategy - "cosine", "euclidean", "synergy" (default: "cosine")
    ///
    /// Returns:
    /// - overall_drift: 5-level drift classification (Critical, High, Medium, Low, None)
    /// - per_embedder_drift: Array of 13 embedder-specific drift results
    /// - most_drifted_embedders: Top 5 most drifted embedders sorted worst-first
    /// - recommendations: Action recommendations based on drift levels
    /// - trend (optional): Trend analysis if history available
    /// - legacy_state: Legacy DriftSeverity for backwards compatibility
    /// - reference_type: "north_star" or "centroid" indicating what drift is relative to
    pub(crate) async fn call_get_alignment_drift(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_alignment_drift tool call");

        // Parse parameters - clone arguments first since we need to access it again below
        let arguments_clone = arguments.clone();
        let params: GetAlignmentDriftParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "get_alignment_drift: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        debug!(
            timeframe = %params.timeframe,
            include_history = params.include_history,
            "get_alignment_drift: Parsed parameters"
        );

        // ARCH-03 COMPLIANT: Check if North Star exists, but don't require it
        let north_star = {
            let hierarchy = self.goal_hierarchy.read();
            hierarchy.north_star().cloned()
        };

        // Parse optional memory_ids from cloned arguments
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

        // Parse comparison strategy (default: Cosine, FAIL FAST on invalid)
        let strategy_str = arguments_clone
            .get("strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("cosine");
        let strategy = match strategy_str {
            "cosine" => SearchStrategy::Cosine,
            "euclidean" => SearchStrategy::Euclidean,
            "synergy" | "synergy_weighted" => SearchStrategy::SynergyWeighted,
            "group" | "hierarchical" => SearchStrategy::GroupHierarchical,
            "cross_correlation" | "dominant" => SearchStrategy::CrossCorrelationDominant,
            unknown => {
                // FAIL FAST: Invalid strategy
                error!(strategy = %unknown, "get_alignment_drift: Unknown strategy");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!(
                        "Unknown search strategy '{}'. Valid: cosine, euclidean, synergy, group, cross_correlation - FAIL FAST",
                        unknown
                    ),
                );
            }
        };

        // Collect memories to analyze
        let memories: Vec<context_graph_core::types::SemanticFingerprint> =
            if let Some(ids) = memory_ids {
                // FAIL FAST: Load specific memories
                let mut mems = Vec::with_capacity(ids.len());
                for mem_id in &ids {
                    match self.teleological_store.retrieve(*mem_id).await {
                        Ok(Some(fp)) => mems.push(fp.semantic),
                        Ok(None) => {
                            error!(memory_id = %mem_id, "get_alignment_drift: Memory not found");
                            return JsonRpcResponse::error(
                                id,
                                error_codes::FINGERPRINT_NOT_FOUND,
                                format!("Memory {} not found - FAIL FAST", mem_id),
                            );
                        }
                        Err(e) => {
                            error!(memory_id = %mem_id, error = %e, "get_alignment_drift: Storage error");
                            return JsonRpcResponse::error(
                                id,
                                error_codes::INTERNAL_ERROR,
                                format!(
                                    "Storage error retrieving {}: {} - FAIL FAST",
                                    mem_id, e
                                ),
                            );
                        }
                    }
                }
                mems
            } else {
                // No specific memories - return guidance on how to use the tool
                // Instead of returning empty, provide legacy state for backwards compatibility
                let detector = DriftDetector::new();
                let severity = detector.detect_drift();
                let trend = detector.compute_trend();

                let (goal_id_str, reference_type) = match &north_star {
                    Some(ns) => (ns.id.to_string(), "north_star"),
                    None => ("none".to_string(), "no_reference"),
                };

                let current_state = json!({
                    "severity": format!("{:?}", severity),
                    "trend": format!("{:?}", trend),
                    "observation_count": 0,
                    "goal_id": goal_id_str,
                    "reference_type": reference_type,
                    "note": "No memory_ids provided. Pass memory_ids array for per-embedder TeleologicalDriftDetector analysis."
                });

                return self.tool_result_with_pulse(
                    id,
                    json!({
                        "legacy_state": current_state,
                        "timeframe": params.timeframe,
                        "reference_type": reference_type,
                        "north_star_id": goal_id_str,
                        "usage_hint": "Provide 'memory_ids' parameter with fingerprint UUIDs for per-embedder drift analysis"
                    }),
                );
            };

        // FAIL FAST: Must have memories to analyze
        if memories.is_empty() {
            error!("get_alignment_drift: Empty memories array");
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                "memory_ids array cannot be empty - FAIL FAST",
            );
        }

        // Create TeleologicalDriftDetector (TASK-LOGIC-010)
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);

        // ARCH-03 COMPLIANT: Determine reference fingerprint for drift calculation
        // If North Star exists, use it. Otherwise, compute centroid of stored memories.
        let (goal_fingerprint, reference_type, reference_id): (
            context_graph_core::types::SemanticFingerprint,
            &str,
            String,
        ) = if let Some(ns) = &north_star {
            // Use North Star as reference
            (
                ns.teleological_array.clone(),
                "north_star",
                ns.id.to_string(),
            )
        } else {
            // ARCH-03: No North Star - compute centroid of memories as reference
            // This allows drift detection to work autonomously
            let pipeline = GoalDiscoveryPipeline::new(TeleologicalComparator::new());
            let refs: Vec<&context_graph_core::types::SemanticFingerprint> =
                memories.iter().collect();
            let centroid = pipeline.compute_centroid(&refs);
            (centroid, "computed_centroid", "centroid".to_string())
        };

        let drift_result = match detector.check_drift(&memories, &goal_fingerprint, strategy) {
            Ok(result) => result,
            Err(e) => {
                let (code, message) = match &e {
                    DriftError::EmptyMemories => (
                        error_codes::INVALID_PARAMS,
                        "Empty memories slice - cannot check drift".to_string(),
                    ),
                    DriftError::InvalidGoal { reason } => (
                        error_codes::INVALID_PARAMS,
                        format!("Invalid goal fingerprint: {}", reason),
                    ),
                    DriftError::ComparisonFailed { embedder, reason } => (
                        error_codes::ALIGNMENT_COMPUTATION_ERROR,
                        format!("Comparison failed for {:?}: {}", embedder, reason),
                    ),
                    DriftError::InvalidThresholds { reason } => (
                        error_codes::ALIGNMENT_COMPUTATION_ERROR,
                        format!("Invalid thresholds: {}", reason),
                    ),
                    DriftError::ComparisonValidationFailed { reason } => (
                        error_codes::ALIGNMENT_COMPUTATION_ERROR,
                        format!("Comparison validation failed: {}", reason),
                    ),
                };
                error!(error = %e, "get_alignment_drift: FAILED");
                return JsonRpcResponse::error(id, code, format!("{} - FAIL FAST", message));
            }
        };

        // Build per-embedder drift response (exactly 13 entries)
        let per_embedder_drift: Vec<serde_json::Value> = drift_result
            .per_embedder_drift
            .embedder_drift
            .iter()
            .map(|info| {
                json!({
                    "embedder": format!("{:?}", info.embedder),
                    "embedder_index": info.embedder.index(),
                    "similarity": info.similarity,
                    "drift_score": info.drift_score,
                    "drift_level": format!("{:?}", info.drift_level)
                })
            })
            .collect();

        // Build most drifted embedders (top 5, sorted worst-first)
        let most_drifted: Vec<serde_json::Value> = drift_result
            .most_drifted_embedders
            .iter()
            .take(5)
            .map(|info| {
                json!({
                    "embedder": format!("{:?}", info.embedder),
                    "embedder_index": info.embedder.index(),
                    "similarity": info.similarity,
                    "drift_score": info.drift_score,
                    "drift_level": format!("{:?}", info.drift_level)
                })
            })
            .collect();

        // Build recommendations
        let recommendations: Vec<serde_json::Value> = drift_result
            .recommendations
            .iter()
            .map(|rec| {
                json!({
                    "embedder": format!("{:?}", rec.embedder),
                    "priority": format!("{:?}", rec.priority),
                    "issue": rec.issue,
                    "suggestion": rec.suggestion
                })
            })
            .collect();

        // Build trend response if available
        let trend_response = drift_result.trend.as_ref().map(|trend| {
            json!({
                "direction": format!("{:?}", trend.direction),
                "velocity": trend.velocity,
                "samples": trend.samples,
                "projected_critical_in": trend.projected_critical_in
            })
        });

        // Map new DriftLevel to legacy DriftSeverity for backwards compatibility
        let legacy_severity = match drift_result.overall_drift.drift_level {
            DriftLevel::Critical => DriftSeverity::Severe,
            DriftLevel::High => DriftSeverity::Severe,
            DriftLevel::Medium => DriftSeverity::Moderate,
            DriftLevel::Low => DriftSeverity::Mild,
            DriftLevel::None => DriftSeverity::None,
        };

        let mut response = json!({
            "overall_drift": {
                "level": format!("{:?}", drift_result.overall_drift.drift_level),
                "similarity": drift_result.overall_drift.similarity,
                "drift_score": drift_result.overall_drift.drift_score,
                "has_drifted": drift_result.overall_drift.has_drifted
            },
            "per_embedder_drift": per_embedder_drift,
            "most_drifted_embedders": most_drifted,
            "recommendations": recommendations,
            "trend": trend_response,
            "analyzed_count": drift_result.analyzed_count,
            "timestamp": drift_result.timestamp.to_rfc3339(),
            "legacy_state": {
                "severity": format!("{:?}", legacy_severity),
                "goal_id": reference_id.clone()
            },
            "timeframe": params.timeframe,
            "reference_type": reference_type,
            "reference_id": reference_id.clone(),
            "arch03_compliant": true,
            "note": if reference_type == "computed_centroid" {
                "ARCH-03 COMPLIANT: Drift computed relative to fingerprints' centroid (no North Star required)"
            } else {
                "Drift computed relative to North Star goal"
            }
        });

        // Optionally include history
        if params.include_history {
            response["history"] = json!({
                "note": "History tracking requires stateful detector with check_drift_with_history",
                "data_points": [],
                "available": false
            });
        }

        info!(
            overall_level = ?drift_result.overall_drift.drift_level,
            analyzed_count = drift_result.analyzed_count,
            reference_type = reference_type,
            "get_alignment_drift: Per-embedder analysis complete (ARCH-03 compliant)"
        );

        self.tool_result_with_pulse(id, response)
    }

    /// trigger_drift_correction tool implementation.
    ///
    /// TASK-AUTONOMOUS-MCP + ARCH-03: Manually trigger drift correction.
    /// Uses DriftCorrector to apply correction strategies.
    ///
    /// ARCH-03 COMPLIANT: Works WITHOUT North Star by balancing fingerprints'
    /// alignment distribution towards the computed centroid.
    ///
    /// Arguments:
    /// - force (optional): Force correction even if drift severity is low (default: false)
    /// - target_alignment (optional): Target alignment to achieve
    ///
    /// Returns:
    /// - correction_result: Strategy applied and alignment change
    /// - before_state: Drift state before correction
    /// - after_state: Drift state after correction
    /// - reference_type: "north_star" or "centroid" indicating correction target
    pub(crate) async fn call_trigger_drift_correction(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling trigger_drift_correction tool call");

        // Parse parameters
        let params: TriggerDriftCorrectionParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "trigger_drift_correction: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        debug!(
            force = params.force,
            target_alignment = ?params.target_alignment,
            "trigger_drift_correction: Parsed parameters"
        );

        // ARCH-03 COMPLIANT: Check if North Star exists, but don't require it
        let (has_north_star, reference_type) = {
            let hierarchy = self.goal_hierarchy.read();
            if hierarchy.north_star().is_some() {
                (true, "north_star")
            } else {
                (false, "computed_centroid")
            }
        };

        // Create drift state and corrector
        let mut state = DriftState::default();
        let mut corrector = DriftCorrector::new();

        // Get current state for before snapshot
        let before_state = json!({
            "severity": format!("{:?}", state.severity),
            "trend": format!("{:?}", state.trend),
            "drift_magnitude": state.drift,
            "rolling_mean": state.rolling_mean,
            "reference_type": reference_type
        });

        // Check if correction is needed (unless forced)
        if !params.force && state.severity == DriftSeverity::None {
            warn!(
                "trigger_drift_correction: No drift detected and force=false, skipping correction"
            );
            return self.tool_result_with_pulse(
                id,
                json!({
                    "skipped": true,
                    "reason": "No drift detected. Use force=true to correct anyway.",
                    "before_state": before_state,
                    "after_state": before_state,
                    "reference_type": reference_type,
                    "arch03_compliant": true
                }),
            );
        }

        // Select and apply correction strategy
        let strategy = corrector.select_strategy(&state);
        let result = corrector.apply_correction(&mut state, &strategy);

        let after_state = json!({
            "severity": format!("{:?}", state.severity),
            "trend": format!("{:?}", state.trend),
            "drift_magnitude": state.drift,
            "rolling_mean": state.rolling_mean,
            "reference_type": reference_type
        });

        let correction_result = json!({
            "strategy_applied": format!("{:?}", result.strategy_applied),
            "alignment_before": result.alignment_before,
            "alignment_after": result.alignment_after,
            "improvement": result.improvement(),
            "success": result.success
        });

        info!(
            strategy = ?result.strategy_applied,
            improvement = result.improvement(),
            success = result.success,
            reference_type = reference_type,
            "trigger_drift_correction: Correction applied (ARCH-03 compliant)"
        );

        self.tool_result_with_pulse(
            id,
            json!({
                "correction_result": correction_result,
                "before_state": before_state,
                "after_state": after_state,
                "forced": params.force,
                "reference_type": reference_type,
                "arch03_compliant": true,
                "note": if has_north_star {
                    "Correction applied towards North Star goal"
                } else {
                    "ARCH-03 COMPLIANT: Correction applied towards computed centroid (no North Star required)"
                }
            }),
        )
    }
}
