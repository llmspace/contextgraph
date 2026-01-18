//! Health metrics handler for Meta-UTL.
//!
//! TASK-S005: Implements health_metrics handler.

use serde_json::json;
use tracing::{debug, error};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;
use super::constants::{
    ATTACK_DETECTION_TARGET, COHERENCE_RECOVERY_TARGET_MS, FALSE_POSITIVE_TARGET,
    LEARNING_SCORE_TARGET, NUM_EMBEDDERS,
};

impl Handlers {
    /// Handle meta_utl/health_metrics request.
    ///
    /// Returns system health metrics with constitution.yaml targets.
    /// TASK-S005: Hardcoded targets from constitution.yaml.
    pub(crate) async fn handle_meta_utl_health_metrics(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("meta_utl/health_metrics: starting");

        let params = params.unwrap_or(json!({}));

        let include_targets = params
            .get("include_targets")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let include_recommendations = params
            .get("include_recommendations")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Get tracker for per-space accuracy
        let (per_space_accuracy, learning_score) = {
            let tracker = self.meta_utl_tracker.read();
            let mut per_space_accuracy = Vec::with_capacity(NUM_EMBEDDERS);
            let mut total_accuracy = 0.0f32;
            let mut accuracy_count = 0usize;

            for i in 0..NUM_EMBEDDERS {
                let acc = tracker.get_embedder_accuracy(i).unwrap_or(0.0);
                per_space_accuracy.push(acc);
                if tracker.accuracy_counts[i] > 0 {
                    total_accuracy += acc;
                    accuracy_count += 1;
                }
            }

            let learning_score = if accuracy_count > 0 {
                total_accuracy / accuracy_count as f32
            } else {
                0.5
            };

            (per_space_accuracy, learning_score)
        };

        // Get metrics from SystemMonitor - FAIL FAST if not configured
        let coherence_recovery_time_ms = match self
            .system_monitor
            .coherence_recovery_time_ms()
            .await
        {
            Ok(v) => v,
            Err(e) => {
                error!(error = %e, "meta_utl/health_metrics: coherence_recovery_time_ms FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::SYSTEM_MONITOR_ERROR,
                    format!("Failed to get coherence_recovery_time_ms: {}", e),
                );
            }
        };

        let attack_detection_rate = match self.system_monitor.attack_detection_rate().await {
            Ok(v) => v,
            Err(e) => {
                error!(error = %e, "meta_utl/health_metrics: attack_detection_rate FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::SYSTEM_MONITOR_ERROR,
                    format!("Failed to get attack_detection_rate: {}", e),
                );
            }
        };

        let false_positive_rate = match self.system_monitor.false_positive_rate().await {
            Ok(v) => v,
            Err(e) => {
                error!(error = %e, "meta_utl/health_metrics: false_positive_rate FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::SYSTEM_MONITOR_ERROR,
                    format!("Failed to get false_positive_rate: {}", e),
                );
            }
        };

        // Check against targets
        let learning_score_status = if learning_score >= LEARNING_SCORE_TARGET {
            "passing"
        } else {
            "failing"
        };
        let coherence_recovery_status = if coherence_recovery_time_ms < COHERENCE_RECOVERY_TARGET_MS
        {
            "passing"
        } else {
            "failing"
        };
        let attack_detection_status = if attack_detection_rate >= ATTACK_DETECTION_TARGET {
            "passing"
        } else {
            "failing"
        };
        let false_positive_status = if false_positive_rate < FALSE_POSITIVE_TARGET {
            "passing"
        } else {
            "failing"
        };

        // Determine failed targets
        let mut failed_targets: Vec<&str> = Vec::new();
        if learning_score < LEARNING_SCORE_TARGET {
            failed_targets.push("learning_score");
        }
        if coherence_recovery_time_ms >= COHERENCE_RECOVERY_TARGET_MS {
            failed_targets.push("coherence_recovery_time_ms");
        }
        if attack_detection_rate < ATTACK_DETECTION_TARGET {
            failed_targets.push("attack_detection_rate");
        }
        if false_positive_rate >= FALSE_POSITIVE_TARGET {
            failed_targets.push("false_positive_rate");
        }

        let overall_status = if failed_targets.is_empty() {
            "healthy"
        } else if failed_targets.len() <= 1 {
            "degraded"
        } else {
            "unhealthy"
        };

        // Build recommendations if requested
        let recommendations: Vec<&str> = if include_recommendations && !failed_targets.is_empty() {
            failed_targets
                .iter()
                .map(|t| match *t {
                    "learning_score" => "Increase training data quality or quantity",
                    "coherence_recovery_time_ms" => "Optimize cache invalidation strategy",
                    "attack_detection_rate" => "Enhance anomaly detection thresholds",
                    "false_positive_rate" => "Adjust classification sensitivity",
                    _ => "Review system configuration",
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut metrics = json!({
            "learning_score": learning_score,
            "coherence_recovery_time_ms": coherence_recovery_time_ms,
            "attack_detection_rate": attack_detection_rate,
            "false_positive_rate": false_positive_rate,
            "per_space_accuracy": per_space_accuracy,
        });

        if include_targets {
            metrics["learning_score_target"] = json!(LEARNING_SCORE_TARGET);
            metrics["learning_score_status"] = json!(learning_score_status);
            metrics["coherence_recovery_target_ms"] = json!(COHERENCE_RECOVERY_TARGET_MS);
            metrics["coherence_recovery_status"] = json!(coherence_recovery_status);
            metrics["attack_detection_target"] = json!(ATTACK_DETECTION_TARGET);
            metrics["attack_detection_status"] = json!(attack_detection_status);
            metrics["false_positive_target"] = json!(FALSE_POSITIVE_TARGET);
            metrics["false_positive_status"] = json!(false_positive_status);
        }

        debug!(
            "meta_utl/health_metrics: overall_status={}, failed={}",
            overall_status,
            failed_targets.len()
        );

        JsonRpcResponse::success(
            id,
            json!({
                "metrics": metrics,
                "overall_status": overall_status,
                "failed_targets": failed_targets,
                "recommendations": recommendations,
            }),
        )
    }
}
