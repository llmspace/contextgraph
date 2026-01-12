//! Drift response building utilities.
//!
//! Provides functions to build JSON responses from DriftResult data.

use serde_json::json;
use tracing::error;

use context_graph_core::autonomous::drift::{DriftError, DriftResult};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

/// Build per-embedder drift response (exactly 13 entries).
pub(super) fn build_per_embedder_drift(drift_result: &DriftResult) -> Vec<serde_json::Value> {
    drift_result
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
        .collect()
}

/// Build most drifted embedders (top 5, sorted worst-first).
pub(super) fn build_most_drifted(drift_result: &DriftResult) -> Vec<serde_json::Value> {
    drift_result
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
        .collect()
}

/// Build recommendations (fields: embedder, issue, suggestion, priority).
pub(super) fn build_recommendations(drift_result: &DriftResult) -> Vec<serde_json::Value> {
    drift_result
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
        .collect()
}

/// Build trend response if available.
pub(super) fn build_trend_response(drift_result: &DriftResult) -> Option<serde_json::Value> {
    drift_result.trend.as_ref().map(|trend| {
        json!({
            "direction": format!("{:?}", trend.direction),
            "velocity": trend.velocity,
            "samples": trend.samples,
            "projected_critical_in": trend.projected_critical_in
        })
    })
}

/// Build the complete drift check success response.
pub(super) fn build_drift_response(
    drift_result: &DriftResult,
    check_time_ms: u128,
) -> serde_json::Value {
    let per_embedder_drift = build_per_embedder_drift(drift_result);
    let most_drifted = build_most_drifted(drift_result);
    let recommendations = build_recommendations(drift_result);
    let trend_response = build_trend_response(drift_result);

    json!({
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
        "check_time_ms": check_time_ms
    })
}

/// Handle drift check errors and map to appropriate JSON-RPC response.
pub(super) fn handle_drift_error(id: &Option<JsonRpcId>, e: DriftError) -> JsonRpcResponse {
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
    error!(error = %e, "purpose/drift_check: FAILED");
    JsonRpcResponse::error(id.clone(), code, format!("{} - FAIL FAST", message))
}
