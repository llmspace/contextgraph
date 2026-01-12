//! GWT compute delta SC handler.
//!
//! TASK-UTL-P1-001: Computes delta S (entropy change) and delta C (coherence change)
//! for UTL learning.

use serde_json::json;
use tracing::{debug, error};
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;
use context_graph_core::types::fingerprint::TeleologicalFingerprint;
use context_graph_core::types::JohariQuadrant;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::constants::{ALPHA, BETA, GAMMA};
use super::gwt_compute::{compute_delta_c, compute_delta_s};
use super::helpers::classify_johari;
use super::super::Handlers;

impl Handlers {
    /// Handle gwt/compute_delta_sc request.
    ///
    /// Computes delta S (entropy change) and delta C (coherence change) for UTL learning.
    /// Per constitution.yaml AP-32, this tool MUST exist.
    ///
    /// # Parameters
    /// - `vertex_id`: UUID of the vertex being updated
    /// - `old_fingerprint`: Previous TeleologicalFingerprint (13 embeddings)
    /// - `new_fingerprint`: New TeleologicalFingerprint (13 embeddings)
    /// - `include_diagnostics`: Optional, include detailed breakdown
    /// - `johari_threshold`: Optional, override threshold (default 0.5)
    ///
    /// # Response
    /// Returns delta_s_per_embedder, delta_s_aggregate, delta_c, johari_quadrants,
    /// johari_aggregate, utl_learning_potential, and optional diagnostics.
    pub(crate) async fn handle_gwt_compute_delta_sc(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("gwt/compute_delta_sc: starting");

        // FAIL FAST: params required
        let params = match params {
            Some(p) => p,
            None => {
                error!("gwt/compute_delta_sc: missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        // Parse vertex_id
        let vertex_id_str = match params.get("vertex_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                error!("gwt/compute_delta_sc: missing 'vertex_id' parameter");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'vertex_id' parameter",
                );
            }
        };

        let _vertex_id = match Uuid::parse_str(vertex_id_str) {
            Ok(uuid) => uuid,
            Err(_) => {
                error!(
                    "gwt/compute_delta_sc: invalid UUID format: {}",
                    vertex_id_str
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid UUID format: {}", vertex_id_str),
                );
            }
        };

        // Parse old_fingerprint
        let old_fingerprint_value = match params.get("old_fingerprint") {
            Some(v) => v.clone(),
            None => {
                error!("gwt/compute_delta_sc: missing 'old_fingerprint' parameter");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'old_fingerprint' parameter",
                );
            }
        };

        let old_fp: TeleologicalFingerprint = match serde_json::from_value(old_fingerprint_value) {
            Ok(fp) => fp,
            Err(e) => {
                error!(
                    "gwt/compute_delta_sc: failed to parse old_fingerprint: {}",
                    e
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Failed to parse old_fingerprint: {}", e),
                );
            }
        };

        // Parse new_fingerprint
        let new_fingerprint_value = match params.get("new_fingerprint") {
            Some(v) => v.clone(),
            None => {
                error!("gwt/compute_delta_sc: missing 'new_fingerprint' parameter");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'new_fingerprint' parameter",
                );
            }
        };

        let new_fp: TeleologicalFingerprint = match serde_json::from_value(new_fingerprint_value) {
            Ok(fp) => fp,
            Err(e) => {
                error!(
                    "gwt/compute_delta_sc: failed to parse new_fingerprint: {}",
                    e
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Failed to parse new_fingerprint: {}", e),
                );
            }
        };

        // Parse optional parameters
        let include_diagnostics = params
            .get("include_diagnostics")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let johari_threshold = params
            .get("johari_threshold")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.5)
            .clamp(0.35, 0.65); // Per constitution.yaml adaptive_thresholds.priors.theta_joh

        // Compute delta S and delta C
        let delta_s_result = compute_delta_s(&old_fp, &new_fp, include_diagnostics);
        let delta_c_result = compute_delta_c(&old_fp, &new_fp);

        // Classify Johari quadrants
        let johari_quadrants: [JohariQuadrant; NUM_EMBEDDERS] = std::array::from_fn(|i| {
            classify_johari(delta_s_result.per_embedder[i], delta_c_result.delta_c, johari_threshold)
        });

        let johari_aggregate =
            classify_johari(delta_s_result.aggregate, delta_c_result.delta_c, johari_threshold);

        // Build response
        let utl_learning_potential =
            (delta_s_result.aggregate * delta_c_result.delta_c).clamp(0.0, 1.0);

        let johari_quadrant_strings: Vec<String> =
            johari_quadrants.iter().map(|q| q.to_string()).collect();

        let mut response = json!({
            "delta_s_per_embedder": delta_s_result.per_embedder.to_vec(),
            "delta_s_aggregate": delta_s_result.aggregate,
            "delta_c": delta_c_result.delta_c,
            "johari_quadrants": johari_quadrant_strings,
            "johari_aggregate": johari_aggregate.to_string(),
            "utl_learning_potential": utl_learning_potential,
        });

        if include_diagnostics {
            response["diagnostics"] = json!({
                "per_embedder": delta_s_result.diagnostics,
                "johari_threshold": johari_threshold,
                "delta_c_components": {
                    "connectivity": delta_c_result.connectivity,
                    "cluster_fit": delta_c_result.cluster_fit,
                    "consistency": delta_c_result.consistency,
                    "weights": {
                        "alpha_connectivity": ALPHA,
                        "beta_cluster_fit": BETA,
                        "gamma_consistency": GAMMA,
                    },
                },
                "cluster_fit_details": {
                    "silhouette": delta_c_result.cluster_fit_result.silhouette,
                    "intra_distance": delta_c_result.cluster_fit_result.intra_distance,
                    "inter_distance": delta_c_result.cluster_fit_result.inter_distance,
                },
                "coherence_config": {
                    "similarity_weight": delta_c_result.similarity_weight,
                    "consistency_weight": delta_c_result.consistency_weight,
                },
            });
        }

        debug!(
            "gwt/compute_delta_sc: completed - delta_S_agg={:.4}, delta_C={:.4}, L_pot={:.4}, quadrant={}",
            delta_s_result.aggregate, delta_c_result.delta_c, utl_learning_potential, johari_aggregate
        );

        JsonRpcResponse::success(id, response)
    }
}
