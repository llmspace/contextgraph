//! Status query tool implementations (get_memetic_status, get_graph_manifest, utl_status).

use serde_json::json;
use tracing::{debug, error};

use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

impl Handlers {
    /// get_memetic_status tool implementation.
    ///
    /// TASK-S001: Updated to use TeleologicalMemoryStore count.
    ///
    /// Returns comprehensive system status including:
    /// - Fingerprint count from TeleologicalMemoryStore
    /// - Live UTL metrics from UtlProcessor (NOT hardcoded)
    /// - 5-layer bio-nervous system status
    /// - `_cognitive_pulse` with live system state
    ///
    /// # Constitution References
    /// - UTL formula: constitution.yaml:152
    /// - Johari quadrant actions: constitution.yaml:159-163
    pub(crate) async fn call_get_memetic_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        let fingerprint_count = match self.teleological_store.count().await {
            Ok(count) => count,
            Err(e) => {
                error!(error = %e, "get_memetic_status: TeleologicalStore.count() FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Failed to get fingerprint count: {}", e),
                );
            }
        };

        // Get LIVE UTL status from the processor
        let utl_status = self.utl_processor.get_status();

        // FAIL-FAST: UTL processor MUST return all required fields.
        // Per constitution AP-007: No stubs or fallbacks in production code paths.
        // If the UTL processor doesn't have these fields, the system is broken.
        let lifecycle_phase = match utl_status.get("lifecycle_phase").and_then(|v| v.as_str()) {
            Some(phase) => phase,
            None => {
                error!("get_memetic_status: UTL processor missing 'lifecycle_phase' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'lifecycle_phase'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let entropy = match utl_status.get("entropy").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!(
                    "get_memetic_status: UTL processor missing 'entropy' field - system is broken"
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'entropy'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let coherence = match utl_status.get("coherence").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!("get_memetic_status: UTL processor missing 'coherence' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'coherence'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let learning_score = match utl_status.get("learning_score").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!("get_memetic_status: UTL processor missing 'learning_score' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'learning_score'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let johari_quadrant = match utl_status.get("johari_quadrant").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                error!("get_memetic_status: UTL processor missing 'johari_quadrant' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'johari_quadrant'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let consolidation_phase = match utl_status
            .get("consolidation_phase")
            .and_then(|v| v.as_str())
        {
            Some(phase) => phase,
            None => {
                error!("get_memetic_status: UTL processor missing 'consolidation_phase' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'consolidation_phase'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        // Map Johari quadrant to suggested action per constitution.yaml:159-163
        let suggested_action = match johari_quadrant {
            "Open" => "direct_recall",
            "Blind" => "trigger_dream",
            "Hidden" => "get_neighborhood",
            "Unknown" => "epistemic_action",
            _ => "continue",
        };

        // Get quadrant counts from teleological store
        let quadrant_counts = match self.teleological_store.count_by_quadrant().await {
            Ok(counts) => counts,
            Err(e) => {
                error!(error = %e, "get_memetic_status: count_by_quadrant() FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Failed to get quadrant counts: {}", e),
                );
            }
        };

        // TASK-EMB-024: Get REAL layer statuses from LayerStatusProvider
        let perception_status = self
            .layer_status_provider
            .perception_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: perception_status FAILED");
                "error".to_string()
            });
        let memory_status = self
            .layer_status_provider
            .memory_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: memory_status FAILED");
                "error".to_string()
            });
        let reasoning_status = self
            .layer_status_provider
            .reasoning_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: reasoning_status FAILED");
                "error".to_string()
            });
        let action_status = self
            .layer_status_provider
            .action_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: action_status FAILED");
                "error".to_string()
            });
        let meta_status = self
            .layer_status_provider
            .meta_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: meta_status FAILED");
                "error".to_string()
            });

        self.tool_result_with_pulse(
            id,
            json!({
                "phase": lifecycle_phase,
                "fingerprintCount": fingerprint_count,
                "embedderCount": NUM_EMBEDDERS,
                "storageBackend": self.teleological_store.backend_type().to_string(),
                "storageSizeBytes": self.teleological_store.storage_size_bytes(),
                "quadrantCounts": {
                    "open": quadrant_counts[0],
                    "hidden": quadrant_counts[1],
                    "blind": quadrant_counts[2],
                    "unknown": quadrant_counts[3]
                },
                "utl": {
                    "entropy": entropy,
                    "coherence": coherence,
                    "learningScore": learning_score,
                    "johariQuadrant": johari_quadrant,
                    "consolidationPhase": consolidation_phase,
                    "suggestedAction": suggested_action
                },
                "layers": {
                    "perception": perception_status,
                    "memory": memory_status,
                    "reasoning": reasoning_status,
                    "action": action_status,
                    "meta": meta_status
                }
            }),
        )
    }

    /// get_graph_manifest tool implementation.
    ///
    /// Returns the 5-layer bio-nervous architecture manifest.
    /// Response includes `_cognitive_pulse` with live system state.
    ///
    /// TASK-EMB-024: Layer statuses now come from LayerStatusProvider.
    pub(crate) async fn call_get_graph_manifest(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        // TASK-EMB-024: Get REAL layer statuses from LayerStatusProvider
        let perception_status = self
            .layer_status_provider
            .perception_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: perception_status FAILED");
                "error".to_string()
            });
        let memory_status = self
            .layer_status_provider
            .memory_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: memory_status FAILED");
                "error".to_string()
            });
        let reasoning_status = self
            .layer_status_provider
            .reasoning_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: reasoning_status FAILED");
                "error".to_string()
            });
        let action_status = self
            .layer_status_provider
            .action_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: action_status FAILED");
                "error".to_string()
            });
        let meta_status = self
            .layer_status_provider
            .meta_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: meta_status FAILED");
                "error".to_string()
            });

        self.tool_result_with_pulse(
            id,
            json!({
                "architecture": "5-layer-bio-nervous",
                "fingerprintType": "TeleologicalFingerprint",
                "embedderCount": NUM_EMBEDDERS,
                "layers": [
                    {
                        "name": "Perception",
                        "description": "Sensory input processing and feature extraction",
                        "status": perception_status
                    },
                    {
                        "name": "Memory",
                        "description": "Teleological memory with 13-embedding semantic fingerprints",
                        "status": memory_status
                    },
                    {
                        "name": "Reasoning",
                        "description": "Inference, planning, and decision making",
                        "status": reasoning_status
                    },
                    {
                        "name": "Action",
                        "description": "Response generation and motor control",
                        "status": action_status
                    },
                    {
                        "name": "Meta",
                        "description": "Self-monitoring, learning rate control, and system optimization",
                        "status": meta_status
                    }
                ],
                "utl": {
                    "description": "Universal Transfer Learning - measures learning potential",
                    "formula": "L(x) = H(P) - H(P|x) + alpha * C(x)"
                },
                "teleological": {
                    "description": "Purpose-aware retrieval with North Star alignment",
                    "purposeVectorDimension": NUM_EMBEDDERS,
                    "johariQuadrants": ["Open", "Hidden", "Blind", "Unknown"]
                }
            }),
        )
    }

    /// utl_status tool implementation.
    ///
    /// Returns current UTL system state including lifecycle phase, entropy,
    /// coherence, learning score, Johari quadrant, and consolidation phase.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(crate) async fn call_utl_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("Handling utl_status tool call");

        // Get status from UTL processor (returns serde_json::Value)
        let status = self.utl_processor.get_status();

        self.tool_result_with_pulse(id, status)
    }
}
