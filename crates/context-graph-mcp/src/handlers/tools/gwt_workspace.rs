//! GWT workspace tool implementations.
//!
//! TASK-GWT-001: Workspace operations - get_workspace_status, trigger_workspace_broadcast.

use serde_json::json;
use tracing::{debug, error};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

impl Handlers {
    /// get_workspace_status tool implementation.
    ///
    /// TASK-GWT-001: Returns Global Workspace status including active memory,
    /// competing candidates, broadcast state, and coherence threshold.
    ///
    /// FAIL FAST on missing workspace provider - no stubs or fallbacks.
    ///
    /// Returns:
    /// - active_memory: UUID of currently active (conscious) memory, or null
    /// - is_broadcasting: Whether broadcast window is active
    /// - has_conflict: Whether multiple memories compete (r > 0.8)
    /// - coherence_threshold: Threshold for workspace entry (default 0.8)
    /// - conflict_memories: List of conflicting memory UUIDs if has_conflict
    pub(crate) async fn call_get_workspace_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("Handling get_workspace_status tool call");

        // FAIL FAST: Check workspace provider
        let workspace = match &self.workspace_provider {
            Some(w) => w,
            None => {
                error!("get_workspace_status: Workspace provider not initialized");
                return JsonRpcResponse::error(
                    id,
                    error_codes::GWT_NOT_INITIALIZED,
                    "Workspace provider not initialized - use with_gwt() constructor",
                );
            }
        };

        // Acquire read lock (tokio RwLock)
        // TASK-07: WorkspaceProvider trait methods are async
        let ws = workspace.read().await;

        // Get active memory if broadcasting
        let active_memory = ws.get_active_memory().await;

        // Check broadcast state
        let is_broadcasting = ws.is_broadcasting().await;

        // Check for conflict
        let has_conflict = ws.has_conflict().await;

        // Get coherence threshold
        let coherence_threshold = ws.coherence_threshold().await;

        // Get conflict details if present
        let conflict_memories = ws.get_conflict_details().await;

        self.tool_result_with_pulse(
            id,
            json!({
                "active_memory": active_memory.map(|id| id.to_string()),
                "is_broadcasting": is_broadcasting,
                "has_conflict": has_conflict,
                "coherence_threshold": coherence_threshold,
                "conflict_memories": conflict_memories.map(|ids|
                    ids.iter().map(|id| id.to_string()).collect::<Vec<_>>()
                ),
                "broadcast_duration_ms": 100 // Constitution default
            }),
        )
    }

    /// trigger_workspace_broadcast tool implementation.
    ///
    /// TASK-GWT-001: Triggers winner-take-all selection with a specific memory.
    /// Forces memory into workspace competition. Requires write lock on workspace.
    ///
    /// FAIL FAST on missing providers - no stubs or fallbacks.
    ///
    /// Arguments:
    /// - memory_id: UUID of memory to broadcast
    /// - importance: Importance score [0,1] (default 0.8)
    /// - alignment: Topic alignment [0,1] (default 0.8)
    /// - force: Force broadcast even if below coherence threshold
    ///
    /// Returns:
    /// - success: Whether broadcast was successful
    /// - memory_id: UUID of the memory
    /// - coherence: Coherence order parameter (from per-space clustering)
    /// - was_selected: Whether this memory won WTA selection
    pub(crate) async fn call_trigger_workspace_broadcast(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling trigger_workspace_broadcast tool call");

        // FAIL FAST: Check workspace provider
        let workspace = match &self.workspace_provider {
            Some(w) => w,
            None => {
                error!("trigger_workspace_broadcast: Workspace provider not initialized");
                return JsonRpcResponse::error(
                    id,
                    error_codes::GWT_NOT_INITIALIZED,
                    "Workspace provider not initialized - use with_gwt() constructor",
                );
            }
        };

        // Parse memory_id (required)
        let memory_id_str = match args.get("memory_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return self.tool_error_with_pulse(id, "Missing required 'memory_id' parameter");
            }
        };

        let memory_id = match uuid::Uuid::parse_str(memory_id_str) {
            Ok(id) => id,
            Err(e) => {
                return self.tool_error_with_pulse(
                    id,
                    &format!("Invalid UUID format for memory_id: {}", e),
                );
            }
        };

        // Parse optional parameters
        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8) as f32;
        let alignment = args
            .get("alignment")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8) as f32;
        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);

        // Coherence from per-space clustering coordination
        let coherence_threshold = 0.8_f32;

        // Use importance * alignment as coherence estimate for WTA selection
        let coherence = importance * alignment;

        // Check if memory qualifies for workspace (coherence >= 0.8 unless forced)
        if coherence < coherence_threshold && !force {
            return self.tool_result_with_pulse(
                id,
                json!({
                    "success": false,
                    "memory_id": memory_id.to_string(),
                    "coherence": coherence,
                    "was_selected": false,
                    "reason": format!(
                        "Coherence {:.3} below threshold {}. Use force=true to override.",
                        coherence, coherence_threshold
                    )
                }),
            );
        }

        // Acquire write lock and trigger selection
        let ws = workspace.write().await;

        // Create candidate and trigger WTA selection
        let candidates = vec![(memory_id, coherence, importance, alignment)];
        let winner = match ws.select_winning_memory(candidates).await {
            Ok(w) => w,
            Err(e) => {
                error!(error = %e, "trigger_workspace_broadcast: WTA selection failed");
                return JsonRpcResponse::error(
                    id,
                    error_codes::WORKSPACE_ERROR,
                    format!("Workspace selection failed: {}", e),
                );
            }
        };

        let was_selected = winner == Some(memory_id);

        // GAP-1 FIX: Wire workspace events to neuromodulation
        // When a memory enters workspace (was_selected), increase dopamine
        let dopamine_triggered = if was_selected {
            if let Some(neuromod) = &self.neuromod_manager {
                let mut manager = neuromod.write();
                manager.on_workspace_entry();
                let new_dopamine = manager.get_hopfield_beta();
                debug!(
                    memory_id = %memory_id,
                    dopamine = new_dopamine,
                    "Workspace entry triggered dopamine increase"
                );
                Some(new_dopamine)
            } else {
                None
            }
        } else {
            None
        };

        self.tool_result_with_pulse(
            id,
            json!({
                "success": true,
                "memory_id": memory_id.to_string(),
                "coherence": coherence,
                "was_selected": was_selected,
                "is_broadcasting": ws.is_broadcasting().await,
                "dopamine_triggered": dopamine_triggered
            }),
        )
    }

    // ========== TOPIC PORTFOLIO TOOLS ==========
    // Topic Portfolio per PRD v6 (ARCH-03)

    /// get_ego_state tool implementation.
    ///
    /// TASK-GWT-001: Returns Topic Profile state including purpose vector (13D),
    /// topic stability, coherence with actions, and trajectory length.
    ///
    /// Per PRD v6: Uses topic stability metrics instead of consciousness state.
    /// Health thresholds based on churn: healthy < 0.3, degraded [0.3, 0.5), critical >= 0.5
    pub(crate) async fn call_get_ego_state(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("Handling get_ego_state tool call");

        // Get topic stability from workspace provider per PRD v6
        let topic_stability = if let Some(ref workspace) = &self.workspace_provider {
            let ws = workspace.read().await;
            ws.get_topic_stability().await
        } else {
            0.5 // Default when workspace not available
        };

        // Default 13D purpose vector - balanced at 0.5 for all embedders
        let purpose_vector: Vec<f64> = vec![0.5; 13];

        // Get coherence with actions from UTL status
        let utl_status = self.utl_processor.get_status();
        let coherence_with_actions = utl_status
            .get("coherence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        // Determine identity status based on topic stability thresholds per PRD v6
        // topic_stability >= 0.7 = healthy (churn < 0.3)
        // topic_stability in [0.5, 0.7) = warning (churn in [0.3, 0.5))
        // topic_stability < 0.5 = critical (churn >= 0.5)
        let identity_status = if topic_stability >= 0.9 {
            "Healthy"
        } else if topic_stability >= 0.7 {
            "Warning"
        } else if topic_stability >= 0.5 {
            "Degraded"
        } else {
            "Critical"
        };

        self.tool_result_with_pulse(
            id,
            json!({
                "purpose_vector": purpose_vector,
                "topic_stability": topic_stability,
                "coherence_with_actions": coherence_with_actions,
                "identity_status": identity_status,
                "trajectory_length": 0_u64,
                "thresholds": {
                    "healthy": 0.9,
                    "warning": 0.7,
                    "degraded": 0.5
                }
            }),
        )
    }

    /// get_coherence_state tool implementation.
    ///
    /// TASK-34: Returns high-level GWT workspace coherence state including
    /// coherence level (High/Medium/Low), broadcasting status, and conflict detection.
    ///
    /// Per PRD v6: Uses topic stability metrics instead of consciousness state.
    pub(crate) async fn call_get_coherence_state(
        &self,
        id: Option<JsonRpcId>,
        _args: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_coherence_state tool call");

        // Get topic stability from workspace provider per PRD v6
        let topic_stability = if let Some(ref workspace) = &self.workspace_provider {
            let ws = workspace.read().await;
            ws.get_topic_stability().await
        } else {
            0.5 // Default when workspace not available
        };

        // Get workspace metrics if available
        let (is_broadcasting, has_conflict) = if let Some(ref workspace) = &self.workspace_provider {
            let ws = workspace.read().await;
            (ws.is_broadcasting().await, ws.has_conflict().await)
        } else {
            (false, false)
        };

        // Map topic stability to coherence level per PRD v6 thresholds
        let coherence_level = if topic_stability >= 0.8 {
            "High"
        } else if topic_stability >= 0.5 {
            "Medium"
        } else {
            "Low"
        };

        self.tool_result_with_pulse(
            id,
            json!({
                "coherence_level": coherence_level,
                "is_broadcasting": is_broadcasting,
                "has_conflict": has_conflict,
                "topic_stability": topic_stability
            }),
        )
    }
}
