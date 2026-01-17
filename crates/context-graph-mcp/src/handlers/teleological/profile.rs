//! TELEO-H5: manage_teleological_profile handler.
//!
//! CRUD operations for task-specific teleological profiles.

use super::types::ManageTeleologicalProfileParams;
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcResponse};
use context_graph_core::teleological::{services::ProfileManager, ProfileId};
use serde_json::json;
use tracing::{debug, error, info};

impl Handlers {
    /// Handle manage_teleological_profile tool call.
    ///
    /// CRUD operations for task-specific teleological profiles.
    pub(in crate::handlers) async fn call_manage_teleological_profile(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("manage_teleological_profile called with: {:?}", arguments);

        // Parse parameters
        let params: ManageTeleologicalProfileParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to parse manage_teleological_profile params: {}", e);
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        // Create a local ProfileManager (stateless per request)
        // In production, this should be shared state with persistence
        let mut manager = ProfileManager::new();

        match params.action.to_lowercase().as_str() {
            "create" => self.handle_profile_create(id, &params, &mut manager).await,
            "get" => self.handle_profile_get(id, &params, &manager).await,
            "update" => self.handle_profile_update(id, &params, &mut manager).await,
            "delete" => self.handle_profile_delete(id, &params, &mut manager).await,
            "list" => self.handle_profile_list(id, &manager).await,
            "find_best" => self.handle_profile_find_best(id, &params, &manager).await,
            _ => self.tool_error_with_pulse(
                id,
                &format!(
                    "Unknown action: {}. Valid actions: create, get, update, delete, list, find_best",
                    params.action
                ),
            ),
        }
    }

    async fn handle_profile_create(
        &self,
        id: Option<JsonRpcId>,
        params: &ManageTeleologicalProfileParams,
        manager: &mut ProfileManager,
    ) -> JsonRpcResponse {
        let profile_id = match &params.profile_id {
            Some(pid) => pid.clone(),
            None => return self.tool_error_with_pulse(id, "profile_id required for create"),
        };
        let weights = match params.weights {
            Some(w) => w,
            None => return self.tool_error_with_pulse(id, "weights required for create"),
        };

        let profile = manager.create_profile(&profile_id, weights);

        info!("Created profile: {}", profile_id);
        self.tool_result_with_pulse(
            id,
            json!({
                "success": true,
                "action": "create",
                "profile_id": profile_id,
                "weights": profile.embedding_weights,
            }),
        )
    }

    async fn handle_profile_get(
        &self,
        id: Option<JsonRpcId>,
        params: &ManageTeleologicalProfileParams,
        manager: &ProfileManager,
    ) -> JsonRpcResponse {
        let profile_id = match &params.profile_id {
            Some(pid) => pid.clone(),
            None => return self.tool_error_with_pulse(id, "profile_id required for get"),
        };

        let pid = ProfileId::new(&profile_id);

        match manager.get_profile(&pid) {
            Some(profile) => self.tool_result_with_pulse(
                id,
                json!({
                    "success": true,
                    "action": "get",
                    "profile_id": profile_id,
                    "weights": profile.embedding_weights,
                    "task_type": format!("{:?}", profile.task_type),
                    "name": profile.name,
                }),
            ),
            None => self.tool_error_with_pulse(id, &format!("Profile '{}' not found", profile_id)),
        }
    }

    async fn handle_profile_update(
        &self,
        id: Option<JsonRpcId>,
        params: &ManageTeleologicalProfileParams,
        manager: &mut ProfileManager,
    ) -> JsonRpcResponse {
        let profile_id = match &params.profile_id {
            Some(pid) => pid.clone(),
            None => return self.tool_error_with_pulse(id, "profile_id required for update"),
        };
        let weights = match params.weights {
            Some(w) => w,
            None => return self.tool_error_with_pulse(id, "weights required for update"),
        };

        let pid = ProfileId::new(&profile_id);
        let updated = manager.update_profile(&pid, weights);

        if updated {
            info!("Updated profile: {}", profile_id);
            self.tool_result_with_pulse(
                id,
                json!({
                    "success": true,
                    "action": "update",
                    "profile_id": profile_id,
                    "weights": weights,
                }),
            )
        } else {
            self.tool_error_with_pulse(id, &format!("Profile '{}' not found", profile_id))
        }
    }

    async fn handle_profile_delete(
        &self,
        id: Option<JsonRpcId>,
        params: &ManageTeleologicalProfileParams,
        manager: &mut ProfileManager,
    ) -> JsonRpcResponse {
        let profile_id = match &params.profile_id {
            Some(pid) => pid.clone(),
            None => return self.tool_error_with_pulse(id, "profile_id required for delete"),
        };

        let pid = ProfileId::new(&profile_id);
        let deleted = manager.delete_profile(&pid);

        if deleted {
            info!("Deleted profile: {}", profile_id);
            self.tool_result_with_pulse(
                id,
                json!({
                    "success": true,
                    "action": "delete",
                    "profile_id": profile_id,
                }),
            )
        } else {
            self.tool_error_with_pulse(
                id,
                &format!("Profile '{}' not found or cannot delete", profile_id),
            )
        }
    }

    async fn handle_profile_list(
        &self,
        id: Option<JsonRpcId>,
        manager: &ProfileManager,
    ) -> JsonRpcResponse {
        let profiles: Vec<String> = manager
            .list_profiles()
            .into_iter()
            .map(|p| p.as_str().to_string())
            .collect();

        self.tool_result_with_pulse(
            id,
            json!({
                "success": true,
                "action": "list",
                "profiles": profiles,
                "count": profiles.len(),
            }),
        )
    }

    async fn handle_profile_find_best(
        &self,
        id: Option<JsonRpcId>,
        params: &ManageTeleologicalProfileParams,
        manager: &ProfileManager,
    ) -> JsonRpcResponse {
        let context = match &params.context {
            Some(c) => c.clone(),
            None => return self.tool_error_with_pulse(id, "context required for find_best"),
        };

        match manager.find_best_match(&context) {
            Some(match_result) => self.tool_result_with_pulse(
                id,
                json!({
                    "success": true,
                    "action": "find_best",
                    "profile_id": match_result.profile_id.as_str(),
                    "similarity": match_result.similarity,
                    "reason": match_result.reason,
                }),
            ),
            None => self.tool_result_with_pulse(
                id,
                json!({
                    "success": true,
                    "action": "find_best",
                    "profile_id": null,
                    "message": "No matching profile found",
                }),
            ),
        }
    }
}
