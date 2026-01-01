//! System status and health handlers.

use serde_json::json;

use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::Handlers;

impl Handlers {
    /// Handle system/status request.
    pub(super) async fn handle_system_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        let node_count = self.memory_store.count().await.unwrap_or(0);

        JsonRpcResponse::success(
            id,
            json!({
                "status": "running",
                "phase": "ghost-system",
                "nodeCount": node_count,
                "gpuAvailable": false,
            }),
        )
    }

    /// Handle system/health request.
    pub(super) async fn handle_system_health(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "healthy": true,
                "components": {
                    "memory": "healthy",
                    "utl": "healthy",
                    "graph": "healthy"
                }
            }),
        )
    }
}
