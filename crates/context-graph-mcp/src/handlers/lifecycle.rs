//! MCP lifecycle handlers.

use serde_json::json;
use tracing::info;

use context_graph_core::types::{CognitivePulse, SuggestedAction};

use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::Handlers;

impl Handlers {
    /// Handle MCP initialize request.
    ///
    /// Returns server capabilities following MCP 2024-11-05 protocol specification.
    pub(super) async fn handle_initialize(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("Handling initialize request");

        let pulse = CognitivePulse::new(0.5, 0.8, 0.0, 1.0, SuggestedAction::Ready, None);

        // MCP-compliant initialize response
        JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": true }
                },
                "serverInfo": {
                    "name": "context-graph-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
        .with_pulse(pulse)
    }

    /// Handle notifications/initialized - this is a notification, not a request.
    ///
    /// Notifications don't require a response per JSON-RPC 2.0 spec.
    pub(super) fn handle_initialized_notification(&self) -> JsonRpcResponse {
        info!("Client initialized notification received");

        // Return a response with no id, result, or error to signal "no response needed"
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: None,
            result: None,
            error: None,
            cognitive_pulse: None,
        }
    }

    /// Handle MCP shutdown request.
    pub(super) async fn handle_shutdown(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("Handling shutdown request");
        JsonRpcResponse::success(id, json!(null))
    }
}
