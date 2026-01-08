//! MCP lifecycle handlers.

use serde_json::json;
use tracing::info;

use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::Handlers;

impl Handlers {
    /// Handle MCP initialize request.
    ///
    /// Returns server capabilities following MCP 2024-11-05 protocol specification.
    ///
    /// **CRITICAL**: The initialize response MUST be strict JSON-RPC 2.0 format with
    /// no extension fields. Claude Code's MCP client rejects responses with extra fields
    /// like X-Cognitive-Pulse during the handshake. Cognitive pulse is ONLY included in
    /// tool call responses, NOT in initialize/shutdown/tools_list.
    pub(super) async fn handle_initialize(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("Handling initialize request");

        // MCP-compliant initialize response - NO EXTENSIONS ALLOWED
        // Claude Code's MCP client requires strict JSON-RPC 2.0 format for handshake
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
        // NOTE: DO NOT add .with_pulse() here - it breaks Claude Code MCP connection!
        // The X-Cognitive-Pulse extension is only for tool call responses.
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
