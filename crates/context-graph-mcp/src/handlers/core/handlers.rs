//! Handlers struct definition and constructors.
//!
//! PRD v6 Section 10 - Minimal Handlers for 6 MCP tools.

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::json;
use tracing::info;

use context_graph_core::monitoring::LayerStatusProvider;
use context_graph_core::purpose::GoalHierarchy;
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
};

use crate::protocol::{JsonRpcId, JsonRpcResponse};

/// Request handlers for MCP protocol.
///
/// PRD v6 Section 10 - Supports only:
/// - inject_context, store_memory, get_memetic_status, search_graph
/// - trigger_consolidation
/// - merge_concepts
pub struct Handlers {
    /// Teleological memory store - stores TeleologicalFingerprint with 13 embeddings.
    pub(in crate::handlers) teleological_store: Arc<dyn TeleologicalMemoryStore>,

    /// UTL processor for computing learning metrics.
    pub(in crate::handlers) utl_processor: Arc<dyn UtlProcessor>,

    /// Multi-array embedding provider - generates all 13 embeddings per content.
    pub(in crate::handlers) multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,

    /// Goal hierarchy - defines strategic goals and sub-goals.
    pub(in crate::handlers) goal_hierarchy: Arc<RwLock<GoalHierarchy>>,

    /// Layer status provider for get_memetic_status.
    pub(in crate::handlers) layer_status_provider: Arc<dyn LayerStatusProvider>,
}

impl Handlers {
    /// Create handlers with all dependencies explicitly provided.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint
    /// * `utl_processor` - UTL processor for learning metrics
    /// * `multi_array_provider` - 13-embedding generator
    /// * `goal_hierarchy` - Goal hierarchy (can be empty initially)
    /// * `layer_status_provider` - Provider for layer status information
    pub fn with_all(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        layer_status_provider: Arc<dyn LayerStatusProvider>,
    ) -> Self {
        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            goal_hierarchy,
            layer_status_provider,
        }
    }

    /// Handle MCP initialize request.
    ///
    /// Returns server capabilities per MCP protocol.
    pub async fn handle_initialize(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("MCP initialize request received");

        let capabilities = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": false
                }
            },
            "serverInfo": {
                "name": "context-graph",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        JsonRpcResponse::success(id, capabilities)
    }

    /// Handle MCP initialized notification.
    ///
    /// This is a notification (no response expected), but we return
    /// an empty success for consistency in dispatch.
    pub fn handle_initialized_notification(&self) -> JsonRpcResponse {
        info!("MCP initialized notification received");
        JsonRpcResponse::success(None, json!({}))
    }

    /// Handle MCP shutdown request.
    ///
    /// Performs graceful shutdown of handlers.
    pub async fn handle_shutdown(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("MCP shutdown request received");
        JsonRpcResponse::success(id, json!({}))
    }
}
