//! Legacy memory operation handlers.

use serde_json::json;

use context_graph_core::traits::SearchOptions;
use context_graph_core::types::{CognitivePulse, MemoryNode, SuggestedAction};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::Handlers;

impl Handlers {
    /// Handle memory/store request.
    pub(super) async fn handle_memory_store(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'content' parameter",
                );
            }
        };

        // Create node with stub embedding
        let embedding = vec![0.1; 1536]; // Stub embedding
        let node = MemoryNode::new(content, embedding);
        let node_id = node.id;

        match self.memory_store.store(node).await {
            Ok(_) => {
                let pulse =
                    CognitivePulse::new(0.6, 0.75, 0.0, 1.0, SuggestedAction::Continue, None);
                JsonRpcResponse::success(id, json!({ "nodeId": node_id.to_string() }))
                    .with_pulse(pulse)
            }
            Err(e) => JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e.to_string()),
        }
    }

    /// Handle memory/retrieve request.
    pub(super) async fn handle_memory_retrieve(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let node_id_str = match params.get("nodeId").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'nodeId' parameter",
                );
            }
        };

        let node_id = match uuid::Uuid::parse_str(node_id_str) {
            Ok(u) => u,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Invalid UUID format",
                );
            }
        };

        match self.memory_store.retrieve(node_id).await {
            Ok(Some(node)) => JsonRpcResponse::success(
                id,
                json!({
                    "node": {
                        "id": node.id.to_string(),
                        "content": node.content,
                        "modality": format!("{:?}", node.metadata.modality),
                        "johariQuadrant": format!("{:?}", node.quadrant),
                        "importance": node.importance,
                        "createdAt": node.created_at.to_rfc3339(),
                    }
                }),
            ),
            Ok(None) => JsonRpcResponse::error(id, error_codes::NODE_NOT_FOUND, "Node not found"),
            Err(e) => JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e.to_string()),
        }
    }

    /// Handle memory/search request.
    pub(super) async fn handle_memory_search(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'query' parameter",
                );
            }
        };

        let top_k = params.get("topK").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let options = SearchOptions::new(top_k);

        match self.memory_store.search_text(query, options).await {
            Ok(results) => {
                let results_json: Vec<_> = results
                    .iter()
                    .map(|(node, score)| {
                        json!({
                            "nodeId": node.id.to_string(),
                            "content": node.content,
                            "score": score,
                        })
                    })
                    .collect();

                let pulse = CognitivePulse::new(0.4, 0.8, 0.0, 1.0, SuggestedAction::Continue, None);
                JsonRpcResponse::success(id, json!({ "results": results_json })).with_pulse(pulse)
            }
            Err(e) => JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e.to_string()),
        }
    }

    /// Handle memory/delete request.
    pub(super) async fn handle_memory_delete(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let node_id_str = match params.get("nodeId").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'nodeId' parameter",
                );
            }
        };

        let node_id = match uuid::Uuid::parse_str(node_id_str) {
            Ok(u) => u,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Invalid UUID format",
                );
            }
        };

        let soft = params.get("soft").and_then(|v| v.as_bool()).unwrap_or(true);

        match self.memory_store.delete(node_id, soft).await {
            Ok(deleted) => JsonRpcResponse::success(id, json!({ "deleted": deleted })),
            Err(e) => JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e.to_string()),
        }
    }
}
