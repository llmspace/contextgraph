//! Common test helpers for Full State Verification tests.
//!
//! This module provides setup functions and utilities shared across FSV tests.

use std::sync::Arc;

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::stubs::{
    InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor,
};
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
};

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

// Import shared test helpers for RocksDB integration tests
use super::super::create_test_hierarchy;

/// Create test handlers AND return direct access to the store for verification.
///
/// TASK-S003: Updated to include GoalAlignmentCalculator and GoalHierarchy.
pub(crate) fn create_handlers_with_store_access() -> (
    Handlers,
    Arc<dyn TeleologicalMemoryStore>,
    Arc<dyn MultiArrayEmbeddingProvider>,
) {
    let teleological_store: Arc<dyn TeleologicalMemoryStore> =
        Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());
    // Must use test hierarchy with strategic goal - store handler requires it (AP-007)
    let goal_hierarchy = create_test_hierarchy();

    let handlers = Handlers::new(
        Arc::clone(&teleological_store),
        Arc::clone(&utl_processor),
        Arc::clone(&multi_array_provider),
        alignment_calculator,
        goal_hierarchy,
    );

    (handlers, teleological_store, multi_array_provider)
}

/// Create a JSON-RPC request for testing.
pub(crate) fn make_request(
    method: &str,
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id,
        method: method.to_string(),
        params,
    }
}

/// Helper: Check if fingerprint exists in store (via retrieve).
pub(crate) async fn exists_in_store(
    store: &Arc<dyn TeleologicalMemoryStore>,
    id: uuid::Uuid,
) -> bool {
    store
        .retrieve(id)
        .await
        .map(|opt| opt.is_some())
        .unwrap_or(false)
}
