//! Helper functions for Full State Verification Meta-UTL tests.
//!
//! Provides setup functions for creating verifiable handlers with shared access
//! to underlying stores and trackers for direct inspection.

use std::sync::Arc;

use parking_lot::RwLock;

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::purpose::GoalHierarchy;
use context_graph_core::stubs::{
    InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor,
};
use context_graph_core::types::fingerprint::{
    PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};

use crate::handlers::core::MetaUtlTracker;
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

/// Create test handlers with SHARED access for direct verification.
///
/// Returns the handlers plus the underlying store and tracker for direct inspection.
pub fn create_verifiable_handlers_with_tracker() -> (
    Handlers,
    Arc<InMemoryTeleologicalStore>,
    Arc<RwLock<MetaUtlTracker>>,
) {
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor = Arc::new(StubUtlProcessor::new());
    let multi_array = Arc::new(StubMultiArrayProvider::new());
    let alignment_calc: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());
    let goal_hierarchy = Arc::new(RwLock::new(GoalHierarchy::default()));

    // Create MetaUtlTracker with SHARED access
    let meta_utl_tracker = Arc::new(RwLock::new(MetaUtlTracker::new()));

    let handlers = Handlers::with_meta_utl_tracker(
        store.clone(),
        utl_processor,
        multi_array,
        alignment_calc,
        goal_hierarchy,
        meta_utl_tracker.clone(),
    );

    (handlers, store, meta_utl_tracker)
}

/// Create a test fingerprint.
pub fn create_test_fingerprint() -> TeleologicalFingerprint {
    TeleologicalFingerprint::new(
        SemanticFingerprint::zeroed(),
        PurposeVector::default(),
        [0u8; 32],
    )
}

/// Build JSON-RPC request.
pub fn make_request(method: &str, params: serde_json::Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: method.to_string(),
        params: Some(params),
    }
}

/// Build JSON-RPC request with no params.
pub fn make_request_no_params(method: &str) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: method.to_string(),
        params: None,
    }
}
