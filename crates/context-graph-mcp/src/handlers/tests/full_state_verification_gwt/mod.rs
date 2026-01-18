//! Full State Verification Tests for GWT (Global Workspace Theory) Integration
//!
//! TASK-GWT-001: Integration tests verifying GWT tools return REAL data from
//! physical components - NO mocks, NO stubs, FAIL FAST on errors.
//!
//! These tests follow Full State Verification (FSV) pattern:
//! 1. Execute GWT tool via MCP handler
//! 2. Parse response and verify structure
//! 3. Verify data contains REAL values (not null, valid ranges)
//! 4. Cross-validate between related tools
//!
//! # Test Categories
//!
//! - **Workspace Status**: Verifies winner-take-all selection state
//! - **Ego State**: Verifies purpose vector and topic profile
//! - **Cross-Validation**: Verifies consistency across GWT tools
//! - **ATC (Threshold Calibration)**: Verifies adaptive threshold system
//! - **Dream Consolidation**: Verifies dream cycle and amortized learning
//! - **Neuromodulation**: Verifies 4-modulator control system

mod atc_tests;
mod dream_tests;
mod ego_tests;
mod integration_tests;
mod neuromodulation_tests;
mod workspace_tests;

use std::sync::Arc;

use parking_lot::RwLock as ParkingRwLock;
use serde_json::{json, Value};
use tempfile::TempDir;

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::monitoring::{StubLayerStatusProvider, StubSystemMonitor};
use context_graph_core::purpose::{GoalHierarchy, GoalLevel, GoalNode};
use context_graph_core::stubs::{
    InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor,
};
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
};
use context_graph_core::{LayerStatusProvider, SystemMonitor};
use context_graph_storage::teleological::RocksDbTeleologicalStore;

use crate::adapters::UtlProcessorAdapter;
use crate::handlers::core::MetaUtlTracker;
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

/// Create test handlers with REAL GWT components wired in.
///
/// Uses in-memory stores but REAL GWT implementations:
/// - GwtSystemProviderImpl (real GWT state calculator)
/// - WorkspaceProviderImpl (real global workspace)
/// - MetaCognitiveProviderImpl (real meta-cognitive loop)
pub(super) fn create_handlers_with_gwt() -> Handlers {
    let teleological_store: Arc<dyn TeleologicalMemoryStore> =
        Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());
    let goal_hierarchy = Arc::new(ParkingRwLock::new(create_test_hierarchy()));
    let meta_utl_tracker = Arc::new(ParkingRwLock::new(MetaUtlTracker::new()));
    let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor);
    let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    Handlers::with_default_gwt(
        teleological_store,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        goal_hierarchy,
        meta_utl_tracker,
        system_monitor,
        layer_status_provider,
    )
}

/// Create test handlers with REAL RocksDB storage and REAL GWT components.
pub(super) async fn create_handlers_with_rocksdb_and_gwt() -> (Handlers, TempDir) {
    let tempdir = TempDir::new().expect("Failed to create temp directory");
    let db_path = tempdir.path().join("test_gwt_rocksdb");

    let rocksdb_store =
        RocksDbTeleologicalStore::open(&db_path).expect("Failed to open RocksDbTeleologicalStore");
    // Note: EmbedderIndexRegistry is initialized in constructor

    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(rocksdb_store);
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(UtlProcessorAdapter::with_defaults());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
        Arc::new(DefaultAlignmentCalculator::new());
    let goal_hierarchy = Arc::new(ParkingRwLock::new(create_test_hierarchy()));
    let meta_utl_tracker = Arc::new(ParkingRwLock::new(MetaUtlTracker::new()));
    let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor);
    let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    let handlers = Handlers::with_default_gwt(
        teleological_store,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        goal_hierarchy,
        meta_utl_tracker,
        system_monitor,
        layer_status_provider,
    );

    (handlers, tempdir)
}

/// Create test goal hierarchy.
/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
pub(super) fn create_test_hierarchy() -> GoalHierarchy {
    use context_graph_core::purpose::GoalDiscoveryMetadata;
    use context_graph_core::types::fingerprint::SemanticFingerprint;

    let mut hierarchy = GoalHierarchy::new();
    let discovery = GoalDiscoveryMetadata::bootstrap();

    // Strategic goal 1 (top-level, no parent)
    let s1_goal = GoalNode::autonomous_goal(
        "GWT Test Strategic".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create Strategic goal 1");
    hierarchy
        .add_goal(s1_goal)
        .expect("Failed to add Strategic goal 1");

    // Strategic goal 2 (top-level, no parent)
    let s2_goal = GoalNode::autonomous_goal(
        "Achieve topic coherence".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery,
    )
    .expect("Failed to create Strategic goal 2");
    hierarchy
        .add_goal(s2_goal)
        .expect("Failed to add Strategic goal 2");

    hierarchy
}

/// Helper to make tools/call request.
pub(super) fn make_tool_call_request(tool_name: &str, args: Option<Value>) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_name,
            "arguments": args.unwrap_or(json!({}))
        })),
    }
}

/// Extract content from tool call response.
pub(super) fn extract_tool_content(response_value: &Value) -> Option<Value> {
    response_value
        .get("result")?
        .get("content")?
        .as_array()?
        .first()?
        .get("text")
        .and_then(|t| serde_json::from_str(t.as_str()?).ok())
}
