//! Test Infrastructure for Integration E2E Tests
//!
//! Provides shared test context, helpers, and utilities used across
//! all FSV (Full State Verification) integration tests.

use std::sync::Arc;

use parking_lot::RwLock;

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::johari::{DynDefaultJohariManager, JohariTransitionManager};
use context_graph_core::purpose::{GoalDiscoveryMetadata, GoalHierarchy, GoalLevel, GoalNode};
use context_graph_core::stubs::{
    InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor,
};
use context_graph_core::traits::{MultiArrayEmbeddingProvider, UtlProcessor};
use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};

use crate::handlers::core::MetaUtlTracker;
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

// Re-export commonly used items for other test modules
pub use crate::protocol::error_codes;
pub use context_graph_core::johari::NUM_EMBEDDERS;
pub use context_graph_core::types::JohariQuadrant;
pub use serde_json::json;
pub use sha2::{Digest, Sha256};
pub use uuid::Uuid;

/// Comprehensive test context with shared access to all Sources of Truth.
///
/// This struct provides direct access to:
/// - InMemoryTeleologicalStore (fingerprint storage)
/// - GoalHierarchy (purpose/goal tree)
/// - JohariTransitionManager (Johari window operations)
/// - MetaUtlTracker (Meta-UTL predictions)
pub struct TestContext {
    pub handlers: Handlers,
    pub store: Arc<InMemoryTeleologicalStore>,
    pub hierarchy: Arc<RwLock<GoalHierarchy>>,
    #[allow(dead_code)]
    pub johari_manager: Arc<dyn JohariTransitionManager>,
    pub meta_utl_tracker: Arc<RwLock<MetaUtlTracker>>,
}

impl TestContext {
    /// Create a new test context with full access to all Sources of Truth.
    pub fn new() -> Self {
        let store = Arc::new(InMemoryTeleologicalStore::new());
        let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
        let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
            Arc::new(StubMultiArrayProvider::new());
        let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
            Arc::new(DefaultAlignmentCalculator::new());

        // Create goal hierarchy with North Star and sub-goals
        let hierarchy = Arc::new(RwLock::new(create_test_hierarchy()));

        // Create JohariTransitionManager with SHARED store reference
        let johari_manager: Arc<dyn JohariTransitionManager> =
            Arc::new(DynDefaultJohariManager::new(store.clone()));

        // Create MetaUtlTracker with SHARED access
        let meta_utl_tracker = Arc::new(RwLock::new(MetaUtlTracker::new()));

        let handlers = Handlers::with_meta_utl_tracker(
            store.clone(),
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            hierarchy.clone(),
            johari_manager.clone(),
            meta_utl_tracker.clone(),
        );

        Self {
            handlers,
            store,
            hierarchy,
            johari_manager,
            meta_utl_tracker,
        }
    }

    /// Create a test context WITHOUT a North Star (for error testing).
    pub fn new_without_north_star() -> Self {
        let store = Arc::new(InMemoryTeleologicalStore::new());
        let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
        let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
            Arc::new(StubMultiArrayProvider::new());
        let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
            Arc::new(DefaultAlignmentCalculator::new());

        // Empty hierarchy - no North Star
        let hierarchy = Arc::new(RwLock::new(GoalHierarchy::new()));

        let johari_manager: Arc<dyn JohariTransitionManager> =
            Arc::new(DynDefaultJohariManager::new(store.clone()));

        let meta_utl_tracker = Arc::new(RwLock::new(MetaUtlTracker::new()));

        let handlers = Handlers::with_meta_utl_tracker(
            store.clone(),
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            hierarchy.clone(),
            johari_manager.clone(),
            meta_utl_tracker.clone(),
        );

        Self {
            handlers,
            store,
            hierarchy,
            johari_manager,
            meta_utl_tracker,
        }
    }
}

/// Create a test goal hierarchy with 3 levels.
/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
pub fn create_test_hierarchy() -> GoalHierarchy {
    let mut hierarchy = GoalHierarchy::new();
    let discovery = GoalDiscoveryMetadata::bootstrap();

    // Strategic goal 1 (top-level, no parent)
    let s1_goal = GoalNode::autonomous_goal(
        "Build the best ML learning system".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create Strategic goal 1");
    let s1_id = s1_goal.id;
    hierarchy
        .add_goal(s1_goal)
        .expect("Failed to add Strategic goal 1");

    // Strategic goal 2 (top-level, no parent)
    let s2_goal = GoalNode::autonomous_goal(
        "Enhance user experience".into(),
        GoalLevel::Strategic,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create Strategic goal 2");
    hierarchy
        .add_goal(s2_goal)
        .expect("Failed to add Strategic goal 2");

    // Tactical goal - child of Strategic goal 1
    let t1_goal = GoalNode::child_goal(
        "Implement semantic search".into(),
        GoalLevel::Tactical,
        s1_id,
        SemanticFingerprint::zeroed(),
        discovery.clone(),
    )
    .expect("Failed to create tactical goal");
    let t1_id = t1_goal.id;
    hierarchy
        .add_goal(t1_goal)
        .expect("Failed to add tactical goal");

    // Immediate goal - child of Tactical goal
    let i1_goal = GoalNode::child_goal(
        "Add vector similarity".into(),
        GoalLevel::Immediate,
        t1_id,
        SemanticFingerprint::zeroed(),
        discovery,
    )
    .expect("Failed to create immediate goal");
    hierarchy
        .add_goal(i1_goal)
        .expect("Failed to add immediate goal");

    hierarchy
}

/// Build JSON-RPC request with ID and params.
pub fn make_request(method: &str, id: i64, params: serde_json::Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(id)),
        method: method.to_string(),
        params: Some(params),
    }
}

/// Build JSON-RPC request with no params.
pub fn make_request_no_params(method: &str, id: i64) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(id)),
        method: method.to_string(),
        params: None,
    }
}

/// Create a test fingerprint with specific Johari configuration.
pub fn create_fingerprint_with_johari(
    quadrants: [JohariQuadrant; NUM_EMBEDDERS],
) -> TeleologicalFingerprint {
    let mut johari = JohariFingerprint::zeroed();
    for (idx, quadrant) in quadrants.iter().enumerate() {
        match quadrant {
            JohariQuadrant::Open => johari.set_quadrant(idx, 1.0, 0.0, 0.0, 0.0, 1.0),
            JohariQuadrant::Hidden => johari.set_quadrant(idx, 0.0, 1.0, 0.0, 0.0, 1.0),
            JohariQuadrant::Blind => johari.set_quadrant(idx, 0.0, 0.0, 1.0, 0.0, 1.0),
            JohariQuadrant::Unknown => johari.set_quadrant(idx, 0.0, 0.0, 0.0, 1.0, 1.0),
        }
    }

    TeleologicalFingerprint::new(
        SemanticFingerprint::zeroed(),
        PurposeVector::default(),
        johari,
        [0u8; 32],
    )
}
