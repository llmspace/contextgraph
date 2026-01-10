//! Full State Verification Tests for STUBFIX Handlers
//!
//! These tests verify that get_steering_feedback, get_pruning_candidates, and
//! trigger_consolidation return REAL computed data from the database, not stubs.
//!
//! SPEC-STUBFIX-001: get_steering_feedback computes real orphan_count and connectivity
//! SPEC-STUBFIX-002: get_pruning_candidates returns real candidates from PruningService
//! SPEC-STUBFIX-003: trigger_consolidation evaluates real pairs with ConsolidationService
//!
//! Test Pattern:
//! 1. Create handlers with RocksDB store access
//! 2. Store synthetic fingerprints DIRECTLY to store (known data)
//! 3. Call handler via MCP dispatch
//! 4. Verify output matches expected computation from stored data

use serde_json::json;
use sha2::{Digest, Sha256};

use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint,
};

use crate::protocol::JsonRpcId;

use super::{create_test_handlers_with_rocksdb_store_access, extract_mcp_tool_data, make_request};

// ============================================================================
// Synthetic Data Helpers
// ============================================================================

/// Embedding dimensions (subset needed for tests)
const E1_DIM: usize = 1024;
const E2_DIM: usize = 512;
const E3_DIM: usize = 512;
const E4_DIM: usize = 512;
const E5_DIM: usize = 768;
const E7_DIM: usize = 1536;
const E8_DIM: usize = 384;
const E9_DIM: usize = 1024;
const E10_DIM: usize = 768;
const E11_DIM: usize = 384;
const E12_TOKEN_DIM: usize = 128;
const NUM_EMBEDDERS: usize = 13;

/// Create a synthetic fingerprint with configurable alignment and access count.
///
/// Parameters:
/// - content: Used to generate content hash and embeddings deterministically
/// - theta: The theta_to_north_star alignment value [0.0, 1.0]
/// - access_count: Number of times this fingerprint has been accessed
fn create_test_fingerprint(content: &str, theta: f32, access_count: u64) -> TeleologicalFingerprint {
    let content_hash = {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hasher.finalize().into()
    };

    let semantic = create_test_semantic(&content_hash);
    let purpose_vector = create_test_purpose_vector(theta);
    let johari = JohariFingerprint::zeroed();

    let mut fp = TeleologicalFingerprint::new(semantic, purpose_vector, johari, content_hash);
    // Override theta and access_count for test control
    fp.theta_to_north_star = theta;
    fp.access_count = access_count;
    fp
}

/// Create a synthetic fingerprint with specific created_at time for temporal tests.
fn create_test_fingerprint_with_age(
    content: &str,
    theta: f32,
    access_count: u64,
    days_old: i64,
) -> TeleologicalFingerprint {
    let mut fp = create_test_fingerprint(content, theta, access_count);
    fp.created_at = chrono::Utc::now() - chrono::Duration::days(days_old);
    fp.last_updated = fp.created_at;
    fp
}

/// Create semantic fingerprint with deterministic embeddings from hash.
fn create_test_semantic(hash: &[u8; 32]) -> SemanticFingerprint {
    let seed = |offset: usize| -> f32 {
        let idx = offset % 32;
        let val = hash[idx] as f32 / 255.0;
        (val * 2.0) - 1.0
    };

    let e1_semantic: Vec<f32> = (0..E1_DIM).map(|i| seed(i) * 0.5).collect();
    let e2_temporal_recent: Vec<f32> = (0..E2_DIM).map(|i| seed(i + 1000)).collect();
    let e3_temporal_periodic: Vec<f32> = (0..E3_DIM).map(|i| seed(i + 2000) * 0.8).collect();
    let e4_temporal_positional: Vec<f32> = (0..E4_DIM).map(|i| seed(i + 3000) * 0.6).collect();
    let e5_causal: Vec<f32> = (0..E5_DIM).map(|i| seed(i + 4000)).collect();
    let e7_code: Vec<f32> = (0..E7_DIM).map(|i| seed(i + 6000)).collect();
    let e8_graph: Vec<f32> = (0..E8_DIM).map(|i| seed(i + 7000)).collect();
    let e9_hdc: Vec<f32> = (0..E9_DIM)
        .map(|i| if seed(i + 8000) > 0.0 { 1.0 } else { -1.0 })
        .collect();
    let e10_multimodal: Vec<f32> = (0..E10_DIM).map(|i| seed(i + 9000)).collect();
    let e11_entity: Vec<f32> = (0..E11_DIM).map(|i| seed(i + 10000)).collect();

    // E12: Late interaction - 10 tokens
    let e12_late_interaction: Vec<Vec<f32>> = (0..10)
        .map(|t| (0..E12_TOKEN_DIM).map(|i| seed(i + t * 128 + 11000)).collect())
        .collect();

    // Sparse vectors
    let mut e6_indices: Vec<u16> = (0..20)
        .map(|i| ((hash[i % 32] as u32 * 100) % 30000) as u16)
        .collect();
    e6_indices.sort();
    e6_indices.dedup();
    let e6_values: Vec<f32> = (0..e6_indices.len())
        .map(|i| seed(i + 5000).abs() * 2.0)
        .collect();

    let mut e13_indices: Vec<u16> = (0..30)
        .map(|i| ((hash[(i + 10) % 32] as u32 * 200) % 30000) as u16)
        .collect();
    e13_indices.sort();
    e13_indices.dedup();
    let e13_values: Vec<f32> = (0..e13_indices.len())
        .map(|i| seed(i + 12000).abs() * 3.0)
        .collect();

    SemanticFingerprint {
        e1_semantic,
        e2_temporal_recent,
        e3_temporal_periodic,
        e4_temporal_positional,
        e5_causal,
        e6_sparse: SparseVector {
            indices: e6_indices,
            values: e6_values,
        },
        e7_code,
        e8_graph,
        e9_hdc,
        e10_multimodal,
        e11_entity,
        e12_late_interaction,
        e13_splade: SparseVector {
            indices: e13_indices,
            values: e13_values,
        },
    }
}

/// Create purpose vector with specified theta alignment.
fn create_test_purpose_vector(theta: f32) -> PurposeVector {
    // Create alignments that produce the desired theta
    let alignments: [f32; NUM_EMBEDDERS] = [
        theta * 0.9,
        theta * 0.7,
        theta * 0.65,
        theta * 0.6,
        theta * 0.75,
        theta * 0.5,
        theta * 0.8,
        theta * 0.7,
        theta * 0.4,
        theta * 0.55,
        theta * 0.6,
        theta * 0.85,
        theta * 0.5,
    ];
    PurposeVector::new(alignments)
}

// ============================================================================
// SPEC-STUBFIX-001: get_steering_feedback FSV Tests
// ============================================================================

/// FSV-STEERING-001: Empty store returns zero metrics.
///
/// Verifies get_steering_feedback with empty store returns:
/// - orphan_count = 0
/// - connectivity = 0.0 (no nodes to be connected)
#[tokio::test]
async fn test_steering_feedback_empty_store_returns_zero_metrics() {
    println!("\n================================================================================");
    println!("FSV-STEERING-001: Empty Store Returns Zero Metrics");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Verify store is empty
    let count = store.count().await.expect("count works");
    assert_eq!(count, 0, "Store must be empty");
    println!("[BEFORE] Store count: {}", count);

    // Call get_steering_feedback
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_steering_feedback",
            "arguments": {}
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // Verify gardener details
    let gardener = data.get("gardener_details").expect("Should have gardener_details");
    let connectivity = gardener.get("connectivity").and_then(|v| v.as_f64()).unwrap_or(-1.0);
    let dead_ends_removed = gardener.get("dead_ends_removed").and_then(|v| v.as_u64()).unwrap_or(999);

    println!("[RESULT] gardener_details:");
    println!("  - connectivity: {}", connectivity);
    println!("  - dead_ends_removed: {}", dead_ends_removed);

    // With empty store: connectivity should be 0.0 (no aligned nodes)
    assert!(
        connectivity >= 0.0 && connectivity <= 0.01,
        "Empty store connectivity should be ~0.0, got {}",
        connectivity
    );

    println!("\n[FSV-STEERING-001 PASSED] Empty store returns zero connectivity");
    println!("================================================================================\n");
}

/// FSV-STEERING-002: All orphans produces low connectivity.
///
/// Stores N fingerprints with access_count=0 (orphan proxy) and theta < 0.5.
/// Verifies:
/// - Handler reads all fingerprints
/// - orphan_count matches stored count
/// - connectivity is low
#[tokio::test]
async fn test_steering_feedback_all_orphans_low_connectivity() {
    println!("\n================================================================================");
    println!("FSV-STEERING-002: All Orphans Produces Low Connectivity");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store 5 fingerprints with access_count=0 (orphan proxy) and low alignment
    let test_data = [
        ("orphan_memory_1", 0.2_f32, 0_u64),
        ("orphan_memory_2", 0.3_f32, 0_u64),
        ("orphan_memory_3", 0.1_f32, 0_u64),
        ("orphan_memory_4", 0.25_f32, 0_u64),
        ("orphan_memory_5", 0.15_f32, 0_u64),
    ];

    println!("[SETUP] Storing {} orphan fingerprints:", test_data.len());
    for (content, theta, access_count) in test_data.iter() {
        let fp = create_test_fingerprint(content, *theta, *access_count);
        let id = store.store(fp).await.expect("Store must succeed");
        println!("  - {} (theta={}, access={})", id, theta, access_count);
    }

    let count = store.count().await.expect("count works");
    assert_eq!(count, test_data.len(), "All fingerprints must be stored");

    // Call get_steering_feedback
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_steering_feedback",
            "arguments": {}
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let gardener = data.get("gardener_details").expect("Should have gardener_details");
    let connectivity = gardener.get("connectivity").and_then(|v| v.as_f64()).unwrap_or(-1.0);
    let dead_ends_removed = gardener.get("dead_ends_removed").and_then(|v| v.as_u64()).unwrap_or(999);

    println!("\n[RESULT] gardener_details:");
    println!("  - connectivity: {}", connectivity);
    println!("  - dead_ends_removed (orphan_count): {}", dead_ends_removed);

    // All orphans with theta < 0.5 means connectivity = 0 (none aligned to North Star)
    assert!(
        connectivity >= 0.0 && connectivity < 0.1,
        "All orphans should have low connectivity, got {}",
        connectivity
    );

    // dead_ends_removed should match orphan_count (all 5 have access_count=0)
    assert_eq!(
        dead_ends_removed, 5,
        "dead_ends_removed should equal orphan count"
    );

    println!("\n[FSV-STEERING-002 PASSED] All orphans produces low connectivity and correct orphan count");
    println!("================================================================================\n");
}

/// FSV-STEERING-003: Mixed data produces accurate metrics.
///
/// Stores fingerprints with varied alignment and access patterns:
/// - 3 aligned (theta >= 0.5) with access_count > 0
/// - 2 orphans (access_count = 0) with low alignment
///
/// Verifies connectivity reflects aligned ratio.
#[tokio::test]
async fn test_steering_feedback_mixed_data_accurate_metrics() {
    println!("\n================================================================================");
    println!("FSV-STEERING-003: Mixed Data Produces Accurate Metrics");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store mixed fingerprints
    let test_data = [
        // Aligned and accessed (connected)
        ("aligned_memory_1", 0.8_f32, 5_u64),
        ("aligned_memory_2", 0.75_f32, 3_u64),
        ("aligned_memory_3", 0.6_f32, 2_u64),
        // Orphans (low alignment, never accessed)
        ("orphan_memory_1", 0.2_f32, 0_u64),
        ("orphan_memory_2", 0.3_f32, 0_u64),
    ];

    println!("[SETUP] Storing {} fingerprints:", test_data.len());
    for (content, theta, access_count) in test_data.iter() {
        let fp = create_test_fingerprint(content, *theta, *access_count);
        let id = store.store(fp).await.expect("Store must succeed");
        println!(
            "  - {} (theta={:.2}, access={}) {}",
            id,
            theta,
            access_count,
            if *theta >= 0.5 { "ALIGNED" } else { "NOT ALIGNED" }
        );
    }

    // Expected metrics:
    // - 3 out of 5 are aligned (theta >= 0.5) -> connectivity = 0.6
    // - 2 orphans (access_count = 0)
    let expected_aligned = 3;
    let expected_orphans = 2;
    let expected_connectivity = expected_aligned as f64 / test_data.len() as f64;

    println!("\n[EXPECTED]");
    println!("  - Aligned nodes: {}", expected_aligned);
    println!("  - Orphan nodes: {}", expected_orphans);
    println!("  - Connectivity: {:.2}", expected_connectivity);

    // Call get_steering_feedback
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_steering_feedback",
            "arguments": {}
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let gardener = data.get("gardener_details").expect("Should have gardener_details");
    let connectivity = gardener.get("connectivity").and_then(|v| v.as_f64()).unwrap_or(-1.0);
    let dead_ends_removed = gardener.get("dead_ends_removed").and_then(|v| v.as_u64()).unwrap_or(999);

    println!("\n[RESULT] gardener_details:");
    println!("  - connectivity: {:.4}", connectivity);
    println!("  - dead_ends_removed (orphan_count): {}", dead_ends_removed);

    // Verify connectivity matches expected (with tolerance for floating point)
    let tolerance = 0.05;
    assert!(
        (connectivity - expected_connectivity).abs() < tolerance,
        "Connectivity should be ~{:.2}, got {:.4}",
        expected_connectivity,
        connectivity
    );

    // Verify orphan count
    assert_eq!(
        dead_ends_removed, expected_orphans as u64,
        "Orphan count should match"
    );

    println!("\n[FSV-STEERING-003 PASSED] Mixed data produces accurate connectivity and orphan count");
    println!("================================================================================\n");
}

// ============================================================================
// SPEC-STUBFIX-002: get_pruning_candidates FSV Tests
// ============================================================================

/// FSV-PRUNING-001: Empty store returns empty candidates.
#[tokio::test]
async fn test_pruning_candidates_empty_store_returns_empty() {
    println!("\n================================================================================");
    println!("FSV-PRUNING-001: Empty Store Returns Empty Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let count = store.count().await.expect("count works");
    assert_eq!(count, 0, "Store must be empty");

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_pruning_candidates",
            "arguments": {}
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let candidates = data.get("candidates").and_then(|v| v.as_array());
    let summary = data.get("summary").expect("Should have summary");
    let total = summary.get("total_candidates").and_then(|v| v.as_u64()).unwrap_or(999);

    println!("[RESULT]");
    println!("  - candidates: {:?}", candidates.map(|c| c.len()));
    println!("  - total_candidates: {}", total);

    assert_eq!(total, 0, "Empty store should have 0 candidates");

    println!("\n[FSV-PRUNING-001 PASSED] Empty store returns empty candidates");
    println!("================================================================================\n");
}

/// FSV-PRUNING-002: Fresh data produces no candidates.
///
/// Stores fingerprints that are all:
/// - Recently created (< 30 days)
/// - High alignment (> 0.5)
/// - Non-orphan (access_count > 0)
#[tokio::test]
async fn test_pruning_candidates_fresh_data_no_candidates() {
    println!("\n================================================================================");
    println!("FSV-PRUNING-002: Fresh High-Quality Data Produces No Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fresh, high-quality fingerprints
    let test_data = [
        ("fresh_memory_1", 0.8_f32, 5_u64, 0_i64),  // 0 days old
        ("fresh_memory_2", 0.75_f32, 3_u64, 1_i64), // 1 day old
        ("fresh_memory_3", 0.9_f32, 10_u64, 5_i64), // 5 days old
    ];

    println!("[SETUP] Storing {} fresh fingerprints:", test_data.len());
    for (content, theta, access_count, days_old) in test_data.iter() {
        let fp = create_test_fingerprint_with_age(content, *theta, *access_count, *days_old);
        let id = store.store(fp).await.expect("Store must succeed");
        println!(
            "  - {} (theta={:.2}, access={}, age={}d)",
            id, theta, access_count, days_old
        );
    }

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_pruning_candidates",
            "arguments": {
                "min_staleness_days": 30,
                "min_alignment": 0.4
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let summary = data.get("summary").expect("Should have summary");
    let total = summary.get("total_candidates").and_then(|v| v.as_u64()).unwrap_or(999);

    println!("\n[RESULT]");
    println!("  - total_candidates: {}", total);

    // Fresh, high-quality data should not be pruning candidates
    assert_eq!(
        total, 0,
        "Fresh high-quality data should produce no candidates"
    );

    println!("\n[FSV-PRUNING-002 PASSED] Fresh data produces no candidates");
    println!("================================================================================\n");
}

/// FSV-PRUNING-003: Stale/low-alignment data produces candidates.
///
/// Stores mix of:
/// - Old, low-alignment fingerprints (should be candidates)
/// - Fresh, high-alignment fingerprints (should NOT be candidates)
#[tokio::test]
async fn test_pruning_candidates_stale_data_produces_candidates() {
    println!("\n================================================================================");
    println!("FSV-PRUNING-003: Stale/Low-Alignment Data Produces Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store mix of fingerprints
    let test_data = [
        // Should be candidates (stale OR low alignment)
        ("stale_memory_1", 0.2_f32, 0_u64, 100_i64),  // Old + low alignment + orphan
        ("stale_memory_2", 0.3_f32, 0_u64, 120_i64),  // Old + low alignment + orphan
        ("low_align_memory", 0.1_f32, 1_u64, 50_i64), // Low alignment
        // Should NOT be candidates (fresh + high alignment)
        ("fresh_memory_1", 0.8_f32, 5_u64, 5_i64),
        ("fresh_memory_2", 0.9_f32, 10_u64, 2_i64),
    ];

    println!("[SETUP] Storing {} fingerprints:", test_data.len());
    let mut stored_ids = Vec::new();
    for (content, theta, access_count, days_old) in test_data.iter() {
        let fp = create_test_fingerprint_with_age(content, *theta, *access_count, *days_old);
        let id = store.store(fp).await.expect("Store must succeed");
        stored_ids.push(id);
        let is_candidate = *theta < 0.4 || *days_old > 90 || *access_count == 0;
        println!(
            "  - {} (theta={:.2}, access={}, age={}d) {}",
            id,
            theta,
            access_count,
            days_old,
            if is_candidate { "CANDIDATE" } else { "KEEP" }
        );
    }

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_pruning_candidates",
            "arguments": {
                "min_staleness_days": 90,
                "min_alignment": 0.4,
                "limit": 10
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let candidates = data
        .get("candidates")
        .and_then(|v| v.as_array())
        .expect("Should have candidates array");
    let summary = data.get("summary").expect("Should have summary");
    let total = summary.get("total_candidates").and_then(|v| v.as_u64()).unwrap_or(0);

    println!("\n[RESULT]");
    println!("  - total_candidates: {}", total);
    println!("  - candidates returned: {}", candidates.len());

    for (i, candidate) in candidates.iter().enumerate() {
        let memory_id = candidate.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
        let reason = candidate.get("reason").and_then(|v| v.as_str()).unwrap_or("?");
        let alignment = candidate.get("alignment").and_then(|v| v.as_f64()).unwrap_or(-1.0);
        let age_days = candidate.get("age_days").and_then(|v| v.as_u64()).unwrap_or(0);
        println!(
            "    [{}] {} - reason: {}, alignment: {:.2}, age: {}d",
            i + 1,
            memory_id,
            reason,
            alignment,
            age_days
        );
    }

    // Should have at least some candidates (the stale/low-alignment ones)
    assert!(
        total >= 1,
        "Should have at least 1 pruning candidate, got {}",
        total
    );

    // Verify each returned candidate has a valid reason
    for candidate in candidates {
        let reason = candidate.get("reason").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            !reason.is_empty(),
            "Each candidate must have a reason"
        );
    }

    println!("\n[FSV-PRUNING-003 PASSED] Stale/low-alignment data produces candidates with reasons");
    println!("================================================================================\n");
}

// ============================================================================
// SPEC-STUBFIX-003: trigger_consolidation FSV Tests
// ============================================================================

/// FSV-CONSOLIDATION-001: Empty store returns empty candidates.
#[tokio::test]
async fn test_consolidation_empty_store_returns_empty() {
    println!("\n================================================================================");
    println!("FSV-CONSOLIDATION-001: Empty Store Returns Empty Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let count = store.count().await.expect("count works");
    assert_eq!(count, 0, "Store must be empty");

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "trigger_consolidation",
            "arguments": {
                "strategy": "similarity"
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let statistics = data.get("statistics").expect("Should have statistics");
    let pairs_evaluated = statistics
        .get("pairs_evaluated")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("[RESULT]");
    println!("  - pairs_evaluated: {}", pairs_evaluated);

    assert_eq!(pairs_evaluated, 0, "Empty store should evaluate 0 pairs");

    println!("\n[FSV-CONSOLIDATION-001 PASSED] Empty store returns no pairs");
    println!("================================================================================\n");
}

/// FSV-CONSOLIDATION-002: Orthogonal embeddings produce no candidates.
///
/// Stores fingerprints with very different content (different hashes = different embeddings).
/// Since synthetic embeddings are hash-based, different content produces orthogonal vectors.
#[tokio::test]
async fn test_consolidation_orthogonal_no_candidates() {
    println!("\n================================================================================");
    println!("FSV-CONSOLIDATION-002: Orthogonal Embeddings Produce No Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fingerprints with very different content (different hashes)
    let test_data = [
        "machine learning optimization with gradient descent",
        "distributed systems consensus with Raft protocol",
        "database indexing using B-trees and hash indexes",
        "rust memory safety with ownership and borrowing",
        "natural language processing with transformers",
    ];

    println!("[SETUP] Storing {} diverse fingerprints:", test_data.len());
    for content in test_data.iter() {
        let fp = create_test_fingerprint(content, 0.7, 5);
        let id = store.store(fp).await.expect("Store must succeed");
        println!("  - {} (content: {}...)", id, &content[..30.min(content.len())]);
    }

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "trigger_consolidation",
            "arguments": {
                "strategy": "similarity",
                "min_similarity": 0.95,
                "max_memories": 10
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let result_status = data.get("consolidation_result").expect("Should have result");
    let candidate_count = result_status
        .get("candidate_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);
    let statistics = data.get("statistics").expect("Should have statistics");
    let pairs_evaluated = statistics
        .get("pairs_evaluated")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!("\n[RESULT]");
    println!("  - pairs_evaluated: {}", pairs_evaluated);
    println!("  - candidate_count: {}", candidate_count);

    // With 5 items, we have 5*4/2 = 10 pairs
    assert!(
        pairs_evaluated > 0,
        "Should evaluate some pairs, got {}",
        pairs_evaluated
    );

    // Different content should produce low similarity, so no candidates at 0.95 threshold
    assert_eq!(
        candidate_count, 0,
        "Orthogonal embeddings should produce no candidates at 0.95 threshold"
    );

    println!("\n[FSV-CONSOLIDATION-002 PASSED] Orthogonal embeddings produce no candidates");
    println!("================================================================================\n");
}

/// FSV-CONSOLIDATION-003: Identical content produces candidates.
///
/// Stores fingerprints with identical content (same hash = same embeddings).
#[tokio::test]
async fn test_consolidation_identical_produces_candidates() {
    println!("\n================================================================================");
    println!("FSV-CONSOLIDATION-003: Identical Content Produces Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fingerprints with IDENTICAL content (same hash = same embeddings)
    let identical_content = "machine learning optimization with neural networks";

    println!("[SETUP] Storing 3 fingerprints with IDENTICAL content:");
    for i in 0..3 {
        let fp = create_test_fingerprint(identical_content, 0.7, 5);
        let id = store.store(fp).await.expect("Store must succeed");
        println!("  - [{}] {} (identical content)", i + 1, id);
    }

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "trigger_consolidation",
            "arguments": {
                "strategy": "similarity",
                "min_similarity": 0.9,
                "max_memories": 10
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let result_status = data.get("consolidation_result").expect("Should have result");
    let candidate_count = result_status
        .get("candidate_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let statistics = data.get("statistics").expect("Should have statistics");
    let pairs_evaluated = statistics
        .get("pairs_evaluated")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!("\n[RESULT]");
    println!("  - pairs_evaluated: {}", pairs_evaluated);
    println!("  - candidate_count: {}", candidate_count);

    // Check candidates_sample if available
    if let Some(sample) = data.get("candidates_sample").and_then(|v| v.as_array()) {
        println!("  - candidates_sample count: {}", sample.len());
        for (i, c) in sample.iter().enumerate() {
            let similarity = c.get("similarity").and_then(|v| v.as_f64()).unwrap_or(-1.0);
            println!("    [{}] similarity: {:.4}", i + 1, similarity);
        }
    }

    // With 3 identical items, we have 3 pairs, and all should have similarity = 1.0
    assert!(
        pairs_evaluated > 0,
        "Should evaluate some pairs"
    );

    // Identical content should produce candidates
    assert!(
        candidate_count >= 1,
        "Identical content should produce at least 1 consolidation candidate, got {}",
        candidate_count
    );

    println!("\n[FSV-CONSOLIDATION-003 PASSED] Identical content produces consolidation candidates");
    println!("================================================================================\n");
}

// ============================================================================
// Full State Verification - RocksDB Direct Inspection Tests
// ============================================================================

/// FSV-ROCKSDB-STEERING-001: Verify steering data matches direct RocksDB state.
///
/// This test directly inspects the store to verify:
/// - Count matches what handler reports
/// - Each fingerprint's theta and access_count are correctly read
#[tokio::test]
async fn test_fsv_rocksdb_steering_data_matches_store() {
    println!("\n================================================================================");
    println!("FSV-ROCKSDB-STEERING-001: Verify Steering Data Matches RocksDB State");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fingerprints with known values
    let test_data = [
        ("known_fp_1", 0.8_f32, 5_u64),  // aligned, accessed
        ("known_fp_2", 0.3_f32, 0_u64),  // not aligned, orphan
        ("known_fp_3", 0.6_f32, 3_u64),  // aligned, accessed
    ];

    let mut stored_ids = Vec::new();
    println!("[SETUP] Storing fingerprints with known values:");
    for (content, theta, access_count) in test_data.iter() {
        let fp = create_test_fingerprint(content, *theta, *access_count);
        let id = store.store(fp).await.expect("Store must succeed");
        stored_ids.push(id);
        println!("  - {} (theta={:.2}, access={})", id, theta, access_count);
    }

    // DIRECT ROCKSDB VERIFICATION: Read each fingerprint directly
    println!("\n[DIRECT ROCKSDB VERIFICATION] Reading fingerprints from store:");
    let mut aligned_count = 0;
    let mut orphan_count = 0;

    for id in stored_ids.iter() {
        let fp = store
            .retrieve(*id)
            .await
            .expect("retrieve works")
            .expect("fingerprint exists");

        println!(
            "  - {} theta={:.4}, access={}, aligned={}, orphan={}",
            id,
            fp.theta_to_north_star,
            fp.access_count,
            fp.theta_to_north_star >= 0.5,
            fp.access_count == 0
        );

        if fp.theta_to_north_star >= 0.5 {
            aligned_count += 1;
        }
        if fp.access_count == 0 {
            orphan_count += 1;
        }
    }

    let expected_connectivity = aligned_count as f64 / test_data.len() as f64;
    println!("\n[EXPECTED from direct inspection]");
    println!("  - aligned_count: {}", aligned_count);
    println!("  - orphan_count: {}", orphan_count);
    println!("  - connectivity: {:.4}", expected_connectivity);

    // Call handler and verify it matches
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_steering_feedback",
            "arguments": {}
        })),
    );
    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let gardener = data.get("gardener_details").expect("Should have gardener_details");
    let handler_connectivity = gardener.get("connectivity").and_then(|v| v.as_f64()).unwrap_or(-1.0);
    let handler_orphans = gardener.get("dead_ends_removed").and_then(|v| v.as_u64()).unwrap_or(999);

    println!("\n[HANDLER RESULT]");
    println!("  - connectivity: {:.4}", handler_connectivity);
    println!("  - dead_ends_removed: {}", handler_orphans);

    // Verify handler matches direct inspection
    let tolerance = 0.01;
    assert!(
        (handler_connectivity - expected_connectivity).abs() < tolerance,
        "Handler connectivity {:.4} must match direct inspection {:.4}",
        handler_connectivity,
        expected_connectivity
    );
    assert_eq!(
        handler_orphans, orphan_count as u64,
        "Handler orphan count must match direct inspection"
    );

    println!("\n[FSV-ROCKSDB-STEERING-001 PASSED] Handler data matches RocksDB state");
    println!("================================================================================\n");
}

/// FSV-ROCKSDB-PRUNING-001: Verify pruning candidates match direct RocksDB state.
#[tokio::test]
async fn test_fsv_rocksdb_pruning_candidates_match_store() {
    println!("\n================================================================================");
    println!("FSV-ROCKSDB-PRUNING-001: Verify Pruning Candidates Match RocksDB State");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fingerprints - some should be candidates, some not
    let test_data = [
        ("orphan_1", 0.2_f32, 0_u64),  // Candidate: orphan + low alignment
        ("orphan_2", 0.15_f32, 0_u64), // Candidate: orphan + low alignment
        ("good_1", 0.9_f32, 10_u64),   // Not candidate: high alignment + accessed
    ];

    let mut candidate_ids = Vec::new();
    println!("[SETUP] Storing fingerprints:");
    for (content, theta, access_count) in test_data.iter() {
        let fp = create_test_fingerprint(content, *theta, *access_count);
        let id = fp.id;
        store.store(fp).await.expect("Store must succeed");
        // Orphans with low alignment should be candidates
        if *access_count == 0 {
            candidate_ids.push(id);
        }
        println!(
            "  - {} (theta={:.2}, access={}) {}",
            id,
            theta,
            access_count,
            if *access_count == 0 { "CANDIDATE" } else { "KEEP" }
        );
    }

    // DIRECT VERIFICATION: Each candidate ID should exist in store
    println!("\n[DIRECT ROCKSDB VERIFICATION] Confirming candidates exist:");
    for id in candidate_ids.iter() {
        let exists = store.retrieve(*id).await.expect("retrieve works").is_some();
        println!("  - {} exists: {}", id, exists);
        assert!(exists, "Candidate must exist in store");
    }

    // Call handler
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_pruning_candidates",
            "arguments": {
                "min_staleness_days": 0,
                "min_alignment": 0.5,
                "limit": 10
            }
        })),
    );
    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let candidates = data
        .get("candidates")
        .and_then(|v| v.as_array())
        .expect("Should have candidates");

    println!("\n[HANDLER RESULT] {} candidates returned:", candidates.len());
    let mut handler_ids: Vec<uuid::Uuid> = Vec::new();
    for c in candidates {
        let id_str = c.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
        let reason = c.get("reason").and_then(|v| v.as_str()).unwrap_or("?");
        println!("  - {} (reason: {})", id_str, reason);
        if let Ok(id) = uuid::Uuid::parse_str(id_str) {
            handler_ids.push(id);
        }
    }

    // Verify handler returned the expected candidates
    // (handler may return additional candidates due to PruningService logic)
    for expected_id in candidate_ids.iter() {
        // The handler should return orphans
        let found = handler_ids.contains(expected_id);
        println!(
            "\n[VERIFY] Expected candidate {} found in handler result: {}",
            expected_id, found
        );
        assert!(
            found,
            "Expected candidate {} must be in handler result",
            expected_id
        );
    }

    println!("\n[FSV-ROCKSDB-PRUNING-001 PASSED] Pruning candidates match RocksDB state");
    println!("================================================================================\n");
}

/// FSV-CONSOLIDATION-004: Limit parameter is respected.
#[tokio::test]
async fn test_consolidation_limit_respected() {
    println!("\n================================================================================");
    println!("FSV-CONSOLIDATION-004: Limit Parameter Is Respected");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store 10 fingerprints
    println!("[SETUP] Storing 10 fingerprints:");
    for i in 0..10 {
        let content = format!("memory content number {} for testing", i);
        let fp = create_test_fingerprint(&content, 0.7, 5);
        let id = store.store(fp).await.expect("Store must succeed");
        println!("  - [{}] {}", i + 1, id);
    }

    // Request with max_memories = 5 (should only consider 5 memories for pairs)
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "trigger_consolidation",
            "arguments": {
                "strategy": "similarity",
                "min_similarity": 0.5,
                "max_memories": 5
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let statistics = data.get("statistics").expect("Should have statistics");
    let max_memories_limit = statistics
        .get("max_memories_limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("\n[RESULT]");
    println!("  - max_memories_limit in response: {}", max_memories_limit);

    assert_eq!(
        max_memories_limit, 5,
        "max_memories_limit should be 5"
    );

    println!("\n[FSV-CONSOLIDATION-004 PASSED] Limit parameter is respected");
    println!("================================================================================\n");
}
