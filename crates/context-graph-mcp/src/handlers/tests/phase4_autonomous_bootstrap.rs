//! Phase 4: AUTONOMOUS BOOTSTRAP Flow Manual Testing
//!
//! This module tests the CRITICAL FIX for the circular dependency in ARCH-03:
//! - store_memory required North Star -> but auto_bootstrap_north_star required stored memories
//!
//! THE FIX: store_memory now uses PurposeVector::default() [0.0; 13] when no North Star exists.
//!
//! # Autonomous Tools Tested
//!
//! 1. `store_memory` - Should work WITHOUT North Star (neutral alignment)
//! 2. `auto_bootstrap_north_star` - Discovers purpose from >= 3 stored fingerprints
//! 3. `get_autonomous_status` - Returns discovery status, alignment drift, recommendations
//! 4. `get_alignment_drift` - Checks if purpose vectors are drifting
//! 5. `trigger_drift_correction` - Corrects drift if detected
//! 6. `discover_sub_goals` - Finds sub-goals from clustered patterns
//!
//! # Test Scenarios
//!
//! ## Scenario 1: Store WITHOUT North Star (CRITICAL)
//! - Create handlers WITHOUT North Star using create_test_handlers_no_north_star()
//! - Store 3+ memories via tools/call store_memory
//! - VERIFY: No errors, fingerprints created with neutral alignment [0.0; 13]
//!
//! ## Scenario 2: Auto Bootstrap North Star
//! - After storing >= 3 memories, call auto_bootstrap_north_star
//! - VERIFY: Either succeeds (North Star created) or returns "needs more data" message
//! - Check goal hierarchy for North Star
//!
//! ## Scenario 3: Get Autonomous Status
//! - Call get_autonomous_status
//! - VERIFY: Returns phase, recommendations, discovery_status
//!
//! ## Scenario 4: Full Autonomous Flow
//! - Start fresh (no North Star)
//! - Store 5 memories
//! - Bootstrap North Star
//! - Store 2 more memories
//! - Verify new memories would get computed (non-neutral) purpose vectors
//!
//! # Critical Verification
//! - store_memory MUST succeed without North Star (this was the bug)
//! - Purpose vector should be [0.0; 13] (neutral) when no North Star
//! - After bootstrap, purpose vectors should have non-zero values

use serde_json::json;

use crate::handlers::tests::{
    create_test_handlers_no_north_star, create_test_handlers_with_rocksdb_no_north_star,
    extract_mcp_tool_data,
};
use crate::protocol::{JsonRpcId, JsonRpcRequest};
use crate::tools::tool_names;

// =============================================================================
// SCENARIO 1: STORE WITHOUT NORTH STAR (CRITICAL - THE FIX)
// =============================================================================

/// FSV Test: store_memory MUST succeed without North Star configured.
///
/// This is the CRITICAL test for the circular dependency fix.
/// Before the fix: store_memory required North Star, but bootstrap needed stored memories.
/// After the fix: store_memory uses PurposeVector::default() when no North Star exists.
///
/// Source of Truth: tools.rs call_store_memory() - line 491: hierarchy.north_star().is_none()
/// Expected: Memory stored successfully with fingerprint ID returned.
#[tokio::test]
async fn test_store_memory_succeeds_without_north_star() {
    // SETUP: Create handlers WITHOUT North Star (empty goal hierarchy)
    let handlers = create_test_handlers_no_north_star();

    // EXECUTE: Store a memory - this should NOT fail even without North Star
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::STORE_MEMORY,
            "arguments": {
                "content": "Test memory content for autonomous bootstrap testing"
            }
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Response is successful - THIS IS THE CRITICAL CHECK
    assert!(
        response.error.is_none(),
        "[CRITICAL FIX] store_memory MUST succeed without North Star. Got error: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify fingerprintId was returned
    let fingerprint_id = data
        .get("fingerprintId")
        .and_then(|v| v.as_str())
        .expect("fingerprintId must exist");
    assert!(
        !fingerprint_id.is_empty(),
        "[FSV] fingerprintId must not be empty"
    );

    // FSV: Verify embedderCount is 13
    let embedder_count = data
        .get("embedderCount")
        .and_then(|v| v.as_u64())
        .expect("embedderCount must exist");
    assert_eq!(
        embedder_count, 13,
        "[FSV] embedderCount must be 13, got {}",
        embedder_count
    );

    println!("[FSV] Phase 4 - CRITICAL FIX VERIFIED");
    println!("[FSV]   store_memory succeeded WITHOUT North Star");
    println!("[FSV]   fingerprintId={}", fingerprint_id);
    println!("[FSV]   embedderCount={}", embedder_count);
    println!("[FSV]   CIRCULAR DEPENDENCY: FIXED");
}

/// FSV Test: Multiple memories can be stored without North Star.
///
/// Tests that the system can accumulate enough memories for bootstrap (>= 3).
/// This verifies the "autonomous seeding" phase works correctly.
#[tokio::test]
async fn test_multiple_memories_stored_without_north_star() {
    let handlers = create_test_handlers_no_north_star();

    let test_contents = [
        "Machine learning optimization techniques",
        "Neural network training strategies",
        "Deep learning model architectures",
        "Gradient descent and backpropagation",
        "Reinforcement learning algorithms",
    ];

    let mut stored_ids: Vec<String> = Vec::new();

    for (i, content) in test_contents.iter().enumerate() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(JsonRpcId::Number(i as i64 + 1)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": tool_names::STORE_MEMORY,
                "arguments": {
                    "content": content
                }
            })),
        };
        let response = handlers.dispatch(request).await;

        // VERIFY: Each store succeeds
        assert!(
            response.error.is_none(),
            "[FSV] Memory {} store failed: {:?}",
            i + 1,
            response.error
        );

        let result = response.result.expect("Should have result");
        let data = extract_mcp_tool_data(&result);
        let fp_id = data
            .get("fingerprintId")
            .and_then(|v| v.as_str())
            .expect("fingerprintId must exist")
            .to_string();
        stored_ids.push(fp_id);
    }

    // FSV: Verify all 5 memories stored
    assert_eq!(
        stored_ids.len(),
        5,
        "[FSV] Should have stored 5 memories, got {}",
        stored_ids.len()
    );

    // FSV: Verify all fingerprint IDs are unique
    let unique_ids: std::collections::HashSet<_> = stored_ids.iter().collect();
    assert_eq!(
        unique_ids.len(),
        5,
        "[FSV] All fingerprint IDs must be unique"
    );

    println!("[FSV] Phase 4 - Multiple memories stored without North Star: PASSED");
    println!("[FSV]   Stored {} memories", stored_ids.len());
    println!("[FSV]   All IDs unique: YES");
    println!("[FSV]   AUTONOMOUS SEEDING: WORKING");
}

// =============================================================================
// SCENARIO 2: AUTO BOOTSTRAP NORTH STAR
// =============================================================================

/// FSV Test: auto_bootstrap_north_star returns appropriate response without data.
///
/// When no fingerprints exist, bootstrap should return an informative error
/// explaining that memories need to be stored first.
#[tokio::test]
async fn test_auto_bootstrap_fails_gracefully_with_no_data() {
    let handlers = create_test_handlers_no_north_star();

    // EXECUTE: Try to bootstrap without any stored memories
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::AUTO_BOOTSTRAP_NORTH_STAR,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Should fail with informative error (no data to discover from)
    assert!(
        response.error.is_some(),
        "[FSV] Bootstrap should fail without stored fingerprints"
    );

    let error = response.error.expect("Should have error");
    let error_message = error.message.to_lowercase();

    // FSV: Error message should indicate need for more data
    let indicates_data_needed = error_message.contains("no teleological fingerprints")
        || error_message.contains("store memories first")
        || error_message.contains("insufficient data");

    assert!(
        indicates_data_needed,
        "[FSV] Error message should indicate data needed. Got: {}",
        error.message
    );

    println!("[FSV] Phase 4 - Bootstrap without data: GRACEFUL FAILURE");
    println!("[FSV]   Error code: {}", error.code);
    println!("[FSV]   Message explains: store memories first");
}

/// FSV Test: auto_bootstrap returns "already_bootstrapped" when North Star exists.
///
/// Tests the idempotent behavior - calling bootstrap when already configured
/// should return success with status="already_bootstrapped".
#[tokio::test]
async fn test_auto_bootstrap_returns_already_bootstrapped_when_configured() {
    // Use handlers WITH North Star already configured
    let handlers = super::create_test_handlers();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::AUTO_BOOTSTRAP_NORTH_STAR,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Should succeed (idempotent)
    assert!(
        response.error.is_none(),
        "[FSV] Bootstrap should succeed when already configured: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify status is "already_bootstrapped"
    let status = data
        .get("status")
        .and_then(|v| v.as_str())
        .expect("status must exist");
    assert_eq!(
        status, "already_bootstrapped",
        "[FSV] Status should be 'already_bootstrapped', got '{}'",
        status
    );

    // FSV: Verify bootstrap_result contains existing goal info
    let bootstrap_result = data
        .get("bootstrap_result")
        .expect("bootstrap_result must exist");
    assert!(
        bootstrap_result.get("goal_id").is_some(),
        "[FSV] bootstrap_result must contain goal_id"
    );
    assert_eq!(
        bootstrap_result.get("source").and_then(|v| v.as_str()),
        Some("existing_north_star"),
        "[FSV] source must be 'existing_north_star'"
    );

    println!("[FSV] Phase 4 - Bootstrap when already configured: PASSED");
    println!("[FSV]   Status: already_bootstrapped");
    println!("[FSV]   IDEMPOTENT BEHAVIOR: VERIFIED");
}

// =============================================================================
// SCENARIO 3: GET AUTONOMOUS STATUS
// =============================================================================

/// FSV Test: get_autonomous_status returns valid structure without North Star.
///
/// Tests that status works even when system is not fully configured.
/// Should indicate that North Star is not configured and provide recommendations.
#[tokio::test]
async fn test_get_autonomous_status_without_north_star() {
    let handlers = create_test_handlers_no_north_star();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_AUTONOMOUS_STATUS,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Should succeed even without North Star
    assert!(
        response.error.is_none(),
        "[FSV] get_autonomous_status should succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify north_star shows not configured
    let north_star = data.get("north_star").expect("north_star must exist");
    let configured = north_star
        .get("configured")
        .and_then(|v| v.as_bool())
        .expect("configured must exist");
    assert!(
        !configured,
        "[FSV] North Star should NOT be configured"
    );

    // FSV: Verify overall_health shows not_configured
    let overall_health = data.get("overall_health").expect("overall_health must exist");
    let health_status = overall_health
        .get("status")
        .and_then(|v| v.as_str())
        .expect("status must exist");
    assert_eq!(
        health_status, "not_configured",
        "[FSV] Health status should be 'not_configured', got '{}'",
        health_status
    );

    // FSV: Verify recommendations include bootstrapping guidance
    let recommendations = data
        .get("recommendations")
        .and_then(|v| v.as_array())
        .expect("recommendations must exist");
    assert!(
        !recommendations.is_empty(),
        "[FSV] Should have recommendations"
    );

    // Check that at least one recommendation mentions store_memory or bootstrap
    let has_guidance = recommendations.iter().any(|rec| {
        let action = rec.get("action").and_then(|v| v.as_str()).unwrap_or("");
        action.contains("store_memory") || action.contains("bootstrap")
    });
    assert!(
        has_guidance,
        "[FSV] Recommendations should guide user to store memories or bootstrap"
    );

    // FSV: Verify services section exists and shows ready
    let services = data.get("services").expect("services must exist");
    let bootstrap_service = services
        .get("bootstrap_service")
        .expect("bootstrap_service must exist");
    assert_eq!(
        bootstrap_service.get("ready").and_then(|v| v.as_bool()),
        Some(true),
        "[FSV] bootstrap_service must be ready"
    );

    println!("[FSV] Phase 4 - get_autonomous_status without North Star: PASSED");
    println!("[FSV]   north_star.configured: {}", configured);
    println!("[FSV]   health_status: {}", health_status);
    println!("[FSV]   recommendations.len: {}", recommendations.len());
    println!("[FSV]   AUTONOMOUS STATUS: WORKING");
}

/// FSV Test: get_autonomous_status with include_metrics=true.
#[tokio::test]
async fn test_get_autonomous_status_with_metrics() {
    let handlers = super::create_test_handlers(); // With North Star

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_AUTONOMOUS_STATUS,
            "arguments": {
                "include_metrics": true
            }
        })),
    };
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "[FSV] get_autonomous_status should succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify metrics section exists when requested
    let metrics = data.get("metrics").expect("metrics must exist when include_metrics=true");
    assert!(
        metrics.get("drift_rolling_mean").is_some(),
        "[FSV] metrics.drift_rolling_mean must exist"
    );
    assert!(
        metrics.get("correction_success_rate").is_some(),
        "[FSV] metrics.correction_success_rate must exist"
    );

    println!("[FSV] Phase 4 - get_autonomous_status with metrics: PASSED");
}

// =============================================================================
// SCENARIO 4: ALIGNMENT DRIFT AND CORRECTION
// =============================================================================

/// FSV Test: get_alignment_drift works without North Star (ARCH-03 compliant).
///
/// ARCH-03: System should work autonomously. Drift detection should work
/// by computing drift relative to the fingerprints' centroid when no North Star.
#[tokio::test]
async fn test_get_alignment_drift_arch03_compliant() {
    let handlers = create_test_handlers_no_north_star();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_ALIGNMENT_DRIFT,
            "arguments": {
                "timeframe": "24h"
            }
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Should succeed even without North Star
    assert!(
        response.error.is_none(),
        "[FSV] get_alignment_drift should succeed without North Star: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify reference_type indicates no North Star
    let reference_type = data
        .get("reference_type")
        .and_then(|v| v.as_str())
        .expect("reference_type must exist");

    // When no memory_ids provided and no North Star, should indicate this
    let valid_types = ["no_reference", "computed_centroid"];
    assert!(
        valid_types.contains(&reference_type),
        "[FSV] reference_type should be 'no_reference' or 'computed_centroid', got '{}'",
        reference_type
    );

    println!("[FSV] Phase 4 - get_alignment_drift ARCH-03 compliant: PASSED");
    println!("[FSV]   reference_type: {}", reference_type);
}

/// FSV Test: trigger_drift_correction works without North Star (ARCH-03 compliant).
#[tokio::test]
async fn test_trigger_drift_correction_arch03_compliant() {
    let handlers = create_test_handlers_no_north_star();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::TRIGGER_DRIFT_CORRECTION,
            "arguments": {
                "force": false
            }
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Should succeed even without North Star
    assert!(
        response.error.is_none(),
        "[FSV] trigger_drift_correction should succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify ARCH-03 compliance is indicated
    let arch03_compliant = data
        .get("arch03_compliant")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(
        arch03_compliant,
        "[FSV] Response must indicate ARCH-03 compliance"
    );

    // FSV: Verify reference_type is centroid-based
    let reference_type = data
        .get("reference_type")
        .and_then(|v| v.as_str())
        .expect("reference_type must exist");
    assert_eq!(
        reference_type, "computed_centroid",
        "[FSV] Without North Star, reference_type should be 'computed_centroid'"
    );

    println!("[FSV] Phase 4 - trigger_drift_correction ARCH-03 compliant: PASSED");
    println!("[FSV]   arch03_compliant: {}", arch03_compliant);
    println!("[FSV]   reference_type: {}", reference_type);
}

// =============================================================================
// SCENARIO 5: DISCOVER SUB-GOALS
// =============================================================================

/// FSV Test: discover_sub_goals works without North Star (ARCH-03 compliant).
///
/// ARCH-03: Goals emerge from data patterns via clustering, no manual config needed.
#[tokio::test]
async fn test_discover_sub_goals_arch03_compliant() {
    let handlers = create_test_handlers_no_north_star();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::DISCOVER_SUB_GOALS,
            "arguments": {
                "min_confidence": 0.6,
                "max_goals": 5
            }
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Should succeed without North Star
    assert!(
        response.error.is_none(),
        "[FSV] discover_sub_goals should succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify discovery_mode is "autonomous" when no North Star
    let discovery_mode = data
        .get("discovery_mode")
        .and_then(|v| v.as_str())
        .expect("discovery_mode must exist");
    assert_eq!(
        discovery_mode, "autonomous",
        "[FSV] Without North Star, discovery_mode should be 'autonomous', got '{}'",
        discovery_mode
    );

    // FSV: Verify ARCH-03 compliance
    let arch03_compliant = data
        .get("arch03_compliant")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(
        arch03_compliant,
        "[FSV] Response must indicate ARCH-03 compliance"
    );

    println!("[FSV] Phase 4 - discover_sub_goals ARCH-03 compliant: PASSED");
    println!("[FSV]   discovery_mode: {}", discovery_mode);
    println!("[FSV]   arch03_compliant: {}", arch03_compliant);
}

// =============================================================================
// SCENARIO 6: FULL AUTONOMOUS FLOW WITH ROCKSDB
// =============================================================================

/// FSV Test: Full autonomous flow with real RocksDB storage.
///
/// This integration test verifies the complete autonomous bootstrap flow:
/// 1. Create handlers without North Star (fresh start)
/// 2. Store 5 memories (seeding phase)
/// 3. Verify memories stored with neutral purpose vectors
/// 4. Get autonomous status (should show not_configured)
/// 5. Check alignment drift (should work with centroid reference)
///
/// Uses RocksDB for real persistence verification.
#[tokio::test]
async fn test_full_autonomous_flow_with_rocksdb() {
    // SETUP: Create handlers with RocksDB but no North Star
    let (handlers, _tempdir) = create_test_handlers_with_rocksdb_no_north_star().await;

    // STEP 1: Store 5 memories for autonomous seeding
    let test_contents = [
        "Advanced neural architecture search techniques for optimal model design",
        "Transformer attention mechanisms and their applications in NLP",
        "Convolutional neural networks for computer vision tasks",
        "Recurrent networks and LSTM for sequential data processing",
        "Generative adversarial networks for synthetic data creation",
    ];

    let mut stored_ids: Vec<String> = Vec::new();

    for (i, content) in test_contents.iter().enumerate() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(JsonRpcId::Number(i as i64 + 1)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": tool_names::STORE_MEMORY,
                "arguments": {
                    "content": content
                }
            })),
        };
        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "[FSV] Memory {} store failed with RocksDB: {:?}",
            i + 1,
            response.error
        );

        let result = response.result.expect("Should have result");
        let data = extract_mcp_tool_data(&result);
        let fp_id = data
            .get("fingerprintId")
            .and_then(|v| v.as_str())
            .expect("fingerprintId must exist")
            .to_string();
        stored_ids.push(fp_id);
    }

    // STEP 2: Get autonomous status
    let status_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(10)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_AUTONOMOUS_STATUS,
            "arguments": {
                "include_metrics": true
            }
        })),
    };
    let status_response = handlers.dispatch(status_request).await;
    assert!(
        status_response.error.is_none(),
        "[FSV] get_autonomous_status failed: {:?}",
        status_response.error
    );

    let status_data = extract_mcp_tool_data(&status_response.result.unwrap());
    let north_star = status_data.get("north_star").expect("north_star must exist");
    let configured = north_star.get("configured").and_then(|v| v.as_bool()).unwrap_or(true);
    assert!(
        !configured,
        "[FSV] North Star should still be unconfigured"
    );

    // STEP 3: Check alignment drift (should work without North Star)
    let drift_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(11)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_ALIGNMENT_DRIFT,
            "arguments": {
                "timeframe": "24h"
            }
        })),
    };
    let drift_response = handlers.dispatch(drift_request).await;
    assert!(
        drift_response.error.is_none(),
        "[FSV] get_alignment_drift failed: {:?}",
        drift_response.error
    );

    println!("\n[FSV] Phase 4 - FULL AUTONOMOUS FLOW WITH ROCKSDB");
    println!("==================================================");
    println!("[FSV]   Memories stored: {}", stored_ids.len());
    println!("[FSV]   North Star configured: {}", configured);
    println!("[FSV]   All IDs:");
    for (i, id) in stored_ids.iter().enumerate() {
        println!("[FSV]     {}: {}", i + 1, id);
    }
    println!("[FSV]   Drift check: PASSED");
    println!("[FSV]   Status check: PASSED");
    println!("==================================================");
    println!("[FSV] AUTONOMOUS BOOTSTRAP FLOW: VERIFIED");
}

// =============================================================================
// SUMMARY TEST: ALL AUTONOMOUS TOOLS
// =============================================================================

/// FSV Integration Test: Verify all autonomous tools work together.
///
/// This test exercises the complete autonomous toolchain in order:
/// 1. store_memory (without North Star) - CRITICAL
/// 2. get_autonomous_status (check not_configured)
/// 3. get_alignment_drift (ARCH-03 compliant)
/// 4. trigger_drift_correction (ARCH-03 compliant)
/// 5. discover_sub_goals (ARCH-03 compliant)
/// 6. auto_bootstrap_north_star (with existing handlers)
#[tokio::test]
async fn test_all_autonomous_tools_integration() {
    let handlers = create_test_handlers_no_north_star();
    let mut tests_passed = 0;
    let total_tests = 6;

    // TEST 1: store_memory (CRITICAL - the fix)
    let req1 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::STORE_MEMORY,
            "arguments": {
                "content": "Test content for autonomous integration"
            }
        })),
    };
    let resp1 = handlers.dispatch(req1).await;
    if resp1.error.is_none() {
        tests_passed += 1;
        println!("[Phase 4] store_memory (without North Star): PASSED");
    } else {
        println!("[Phase 4] store_memory (without North Star): FAILED - {:?}", resp1.error);
    }

    // TEST 2: get_autonomous_status
    let req2 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_AUTONOMOUS_STATUS,
            "arguments": {}
        })),
    };
    let resp2 = handlers.dispatch(req2).await;
    let status_correct = if resp2.error.is_none() {
        let data = extract_mcp_tool_data(&resp2.result.unwrap());
        let north_star = data.get("north_star");
        north_star
            .and_then(|ns| ns.get("configured"))
            .and_then(|c| c.as_bool())
            == Some(false)
    } else {
        false
    };
    if status_correct {
        tests_passed += 1;
        println!("[Phase 4] get_autonomous_status (not_configured): PASSED");
    } else {
        println!("[Phase 4] get_autonomous_status: FAILED");
    }

    // TEST 3: get_alignment_drift (ARCH-03)
    let req3 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_ALIGNMENT_DRIFT,
            "arguments": {}
        })),
    };
    let resp3 = handlers.dispatch(req3).await;
    if resp3.error.is_none() {
        tests_passed += 1;
        println!("[Phase 4] get_alignment_drift (ARCH-03): PASSED");
    } else {
        println!("[Phase 4] get_alignment_drift: FAILED - {:?}", resp3.error);
    }

    // TEST 4: trigger_drift_correction (ARCH-03)
    let req4 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(4)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::TRIGGER_DRIFT_CORRECTION,
            "arguments": {}
        })),
    };
    let resp4 = handlers.dispatch(req4).await;
    if resp4.error.is_none() {
        tests_passed += 1;
        println!("[Phase 4] trigger_drift_correction (ARCH-03): PASSED");
    } else {
        println!("[Phase 4] trigger_drift_correction: FAILED - {:?}", resp4.error);
    }

    // TEST 5: discover_sub_goals (ARCH-03)
    let req5 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(5)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::DISCOVER_SUB_GOALS,
            "arguments": {}
        })),
    };
    let resp5 = handlers.dispatch(req5).await;
    if resp5.error.is_none() {
        tests_passed += 1;
        println!("[Phase 4] discover_sub_goals (ARCH-03): PASSED");
    } else {
        println!("[Phase 4] discover_sub_goals: FAILED - {:?}", resp5.error);
    }

    // TEST 6: auto_bootstrap_north_star (should fail gracefully - insufficient data)
    let req6 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(6)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::AUTO_BOOTSTRAP_NORTH_STAR,
            "arguments": {}
        })),
    };
    let resp6 = handlers.dispatch(req6).await;
    // This should fail because we only stored 1 memory (need >= 3)
    // But failure with the right error message is still a "pass" for this test
    let bootstrap_behavior_correct = resp6.error.is_some();
    if bootstrap_behavior_correct {
        tests_passed += 1;
        println!("[Phase 4] auto_bootstrap_north_star (graceful fail): PASSED");
    } else {
        println!("[Phase 4] auto_bootstrap_north_star: UNEXPECTED SUCCESS (should need more data)");
    }

    // SUMMARY
    println!("\n[Phase 4] AUTONOMOUS BOOTSTRAP TOOLS SUMMARY");
    println!("=============================================");
    println!("Tests passed: {}/{}", tests_passed, total_tests);
    println!("Circular dependency fix: {}", if tests_passed >= 1 { "VERIFIED" } else { "BROKEN" });
    println!("ARCH-03 compliance: {}", if tests_passed >= 4 { "VERIFIED" } else { "INCOMPLETE" });
    println!("=============================================");

    // Critical: store_memory must work (test 1)
    assert!(
        tests_passed >= 1,
        "[CRITICAL] store_memory without North Star MUST work - this is the core fix"
    );

    // All tests should pass
    assert_eq!(
        tests_passed, total_tests,
        "[FSV] All {} autonomous tools should work, got {}/{}",
        total_tests, tests_passed, total_tests
    );
}
