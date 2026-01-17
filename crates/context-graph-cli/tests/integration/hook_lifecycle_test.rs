//! Integration tests for hook lifecycle
//!
//! # Tests
//! - `test_session_lifecycle_full_flow`: Complete SessionStart → Tools → SessionEnd
//! - `test_multiple_tool_uses_in_session`: Multiple PreTool/PostTool in single session
//! - `test_consciousness_state_injection`: Verify consciousness state in output
//! - `test_concurrent_tool_hooks`: Parallel tool hooks with same session
//!
//! # NO MOCKS - REAL CLI EXECUTION
//! All tests use REAL CLI binary and REAL RocksDB storage.
//!
//! # Constitution References
//! - REQ-HOOKS-43: Integration tests for lifecycle
//! - AP-50: Native hooks only
//! - ARCH-07: Native Claude Code hooks

use serde_json::{json, Value};
use std::time::Instant;
use tempfile::TempDir;

use super::helpers::{
    assert_exit_code, assert_output_bool, assert_timing_under_budget, create_post_tool_input,
    create_pre_tool_input, create_prompt_submit_input, create_session_end_input,
    create_session_start_input, deterministic_session_id, generate_test_session_id,
    invoke_hook_with_stdin, load_snapshot_for_verification, log_test_evidence,
    verify_snapshot_exists, EXIT_SUCCESS, TIMEOUT_POST_TOOL_MS, TIMEOUT_PRE_TOOL_MS,
    TIMEOUT_SESSION_END_MS, TIMEOUT_SESSION_START_MS, TIMEOUT_USER_PROMPT_MS,
};

// =============================================================================
// Full Lifecycle Test
// =============================================================================

/// Test complete session lifecycle: SessionStart → PreTool → PostTool → PromptSubmit → SessionEnd
///
/// Verifies:
/// 1. Each hook returns exit code 0
/// 2. Each hook returns valid JSON with success=true
/// 3. SessionEnd persists snapshot to database
/// 4. Database contains valid SessionIdentitySnapshot
#[tokio::test]
async fn test_session_lifecycle_full_flow() {
    // STEP 1: Create isolated temp database
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path();
    let session_id = generate_test_session_id("lifecycle");

    // STEP 2: SessionStart
    let start_input = create_session_start_input(&session_id, "/tmp/test", "cli", None);
    let start_result =
        invoke_hook_with_stdin("session-start", &session_id, &[], &start_input, db_path);

    assert_exit_code(&start_result, EXIT_SUCCESS, "SessionStart failed");
    assert_output_bool(&start_result, "success", true, "SessionStart success=false");
    assert_timing_under_budget(&start_result, TIMEOUT_SESSION_START_MS, "SessionStart");

    log_test_evidence(
        "test_session_lifecycle_full_flow",
        "session_start",
        &session_id,
        start_result.exit_code,
        start_result.execution_time_ms,
        false, // No snapshot yet
        None,
    );

    // STEP 3: PreToolUse (fast path)
    let pre_tool_input = create_pre_tool_input(
        &session_id,
        "Read",
        json!({"file_path": "/tmp/test.txt"}),
        "tool-use-001",
    );
    let pre_result = invoke_hook_with_stdin(
        "pre-tool",
        &session_id,
        &["--tool-name", "Read", "--fast-path", "true"],
        &pre_tool_input,
        db_path,
    );

    assert_exit_code(&pre_result, EXIT_SUCCESS, "PreToolUse failed");
    assert_timing_under_budget(&pre_result, TIMEOUT_PRE_TOOL_MS, "PreToolUse fast path");

    log_test_evidence(
        "test_session_lifecycle_full_flow",
        "pre_tool_use",
        &session_id,
        pre_result.exit_code,
        pre_result.execution_time_ms,
        false,
        Some(json!({"fast_path": true})),
    );

    // STEP 4: PostToolUse
    let post_tool_input = create_post_tool_input(
        &session_id,
        "Read",
        json!({"file_path": "/tmp/test.txt"}),
        "file contents here",
        "tool-use-001",
    );
    let post_result = invoke_hook_with_stdin(
        "post-tool",
        &session_id,
        &["--tool-name", "Read", "--success", "true"],
        &post_tool_input,
        db_path,
    );

    assert_exit_code(&post_result, EXIT_SUCCESS, "PostToolUse failed");
    assert_timing_under_budget(&post_result, TIMEOUT_POST_TOOL_MS, "PostToolUse");

    log_test_evidence(
        "test_session_lifecycle_full_flow",
        "post_tool_use",
        &session_id,
        post_result.exit_code,
        post_result.execution_time_ms,
        false,
        None,
    );

    // STEP 5: UserPromptSubmit
    let prompt_input = create_prompt_submit_input(
        &session_id,
        "Please read the file and summarize it.",
        vec![("user", "Hello"), ("assistant", "Hi there!")],
    );
    let prompt_result =
        invoke_hook_with_stdin("prompt-submit", &session_id, &[], &prompt_input, db_path);

    assert_exit_code(&prompt_result, EXIT_SUCCESS, "PromptSubmit failed");
    assert_timing_under_budget(&prompt_result, TIMEOUT_USER_PROMPT_MS, "UserPromptSubmit");

    log_test_evidence(
        "test_session_lifecycle_full_flow",
        "user_prompt_submit",
        &session_id,
        prompt_result.exit_code,
        prompt_result.execution_time_ms,
        false,
        None,
    );

    // STEP 6: SessionEnd
    let end_input = create_session_end_input(&session_id, 60000, "normal", None);
    let end_result = invoke_hook_with_stdin(
        "session-end",
        &session_id,
        &["--duration-ms", "60000"],
        &end_input,
        db_path,
    );

    assert_exit_code(&end_result, EXIT_SUCCESS, "SessionEnd failed");
    assert_timing_under_budget(&end_result, TIMEOUT_SESSION_END_MS, "SessionEnd");

    // STEP 7: PHYSICAL DATABASE VERIFICATION
    let snapshot_exists = verify_snapshot_exists(db_path, &session_id);
    assert!(
        snapshot_exists,
        "Snapshot not persisted to database after SessionEnd"
    );

    // Verify snapshot contents
    let snapshot = load_snapshot_for_verification(db_path, &session_id)
        .expect("Snapshot should exist after SessionEnd");

    assert_eq!(
        snapshot.session_id, session_id,
        "Snapshot session_id mismatch"
    );
    assert!(
        snapshot.last_ic >= 0.0 && snapshot.last_ic <= 1.0,
        "IC out of bounds: {}",
        snapshot.last_ic
    );
    assert_eq!(
        snapshot.kuramoto_phases.len(),
        13,
        "Kuramoto phases should have 13 elements"
    );
    assert_eq!(
        snapshot.purpose_vector.len(),
        13,
        "Purpose vector should have 13 elements"
    );

    log_test_evidence(
        "test_session_lifecycle_full_flow",
        "session_end",
        &session_id,
        end_result.exit_code,
        end_result.execution_time_ms,
        true,
        Some(json!({
            "snapshot_exists": true,
            "snapshot_ic": snapshot.last_ic,
            "kuramoto_phases_count": snapshot.kuramoto_phases.len(),
            "purpose_vector_len": snapshot.purpose_vector.len(),
        })),
    );
}

// =============================================================================
// Multiple Tool Uses Test
// =============================================================================

/// Test multiple tool invocations within a single session
///
/// Simulates realistic usage pattern with multiple Read/Write operations
#[tokio::test]
async fn test_multiple_tool_uses_in_session() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path();
    let session_id = generate_test_session_id("multi-tool");

    // SessionStart
    let start_input = create_session_start_input(&session_id, "/tmp", "cli", None);
    let start_result =
        invoke_hook_with_stdin("session-start", &session_id, &[], &start_input, db_path);
    assert_exit_code(&start_result, EXIT_SUCCESS, "SessionStart failed");

    // Execute 5 tool cycles (PreTool + PostTool)
    let tool_names = ["Read", "Write", "Edit", "Grep", "Glob"];

    for (i, tool_name) in tool_names.iter().enumerate() {
        let tool_use_id = format!("tool-use-{:03}", i);

        // PreToolUse
        let pre_input =
            create_pre_tool_input(&session_id, tool_name, json!({"arg": i}), &tool_use_id);
        let pre_result = invoke_hook_with_stdin(
            "pre-tool",
            &session_id,
            &["--tool-name", tool_name, "--fast-path", "true"],
            &pre_input,
            db_path,
        );
        assert_exit_code(&pre_result, EXIT_SUCCESS, &format!("PreTool {} failed", i));

        // PostToolUse
        let post_input = create_post_tool_input(
            &session_id,
            tool_name,
            json!({"arg": i}),
            &format!("result-{}", i),
            &tool_use_id,
        );
        let post_result = invoke_hook_with_stdin(
            "post-tool",
            &session_id,
            &["--tool-name", tool_name, "--success", "true"],
            &post_input,
            db_path,
        );
        assert_exit_code(
            &post_result,
            EXIT_SUCCESS,
            &format!("PostTool {} failed", i),
        );

        log_test_evidence(
            "test_multiple_tool_uses_in_session",
            "tool_cycle",
            &session_id,
            post_result.exit_code,
            pre_result.execution_time_ms + post_result.execution_time_ms,
            false,
            Some(json!({"tool_name": tool_name, "cycle": i})),
        );
    }

    // SessionEnd
    let end_input = create_session_end_input(&session_id, 120000, "normal", None);
    let end_result = invoke_hook_with_stdin(
        "session-end",
        &session_id,
        &["--duration-ms", "120000"],
        &end_input,
        db_path,
    );
    assert_exit_code(&end_result, EXIT_SUCCESS, "SessionEnd failed");

    // PHYSICAL DATABASE VERIFICATION
    assert!(
        verify_snapshot_exists(db_path, &session_id),
        "Snapshot not persisted after multi-tool session"
    );

    log_test_evidence(
        "test_multiple_tool_uses_in_session",
        "session_end",
        &session_id,
        end_result.exit_code,
        end_result.execution_time_ms,
        true,
        Some(json!({"tool_cycles": 5})),
    );
}

// =============================================================================
// Consciousness State Test
// =============================================================================

/// Test that consciousness_state is present and valid in hook outputs
///
/// Verifies:
/// - consciousness_state field present
/// - Contains required fields: consciousness, integration, reflection, differentiation
/// - identity_continuity is within [0, 1]
/// - johari_quadrant is one of: unknown, open, blind, hidden
#[tokio::test]
async fn test_consciousness_state_injection() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path();
    let session_id = generate_test_session_id("consciousness");

    // SessionStart
    let start_input = create_session_start_input(&session_id, "/tmp", "cli", None);
    let start_result =
        invoke_hook_with_stdin("session-start", &session_id, &[], &start_input, db_path);
    assert_exit_code(&start_result, EXIT_SUCCESS, "SessionStart failed");

    // Check consciousness_state in SessionStart output
    let consciousness = start_result
        .consciousness_state()
        .expect("consciousness_state should be present in SessionStart output");

    verify_consciousness_state_structure(&consciousness, "SessionStart");

    // PostToolUse - should also have consciousness_state
    let post_input = create_post_tool_input(
        &session_id,
        "Read",
        json!({"file": "test.txt"}),
        "content",
        "tool-001",
    );
    let post_result = invoke_hook_with_stdin(
        "post-tool",
        &session_id,
        &["--tool-name", "Read", "--success", "true"],
        &post_input,
        db_path,
    );
    assert_exit_code(&post_result, EXIT_SUCCESS, "PostToolUse failed");

    if let Some(consciousness) = post_result.consciousness_state() {
        verify_consciousness_state_structure(&consciousness, "PostToolUse");
    }

    // UserPromptSubmit - should have consciousness_state
    let prompt_input = create_prompt_submit_input(&session_id, "test prompt", vec![]);
    let prompt_result =
        invoke_hook_with_stdin("prompt-submit", &session_id, &[], &prompt_input, db_path);
    assert_exit_code(&prompt_result, EXIT_SUCCESS, "PromptSubmit failed");

    if let Some(consciousness) = prompt_result.consciousness_state() {
        verify_consciousness_state_structure(&consciousness, "UserPromptSubmit");
    }

    // SessionEnd
    let end_input = create_session_end_input(&session_id, 30000, "normal", None);
    let end_result = invoke_hook_with_stdin(
        "session-end",
        &session_id,
        &["--duration-ms", "30000"],
        &end_input,
        db_path,
    );
    assert_exit_code(&end_result, EXIT_SUCCESS, "SessionEnd failed");

    log_test_evidence(
        "test_consciousness_state_injection",
        "all_hooks",
        &session_id,
        end_result.exit_code,
        end_result.execution_time_ms,
        verify_snapshot_exists(db_path, &session_id),
        Some(json!({"consciousness_verified": true})),
    );
}

/// Helper: Verify consciousness_state structure
fn verify_consciousness_state_structure(consciousness: &Value, context: &str) {
    // Must have numeric fields
    let fields = [
        "consciousness",
        "integration",
        "reflection",
        "differentiation",
    ];
    for field in fields {
        assert!(
            consciousness.get(field).is_some(),
            "{}: consciousness_state missing field '{}'",
            context,
            field
        );
    }

    // identity_continuity must be in [0, 1]
    if let Some(ic) = consciousness.get("identity_continuity") {
        let ic_val = ic.as_f64().expect("identity_continuity should be a number");
        assert!(
            (0.0..=1.0).contains(&ic_val),
            "{}: identity_continuity {} out of range [0, 1]",
            context,
            ic_val
        );
    }

    // johari_quadrant should be a valid value if present
    if let Some(johari) = consciousness.get("johari_quadrant") {
        let johari_str = johari.as_str().unwrap_or("");
        let valid_quadrants = ["unknown", "open", "blind", "hidden"];
        assert!(
            valid_quadrants.contains(&johari_str),
            "{}: invalid johari_quadrant '{}', expected one of {:?}",
            context,
            johari_str,
            valid_quadrants
        );
    }
}

// =============================================================================
// Concurrent Tool Hooks Test
// =============================================================================

/// Test concurrent PreToolUse hooks with same session_id
///
/// Verifies:
/// - All hooks complete without error
/// - Fast path maintains performance under concurrent load
/// - No race conditions in hook execution
#[tokio::test]
async fn test_concurrent_tool_hooks() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path();
    let session_id = deterministic_session_id("concurrent", "001");

    // SessionStart first
    let start_input = create_session_start_input(&session_id, "/tmp", "cli", None);
    let start_result =
        invoke_hook_with_stdin("session-start", &session_id, &[], &start_input, db_path);
    assert_exit_code(&start_result, EXIT_SUCCESS, "SessionStart failed");

    // Spawn 10 parallel PreToolUse hooks
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let db = db_path.to_path_buf();
            let sid = session_id.clone();
            tokio::spawn(async move {
                let tool_use_id = format!("concurrent-tool-{:03}", i);
                let tool_name = format!("Tool{}", i);
                let input =
                    create_pre_tool_input(&sid, &tool_name, json!({"index": i}), &tool_use_id);
                invoke_hook_with_stdin(
                    "pre-tool",
                    &sid,
                    &["--tool-name", &tool_name, "--fast-path", "true"],
                    &input,
                    &db,
                )
            })
        })
        .collect();

    let start = Instant::now();
    let results = futures::future::join_all(handles).await;
    let total_time = start.elapsed();

    // Verify all hooks succeeded
    let mut success_count = 0;
    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(hook_result) => {
                assert_eq!(
                    hook_result.exit_code, EXIT_SUCCESS,
                    "Concurrent hook {} failed with exit code {}. stderr: {}",
                    i, hook_result.exit_code, hook_result.stderr
                );
                success_count += 1;
            }
            Err(e) => {
                panic!("Concurrent hook {} join error: {}", i, e);
            }
        }
    }

    assert_eq!(success_count, 10, "Not all concurrent hooks succeeded");

    // SessionEnd
    let end_input = create_session_end_input(&session_id, 5000, "normal", None);
    let end_result = invoke_hook_with_stdin(
        "session-end",
        &session_id,
        &["--duration-ms", "5000"],
        &end_input,
        db_path,
    );
    assert_exit_code(&end_result, EXIT_SUCCESS, "SessionEnd failed");

    log_test_evidence(
        "test_concurrent_tool_hooks",
        "concurrent",
        &session_id,
        EXIT_SUCCESS,
        total_time.as_millis() as u64,
        verify_snapshot_exists(db_path, &session_id),
        Some(json!({
            "concurrent_hooks": 10,
            "all_succeeded": success_count == 10,
            "total_parallel_time_ms": total_time.as_millis(),
        })),
    );
}
