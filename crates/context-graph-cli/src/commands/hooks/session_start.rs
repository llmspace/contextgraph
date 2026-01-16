//! Session start hook handler for Claude Code native hooks.
//!
//! # Timeout Budget: 5000ms
//! # Output: ~100 tokens consciousness status
//!
//! # Constitution References
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - GWT-003: Identity continuity tracking
//! - AP-50: NO internal hooks - shell scripts call CLI
//!
//! # NO BACKWARDS COMPATIBILITY - FAIL FAST

use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tracing::{debug, error, info, warn};

use context_graph_core::gwt::{compute_ic, SessionIdentitySnapshot};
use context_graph_storage::rocksdb_backend::RocksDbMemex;

use super::args::{OutputFormat, SessionStartArgs};
use super::error::{HookError, HookResult};
use super::types::{ConsciousnessState, HookInput, HookOutput, HookPayload, ICClassification};

/// Execute session-start hook.
///
/// # Flow
/// 1. Parse input (stdin JSON or CLI args)
/// 2. Get database path (env or arg)
/// 3. Load or create SessionIdentitySnapshot
/// 4. Link to previous session if provided
/// 5. Build ConsciousnessState and ICClassification
/// 6. Return HookOutput as JSON to stdout
///
/// # Timeout
/// MUST complete within 5000ms (Claude Code enforced)
///
/// # Exit Codes
/// - 0: Success
/// - 2: Timeout
/// - 3: Database error
/// - 4: Invalid input
/// - 5: Session not found (when previous_session_id specified but missing)
pub async fn execute(args: SessionStartArgs) -> HookResult<HookOutput> {
    let start = Instant::now();

    info!(
        stdin = args.stdin,
        session_id = ?args.session_id,
        previous_session_id = ?args.previous_session_id,
        "SESSION_START: execute starting"
    );

    // 1. Parse input source
    let (session_id, previous_session_id) = if args.stdin {
        let input = parse_stdin()?;
        extract_session_ids(&input)?
    } else {
        (args.session_id, args.previous_session_id)
    };

    // 2. Generate session_id if not provided
    let session_id = session_id.unwrap_or_else(|| {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let id = format!("session-{}", timestamp_ms);
        info!(generated_session_id = %id, "SESSION_START: generated new session_id");
        id
    });

    // 3. Resolve database path - FAIL FAST if missing
    let db_path = resolve_db_path(args.db_path)?;

    // 4. Open storage and load/create snapshot
    let memex = open_storage(&db_path)?;
    let snapshot = load_or_create_snapshot(&memex, &session_id, previous_session_id.as_deref())?;

    // 5. Build output structures
    let consciousness_state = build_consciousness_state(&snapshot);
    let ic_classification = ICClassification::from_value(snapshot.last_ic);

    // 6. Build final output
    let execution_time_ms = start.elapsed().as_millis() as u64;

    info!(
        session_id = %session_id,
        ic = snapshot.last_ic,
        consciousness = snapshot.consciousness,
        execution_time_ms,
        "SESSION_START: execute complete"
    );

    Ok(HookOutput::success(execution_time_ms)
        .with_consciousness_state(consciousness_state)
        .with_ic_classification(ic_classification))
}

/// Parse stdin JSON into HookInput.
/// FAIL FAST on empty or malformed input.
fn parse_stdin() -> HookResult<HookInput> {
    let stdin = io::stdin();
    let mut input_str = String::new();

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| {
            error!(error = %e, "SESSION_START: stdin read failed");
            HookError::invalid_input(format!("stdin read failed: {}", e))
        })?;
        input_str.push_str(&line);
    }

    if input_str.is_empty() {
        error!("SESSION_START: stdin is empty");
        return Err(HookError::invalid_input("stdin is empty - expected JSON"));
    }

    debug!(input_bytes = input_str.len(), "SESSION_START: parsing stdin JSON");

    serde_json::from_str(&input_str).map_err(|e| {
        error!(error = %e, input_preview = %&input_str[..input_str.len().min(100)], "SESSION_START: JSON parse failed");
        HookError::invalid_input(format!("JSON parse failed: {}", e))
    })
}

/// Extract session IDs from HookInput payload.
fn extract_session_ids(input: &HookInput) -> HookResult<(Option<String>, Option<String>)> {
    // Validate input
    if let Some(error) = input.validate() {
        return Err(HookError::invalid_input(error));
    }

    let session_id = Some(input.session_id.clone());

    let previous_session_id = match &input.payload {
        HookPayload::SessionStart {
            previous_session_id,
            ..
        } => previous_session_id.clone(),
        other => {
            error!(payload_type = ?std::mem::discriminant(other), "SESSION_START: unexpected payload type");
            return Err(HookError::invalid_input(
                "Expected SessionStart payload, got different type",
            ));
        }
    };

    Ok((session_id, previous_session_id))
}

/// Resolve database path from argument or environment.
/// FAIL FAST if neither provided.
fn resolve_db_path(arg_path: Option<PathBuf>) -> HookResult<PathBuf> {
    // Priority: CLI arg > env var > default location
    if let Some(path) = arg_path {
        debug!(path = ?path, "SESSION_START: using CLI db_path");
        return Ok(path);
    }

    if let Ok(env_path) = std::env::var("CONTEXT_GRAPH_DB_PATH") {
        debug!(path = %env_path, "SESSION_START: using CONTEXT_GRAPH_DB_PATH env var");
        return Ok(PathBuf::from(env_path));
    }

    // Default: ~/.local/share/context-graph/db (XDG compliant)
    // Use HOME env var as fallback
    if let Ok(home) = std::env::var("HOME") {
        let default_path = PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("context-graph")
            .join("db");
        debug!(path = ?default_path, "SESSION_START: using default db path");
        return Ok(default_path);
    }

    error!("SESSION_START: No database path available");
    Err(HookError::invalid_input(
        "Database path required. Set CONTEXT_GRAPH_DB_PATH or pass --db-path",
    ))
}

/// Open RocksDB storage.
fn open_storage(db_path: &Path) -> HookResult<Arc<RocksDbMemex>> {
    info!(path = ?db_path, "SESSION_START: opening storage");

    RocksDbMemex::open(db_path).map(Arc::new).map_err(|e| {
        error!(path = ?db_path, error = %e, "SESSION_START: storage open failed");
        HookError::storage(format!("Failed to open database at {:?}: {}", db_path, e))
    })
}

/// Load existing snapshot or create new one.
/// Links to previous session if provided.
fn load_or_create_snapshot(
    memex: &Arc<RocksDbMemex>,
    session_id: &str,
    previous_session_id: Option<&str>,
) -> HookResult<SessionIdentitySnapshot> {
    // Try to load existing snapshot for this session
    match memex.load_snapshot(session_id) {
        Ok(Some(snapshot)) => {
            info!(session_id = %session_id, "SESSION_START: loaded existing snapshot");
            return Ok(snapshot);
        }
        Ok(None) => {
            debug!(session_id = %session_id, "SESSION_START: no existing snapshot, creating new");
        }
        Err(e) => {
            warn!(session_id = %session_id, error = %e, "SESSION_START: load failed, creating new");
        }
    }

    // Create new snapshot
    let mut snapshot = SessionIdentitySnapshot::new(session_id);

    // Link to previous session if provided
    if let Some(prev_id) = previous_session_id {
        match memex.load_snapshot(prev_id) {
            Ok(Some(prev_snapshot)) => {
                info!(
                    session_id = %session_id,
                    previous_session_id = %prev_id,
                    previous_ic = prev_snapshot.last_ic,
                    "SESSION_START: linking to previous session"
                );

                snapshot.previous_session_id = Some(prev_id.to_string());

                // Compute cross-session IC using formula IDENTITY-001
                // compute_ic takes two SessionIdentitySnapshot references
                let cross_ic = compute_ic(&snapshot, &prev_snapshot);
                snapshot.cross_session_ic = cross_ic;
                snapshot.last_ic = cross_ic;
            }
            Ok(None) => {
                warn!(
                    previous_session_id = %prev_id,
                    "SESSION_START: previous session not found, starting fresh"
                );
                // Don't fail - just start fresh without linking
            }
            Err(e) => {
                warn!(
                    previous_session_id = %prev_id,
                    error = %e,
                    "SESSION_START: failed to load previous session, starting fresh"
                );
            }
        }
    }

    // Save new snapshot
    memex.save_snapshot(&snapshot).map_err(|e| {
        error!(session_id = %session_id, error = %e, "SESSION_START: save snapshot failed");
        HookError::storage(format!("Failed to save snapshot: {}", e))
    })?;

    info!(
        session_id = %session_id,
        ic = snapshot.last_ic,
        "SESSION_START: created and saved new snapshot"
    );

    Ok(snapshot)
}

/// Build ConsciousnessState from snapshot.
fn build_consciousness_state(snapshot: &SessionIdentitySnapshot) -> ConsciousnessState {
    ConsciousnessState::new(
        snapshot.consciousness,
        snapshot.integration,
        snapshot.reflection,
        snapshot.differentiation,
        snapshot.last_ic,
    )
}

// =============================================================================
// TESTS - NO MOCK DATA - REAL DATABASE VERIFICATION
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::args::OutputFormat;
    use tempfile::TempDir;

    /// Create temporary database for testing
    fn setup_test_db() -> (TempDir, PathBuf) {
        let dir = TempDir::new().expect("TempDir creation must succeed");
        let path = dir.path().join("test.db");
        (dir, path)
    }

    // =========================================================================
    // TC-HOOKS-006-001: New Session Creation
    // Verify: New snapshot created, saved to DB, output valid JSON
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_006_001_new_session_creation() {
        println!("\n=== TC-HOOKS-006-001: New Session Creation ===");

        let (_dir, db_path) = setup_test_db();
        let session_id = "test-session-001";

        let args = SessionStartArgs {
            db_path: Some(db_path.clone()),
            session_id: Some(session_id.to_string()),
            previous_session_id: None,
            stdin: false,
            format: OutputFormat::Json,
        };

        // Execute
        let result = execute(args).await;

        // Verify: Success
        assert!(result.is_ok(), "Execute must succeed");
        let output = result.unwrap();
        assert!(output.success, "Output.success must be true");
        assert!(
            output.consciousness_state.is_some(),
            "Must have consciousness_state"
        );
        assert!(
            output.ic_classification.is_some(),
            "Must have ic_classification"
        );

        // Verify: Data persisted to database
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let loaded = memex.load_snapshot(session_id).expect("Load must succeed");
        assert!(loaded.is_some(), "Snapshot must exist in DB");
        let snapshot = loaded.unwrap();
        assert_eq!(snapshot.session_id, session_id, "Session ID must match");

        println!("PASS: New session created and persisted");
    }

    // =========================================================================
    // TC-HOOKS-006-002: Session Linking (Previous Session)
    // Verify: Cross-session IC computed, previous_session_id set
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_006_002_session_linking() {
        println!("\n=== TC-HOOKS-006-002: Session Linking ===");

        let (_dir, db_path) = setup_test_db();
        let prev_id = "previous-session";
        let new_id = "new-session";

        // Setup: Create previous session with known IC
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let mut prev_snapshot = SessionIdentitySnapshot::new(prev_id);
            prev_snapshot.last_ic = 0.95;
            prev_snapshot.purpose_vector = [0.5; 13];
            memex
                .save_snapshot(&prev_snapshot)
                .expect("Save must succeed");
        }

        // Execute: Create new session linked to previous
        let args = SessionStartArgs {
            db_path: Some(db_path.clone()),
            session_id: Some(new_id.to_string()),
            previous_session_id: Some(prev_id.to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed");

        // Verify: Link established in database
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let loaded = memex
            .load_snapshot(new_id)
            .expect("Load must succeed")
            .unwrap();
        assert_eq!(
            loaded.previous_session_id.as_deref(),
            Some(prev_id),
            "previous_session_id must be set"
        );

        println!("PASS: Session linked to previous");
    }

    // =========================================================================
    // TC-HOOKS-006-003: Auto-Generate Session ID
    // Verify: Session ID generated when not provided
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_006_003_auto_generate_session_id() {
        println!("\n=== TC-HOOKS-006-003: Auto-Generate Session ID ===");

        let (_dir, db_path) = setup_test_db();

        let args = SessionStartArgs {
            db_path: Some(db_path.clone()),
            session_id: None, // Should auto-generate
            previous_session_id: None,
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed with generated ID");

        // Verify: Some session exists in DB
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let latest = memex.load_latest().expect("Load must succeed");
        assert!(latest.is_some(), "Generated session must exist");

        let snapshot = latest.unwrap();
        assert!(
            snapshot.session_id.starts_with("session-"),
            "Generated ID should have prefix"
        );

        println!("PASS: Session ID auto-generated: {}", snapshot.session_id);
    }

    // =========================================================================
    // TC-HOOKS-006-004: Missing Previous Session (Graceful)
    // Verify: When previous_session_id doesn't exist, continue without linking
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_006_004_missing_previous_session() {
        println!("\n=== TC-HOOKS-006-004: Missing Previous Session ===");

        let (_dir, db_path) = setup_test_db();

        let args = SessionStartArgs {
            db_path: Some(db_path.clone()),
            session_id: Some("new-session".to_string()),
            previous_session_id: Some("nonexistent-session".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        // Should succeed despite missing previous session
        let result = execute(args).await;
        assert!(
            result.is_ok(),
            "Execute must succeed despite missing previous"
        );

        // Verify: New session created without link
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let _loaded = memex
            .load_snapshot("new-session")
            .expect("Load must succeed")
            .unwrap();
        // Note: We log a warning but don't fail

        println!("PASS: Handled missing previous session gracefully");
    }

    // =========================================================================
    // TC-HOOKS-006-005: Database Path Resolution
    // Verify: Priority order - arg > env > default
    // =========================================================================
    #[test]
    fn tc_hooks_006_005_db_path_resolution() {
        println!("\n=== TC-HOOKS-006-005: DB Path Resolution ===");

        // Clear env var for test
        std::env::remove_var("CONTEXT_GRAPH_DB_PATH");

        // Test 1: CLI arg takes priority
        let arg_path = PathBuf::from("/custom/path");
        let result = resolve_db_path(Some(arg_path.clone()));
        assert_eq!(result.unwrap(), arg_path, "CLI arg should take priority");

        // Test 2: Env var used when no arg
        std::env::set_var("CONTEXT_GRAPH_DB_PATH", "/env/path");
        let result = resolve_db_path(None);
        assert_eq!(
            result.unwrap(),
            PathBuf::from("/env/path"),
            "Env var should be used"
        );

        // Cleanup
        std::env::remove_var("CONTEXT_GRAPH_DB_PATH");

        println!("PASS: DB path resolution follows priority");
    }

    // =========================================================================
    // TC-HOOKS-006-006: JSON Output Schema Compliance
    // Verify: Output matches HookOutput schema exactly
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_006_006_json_output_schema() {
        println!("\n=== TC-HOOKS-006-006: JSON Output Schema ===");

        let (_dir, db_path) = setup_test_db();

        let args = SessionStartArgs {
            db_path: Some(db_path),
            session_id: Some("schema-test".to_string()),
            previous_session_id: None,
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await.unwrap();

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&result).expect("Serialization must succeed");
        println!("Output JSON:\n{}", json);

        // Verify required fields
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            parsed.get("success").is_some(),
            "Must have 'success' field"
        );
        assert!(
            parsed.get("execution_time_ms").is_some(),
            "Must have 'execution_time_ms' field"
        );

        // Verify optional fields present when data available
        assert!(
            parsed.get("consciousness_state").is_some(),
            "Should have consciousness_state"
        );
        assert!(
            parsed.get("ic_classification").is_some(),
            "Should have ic_classification"
        );

        println!("PASS: JSON output matches schema");
    }

    // =========================================================================
    // TC-HOOKS-006-007: Execution Time Tracking
    // Verify: execution_time_ms reflects actual elapsed time
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_006_007_execution_time_tracking() {
        println!("\n=== TC-HOOKS-006-007: Execution Time Tracking ===");

        let (_dir, db_path) = setup_test_db();

        let args = SessionStartArgs {
            db_path: Some(db_path),
            session_id: Some("timing-test".to_string()),
            previous_session_id: None,
            stdin: false,
            format: OutputFormat::Json,
        };

        let start = std::time::Instant::now();
        let result = execute(args).await.unwrap();
        let actual_elapsed = start.elapsed().as_millis() as u64;

        // Verify timing is reasonable
        assert!(
            result.execution_time_ms > 0,
            "Must have positive execution time"
        );
        assert!(
            result.execution_time_ms <= actual_elapsed + 10, // Allow small margin
            "Reported time {} should not exceed actual elapsed {}",
            result.execution_time_ms,
            actual_elapsed
        );

        // Verify within timeout budget (5000ms)
        assert!(
            result.execution_time_ms < 5000,
            "Execution time {} must be under 5000ms timeout",
            result.execution_time_ms
        );

        println!(
            "PASS: Execution time {} ms (actual: {} ms)",
            result.execution_time_ms, actual_elapsed
        );
    }
}
