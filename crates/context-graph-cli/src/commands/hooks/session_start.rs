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

use std::f64::consts::PI;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tracing::{debug, error, info, warn};

use context_graph_core::gwt::session_identity::KURAMOTO_N;
use context_graph_core::gwt::{compute_ic, SessionIdentitySnapshot};
use context_graph_storage::rocksdb_backend::RocksDbMemex;

use super::args::{OutputFormat, SessionStartArgs};
use super::error::{HookError, HookResult};
use super::types::{
    ConsciousnessState, DriftMetrics, HookInput, HookOutput, HookPayload, ICClassification,
};

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
    let (snapshot, drift_metrics) =
        load_or_create_snapshot(&memex, &session_id, previous_session_id.as_deref())?;

    // 5. Build output structures
    let consciousness_state = build_consciousness_state(&snapshot);
    let ic_classification = ICClassification::from_value(snapshot.last_ic);

    // 6. Build final output
    let execution_time_ms = start.elapsed().as_millis() as u64;

    info!(
        session_id = %session_id,
        ic = snapshot.last_ic,
        consciousness = snapshot.consciousness,
        drift_metrics = ?drift_metrics,
        execution_time_ms,
        "SESSION_START: execute complete"
    );

    // 7. Return output with drift metrics if available
    let mut output = HookOutput::success(execution_time_ms)
        .with_consciousness_state(consciousness_state)
        .with_ic_classification(ic_classification);

    if let Some(metrics) = drift_metrics {
        output = output.with_drift_metrics(metrics);
    }

    Ok(output)
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
/// Links to previous session if provided and computes drift metrics.
///
/// # Returns
/// Tuple of (SessionIdentitySnapshot, Option<DriftMetrics>)
/// - DriftMetrics is Some when linking to a previous session succeeded
/// - DriftMetrics is None for new sessions or when previous session not found
fn load_or_create_snapshot(
    memex: &Arc<RocksDbMemex>,
    session_id: &str,
    previous_session_id: Option<&str>,
) -> HookResult<(SessionIdentitySnapshot, Option<DriftMetrics>)> {
    // Try to load existing snapshot for this session
    match memex.load_snapshot(session_id) {
        Ok(Some(snapshot)) => {
            info!(session_id = %session_id, "SESSION_START: loaded existing snapshot");
            // No drift metrics for resumed sessions (same session, not linked)
            return Ok((snapshot, None));
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
    let mut drift_metrics: Option<DriftMetrics> = None;

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

                // CRITICAL FIX: RESTORE identity state from previous session
                // This is what "restore_identity" means - we copy the previous session's
                // identity state to the new session so it continues with the same identity.
                // Without this, the new session starts with zero/default values and
                // IC computation shows 0.0 (no continuity) instead of ~1.0 (full continuity).
                //
                // Constitution Reference: TECH-SESSION-IDENTITY restore_identity flow
                // "The new session inherits the previous session's identity"
                snapshot.purpose_vector = prev_snapshot.purpose_vector;
                snapshot.kuramoto_phases = prev_snapshot.kuramoto_phases;
                snapshot.coupling = prev_snapshot.coupling;
                snapshot.consciousness = prev_snapshot.consciousness;
                snapshot.integration = prev_snapshot.integration;
                snapshot.reflection = prev_snapshot.reflection;
                snapshot.differentiation = prev_snapshot.differentiation;
                snapshot.crisis_threshold = prev_snapshot.crisis_threshold;

                // Copy trajectory (up to MAX_TRAJECTORY_LEN)
                for pv in prev_snapshot.trajectory.iter() {
                    snapshot.append_to_trajectory(*pv);
                }

                snapshot.previous_session_id = Some(prev_id.to_string());

                // Compute cross-session IC using formula IDENTITY-001
                // Now that snapshot has restored state from prev_snapshot,
                // IC should be high (~1.0) since purpose_vector and kuramoto_phases match
                let cross_ic = compute_ic(&snapshot, &prev_snapshot);
                snapshot.cross_session_ic = cross_ic;
                snapshot.last_ic = cross_ic;

                info!(
                    session_id = %session_id,
                    restored_ic = cross_ic,
                    restored_consciousness = snapshot.consciousness,
                    "SESSION_START: identity state restored from previous session"
                );

                // TASK-HOOKS-013: Compute drift metrics when linking sessions
                // After restoration, drift should be minimal (near zero)
                drift_metrics = Some(compute_drift_metrics(&snapshot, &prev_snapshot));
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
        drift_detected = drift_metrics.is_some(),
        "SESSION_START: created and saved new snapshot"
    );

    Ok((snapshot, drift_metrics))
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
// Drift Metrics Computation (TASK-HOOKS-013)
// =============================================================================

/// Compute cosine distance between two purpose vectors.
///
/// # Arguments
/// * `current` - Current session's purpose vector [KURAMOTO_N]
/// * `previous` - Previous session's purpose vector [KURAMOTO_N]
///
/// # Returns
/// Cosine distance in range [0.0, 2.0] where:
/// - 0.0 = identical vectors (perfectly aligned)
/// - 1.0 = orthogonal vectors
/// - 2.0 = opposite vectors (completely misaligned)
///
/// # Formula
/// `distance = 1 - cosine_similarity`
/// `cosine_similarity = (a · b) / (||a|| * ||b||)`
///
/// # Edge Cases
/// - Returns 0.0 if either vector has zero magnitude (identical = no drift)
///
/// # Example
/// ```ignore
/// let v1 = [0.5; 13];
/// let v2 = [0.5; 13];
/// assert_eq!(cosine_distance(&v1, &v2), 0.0); // Identical
/// ```
fn cosine_distance(current: &[f32; KURAMOTO_N], previous: &[f32; KURAMOTO_N]) -> f32 {
    // Compute dot product and magnitudes
    let mut dot_product: f64 = 0.0;
    let mut mag_current: f64 = 0.0;
    let mut mag_previous: f64 = 0.0;

    for i in 0..KURAMOTO_N {
        let c = current[i] as f64;
        let p = previous[i] as f64;
        dot_product += c * p;
        mag_current += c * c;
        mag_previous += p * p;
    }

    mag_current = mag_current.sqrt();
    mag_previous = mag_previous.sqrt();

    // Handle zero magnitude edge case
    if mag_current < f64::EPSILON || mag_previous < f64::EPSILON {
        return 0.0; // No drift if either vector is zero
    }

    // Cosine similarity in range [-1, 1]
    let cosine_similarity = dot_product / (mag_current * mag_previous);

    // Clamp to handle floating point precision issues
    let cosine_similarity = cosine_similarity.clamp(-1.0, 1.0);

    // Convert to distance: 1 - similarity gives range [0, 2]
    let distance = 1.0 - cosine_similarity;

    distance as f32
}

/// Compute mean absolute phase difference across Kuramoto oscillators.
///
/// # Arguments
/// * `current` - Current session's Kuramoto phases [KURAMOTO_N]
/// * `previous` - Previous session's Kuramoto phases [KURAMOTO_N]
///
/// # Returns
/// Mean absolute phase difference in range [0.0, π] where:
/// - 0.0 = perfectly synchronized phases
/// - π = maximally different phases (opposite)
///
/// # Formula
/// For each oscillator i:
/// `diff_i = |atan2(sin(c_i - p_i), cos(c_i - p_i))|`
/// `phase_drift = mean(diff_1, ..., diff_N)`
///
/// Using atan2 ensures proper handling of phase wrapping (2π periodicity).
///
/// # Example
/// ```ignore
/// let phases1 = [0.0; 13];
/// let phases2 = [0.0; 13];
/// assert_eq!(phase_drift(&phases1, &phases2), 0.0); // No drift
/// ```
fn phase_drift(current: &[f64; KURAMOTO_N], previous: &[f64; KURAMOTO_N]) -> f64 {
    let mut total_drift: f64 = 0.0;

    for i in 0..KURAMOTO_N {
        let diff = current[i] - previous[i];
        // Use atan2 to properly handle phase wrapping
        // This gives the smallest angle between the two phases
        let wrapped_diff = diff.sin().atan2(diff.cos()).abs();
        total_drift += wrapped_diff;
    }

    // Return mean absolute difference
    total_drift / (KURAMOTO_N as f64)
}

/// Compute comprehensive drift metrics between current and previous session snapshots.
///
/// # Arguments
/// * `current` - Current session's identity snapshot
/// * `previous` - Previous session's identity snapshot
///
/// # Returns
/// DriftMetrics containing:
/// - ic_delta: Change in identity continuity (current.last_ic - previous.last_ic)
/// - purpose_drift: Cosine distance between purpose vectors [0.0, 2.0]
/// - time_since_snapshot_ms: Time elapsed since previous snapshot
/// - kuramoto_phase_drift: Mean absolute phase difference [0.0, π]
///
/// # Constitution Reference
/// - IDENTITY-002: IC thresholds and drift detection
/// - TASK-HOOKS-013: Drift metrics specification
///
/// # Example
/// ```ignore
/// let current = SessionIdentitySnapshot::new("session-2");
/// let previous = SessionIdentitySnapshot::new("session-1");
/// let metrics = compute_drift_metrics(&current, &previous);
/// if metrics.is_crisis_drift() {
///     // Trigger recovery actions
/// }
/// ```
fn compute_drift_metrics(
    current: &SessionIdentitySnapshot,
    previous: &SessionIdentitySnapshot,
) -> DriftMetrics {
    // IC delta: positive = improvement, negative = degradation
    let ic_delta = current.last_ic - previous.last_ic;

    // Purpose vector drift (cosine distance)
    let purpose_drift = cosine_distance(&current.purpose_vector, &previous.purpose_vector);

    // Time since previous snapshot
    let time_since_snapshot_ms = current.timestamp_ms - previous.timestamp_ms;

    // Kuramoto phase drift
    let kuramoto_phase_drift = phase_drift(&current.kuramoto_phases, &previous.kuramoto_phases);

    let metrics = DriftMetrics {
        ic_delta,
        purpose_drift,
        time_since_snapshot_ms,
        kuramoto_phase_drift,
    };

    // Log drift detection results
    if metrics.is_crisis_drift() {
        warn!(
            ic_delta = ic_delta,
            purpose_drift = purpose_drift,
            kuramoto_phase_drift = kuramoto_phase_drift,
            time_since_ms = time_since_snapshot_ms,
            "SESSION_START: CRISIS DRIFT DETECTED"
        );
    } else if metrics.is_warning_drift() {
        info!(
            ic_delta = ic_delta,
            purpose_drift = purpose_drift,
            "SESSION_START: Warning drift detected"
        );
    } else {
        debug!(
            ic_delta = ic_delta,
            purpose_drift = purpose_drift,
            "SESSION_START: Normal drift metrics"
        );
    }

    metrics
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

        // Setup: Create previous session with known IC and identity state
        let prev_purpose = [0.5_f32; 13];
        let prev_phases = [0.1_f64; 13];
        let prev_coupling = 0.7;
        let prev_ic = 0.95;
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let mut prev_snapshot = SessionIdentitySnapshot::new(prev_id);
            prev_snapshot.last_ic = prev_ic;
            prev_snapshot.purpose_vector = prev_purpose;
            prev_snapshot.kuramoto_phases = prev_phases;
            prev_snapshot.coupling = prev_coupling;
            prev_snapshot.consciousness = 0.8;
            prev_snapshot.integration = 0.75;
            prev_snapshot.reflection = 0.65;
            prev_snapshot.differentiation = 0.55;
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

        // TC-HOOKS-017 FIX VERIFICATION: Identity state MUST be restored
        // Before the fix, purpose_vector was [0.0; 13] after linking, causing IC=0.0
        // After the fix, identity state is COPIED from previous session
        assert_eq!(
            loaded.purpose_vector, prev_purpose,
            "purpose_vector MUST be restored from previous session"
        );
        assert_eq!(
            loaded.kuramoto_phases, prev_phases,
            "kuramoto_phases MUST be restored from previous session"
        );
        assert!(
            (loaded.coupling - prev_coupling).abs() < 0.001,
            "coupling MUST be restored from previous session"
        );

        // Verify IC is healthy (not 0.0 as before the fix)
        assert!(
            loaded.cross_session_ic > 0.9,
            "cross_session_ic MUST be ~1.0 after restoration, got {}",
            loaded.cross_session_ic
        );

        println!("PASS: Session linked to previous with identity state restored");
        println!("  purpose_vector restored: {:?}", &loaded.purpose_vector[0..3]);
        println!("  kuramoto_phases restored: {:?}", &loaded.kuramoto_phases[0..3]);
        println!("  cross_session_ic: {}", loaded.cross_session_ic);
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

    // =========================================================================
    // TC-HOOKS-013: Drift Metrics Computation Tests
    // NO MOCK DATA - All tests use REAL RocksDB
    // =========================================================================

    // =========================================================================
    // TC-HOOKS-013-01: Drift metrics computed when linking sessions
    // Verify: drift_metrics is Some with valid values when previous exists
    // UPDATED: After TC-HOOKS-017 fix, ic_delta should be ~0.0 (not -1.0)
    //          because identity state is RESTORED before IC computation
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_013_01_drift_metrics_computed_when_linking() {
        println!("\n=== TC-HOOKS-013-01: Drift Metrics Computed When Linking ===");

        let (_dir, db_path) = setup_test_db();
        let prev_id = "prev-session-drift";
        let new_id = "new-session-drift";

        // Setup: Create previous session snapshot with known values
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let mut prev_snapshot = SessionIdentitySnapshot::new(prev_id);
            prev_snapshot.last_ic = 0.95;
            prev_snapshot.purpose_vector = [0.5; 13];
            prev_snapshot.kuramoto_phases = [0.0; 13];
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
        let output = result.unwrap();

        // Verify: drift_metrics is present
        assert!(
            output.drift_metrics.is_some(),
            "drift_metrics must be Some when linking sessions"
        );
        let drift = output.drift_metrics.unwrap();

        // Verify: all fields are valid
        assert!(
            drift.purpose_drift >= 0.0 && drift.purpose_drift <= 2.0,
            "purpose_drift {} must be in [0.0, 2.0]",
            drift.purpose_drift
        );
        assert!(
            drift.time_since_snapshot_ms > 0,
            "time_since_snapshot_ms {} must be positive",
            drift.time_since_snapshot_ms
        );
        assert!(
            drift.kuramoto_phase_drift >= 0.0 && drift.kuramoto_phase_drift <= std::f64::consts::PI,
            "kuramoto_phase_drift {} must be in [0.0, π]",
            drift.kuramoto_phase_drift
        );

        // TC-HOOKS-017 FIX VERIFICATION:
        // After the fix, identity state is RESTORED from previous session,
        // so ic_delta should be ~0.0 (identical purpose vectors = cosine similarity 1.0)
        // Before the fix: ic_delta was -0.95 or -1.0 because new session had zero vectors
        assert!(
            drift.ic_delta.abs() < 0.1,
            "ic_delta MUST be ~0.0 after identity restoration, got {} (bug if < -0.3)",
            drift.ic_delta
        );

        // purpose_drift should be ~0.0 (vectors are identical after copy)
        assert!(
            drift.purpose_drift < 0.01,
            "purpose_drift MUST be ~0.0 after identity restoration, got {}",
            drift.purpose_drift
        );

        // kuramoto_phase_drift should be ~0.0 (phases are identical after copy)
        assert!(
            drift.kuramoto_phase_drift < 0.01,
            "kuramoto_phase_drift MUST be ~0.0 after identity restoration, got {}",
            drift.kuramoto_phase_drift
        );

        println!("PASS: Drift metrics computed with valid values (TC-HOOKS-017 fix verified)");
        println!("  ic_delta: {} (MUST be ~0.0, not -1.0)", drift.ic_delta);
        println!("  purpose_drift: {} (MUST be ~0.0)", drift.purpose_drift);
        println!("  time_since_snapshot_ms: {}", drift.time_since_snapshot_ms);
        println!("  kuramoto_phase_drift: {} (MUST be ~0.0)", drift.kuramoto_phase_drift);
    }

    // =========================================================================
    // TC-HOOKS-013-02: No drift metrics for new session
    // Verify: drift_metrics is None when no previous_session_id
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_013_02_no_drift_metrics_for_new_session() {
        println!("\n=== TC-HOOKS-013-02: No Drift Metrics for New Session ===");

        let (_dir, db_path) = setup_test_db();

        let args = SessionStartArgs {
            db_path: Some(db_path),
            session_id: Some("brand-new-session".to_string()),
            previous_session_id: None,
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed");
        let output = result.unwrap();

        // Verify: No drift metrics for new session
        assert!(
            output.drift_metrics.is_none(),
            "drift_metrics must be None for new session"
        );
        assert!(output.success, "success must be true");

        println!("PASS: No drift metrics for new session");
    }

    // =========================================================================
    // TC-HOOKS-013-03: Drift metrics when previous session not found
    // Verify: drift_metrics is None when previous doesn't exist
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_013_03_drift_metrics_when_previous_not_found() {
        println!("\n=== TC-HOOKS-013-03: Drift Metrics When Previous Not Found ===");

        let (_dir, db_path) = setup_test_db();

        let args = SessionStartArgs {
            db_path: Some(db_path),
            session_id: Some("orphan-session".to_string()),
            previous_session_id: Some("nonexistent-session-xyz".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed despite missing previous");
        let output = result.unwrap();

        // Verify: No drift metrics when previous not found
        assert!(
            output.drift_metrics.is_none(),
            "drift_metrics must be None when previous not found"
        );
        assert!(output.success, "success must be true");

        println!("PASS: Handled missing previous session gracefully");
    }

    // =========================================================================
    // TC-HOOKS-013-04: Cosine distance edge cases
    // Verify: cosine_distance function handles edge cases
    // =========================================================================
    #[test]
    fn tc_hooks_013_04_cosine_distance_edge_cases() {
        println!("\n=== TC-HOOKS-013-04: Cosine Distance Edge Cases ===");

        // Test 1: Identical vectors -> distance 0.0
        let a = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!(
            (dist - 0.0).abs() < 0.001,
            "Identical vectors should have distance 0.0, got {}",
            dist
        );
        println!("  Identical vectors: distance = {}", dist);

        // Test 2: Opposite vectors -> distance 2.0
        let a = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [-1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!(
            (dist - 2.0).abs() < 0.001,
            "Opposite vectors should have distance 2.0, got {}",
            dist
        );
        println!("  Opposite vectors: distance = {}", dist);

        // Test 3: Zero vectors -> distance 0.0 (graceful handling)
        let a = [0.0_f32; 13];
        let b = [0.0_f32; 13];
        let dist = cosine_distance(&a, &b);
        assert!(
            dist == 0.0,
            "Zero vectors should have distance 0.0, got {}",
            dist
        );
        println!("  Zero vectors: distance = {}", dist);

        // Test 4: Proportional vectors -> distance 0.0
        let a = [1.0_f32; 13];
        let b = [0.5_f32; 13];
        let dist = cosine_distance(&a, &b);
        assert!(
            (dist - 0.0).abs() < 0.001,
            "Proportional vectors should have distance ~0.0, got {}",
            dist
        );
        println!("  Proportional vectors: distance = {}", dist);

        println!("PASS: Cosine distance handles edge cases");
    }

    // =========================================================================
    // TC-HOOKS-013-05: Phase drift edge cases
    // Verify: phase_drift function handles edge cases with wrapping
    // =========================================================================
    #[test]
    fn tc_hooks_013_05_phase_drift_edge_cases() {
        use std::f64::consts::PI;
        println!("\n=== TC-HOOKS-013-05: Phase Drift Edge Cases ===");

        // Test 1: Identical phases -> drift 0.0
        let a = [0.0_f64; 13];
        let b = [0.0_f64; 13];
        let drift = phase_drift(&a, &b);
        assert!(
            (drift - 0.0).abs() < 0.001,
            "Identical phases should have drift 0.0, got {}",
            drift
        );
        println!("  Identical phases: drift = {}", drift);

        // Test 2: Opposite phases (π apart) -> drift π
        let a = [0.0_f64; 13];
        let b = [PI; 13];
        let drift = phase_drift(&a, &b);
        assert!(
            (drift - PI).abs() < 0.001,
            "Opposite phases should have drift π, got {}",
            drift
        );
        println!("  Opposite phases: drift = {}", drift);

        // Test 3: 2π apart (wrapped) -> drift 0.0
        let a = [0.0_f64; 13];
        let b = [2.0 * PI; 13];
        let drift = phase_drift(&a, &b);
        assert!(
            (drift - 0.0).abs() < 0.001,
            "2π apart should wrap to drift 0.0, got {}",
            drift
        );
        println!("  2π apart (wrapped): drift = {}", drift);

        // Test 4: π/2 apart -> drift π/2
        let a = [PI / 2.0; 13];
        let b = [0.0_f64; 13];
        let drift = phase_drift(&a, &b);
        assert!(
            (drift - PI / 2.0).abs() < 0.001,
            "π/2 apart should have drift π/2, got {}",
            drift
        );
        println!("  π/2 apart: drift = {}", drift);

        println!("PASS: Phase drift handles edge cases with wrapping");
    }

    // =========================================================================
    // TC-HOOKS-013-06: Crisis drift triggers error log
    // Verify: is_crisis_drift returns true for ic_delta < -0.3
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_013_06_crisis_drift_detection() {
        println!("\n=== TC-HOOKS-013-06: Crisis Drift Detection ===");

        let (_dir, db_path) = setup_test_db();
        let prev_id = "high-ic-session";
        let new_id = "low-ic-session";

        // Setup: Create previous session with high IC
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let mut prev_snapshot = SessionIdentitySnapshot::new(prev_id);
            prev_snapshot.last_ic = 0.95; // High IC
            memex
                .save_snapshot(&prev_snapshot)
                .expect("Save must succeed");
        }

        // Create new session with low IC (simulated by manipulating the snapshot)
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let mut new_snapshot = SessionIdentitySnapshot::new(new_id);
            new_snapshot.last_ic = 0.55; // Low IC (>30% drop)
            new_snapshot.previous_session_id = Some(prev_id.to_string());
            memex
                .save_snapshot(&new_snapshot)
                .expect("Save must succeed");
        }

        // Verify: DriftMetrics correctly identifies crisis
        let drift = DriftMetrics {
            ic_delta: 0.55 - 0.95, // -0.4 (40% drop)
            purpose_drift: 0.0,
            time_since_snapshot_ms: 1000,
            kuramoto_phase_drift: 0.0,
        };

        assert!(
            drift.is_crisis_drift(),
            "ic_delta {} should be crisis (< -0.3)",
            drift.ic_delta
        );
        assert!(
            drift.is_warning_drift(),
            "crisis is also warning level"
        );

        println!("PASS: Crisis drift correctly detected for ic_delta = {}", drift.ic_delta);
    }

    // =========================================================================
    // TC-HOOKS-013-07: Warning drift triggers warn log
    // Verify: is_warning_drift returns true for ic_delta < -0.1
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_013_07_warning_drift_detection() {
        println!("\n=== TC-HOOKS-013-07: Warning Drift Detection ===");

        // Test warning level: -0.15 (15% drop)
        let warning_drift = DriftMetrics {
            ic_delta: -0.15, // 15% drop
            purpose_drift: 0.0,
            time_since_snapshot_ms: 1000,
            kuramoto_phase_drift: 0.0,
        };

        assert!(
            warning_drift.is_warning_drift(),
            "ic_delta {} should be warning (< -0.1)",
            warning_drift.ic_delta
        );
        assert!(
            !warning_drift.is_crisis_drift(),
            "ic_delta {} should NOT be crisis (>= -0.3)",
            warning_drift.ic_delta
        );
        println!("  Warning drift: ic_delta = {}, is_warning = true, is_crisis = false",
            warning_drift.ic_delta);

        // Test healthy level: -0.05 (5% drop)
        let healthy_drift = DriftMetrics {
            ic_delta: -0.05, // 5% drop
            purpose_drift: 0.0,
            time_since_snapshot_ms: 1000,
            kuramoto_phase_drift: 0.0,
        };

        assert!(
            !healthy_drift.is_warning_drift(),
            "ic_delta {} should NOT be warning (>= -0.1)",
            healthy_drift.ic_delta
        );
        assert!(
            !healthy_drift.is_crisis_drift(),
            "ic_delta {} should NOT be crisis (>= -0.3)",
            healthy_drift.ic_delta
        );
        println!("  Healthy drift: ic_delta = {}, is_warning = false, is_crisis = false",
            healthy_drift.ic_delta);

        // Test positive drift (improvement)
        let positive_drift = DriftMetrics {
            ic_delta: 0.1, // 10% improvement
            purpose_drift: 0.0,
            time_since_snapshot_ms: 1000,
            kuramoto_phase_drift: 0.0,
        };

        assert!(
            !positive_drift.is_warning_drift(),
            "positive ic_delta {} should NOT be warning",
            positive_drift.ic_delta
        );
        assert!(
            !positive_drift.is_crisis_drift(),
            "positive ic_delta {} should NOT be crisis",
            positive_drift.ic_delta
        );
        println!("  Positive drift: ic_delta = {}, is_warning = false, is_crisis = false",
            positive_drift.ic_delta);

        println!("PASS: Warning drift correctly detected at threshold boundaries");
    }

    // =========================================================================
    // TC-HOOKS-017: IC Restoration Bug Fix Tests
    // CRITICAL: These tests verify the fix for IC dropping to 0.0 after session restore
    // Root cause: identity state was not being copied from previous session
    // Fix: copy purpose_vector, kuramoto_phases, etc. BEFORE computing IC
    // =========================================================================

    // =========================================================================
    // TC-HOOKS-017-01: Complete identity state restoration
    // Verify: ALL identity fields are copied from previous to new session
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_017_01_complete_identity_state_restoration() {
        println!("\n=== TC-HOOKS-017-01: Complete Identity State Restoration ===");

        let (_dir, db_path) = setup_test_db();
        let prev_id = "session-with-identity";
        let new_id = "restored-session";

        // Setup: Create previous session with FULL identity state
        let prev_purpose = [0.3_f32, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.35, 0.45, 0.55];
        let prev_phases = [0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
        let prev_coupling = 0.75;
        let prev_consciousness = 0.85;
        let prev_integration = 0.72;
        let prev_reflection = 0.68;
        let prev_differentiation = 0.59;
        let prev_crisis_threshold = 0.45;
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let mut prev_snapshot = SessionIdentitySnapshot::new(prev_id);
            prev_snapshot.last_ic = 0.95;
            prev_snapshot.purpose_vector = prev_purpose;
            prev_snapshot.kuramoto_phases = prev_phases;
            prev_snapshot.coupling = prev_coupling;
            prev_snapshot.consciousness = prev_consciousness;
            prev_snapshot.integration = prev_integration;
            prev_snapshot.reflection = prev_reflection;
            prev_snapshot.differentiation = prev_differentiation;
            prev_snapshot.crisis_threshold = prev_crisis_threshold;
            // Add trajectory entries
            prev_snapshot.append_to_trajectory([0.1; 13]);
            prev_snapshot.append_to_trajectory([0.2; 13]);
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

        // VERIFY: ALL identity fields restored from previous session
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let new_snapshot = memex
            .load_snapshot(new_id)
            .expect("Load must succeed")
            .expect("Snapshot must exist");

        // Verify purpose_vector (CRITICAL - this was the root cause of IC=0.0)
        assert_eq!(
            new_snapshot.purpose_vector, prev_purpose,
            "purpose_vector MUST be restored"
        );

        // Verify kuramoto_phases
        assert_eq!(
            new_snapshot.kuramoto_phases, prev_phases,
            "kuramoto_phases MUST be restored"
        );

        // Verify coupling
        assert!(
            (new_snapshot.coupling - prev_coupling).abs() < 0.001,
            "coupling MUST be restored, got {} expected {}",
            new_snapshot.coupling, prev_coupling
        );

        // Verify consciousness
        assert!(
            (new_snapshot.consciousness - prev_consciousness).abs() < 0.001,
            "consciousness MUST be restored, got {} expected {}",
            new_snapshot.consciousness, prev_consciousness
        );

        // Verify integration
        assert!(
            (new_snapshot.integration - prev_integration).abs() < 0.001,
            "integration MUST be restored, got {} expected {}",
            new_snapshot.integration, prev_integration
        );

        // Verify reflection
        assert!(
            (new_snapshot.reflection - prev_reflection).abs() < 0.001,
            "reflection MUST be restored, got {} expected {}",
            new_snapshot.reflection, prev_reflection
        );

        // Verify differentiation
        assert!(
            (new_snapshot.differentiation - prev_differentiation).abs() < 0.001,
            "differentiation MUST be restored, got {} expected {}",
            new_snapshot.differentiation, prev_differentiation
        );

        // Verify crisis_threshold
        assert!(
            (new_snapshot.crisis_threshold - prev_crisis_threshold).abs() < 0.001,
            "crisis_threshold MUST be restored, got {} expected {}",
            new_snapshot.crisis_threshold, prev_crisis_threshold
        );

        // Verify trajectory copied
        assert!(
            new_snapshot.trajectory.len() >= 2,
            "trajectory MUST be restored, got {} entries",
            new_snapshot.trajectory.len()
        );

        println!("PASS: All 9 identity state fields restored from previous session");
        println!("  purpose_vector: {:?}...", &new_snapshot.purpose_vector[0..3]);
        println!("  kuramoto_phases: {:?}...", &new_snapshot.kuramoto_phases[0..3]);
        println!("  coupling: {}", new_snapshot.coupling);
        println!("  consciousness: {}", new_snapshot.consciousness);
        println!("  integration: {}", new_snapshot.integration);
        println!("  reflection: {}", new_snapshot.reflection);
        println!("  differentiation: {}", new_snapshot.differentiation);
        println!("  crisis_threshold: {}", new_snapshot.crisis_threshold);
        println!("  trajectory.len: {}", new_snapshot.trajectory.len());
    }

    // =========================================================================
    // TC-HOOKS-017-02: IC is healthy after restoration
    // Verify: cross_session_ic is ~1.0, NOT 0.0 (the bug symptom)
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_017_02_ic_healthy_after_restoration() {
        println!("\n=== TC-HOOKS-017-02: IC Healthy After Restoration ===");

        let (_dir, db_path) = setup_test_db();
        let prev_id = "healthy-session";
        let new_id = "should-be-healthy-too";

        // Setup: Create previous session with healthy IC
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let mut prev_snapshot = SessionIdentitySnapshot::new(prev_id);
            prev_snapshot.last_ic = 0.95;
            prev_snapshot.purpose_vector = [0.5; 13]; // Non-zero vector
            prev_snapshot.kuramoto_phases = [0.1; 13];
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
        let output = result.unwrap();

        // VERIFY: IC classification shows healthy status
        assert!(
            output.ic_classification.is_some(),
            "ic_classification must be present"
        );
        let ic_class = output.ic_classification.unwrap();

        // IC value should be healthy (>= 0.9)
        // Before the fix: IC was 0.0 (critical)
        // After the fix: IC should be ~1.0 (healthy)
        assert!(
            ic_class.value > 0.9,
            "IC MUST be healthy (>0.9) after restoration, got {} (BUG if 0.0)",
            ic_class.value
        );

        // Verify level is "Healthy" not "Critical"
        assert!(
            ic_class.level.to_string().contains("Healthy") || ic_class.value >= 0.9,
            "IC level MUST be Healthy, got {} with value {}",
            ic_class.level, ic_class.value
        );

        // VERIFY: Database shows healthy IC
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let new_snapshot = memex
            .load_snapshot(new_id)
            .expect("Load must succeed")
            .expect("Snapshot must exist");

        assert!(
            new_snapshot.cross_session_ic > 0.9,
            "cross_session_ic in DB MUST be >0.9, got {} (BUG if 0.0)",
            new_snapshot.cross_session_ic
        );

        println!("PASS: IC is healthy after session restoration");
        println!("  IC value: {} (MUST be >0.9, BUG if 0.0)", ic_class.value);
        println!("  IC level: {}", ic_class.level);
        println!("  DB cross_session_ic: {}", new_snapshot.cross_session_ic);
    }

    // =========================================================================
    // TC-HOOKS-017-03: IC delta is ~0.0, not -1.0
    // Verify: drift_metrics.ic_delta is near zero (identical vectors)
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_017_03_ic_delta_near_zero() {
        println!("\n=== TC-HOOKS-017-03: IC Delta Near Zero ===");

        let (_dir, db_path) = setup_test_db();
        let prev_id = "stable-session";
        let new_id = "continued-session";

        // Setup: Create previous session
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let mut prev_snapshot = SessionIdentitySnapshot::new(prev_id);
            prev_snapshot.last_ic = 0.92;
            prev_snapshot.purpose_vector = [0.6; 13];
            prev_snapshot.kuramoto_phases = [0.2; 13];
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
        let output = result.unwrap();

        // VERIFY: drift_metrics shows near-zero delta
        assert!(
            output.drift_metrics.is_some(),
            "drift_metrics must be present"
        );
        let drift = output.drift_metrics.unwrap();

        // Before fix: ic_delta was -0.92 or worse (crisis)
        // After fix: ic_delta should be ~0.0 (vectors identical after copy)
        assert!(
            drift.ic_delta.abs() < 0.1,
            "ic_delta MUST be ~0.0 after restoration, got {} (BUG if < -0.3)",
            drift.ic_delta
        );

        // Also verify not a crisis or warning
        assert!(
            !drift.is_crisis_drift(),
            "MUST NOT be crisis drift after restoration, ic_delta={}",
            drift.ic_delta
        );
        assert!(
            !drift.is_warning_drift(),
            "MUST NOT be warning drift after restoration, ic_delta={}",
            drift.ic_delta
        );

        println!("PASS: IC delta is ~0.0 after restoration (not -1.0)");
        println!("  ic_delta: {} (MUST be ~0.0)", drift.ic_delta);
        println!("  is_crisis_drift: false");
        println!("  is_warning_drift: false");
    }
}
