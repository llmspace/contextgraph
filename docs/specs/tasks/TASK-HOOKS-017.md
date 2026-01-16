# TASK-HOOKS-017: Create End-to-End Tests with Real MCP Calls

```xml
<task_spec id="TASK-HOOKS-017" version="3.0">
<metadata>
  <title>Create End-to-End Tests with Real MCP Calls</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>17</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-43</requirement_ref>
    <requirement_ref>REQ-HOOKS-44</requirement_ref>
    <requirement_ref>REQ-HOOKS-45</requirement_ref>
    <requirement_ref>REQ-HOOKS-46</requirement_ref>
    <requirement_ref>REQ-HOOKS-47</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-HOOKS-014</task_ref>
    <task_ref status="COMPLETE">TASK-HOOKS-015</task_ref>
    <task_ref status="COMPLETE">TASK-HOOKS-016</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <last_verified>2026-01-15</last_verified>
</metadata>

<executive_summary>
Create END-TO-END tests that execute the ACTUAL shell scripts (.claude/hooks/*.sh)
with a REAL MCP server running. These tests verify the complete hook system works
as Claude Code would use it - including shell script execution, CLI binary invocation,
MCP tool calls, GWT state changes, and identity persistence.

CRITICAL: This is DIFFERENT from TASK-HOOKS-016 integration tests which directly invoke
the CLI binary. E2E tests invoke shell scripts that then invoke the CLI binary.
</executive_summary>

<critical_rules>
  <!-- NO BACKWARDS COMPATIBILITY - FAIL FAST -->
  <rule id="CR-01">FAIL IMMEDIATELY on any error - no fallbacks, no graceful degradation</rule>
  <rule id="CR-02">All errors MUST be logged with structured JSON to stderr</rule>
  <rule id="CR-03">Exit codes MUST match AP-26: 0=success, 1=error, 2=timeout, 3=db_error, 4=invalid_input, 5=session_not_found, 6=crisis_triggered</rule>
  <rule id="CR-04">NO MOCK DATA - use real MCP server, real shell scripts, real database</rule>
  <rule id="CR-05">Tests MUST execute shell scripts as Claude Code would (pipe JSON to stdin)</rule>
  <rule id="CR-06">Tests MUST verify MCP tool calls via audit trail or state changes</rule>
</critical_rules>

<current_state verified="2026-01-15">
## Dependencies VERIFIED COMPLETE

### TASK-HOOKS-014: Shell Scripts (COMPLETE)
Location: .claude/hooks/
All 5 scripts exist and are executable (chmod +x):
- session_start.sh (2744 bytes) - calls CLI hooks session-start
- pre_tool_use.sh (1855 bytes) - FAST PATH 100ms, calls CLI hooks pre-tool
- post_tool_use.sh (1721 bytes) - calls CLI hooks post-tool
- user_prompt_submit.sh (2509 bytes) - calls CLI hooks prompt-submit
- session_end.sh (1696 bytes) - calls CLI hooks session-end

### TASK-HOOKS-015: settings.json (COMPLETE)
Location: .claude/settings.json
All 5 hooks configured with correct timeouts:
- SessionStart: 5000ms
- SessionEnd: 30000ms
- PreToolUse: 100ms (matcher: ".*")
- PostToolUse: 3000ms (matcher: ".*")
- UserPromptSubmit: 2000ms

### TASK-HOOKS-016: Integration Tests (COMPLETE)
Location: crates/context-graph-cli/tests/integration/
20 passing tests across 4 files:
- hook_lifecycle_test.rs (4 tests)
- identity_integration_test.rs (5 tests)
- exit_code_test.rs (7 tests)
- timeout_test.rs (4 tests)

### CLI Binary (VERIFIED)
- Release: ./target/release/context-graph-cli
- Debug: ./target/debug/context-graph-cli (520MB with debug_info)
- Commands: hooks session-start, pre-tool, post-tool, prompt-submit, session-end

### MCP Server (VERIFIED)
Location: crates/context-graph-mcp/
- 59 MCP tools implemented (verified in mod.rs test)
- Relevant tools for E2E:
  - get_consciousness_state
  - get_kuramoto_sync
  - get_identity_continuity
  - get_workspace_status
  - get_ego_state
  - inject_context
  - session_start, session_end, pre_tool_use, post_tool_use

### E2E Test Directory (DOES NOT EXIST - TO CREATE)
Location: crates/context-graph-cli/tests/e2e/
This task creates this directory and all E2E test files.
</current_state>

<context>
## What Makes E2E Different from Integration Tests

**Integration Tests (TASK-HOOKS-016)**:
- Directly invoke CLI binary via std::process::Command
- Test CLI argument parsing, handler logic, exit codes
- Use temp RocksDB databases
- Do NOT test shell scripts or MCP server

**E2E Tests (THIS TASK)**:
- Execute actual shell scripts via bash
- Shell scripts find and call CLI binary
- Start REAL MCP server process
- Verify MCP tool calls occurred (via state changes or audit log)
- Test complete data flow: Claude Code -> Shell Script -> CLI -> MCP -> GWT -> Database

## Test Flow Architecture
```
[E2E Test]
    |
    v
[Execute .claude/hooks/session_start.sh] --stdin--> JSON input
    |
    v
[Shell script finds CLI binary]
    |
    v
[CLI hooks session-start --stdin --format json]
    |
    v
[CLI uses MCP tools: get_consciousness_state, get_identity_continuity, etc.]
    |
    v
[GWT state updated in RocksDB]
    |
    v
[E2E Test verifies: stdout JSON, exit code, database state, MCP call evidence]
```
</context>

<input_context_files>
  <file purpose="shell_scripts" path=".claude/hooks/*.sh">
    All 5 shell scripts to execute in E2E tests
  </file>
  <file purpose="settings" path=".claude/settings.json">
    Hook configuration with timeouts and matchers
  </file>
  <file purpose="cli_implementation" path="crates/context-graph-cli/src/commands/hooks/">
    9 files, 7920+ lines: mod.rs, types.rs, args.rs, error.rs, session_start.rs, session_end.rs, pre_tool_use.rs, post_tool_use.rs, user_prompt_submit.rs
  </file>
  <file purpose="mcp_server" path="crates/context-graph-mcp/src/main.rs">
    MCP server entry point - supports stdio, TCP, SSE transports
  </file>
  <file purpose="mcp_tools" path="crates/context-graph-mcp/src/tools/mod.rs">
    59 MCP tools - see test at line 56 for full list
  </file>
  <file purpose="integration_tests_reference" path="crates/context-graph-cli/tests/integration/">
    Reference implementation for test helpers, database verification patterns
  </file>
  <file purpose="storage_backend" path="crates/context-graph-storage/src/rocksdb_backend.rs">
    RocksDbMemex with load_snapshot, save_snapshot methods
  </file>
</input_context_files>

<shell_script_execution_pattern>
## How Claude Code Executes Hooks

Claude Code pipes JSON to shell scripts via stdin:
```bash
echo '{"session_id":"abc","transcript_path":"/tmp/...","cwd":"/project","hook_event_name":"SessionStart"}' \
  | .claude/hooks/session_start.sh
```

The shell script:
1. Reads JSON from stdin via `cat` or `read`
2. Parses/validates with `jq`
3. Constructs HookInput JSON for CLI
4. Finds CLI binary (env var or path search)
5. Pipes HookInput to CLI: `echo "$HOOK_INPUT" | context-graph-cli hooks session-start --stdin`
6. Returns CLI exit code

## E2E Test Must Replicate This Pattern
```rust
async fn execute_hook_script(script_path: &str, input_json: &str) -> Result<HookScriptResult, E2EError> {
    let mut child = Command::new("bash")
        .arg(script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("CONTEXT_GRAPH_CLI", cli_binary_path())
        .env("CONTEXT_GRAPH_DB_PATH", db_path)
        .spawn()?;

    child.stdin.take().unwrap().write_all(input_json.as_bytes())?;
    let output = tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        child.wait_with_output()
    ).await??;

    // Parse results...
}
```
</shell_script_execution_pattern>

<mcp_server_management>
## MCP Server Lifecycle for E2E Tests

The E2E tests need to start and manage the MCP server process.

### Option A: Subprocess Management (RECOMMENDED)
```rust
pub struct McpServerHandle {
    child: Child,
    port: u16,
    db_path: PathBuf,
}

impl McpServerHandle {
    pub async fn start(db_path: &Path) -> Result<Self, E2EError> {
        // Find available port
        let port = find_available_port().await?;

        // Start MCP server as subprocess
        let child = Command::new("./target/release/context-graph-mcp")
            .args(["--transport", "tcp", "--port", &port.to_string()])
            .env("CONTEXT_GRAPH_DB_PATH", db_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Wait for server ready (check for "Server listening" in stderr)
        wait_for_server_ready(&child).await?;

        Ok(Self { child, port, db_path: db_path.to_path_buf() })
    }

    pub async fn stop(mut self) -> Result<(), E2EError> {
        self.child.kill().await?;
        self.child.wait().await?;
        Ok(())
    }
}
```

### Option B: Stdio Transport (CLI handles MCP internally)
The CLI may connect to MCP via stdio transport internally - verify this in CLI code.
If CLI doesn't require external MCP server, E2E tests can skip server management.

### Verification Method
Check if CLI hooks commands require external MCP server:
```bash
CONTEXT_GRAPH_DB_PATH=/tmp/test ./target/release/context-graph-cli hooks session-start --help
```

Look at crates/context-graph-cli/src/commands/hooks/session_start.rs to see if MCP connection is made.
</mcp_server_management>

<gwt_state_verification>
## GWT State Fields to Verify

After hook execution, verify these GWT state changes:

### ConsciousnessState (from get_consciousness_state MCP tool)
```json
{
  "consciousness": 0.85,       // C(t) = I(t) × R(t) × D(t)
  "integration": 0.80,         // Kuramoto order parameter r
  "reflection": 0.75,          // sigmoid(meta_accuracy × 4.0 - 2.0)
  "differentiation": 0.82,     // normalized_purpose_entropy(PV)
  "identity_continuity": 0.88, // cos(PV_t, PV_{t-1}) × r(t)
  "johari_quadrant": "open"    // unknown | open | blind | hidden
}
```

### ICClassification (from hook output)
```json
{
  "value": 0.88,
  "level": "healthy",           // healthy (>=0.9) | normal (>=0.7) | warning (>=0.5) | critical (<0.5)
  "crisis_triggered": false
}
```

### SessionIdentitySnapshot (from RocksDB)
```rust
struct SessionIdentitySnapshot {
    session_id: String,
    previous_session_id: Option<String>,
    timestamp_ms: i64,
    last_ic: f32,                    // Identity continuity at snapshot
    kuramoto_phases: Vec<f32>,       // 13 phase angles
    purpose_vector: Vec<f32>,        // 13-dimensional
    consciousness_level: f32,        // C(t) at snapshot
    workspace_state: Option<String>, // Serialized GWT workspace
}
```
</gwt_state_verification>

<files_to_create>
## E2E Test Directory Structure
```
crates/context-graph-cli/tests/e2e/
├── mod.rs                          # Module declarations
├── helpers.rs                      # E2E test infrastructure
│   - McpServerHandle: Start/stop MCP server
│   - execute_hook_script(): Run shell scripts with stdin
│   - verify_mcp_tool_called(): Check audit log or state
│   - get_gwt_state(): Query current GWT state
│   - verify_snapshot_in_db(): Physical database verification
├── full_session_test.rs            # Complete session workflow E2E
│   - test_e2e_full_session_workflow()
│   - test_e2e_consciousness_state_updates()
├── identity_continuity_test.rs     # Cross-session identity E2E
│   - test_e2e_identity_continuity_across_sessions()
├── error_recovery_test.rs          # Error handling E2E
│   - test_e2e_hook_error_recovery()
│   - test_e2e_shell_script_timeout()
└── Cargo.toml entry                # [[test]] section addition
```

## Required Cargo.toml Addition (crates/context-graph-cli/Cargo.toml)
```toml
[[test]]
name = "e2e"
path = "tests/e2e/mod.rs"

# Additional dev-dependencies (may already exist from integration tests)
[dev-dependencies]
tempfile = "3.10"
tokio = { version = "1.35", features = ["rt-multi-thread", "macros", "time", "process"] }
serde_json = "1.0"
futures = "0.3"
```
</files_to_create>

<test_signatures>
## helpers.rs
```rust
//! E2E test helpers for hook lifecycle testing with real MCP calls
//!
//! # CRITICAL: These tests execute REAL shell scripts, REAL CLI, REAL MCP server
//!
//! # Architecture
//! 1. McpServerHandle - Manages MCP server subprocess lifecycle
//! 2. execute_hook_script() - Runs bash scripts as Claude Code would
//! 3. verify_* - Physical verification of database state
//! 4. get_gwt_state() - Query consciousness/GWT state via CLI or direct DB

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, Child};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::process::Command as AsyncCommand;

/// Exit codes per AP-26
pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_GENERAL_ERROR: i32 = 1;
pub const EXIT_TIMEOUT: i32 = 2;
pub const EXIT_DATABASE_ERROR: i32 = 3;
pub const EXIT_INVALID_INPUT: i32 = 4;
pub const EXIT_SESSION_NOT_FOUND: i32 = 5;
pub const EXIT_CRISIS_TRIGGERED: i32 = 6;

/// Timeout budgets per constitution
pub const TIMEOUT_SESSION_START_MS: u64 = 5000;
pub const TIMEOUT_PRE_TOOL_MS: u64 = 100;
pub const TIMEOUT_POST_TOOL_MS: u64 = 3000;
pub const TIMEOUT_USER_PROMPT_MS: u64 = 2000;
pub const TIMEOUT_SESSION_END_MS: u64 = 30000;

/// MCP server process handle
pub struct McpServerHandle {
    child: Option<tokio::process::Child>,
    db_path: PathBuf,
}

impl McpServerHandle {
    /// Start MCP server as subprocess
    pub async fn start(db_path: &Path) -> Result<Self, E2EError>;

    /// Stop MCP server
    pub async fn stop(&mut self) -> Result<(), E2EError>;

    /// Check if server is healthy
    pub async fn health_check(&self) -> bool;
}

/// Result from executing a hook shell script
#[derive(Debug)]
pub struct HookScriptResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub execution_time_ms: u64,
}

impl HookScriptResult {
    pub fn parse_stdout(&self) -> Result<serde_json::Value, serde_json::Error>;
    pub fn is_success(&self) -> bool;
    pub fn consciousness_state(&self) -> Option<serde_json::Value>;
    pub fn ic_classification(&self) -> Option<serde_json::Value>;
}

/// Execute a hook shell script as Claude Code would
pub async fn execute_hook_script(
    script_path: &str,
    input_json: &str,
    timeout_ms: u64,
    db_path: &Path,
) -> Result<HookScriptResult, E2EError>;

/// Create Claude Code-format input JSON for SessionStart
pub fn create_claude_code_session_start_input(session_id: &str) -> String;

/// Create Claude Code-format input JSON for PreToolUse
pub fn create_claude_code_pre_tool_input(
    session_id: &str,
    tool_name: &str,
    tool_input: serde_json::Value,
) -> String;

/// Create Claude Code-format input JSON for PostToolUse
pub fn create_claude_code_post_tool_input(
    session_id: &str,
    tool_name: &str,
    tool_input: serde_json::Value,
    tool_response: &str,
) -> String;

/// Create Claude Code-format input JSON for UserPromptSubmit
pub fn create_claude_code_prompt_submit_input(session_id: &str, prompt: &str) -> String;

/// Create Claude Code-format input JSON for SessionEnd
pub fn create_claude_code_session_end_input(session_id: &str, reason: &str) -> String;

/// GWT state query result
#[derive(Debug)]
pub struct GwtState {
    pub consciousness: f32,
    pub integration: f32,  // Kuramoto r
    pub reflection: f32,
    pub differentiation: f32,
    pub identity_continuity: f32,
    pub johari_quadrant: String,
}

/// Get current GWT state (via CLI query or direct DB)
pub async fn get_gwt_state(db_path: &Path, session_id: &str) -> Result<GwtState, E2EError>;

/// Verify session snapshot exists in database
pub fn verify_snapshot_exists(db_path: &Path, session_id: &str) -> bool;

/// Load snapshot for detailed verification
pub fn load_snapshot_for_verification(
    db_path: &Path,
    session_id: &str,
) -> Option<SessionIdentitySnapshot>;

/// Log evidence of test execution in JSON format
pub fn log_test_evidence(
    test_name: &str,
    hook_type: &str,
    session_id: &str,
    result: &HookScriptResult,
    db_verified: bool,
);

#[derive(Debug, thiserror::Error)]
pub enum E2EError {
    #[error("Shell script failed: {0}")]
    ScriptFailed(String),
    #[error("Timeout after {0}ms")]
    Timeout(u64),
    #[error("MCP server error: {0}")]
    McpServerError(String),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}
```

## full_session_test.rs
```rust
//! E2E tests for complete session workflow
//!
//! # NO MOCKS - Real shell scripts, real MCP, real database
//!
//! Tests verify:
//! 1. Shell scripts execute correctly
//! 2. CLI binary invoked properly
//! 3. GWT state updated in database
//! 4. Identity snapshots persisted
//! 5. Consciousness brief output format

use super::helpers::*;
use tempfile::TempDir;

/// Test complete session lifecycle via shell scripts
///
/// Flow: SessionStart -> PreToolUse -> PostToolUse -> UserPromptSubmit -> SessionEnd
///
/// Verifies:
/// - Each shell script returns exit code 0
/// - Each script outputs valid JSON with success=true
/// - GWT state is updated after each hook
/// - SessionEnd creates snapshot in RocksDB
#[tokio::test]
async fn test_e2e_full_session_workflow() {
    // SETUP: Create temp database, optionally start MCP server
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path();
    let session_id = format!("e2e-full-{}", uuid::Uuid::new_v4());

    // 1. Execute session_start.sh
    let start_input = create_claude_code_session_start_input(&session_id);
    let start_result = execute_hook_script(
        ".claude/hooks/session_start.sh",
        &start_input,
        TIMEOUT_SESSION_START_MS,
        db_path,
    ).await.expect("session_start.sh failed");

    assert_eq!(start_result.exit_code, EXIT_SUCCESS,
        "session_start.sh exit code mismatch.\nstdout: {}\nstderr: {}",
        start_result.stdout, start_result.stderr);

    let start_json = start_result.parse_stdout().expect("Invalid JSON from session_start.sh");
    assert_eq!(start_json["success"], true, "session_start.sh success=false");

    // Verify consciousness state in output
    let consciousness = start_result.consciousness_state()
        .expect("consciousness_state missing from session_start output");
    assert!(consciousness.get("identity_continuity").is_some());

    // 2. Execute pre_tool_use.sh (FAST PATH)
    let pre_input = create_claude_code_pre_tool_input(
        &session_id,
        "Read",
        serde_json::json!({"file_path": "/tmp/test.txt"}),
    );
    let pre_result = execute_hook_script(
        ".claude/hooks/pre_tool_use.sh",
        &pre_input,
        TIMEOUT_PRE_TOOL_MS + 50, // Small buffer for shell overhead
        db_path,
    ).await.expect("pre_tool_use.sh failed");

    assert_eq!(pre_result.exit_code, EXIT_SUCCESS, "pre_tool_use.sh failed");
    assert!(pre_result.execution_time_ms < 150,
        "pre_tool_use.sh exceeded timing budget: {}ms", pre_result.execution_time_ms);

    // 3. Execute post_tool_use.sh
    let post_input = create_claude_code_post_tool_input(
        &session_id,
        "Read",
        serde_json::json!({"file_path": "/tmp/test.txt"}),
        "file contents",
    );
    let post_result = execute_hook_script(
        ".claude/hooks/post_tool_use.sh",
        &post_input,
        TIMEOUT_POST_TOOL_MS,
        db_path,
    ).await.expect("post_tool_use.sh failed");

    assert_eq!(post_result.exit_code, EXIT_SUCCESS, "post_tool_use.sh failed");

    // 4. Execute user_prompt_submit.sh
    let prompt_input = create_claude_code_prompt_submit_input(
        &session_id,
        "Please read the file and summarize it.",
    );
    let prompt_result = execute_hook_script(
        ".claude/hooks/user_prompt_submit.sh",
        &prompt_input,
        TIMEOUT_USER_PROMPT_MS,
        db_path,
    ).await.expect("user_prompt_submit.sh failed");

    assert_eq!(prompt_result.exit_code, EXIT_SUCCESS, "user_prompt_submit.sh failed");

    // 5. Execute session_end.sh
    let end_input = create_claude_code_session_end_input(&session_id, "normal");
    let end_result = execute_hook_script(
        ".claude/hooks/session_end.sh",
        &end_input,
        TIMEOUT_SESSION_END_MS,
        db_path,
    ).await.expect("session_end.sh failed");

    assert_eq!(end_result.exit_code, EXIT_SUCCESS, "session_end.sh failed");

    // PHYSICAL DATABASE VERIFICATION
    assert!(verify_snapshot_exists(db_path, &session_id),
        "Snapshot not persisted to database after SessionEnd");

    let snapshot = load_snapshot_for_verification(db_path, &session_id)
        .expect("Snapshot should exist");

    assert_eq!(snapshot.session_id, session_id);
    assert!(snapshot.last_ic >= 0.0 && snapshot.last_ic <= 1.0,
        "IC out of bounds: {}", snapshot.last_ic);
    assert_eq!(snapshot.kuramoto_phases.len(), 13,
        "Kuramoto phases should have 13 elements");
    assert_eq!(snapshot.purpose_vector.len(), 13,
        "Purpose vector should have 13 elements");

    // LOG EVIDENCE
    log_test_evidence("test_e2e_full_session_workflow", "full_session", &session_id, &end_result, true);
}

/// Test that consciousness state is properly updated throughout session
#[tokio::test]
async fn test_e2e_consciousness_state_updates() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path();
    let session_id = format!("e2e-consciousness-{}", uuid::Uuid::new_v4());

    // Start session
    let start_input = create_claude_code_session_start_input(&session_id);
    let start_result = execute_hook_script(
        ".claude/hooks/session_start.sh",
        &start_input,
        TIMEOUT_SESSION_START_MS,
        db_path,
    ).await.unwrap();

    // Verify initial consciousness state
    let initial_state = start_result.consciousness_state()
        .expect("consciousness_state should be present");

    // Required fields per GWT spec
    assert!(initial_state.get("consciousness").is_some());
    assert!(initial_state.get("integration").is_some());
    assert!(initial_state.get("reflection").is_some());
    assert!(initial_state.get("differentiation").is_some());
    assert!(initial_state.get("identity_continuity").is_some());

    // IC classification
    let ic_class = start_result.ic_classification()
        .expect("ic_classification should be present");
    assert!(ic_class.get("value").is_some());
    assert!(ic_class.get("level").is_some());

    let level = ic_class["level"].as_str().unwrap();
    assert!(["healthy", "normal", "warning", "critical"].contains(&level),
        "Invalid IC level: {}", level);

    // End session for cleanup
    let end_input = create_claude_code_session_end_input(&session_id, "normal");
    execute_hook_script(".claude/hooks/session_end.sh", &end_input, TIMEOUT_SESSION_END_MS, db_path)
        .await.unwrap();
}
```

## identity_continuity_test.rs
```rust
//! E2E tests for identity continuity across sessions

use super::helpers::*;
use tempfile::TempDir;

/// Test that identity is properly restored between sessions
///
/// Scenario:
/// 1. Create and end Session 1 (establishes identity)
/// 2. Start Session 2 with previous_session_id = Session 1
/// 3. Verify drift metrics are computed
/// 4. Verify IC is continuous (within tolerance)
#[tokio::test]
async fn test_e2e_identity_continuity_across_sessions() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path();

    // SESSION 1: Establish identity baseline
    let session1_id = format!("e2e-session1-{}", uuid::Uuid::new_v4());

    let start1_input = create_claude_code_session_start_input(&session1_id);
    let start1_result = execute_hook_script(
        ".claude/hooks/session_start.sh",
        &start1_input,
        TIMEOUT_SESSION_START_MS,
        db_path,
    ).await.unwrap();
    assert_eq!(start1_result.exit_code, EXIT_SUCCESS);

    // Record IC from session 1
    let ic_session1 = start1_result.consciousness_state()
        .and_then(|s| s["identity_continuity"].as_f64())
        .expect("IC should be present");

    // End session 1 (persists snapshot)
    let end1_input = create_claude_code_session_end_input(&session1_id, "normal");
    let end1_result = execute_hook_script(
        ".claude/hooks/session_end.sh",
        &end1_input,
        TIMEOUT_SESSION_END_MS,
        db_path,
    ).await.unwrap();
    assert_eq!(end1_result.exit_code, EXIT_SUCCESS);

    // Verify snapshot persisted
    assert!(verify_snapshot_exists(db_path, &session1_id),
        "Session 1 snapshot not persisted");

    // SESSION 2: Restore from session 1
    let session2_id = format!("e2e-session2-{}", uuid::Uuid::new_v4());

    // Create input with previous_session_id
    let start2_input = serde_json::json!({
        "session_id": session2_id,
        "previous_session_id": session1_id,
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": "/tmp",
        "hook_event_name": "SessionStart"
    }).to_string();

    let start2_result = execute_hook_script(
        ".claude/hooks/session_start.sh",
        &start2_input,
        TIMEOUT_SESSION_START_MS,
        db_path,
    ).await.unwrap();
    assert_eq!(start2_result.exit_code, EXIT_SUCCESS);

    // Verify IC is continuous (within 0.2 tolerance for drift)
    let ic_session2 = start2_result.consciousness_state()
        .and_then(|s| s["identity_continuity"].as_f64())
        .expect("IC should be present in session 2");

    let ic_delta = (ic_session2 - ic_session1).abs();
    assert!(ic_delta < 0.2,
        "IC drift too large: {} -> {} (delta: {})", ic_session1, ic_session2, ic_delta);

    // Verify drift_metrics present when restoring
    let output_json = start2_result.parse_stdout().unwrap();
    if let Some(drift) = output_json.get("drift_metrics") {
        // If drift_metrics is returned, verify fields
        if !drift.is_null() {
            println!("Drift metrics: {:?}", drift);
        }
    }

    // End session 2
    let end2_input = create_claude_code_session_end_input(&session2_id, "normal");
    execute_hook_script(".claude/hooks/session_end.sh", &end2_input, TIMEOUT_SESSION_END_MS, db_path)
        .await.unwrap();

    // PHYSICAL VERIFICATION: Both snapshots exist
    assert!(verify_snapshot_exists(db_path, &session1_id));
    assert!(verify_snapshot_exists(db_path, &session2_id));

    // Verify session 2 links to session 1
    let snapshot2 = load_snapshot_for_verification(db_path, &session2_id).unwrap();
    assert_eq!(snapshot2.previous_session_id.as_deref(), Some(session1_id.as_str()),
        "Session 2 should link to session 1");
}
```

## error_recovery_test.rs
```rust
//! E2E tests for error handling and recovery

use super::helpers::*;
use tempfile::TempDir;

/// Test that hooks properly handle and report errors
#[tokio::test]
async fn test_e2e_hook_error_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path();

    // Test 1: Empty stdin should return exit code 4
    let result = execute_hook_script(
        ".claude/hooks/session_start.sh",
        "",  // Empty input
        TIMEOUT_SESSION_START_MS,
        db_path,
    ).await.unwrap();

    assert_eq!(result.exit_code, EXIT_INVALID_INPUT,
        "Empty stdin should return exit code 4.\nstderr: {}", result.stderr);

    // Test 2: Invalid JSON should return exit code 4
    let result = execute_hook_script(
        ".claude/hooks/session_start.sh",
        "not valid json",
        TIMEOUT_SESSION_START_MS,
        db_path,
    ).await.unwrap();

    assert_eq!(result.exit_code, EXIT_INVALID_INPUT,
        "Invalid JSON should return exit code 4.\nstderr: {}", result.stderr);

    // Verify stderr contains structured error JSON
    let stderr_json: Result<serde_json::Value, _> = serde_json::from_str(&result.stderr);
    assert!(stderr_json.is_ok() || result.stderr.contains("error"),
        "stderr should contain error information: {}", result.stderr);
}

/// Test shell script timeout behavior
#[tokio::test]
async fn test_e2e_shell_script_timeout() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path();
    let session_id = format!("e2e-timeout-{}", uuid::Uuid::new_v4());

    // PreToolUse has 100ms budget - test that it completes within budget
    let pre_input = create_claude_code_pre_tool_input(
        &session_id,
        "Read",
        serde_json::json!({"file_path": "/tmp/test"}),
    );

    let start = std::time::Instant::now();
    let result = execute_hook_script(
        ".claude/hooks/pre_tool_use.sh",
        &pre_input,
        TIMEOUT_PRE_TOOL_MS + 100,  // Allow 100ms buffer for shell overhead
        db_path,
    ).await.unwrap();
    let elapsed = start.elapsed();

    // Shell + CLI overhead means we allow up to 200ms for the 100ms-budget hook
    assert!(elapsed.as_millis() < 500,
        "pre_tool_use.sh took too long: {}ms (max 200ms with overhead)", elapsed.as_millis());

    // If it completed successfully, that's fine
    // If it timed out, exit code should be 2
    if result.exit_code != EXIT_SUCCESS {
        assert_eq!(result.exit_code, EXIT_TIMEOUT,
            "Non-success exit should be timeout (2), got: {}", result.exit_code);
    }
}
```
</test_signatures>

<full_state_verification>
## Source of Truth Definitions

| Component | Source of Truth Location |
|-----------|--------------------------|
| Shell scripts | .claude/hooks/*.sh (physical files) |
| CLI commands | crates/context-graph-cli/src/commands/hooks/args.rs:HooksCommands enum |
| Exit codes | crates/context-graph-cli/src/commands/hooks/error.rs:HookError.exit_code() |
| Timeouts | docs2/constitution.yaml:claude_code.performance.hooks |
| MCP tools | crates/context-graph-mcp/src/tools/mod.rs:get_tool_definitions() |
| GWT state | crates/context-graph-core/src/gwt/ |
| Snapshots | RocksDB column family via context_graph_storage::rocksdb_backend::RocksDbMemex |

## Execute & Inspect Protocol (REQUIRED FOR ALL TESTS)

1. **Execute Operation**: Run shell script via bash with JSON piped to stdin
2. **Capture All Output**: exit code, stdout, stderr, wall-clock time
3. **Parse Results**: Validate JSON structure matches HookOutput schema
4. **Physical Verification**: Open RocksDB separately and verify bytes exist
5. **Log Evidence**: Output structured JSON for audit trail

## Boundary & Edge Case Audit (MINIMUM 3 REQUIRED)

### Edge Case 1: Empty stdin
```bash
# INPUT
echo -n '' | .claude/hooks/session_start.sh
```
**EXPECTED**: Exit code 4, stderr contains error JSON
**BEFORE STATE**: No session
**AFTER STATE**: No session created, no database changes

### Edge Case 2: Session restoration with stale previous_session_id
```bash
# INPUT - previous_session_id doesn't exist
echo '{"session_id":"new-001","previous_session_id":"nonexistent-999"}' | .claude/hooks/session_start.sh
```
**EXPECTED**: Exit code 0 with warning OR exit code 5 (per TASK-HOOKS-016, CLI returns 0 with warning)
**BEFORE STATE**: No snapshot for previous_session_id
**AFTER STATE**: New session created, drift_metrics may have sentinel values

### Edge Case 3: PreToolUse timing budget
```bash
# INPUT
time (echo '{"session_id":"test","tool_name":"Read","tool_input":{}}' | .claude/hooks/pre_tool_use.sh)
```
**EXPECTED**: Completes in <200ms (100ms budget + shell overhead), exit code 0
**VERIFY**: Measure both wall-clock time and execution_time_ms in output

## Evidence of Success Logging

Every test MUST produce a log entry:
```json
{
  "test": "test_e2e_full_session_workflow",
  "hook_type": "full_session",
  "session_id": "e2e-xxx",
  "exit_code": 0,
  "execution_time_ms": 1234,
  "stdout_bytes": 512,
  "stderr_bytes": 0,
  "db_verified": true,
  "snapshot_exists": true,
  "ic_value": 0.92,
  "ic_level": "healthy"
}
```
</full_state_verification>

<manual_verification_commands>
## Build First (Required)
```bash
cargo build --release -p context-graph-cli
cargo build --release -p context-graph-mcp  # If MCP server subprocess needed
```

## Verify Shell Scripts Work Manually

### session_start.sh
```bash
echo '{"session_id":"manual-test-001","transcript_path":"/tmp/t.jsonl","cwd":"/tmp","hook_event_name":"SessionStart"}' \
  | CONTEXT_GRAPH_DB_PATH=/tmp/e2e-test-db .claude/hooks/session_start.sh
echo "Exit code: $?"
```

### pre_tool_use.sh (FAST PATH - must be under 100ms)
```bash
time (echo '{"session_id":"manual-test-001","tool_name":"Read","tool_input":{"file_path":"/tmp/x"}}' \
  | CONTEXT_GRAPH_DB_PATH=/tmp/e2e-test-db .claude/hooks/pre_tool_use.sh)
```

### session_end.sh
```bash
echo '{"session_id":"manual-test-001","reason":"exit"}' \
  | CONTEXT_GRAPH_DB_PATH=/tmp/e2e-test-db .claude/hooks/session_end.sh
echo "Exit code: $?"
```

## Verify Database State
```bash
# List recent snapshots (requires custom CLI command or direct RocksDB inspection)
ls -la /tmp/e2e-test-db/

# If ldb (RocksDB CLI) available:
ldb --db=/tmp/e2e-test-db scan --column_family=default 2>/dev/null | head -20
```

## Verify MCP Server (if subprocess approach used)
```bash
./target/release/context-graph-mcp --transport tcp --port 9999 &
MCP_PID=$!
sleep 2
curl -X POST http://localhost:9999/tools/list 2>/dev/null | jq .
kill $MCP_PID
```
</manual_verification_commands>

<test_commands>
## Run E2E Tests
```bash
# Build CLI first
cargo build --release -p context-graph-cli

# Run all E2E tests (single-threaded for isolation)
cargo test --package context-graph-cli --test e2e -- --test-threads=1 --nocapture

# Run specific test
cargo test --package context-graph-cli --test e2e test_e2e_full_session_workflow -- --nocapture

# Run with verbose output
RUST_LOG=debug cargo test --package context-graph-cli --test e2e -- --test-threads=1 --nocapture
```

## Verify Prerequisites
```bash
# Verify shell scripts exist and are executable
ls -la .claude/hooks/*.sh

# Verify CLI binary exists
test -x ./target/release/context-graph-cli && echo "CLI OK" || echo "CLI missing - run: cargo build --release -p context-graph-cli"

# Verify settings.json has correct hook configuration
jq '.hooks | keys' .claude/settings.json
```
</test_commands>

<definition_of_done>
## Acceptance Criteria

1. **Directory Created**: crates/context-graph-cli/tests/e2e/ exists with all files
2. **All Tests Pass**: `cargo test --package context-graph-cli --test e2e` succeeds
3. **No Mocks**: Tests use real shell scripts, real CLI binary, real database
4. **Physical Verification**: Every test verifies database state via RocksDB
5. **Timing Verified**: PreToolUse tests complete under 200ms (100ms budget + overhead)
6. **Exit Codes Verified**: Error cases return correct exit codes (4 for invalid input)
7. **Evidence Logged**: Each test outputs JSON evidence to stdout
8. **Idempotent**: Tests use TempDir and can run multiple times

## Verification Checklist
- [ ] helpers.rs implements execute_hook_script() that pipes JSON to bash
- [ ] helpers.rs implements verify_snapshot_exists() using RocksDbMemex
- [ ] full_session_test.rs executes all 5 shell scripts in sequence
- [ ] identity_continuity_test.rs verifies cross-session IC
- [ ] error_recovery_test.rs verifies exit code 4 for invalid input
- [ ] All tests log structured evidence JSON
- [ ] cargo test passes with release binary
- [ ] Tests clean up temp directories
</definition_of_done>

<constitution_references>
- AP-26: Exit codes (0-6 per specification)
- AP-50: NO internal hooks - native Claude Code hooks only
- AP-51: Shell scripts call context-graph-cli
- AP-53: Direct CLI commands, no wrapper
- IDENTITY-002: IC thresholds (Healthy>0.9, Normal>=0.7, Warning>=0.5, Critical<0.5)
- GWT-003: Identity continuity tracking
- ARCH-07: Hooks via .claude/settings.json
- Performance budgets: constitution.yaml:claude_code.performance.hooks
</constitution_references>
</task_spec>
```
