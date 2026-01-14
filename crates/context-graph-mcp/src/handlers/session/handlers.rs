//! Session tool handlers for MCP lifecycle hooks.
//!
//! TASK-014: Implements the 4 session handlers per ARCH-07.
//!
//! # Constitution Compliance
//!
//! - ARCH-07: Hooks control memory lifecycle - SessionStart/PreToolUse/PostToolUse/SessionEnd

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use super::{SessionError, SessionManager};
use crate::protocol::error_codes;

// ============================================================================
// Parameter Structs
// ============================================================================

/// Parameters for session_start tool.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionStartParams {
    /// Optional session ID. If not provided, a UUID will be generated.
    pub session_id: Option<String>,
    /// Session TTL in minutes (default: 30).
    pub ttl_minutes: Option<i64>,
    /// Optional metadata to attach to session.
    pub metadata: Option<Value>,
}

/// Parameters for session_end tool.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionEndParams {
    /// Session ID to terminate.
    pub session_id: String,
}

/// Parameters for pre_tool_use hook.
#[derive(Debug, Clone, Deserialize)]
pub struct PreToolUseParams {
    /// Active session ID.
    pub session_id: String,
    /// Name of tool about to execute.
    pub tool_name: String,
    /// Tool arguments (for context logging).
    pub arguments: Option<Value>,
}

/// Parameters for post_tool_use hook.
#[derive(Debug, Clone, Deserialize)]
pub struct PostToolUseParams {
    /// Active session ID.
    pub session_id: String,
    /// Name of tool that executed.
    pub tool_name: String,
    /// Whether tool execution succeeded.
    pub success: bool,
    /// Execution duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// Brief summary of tool result.
    pub result_summary: Option<String>,
}

// ============================================================================
// Response Structs
// ============================================================================

/// Response for session_start.
#[derive(Debug, Clone, Serialize)]
pub struct SessionStartResponse {
    pub session_id: String,
    pub created_at: String,
    pub expires_at: String,
    pub ttl_minutes: i64,
}

/// Response for session_end.
#[derive(Debug, Clone, Serialize)]
pub struct SessionEndResponse {
    pub session_id: String,
    pub duration_seconds: i64,
    pub tool_count: u64,
    pub status: String,
}

/// Response for pre_tool_use.
#[derive(Debug, Clone, Serialize)]
pub struct PreToolUseResponse {
    pub session_id: String,
    pub tool_name: String,
    pub tool_count: u64,
    pub status: String,
}

/// Response for post_tool_use.
#[derive(Debug, Clone, Serialize)]
pub struct PostToolUseResponse {
    pub session_id: String,
    pub tool_name: String,
    pub success: bool,
    pub status: String,
}

// ============================================================================
// Handler Result Type
// ============================================================================

/// Result type for session handlers.
pub type SessionHandlerResult = Result<Value, (i32, String)>;

// ============================================================================
// Handler Implementations
// ============================================================================

/// Handle session_start tool call.
///
/// Creates a new MCP session. If session_id is not provided, generates a UUID.
/// Returns SESSION_EXISTS error if session already exists.
///
/// # TASK-014
pub fn call_session_start(
    manager: &SessionManager,
    params: SessionStartParams,
) -> SessionHandlerResult {
    // Generate session_id if not provided
    let session_id = params.session_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    tracing::info!(
        session_id = %session_id,
        ttl_minutes = ?params.ttl_minutes,
        "session_start: Creating session"
    );

    match manager.create_session(&session_id) {
        Ok(session) => {
            let response = SessionStartResponse {
                session_id: session.id.clone(),
                created_at: session.created_at.to_rfc3339(),
                expires_at: session.expires_at.to_rfc3339(),
                ttl_minutes: params.ttl_minutes.unwrap_or(30),
            };

            tracing::info!(
                session_id = %session.id,
                expires_at = %session.expires_at,
                "session_start: Session created successfully"
            );

            Ok(json!(response))
        }
        Err(SessionError::SessionExists(id)) => {
            tracing::warn!(
                session_id = %id,
                "session_start: Session already exists"
            );
            Err((
                error_codes::SESSION_EXISTS,
                format!("Session already exists: {}", id),
            ))
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "session_start: Unexpected error"
            );
            Err((
                error_codes::INTERNAL_ERROR,
                format!("Session creation failed: {}", e),
            ))
        }
    }
}

/// Handle session_end tool call.
///
/// Terminates an MCP session and returns summary.
/// Returns SESSION_NOT_FOUND if session doesn't exist.
///
/// # TASK-014
pub fn call_session_end(
    manager: &SessionManager,
    params: SessionEndParams,
) -> SessionHandlerResult {
    tracing::info!(
        session_id = %params.session_id,
        "session_end: Ending session"
    );

    match manager.end_session(&params.session_id) {
        Ok(session) => {
            let duration = chrono::Utc::now() - session.created_at;
            let response = SessionEndResponse {
                session_id: session.id.clone(),
                duration_seconds: duration.num_seconds(),
                tool_count: session.tool_count,
                status: "ended".to_string(),
            };

            tracing::info!(
                session_id = %session.id,
                duration_seconds = duration.num_seconds(),
                tool_count = session.tool_count,
                "session_end: Session ended successfully"
            );

            Ok(json!(response))
        }
        Err(SessionError::SessionNotFound(id)) => {
            tracing::warn!(
                session_id = %id,
                "session_end: Session not found"
            );
            Err((
                error_codes::SESSION_NOT_FOUND,
                format!("Session not found: {}", id),
            ))
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "session_end: Unexpected error"
            );
            Err((
                error_codes::INTERNAL_ERROR,
                format!("Session end failed: {}", e),
            ))
        }
    }
}

/// Handle pre_tool_use hook.
///
/// Records tool invocation in session state.
/// Returns SESSION_NOT_FOUND or SESSION_EXPIRED errors.
///
/// # TASK-014
pub fn call_pre_tool_use(
    manager: &SessionManager,
    params: PreToolUseParams,
) -> SessionHandlerResult {
    tracing::debug!(
        session_id = %params.session_id,
        tool_name = %params.tool_name,
        "pre_tool_use: Recording tool invocation"
    );

    match manager.pre_tool_use(&params.session_id, &params.tool_name) {
        Ok(()) => {
            // Get updated session for response
            match manager.get_session(&params.session_id) {
                Ok(session) => {
                    let response = PreToolUseResponse {
                        session_id: session.id,
                        tool_name: params.tool_name,
                        tool_count: session.tool_count,
                        status: "recorded".to_string(),
                    };
                    Ok(json!(response))
                }
                Err(_) => {
                    // Session was found during pre_tool_use but not now - race condition
                    let response = PreToolUseResponse {
                        session_id: params.session_id,
                        tool_name: params.tool_name,
                        tool_count: 0,
                        status: "recorded".to_string(),
                    };
                    Ok(json!(response))
                }
            }
        }
        Err(SessionError::SessionNotFound(id)) => {
            tracing::warn!(
                session_id = %id,
                "pre_tool_use: Session not found"
            );
            Err((
                error_codes::SESSION_NOT_FOUND,
                format!("Session not found: {}", id),
            ))
        }
        Err(SessionError::SessionExpired(id, expired_at)) => {
            tracing::warn!(
                session_id = %id,
                expired_at = %expired_at,
                "pre_tool_use: Session expired"
            );
            Err((
                error_codes::SESSION_EXPIRED,
                format!("Session expired: {}, at: {}", id, expired_at),
            ))
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "pre_tool_use: Unexpected error"
            );
            Err((
                error_codes::INTERNAL_ERROR,
                format!("Pre-tool use failed: {}", e),
            ))
        }
    }
}

/// Handle post_tool_use hook.
///
/// Records tool completion in session state.
/// Returns SESSION_NOT_FOUND error if session doesn't exist.
///
/// # TASK-014
pub fn call_post_tool_use(
    manager: &SessionManager,
    params: PostToolUseParams,
) -> SessionHandlerResult {
    tracing::debug!(
        session_id = %params.session_id,
        tool_name = %params.tool_name,
        success = params.success,
        duration_ms = ?params.duration_ms,
        "post_tool_use: Recording tool completion"
    );

    let duration_ms = params.duration_ms.unwrap_or(0);

    match manager.post_tool_use(&params.session_id, &params.tool_name, params.success, duration_ms) {
        Ok(()) => {
            let response = PostToolUseResponse {
                session_id: params.session_id,
                tool_name: params.tool_name,
                success: params.success,
                status: "recorded".to_string(),
            };
            Ok(json!(response))
        }
        Err(SessionError::SessionNotFound(id)) => {
            tracing::warn!(
                session_id = %id,
                "post_tool_use: Session not found"
            );
            Err((
                error_codes::SESSION_NOT_FOUND,
                format!("Session not found: {}", id),
            ))
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "post_tool_use: Unexpected error"
            );
            Err((
                error_codes::INTERNAL_ERROR,
                format!("Post-tool use failed: {}", e),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_start_generates_uuid() {
        let manager = SessionManager::new();
        let params = SessionStartParams {
            session_id: None,
            ttl_minutes: None,
            metadata: None,
        };

        let result = call_session_start(&manager, params);
        assert!(result.is_ok());

        let value = result.unwrap();
        let session_id = value["session_id"].as_str().unwrap();
        // UUID format check
        assert!(Uuid::parse_str(session_id).is_ok());
    }

    #[test]
    fn test_session_start_uses_provided_id() {
        let manager = SessionManager::new();
        let params = SessionStartParams {
            session_id: Some("my-session-123".to_string()),
            ttl_minutes: Some(60),
            metadata: None,
        };

        let result = call_session_start(&manager, params);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["session_id"].as_str().unwrap(), "my-session-123");
    }

    #[test]
    fn test_session_start_duplicate_error() {
        let manager = SessionManager::new();
        let params = SessionStartParams {
            session_id: Some("dup-session".to_string()),
            ttl_minutes: None,
            metadata: None,
        };

        // First call succeeds
        let result1 = call_session_start(&manager, params.clone());
        assert!(result1.is_ok());

        // Second call fails with SESSION_EXISTS
        let result2 = call_session_start(&manager, params);
        assert!(result2.is_err());
        let (code, _msg) = result2.unwrap_err();
        assert_eq!(code, error_codes::SESSION_EXISTS);
    }

    #[test]
    fn test_session_end_not_found() {
        let manager = SessionManager::new();
        let params = SessionEndParams {
            session_id: "nonexistent".to_string(),
        };

        let result = call_session_end(&manager, params);
        assert!(result.is_err());
        let (code, _msg) = result.unwrap_err();
        assert_eq!(code, error_codes::SESSION_NOT_FOUND);
    }

    #[test]
    fn test_pre_tool_use_increments_count() {
        let manager = SessionManager::new();

        // Create session
        let start_params = SessionStartParams {
            session_id: Some("tool-test".to_string()),
            ttl_minutes: None,
            metadata: None,
        };
        call_session_start(&manager, start_params).unwrap();

        // Pre-tool use
        let pre_params = PreToolUseParams {
            session_id: "tool-test".to_string(),
            tool_name: "inject_context".to_string(),
            arguments: None,
        };
        let result = call_pre_tool_use(&manager, pre_params);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["tool_count"].as_u64().unwrap(), 1);
    }

    #[test]
    fn test_post_tool_use_records_result() {
        let manager = SessionManager::new();

        // Create session
        let start_params = SessionStartParams {
            session_id: Some("post-test".to_string()),
            ttl_minutes: None,
            metadata: None,
        };
        call_session_start(&manager, start_params).unwrap();

        // Post-tool use
        let post_params = PostToolUseParams {
            session_id: "post-test".to_string(),
            tool_name: "inject_context".to_string(),
            success: true,
            duration_ms: Some(25),
            result_summary: Some("Injected 5 nodes".to_string()),
        };
        let result = call_post_tool_use(&manager, post_params);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["success"].as_bool().unwrap(), true);
        assert_eq!(value["status"].as_str().unwrap(), "recorded");
    }

    #[test]
    fn test_full_session_lifecycle_fsv() {
        println!("\n=== FSV TEST: Full Session Lifecycle (TASK-014) ===");

        let manager = SessionManager::new();

        // 1. Start session
        let start_params = SessionStartParams {
            session_id: Some("fsv-lifecycle".to_string()),
            ttl_minutes: Some(30),
            metadata: None,
        };
        let start_result = call_session_start(&manager, start_params).unwrap();
        println!("FSV-1: Session started: {}", start_result["session_id"]);

        // 2. Pre-tool use
        let pre_params = PreToolUseParams {
            session_id: "fsv-lifecycle".to_string(),
            tool_name: "inject_context".to_string(),
            arguments: Some(json!({"content": "test"})),
        };
        let pre_result = call_pre_tool_use(&manager, pre_params).unwrap();
        println!("FSV-2: Pre-tool recorded, count: {}", pre_result["tool_count"]);

        // 3. Post-tool use
        let post_params = PostToolUseParams {
            session_id: "fsv-lifecycle".to_string(),
            tool_name: "inject_context".to_string(),
            success: true,
            duration_ms: Some(42),
            result_summary: Some("Success".to_string()),
        };
        let post_result = call_post_tool_use(&manager, post_params).unwrap();
        println!("FSV-3: Post-tool recorded: {}", post_result["status"]);

        // 4. End session
        let end_params = SessionEndParams {
            session_id: "fsv-lifecycle".to_string(),
        };
        let end_result = call_session_end(&manager, end_params).unwrap();
        println!(
            "FSV-4: Session ended, tool_count: {}, duration: {}s",
            end_result["tool_count"], end_result["duration_seconds"]
        );

        // 5. Verify session is gone
        let verify_params = SessionEndParams {
            session_id: "fsv-lifecycle".to_string(),
        };
        let verify_result = call_session_end(&manager, verify_params);
        assert!(verify_result.is_err());
        println!("FSV-5: Session correctly not found after end");

        println!("=== FSV TEST PASSED ===\n");
    }
}
