//! Session management for MCP tool lifecycle.
//!
//! TASK-012: Per ARCH-07: Hooks control memory lifecycle - SessionStart/PreToolUse/PostToolUse/SessionEnd
//!
//! # Constitution Compliance
//!
//! - ARCH-07: Session hooks for MCP lifecycle management
//! - SEC-01: Session expiry and cleanup

// TASK-014: Session handler implementations
pub mod handlers;

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Session state for tracking active MCP sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub tool_count: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Session manager for MCP tool lifecycle.
///
/// Thread-safe via RwLock. Default session TTL: 30 minutes.
#[derive(Debug)]
pub struct SessionManager {
    sessions: RwLock<HashMap<String, Session>>,
    default_ttl: Duration,
}

impl SessionManager {
    /// Create new SessionManager with 30-minute default TTL.
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            default_ttl: Duration::minutes(30),
        }
    }

    /// Create SessionManager with custom TTL.
    #[allow(dead_code)]
    pub fn with_ttl(ttl_minutes: i64) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            default_ttl: Duration::minutes(ttl_minutes),
        }
    }

    /// Create a new session.
    ///
    /// Returns SESSION_EXISTS error if session_id already exists.
    pub fn create_session(&self, session_id: &str) -> Result<Session, SessionError> {
        let mut sessions = self.sessions.write();

        if sessions.contains_key(session_id) {
            tracing::warn!(
                session_id = session_id,
                "SessionManager: Session already exists"
            );
            return Err(SessionError::SessionExists(session_id.to_string()));
        }

        let now = Utc::now();
        let session = Session {
            id: session_id.to_string(),
            created_at: now,
            last_activity: now,
            expires_at: now + self.default_ttl,
            tool_count: 0,
            metadata: HashMap::new(),
        };

        sessions.insert(session_id.to_string(), session.clone());

        tracing::info!(
            session_id = session_id,
            expires_at = %session.expires_at,
            "SessionManager: Session created"
        );

        Ok(session)
    }

    /// Get session by ID.
    ///
    /// Returns SESSION_NOT_FOUND if not exists.
    /// Returns SESSION_EXPIRED if past expires_at.
    pub fn get_session(&self, session_id: &str) -> Result<Session, SessionError> {
        let sessions = self.sessions.read();

        match sessions.get(session_id) {
            Some(session) => {
                if Utc::now() > session.expires_at {
                    Err(SessionError::SessionExpired(
                        session_id.to_string(),
                        session.expires_at,
                    ))
                } else {
                    Ok(session.clone())
                }
            }
            None => Err(SessionError::SessionNotFound(session_id.to_string())),
        }
    }

    /// End a session and return final state.
    pub fn end_session(&self, session_id: &str) -> Result<Session, SessionError> {
        let mut sessions = self.sessions.write();

        match sessions.remove(session_id) {
            Some(session) => {
                tracing::info!(
                    session_id = session_id,
                    tool_count = session.tool_count,
                    duration_secs = (Utc::now() - session.created_at).num_seconds(),
                    "SessionManager: Session ended"
                );
                Ok(session)
            }
            None => Err(SessionError::SessionNotFound(session_id.to_string())),
        }
    }

    /// Record tool use in session (pre_tool_use hook).
    pub fn pre_tool_use(&self, session_id: &str, tool_name: &str) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write();

        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_string()))?;

        if Utc::now() > session.expires_at {
            return Err(SessionError::SessionExpired(
                session_id.to_string(),
                session.expires_at,
            ));
        }

        session.last_activity = Utc::now();
        session.tool_count += 1;
        session
            .metadata
            .insert("last_tool".to_string(), serde_json::json!(tool_name));

        tracing::debug!(
            session_id = session_id,
            tool_name = tool_name,
            tool_count = session.tool_count,
            "SessionManager: Pre-tool use recorded"
        );

        Ok(())
    }

    /// Record tool completion (post_tool_use hook).
    pub fn post_tool_use(
        &self,
        session_id: &str,
        tool_name: &str,
        success: bool,
        duration_ms: u64,
    ) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write();

        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::SessionNotFound(session_id.to_string()))?;

        session.last_activity = Utc::now();
        session.metadata.insert(
            "last_tool_result".to_string(),
            serde_json::json!({
                "tool": tool_name,
                "success": success,
                "duration_ms": duration_ms
            }),
        );

        tracing::debug!(
            session_id = session_id,
            tool_name = tool_name,
            success = success,
            duration_ms = duration_ms,
            "SessionManager: Post-tool use recorded"
        );

        Ok(())
    }

    /// Get count of active sessions.
    #[allow(dead_code)]
    pub fn active_session_count(&self) -> usize {
        let now = Utc::now();
        self.sessions
            .read()
            .values()
            .filter(|s| s.expires_at > now)
            .count()
    }

    /// Cleanup expired sessions.
    #[allow(dead_code)]
    pub fn cleanup_expired(&self) -> usize {
        let now = Utc::now();
        let mut sessions = self.sessions.write();

        let expired: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| s.expires_at < now)
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            sessions.remove(&id);
        }

        if count > 0 {
            tracing::info!(
                expired_count = count,
                "SessionManager: Cleaned up expired sessions"
            );
        }

        count
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Error types for session operations.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Session expired: {0}, expired_at: {1}")]
    SessionExpired(String, DateTime<Utc>),

    #[error("Session already exists: {0}")]
    SessionExists(String),

    #[error("No active session. Call session_start first.")]
    #[allow(dead_code)]
    NoActiveSession,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_lifecycle_fsv() {
        println!("\n=== FSV TEST: Session Lifecycle (REQ-ENDSTATE-001) ===");

        let manager = SessionManager::new();

        // Create session
        let session = manager.create_session("test-001").expect("Should create");
        assert_eq!(session.id, "test-001");
        println!("FSV-1: Session created: {}", session.id);

        // Pre-tool use
        manager
            .pre_tool_use("test-001", "inject_context")
            .expect("Should record");
        println!("FSV-2: Pre-tool use recorded");

        // Verify tool count
        let updated = manager.get_session("test-001").expect("Should exist");
        assert_eq!(updated.tool_count, 1);
        println!("FSV-3: Tool count incremented to {}", updated.tool_count);

        // End session
        let final_session = manager.end_session("test-001").expect("Should end");
        assert_eq!(final_session.tool_count, 1);
        println!(
            "FSV-4: Session ended with tool_count={}",
            final_session.tool_count
        );

        // Verify session removed
        let result = manager.get_session("test-001");
        assert!(matches!(result, Err(SessionError::SessionNotFound(_))));
        println!("FSV-5: Session not found after end (correct)");

        println!("=== FSV TEST PASSED ===\n");
    }

    #[test]
    fn test_session_duplicate_error_fsv() {
        println!("\n=== FSV TEST: Session Duplicate Error ===");

        let manager = SessionManager::new();

        manager.create_session("dup-test").expect("Should create");
        let result = manager.create_session("dup-test");
        assert!(matches!(result, Err(SessionError::SessionExists(_))));
        println!("FSV: Duplicate session correctly rejected");

        println!("=== FSV TEST PASSED ===\n");
    }

    #[test]
    fn test_session_not_found_error_fsv() {
        println!("\n=== FSV TEST: Session Not Found Error ===");

        let manager = SessionManager::new();

        let result = manager.get_session("nonexistent");
        assert!(matches!(result, Err(SessionError::SessionNotFound(_))));
        println!("FSV: Nonexistent session correctly returns NotFound");

        println!("=== FSV TEST PASSED ===\n");
    }

    #[test]
    fn test_post_tool_use_fsv() {
        println!("\n=== FSV TEST: Post Tool Use ===");

        let manager = SessionManager::new();

        manager.create_session("post-test").expect("Should create");
        manager
            .post_tool_use("post-test", "inject_context", true, 25)
            .expect("Should record");

        let session = manager.get_session("post-test").expect("Should exist");
        let result = session
            .metadata
            .get("last_tool_result")
            .expect("Should have last_tool_result");
        assert_eq!(result["tool"], "inject_context");
        assert_eq!(result["success"], true);
        assert_eq!(result["duration_ms"], 25);
        println!("FSV: Post tool use recorded correctly");

        println!("=== FSV TEST PASSED ===\n");
    }
}
