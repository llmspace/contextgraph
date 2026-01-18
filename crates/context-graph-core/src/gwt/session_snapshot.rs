//! Session snapshot types for hook handlers.
//!
//! TASK-SESSION-04: SessionSnapshot persistence for cross-session continuity.
//!
//! # Constitution References
//! - ARCH-07: Native Claude Code hooks
//! - AP-50: NO internal hooks - shell scripts call CLI
//!
//! This module focuses on session state management for topic stability tracking.
//! See `clustering/stability.rs` for churn-based stability tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Number of embedder dimensions for purpose vector.
pub const NUM_EMBEDDERS: usize = 13;

/// Maximum trajectory history size.
pub const MAX_TRAJECTORY_SIZE: usize = 100;

/// Global singleton cache for session state.
static GLOBAL_CACHE: OnceLock<SessionCache> = OnceLock::new();

/// Type alias for session manager trait compatibility.
pub type SessionManager = SessionCache;

/// Session snapshot for persistence.
///
/// Stores the essential state needed for cross-session topic continuity.
/// Serialized to/from RocksDB via bincode.
///
/// # Fields
/// - `session_id`: Unique session identifier
/// - `purpose_vector`: 13D teleological alignment
/// - `trajectory`: History of purpose vectors
/// - `created_at`: Unix timestamp of snapshot creation
/// - `updated_at`: Unix timestamp of last update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Unique session identifier.
    pub session_id: String,

    /// Integration metric [0.0, 1.0].
    /// Measures how well different parts of the system are integrated.
    pub integration: f32,

    /// Reflection metric [0.0, 1.0].
    /// Measures self-awareness and meta-cognitive capacity.
    pub reflection: f32,

    /// Differentiation metric [0.0, 1.0].
    /// Measures distinctiveness of concepts and memories.
    pub differentiation: f32,

    /// Purpose vector (13D teleological alignment).
    pub purpose_vector: [f32; NUM_EMBEDDERS],

    /// Previous session ID for continuity tracking.
    pub previous_session_id: Option<String>,

    /// Trajectory history of purpose vectors.
    pub trajectory: Vec<[f32; NUM_EMBEDDERS]>,

    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,

    /// Unix timestamp of snapshot creation.
    pub created_at: u64,

    /// Unix timestamp of last update.
    pub updated_at: u64,
}

impl SessionSnapshot {
    /// Create a new session snapshot.
    ///
    /// Initializes with default values for topic-based stability tracking.
    pub fn new(session_id: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            session_id: session_id.to_string(),
            integration: 0.0,
            reflection: 0.0,
            differentiation: 0.0,
            purpose_vector: [0.5; NUM_EMBEDDERS],
            previous_session_id: None,
            trajectory: Vec::new(),
            timestamp_ms: now_ms,
            created_at: now,
            updated_at: now,
        }
    }

    /// Append a purpose vector to the trajectory history.
    pub fn append_to_trajectory(&mut self, pv: [f32; NUM_EMBEDDERS]) {
        if self.trajectory.len() >= MAX_TRAJECTORY_SIZE {
            self.trajectory.remove(0);
        }
        self.trajectory.push(pv);
    }

    /// Update the timestamp.
    pub fn touch(&mut self) {
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(self.updated_at);
    }
}

impl Default for SessionSnapshot {
    fn default() -> Self {
        Self::new("default")
    }
}

/// Get the global session cache singleton.
fn global_cache() -> &'static SessionCache {
    GLOBAL_CACHE.get_or_init(SessionCache::new)
}

/// Store a snapshot in the global cache and set it as current.
///
/// This function stores the snapshot in the global cache.
///
/// # Arguments
/// - `snapshot`: The session snapshot to cache
pub fn store_in_cache(snapshot: &SessionSnapshot) {
    let cache = global_cache();
    let mut updated_snapshot = snapshot.clone();
    updated_snapshot.touch();
    cache.store(updated_snapshot);
    cache.set_current(&snapshot.session_id);
}

/// In-memory cache for session snapshots.
///
/// Provides fast lookups during session execution, with persistence
/// to RocksDB at session end.
#[derive(Debug)]
pub struct SessionCache {
    /// Cached snapshots by session ID.
    cache: RwLock<HashMap<String, SessionSnapshot>>,

    /// Current active session ID.
    current_session: RwLock<Option<String>>,

    /// Maximum cache entries before eviction.
    max_entries: usize,
}

impl SessionCache {
    /// Create a new session cache.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            current_session: RwLock::new(None),
            max_entries: 100,
        }
    }

    /// Create with custom max entries.
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::with_capacity(max_entries)),
            current_session: RwLock::new(None),
            max_entries,
        }
    }

    /// Get current session snapshot from the global cache.
    ///
    /// Returns the current session snapshot if one exists.
    /// This is a static method that accesses the global cache singleton.
    pub fn get() -> Option<SessionSnapshot> {
        let cache = global_cache();
        let current_session = cache.current()?;
        cache.get_by_id(&current_session)
    }

    /// Check if the global cache has been initialized with a session.
    pub fn is_warm() -> bool {
        global_cache().current().is_some()
    }

    /// Get a snapshot from cache by session ID.
    pub fn get_by_id(&self, session_id: &str) -> Option<SessionSnapshot> {
        self.cache
            .read()
            .ok()
            .and_then(|cache| cache.get(session_id).cloned())
    }

    /// Store a snapshot in cache.
    pub fn store(&self, snapshot: SessionSnapshot) {
        if let Ok(mut cache) = self.cache.write() {
            // Evict oldest if at capacity
            if cache.len() >= self.max_entries {
                // Simple eviction: remove first entry (not optimal but simple)
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }
            cache.insert(snapshot.session_id.clone(), snapshot);
        }
    }

    /// Set the current active session.
    pub fn set_current(&self, session_id: &str) {
        if let Ok(mut current) = self.current_session.write() {
            *current = Some(session_id.to_string());
        }
    }

    /// Get the current active session ID.
    pub fn current(&self) -> Option<String> {
        self.current_session.read().ok().and_then(|c| c.clone())
    }

    /// Clear the cache.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get all cached session IDs.
    pub fn sessions(&self) -> Vec<String> {
        self.cache
            .read()
            .map(|cache| cache.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get cache size.
    pub fn len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SessionCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_creation() {
        let snapshot = SessionSnapshot::new("test-session");
        assert_eq!(snapshot.session_id, "test-session");
        assert!(snapshot.created_at > 0);
    }

    #[test]
    fn test_snapshot_trajectory() {
        let mut snapshot = SessionSnapshot::new("test-session");
        assert!(snapshot.trajectory.is_empty());

        let pv = [0.1; NUM_EMBEDDERS];
        snapshot.append_to_trajectory(pv);
        assert_eq!(snapshot.trajectory.len(), 1);

        // Test max trajectory size
        for i in 0..MAX_TRAJECTORY_SIZE {
            snapshot.append_to_trajectory([i as f32 / 100.0; NUM_EMBEDDERS]);
        }
        assert_eq!(snapshot.trajectory.len(), MAX_TRAJECTORY_SIZE);
    }

    #[test]
    fn test_cache_store_and_get() {
        let cache = SessionCache::new();
        let snapshot = SessionSnapshot::new("test-session");

        cache.store(snapshot.clone());

        let retrieved = cache.get_by_id("test-session");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().session_id, "test-session");
    }

    #[test]
    fn test_cache_current_session() {
        let cache = SessionCache::new();

        assert!(cache.current().is_none());

        cache.set_current("active-session");
        assert_eq!(cache.current(), Some("active-session".to_string()));
    }

    #[test]
    fn test_snapshot_touch() {
        let mut snapshot = SessionSnapshot::new("test-session");
        let original_updated = snapshot.updated_at;

        // Sleep briefly to ensure time difference
        std::thread::sleep(std::time::Duration::from_millis(10));

        snapshot.touch();
        assert!(snapshot.updated_at >= original_updated);
    }

    #[test]
    fn test_snapshot_default() {
        let snapshot = SessionSnapshot::default();
        assert_eq!(snapshot.session_id, "default");
    }
}
