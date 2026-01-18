//! Dream Event Listener
//!
//! Queues exiting memories for dream replay consolidation.
//!
//! # Constitution Compliance (v6.0.0 Topic-Based Architecture)
//!
//! Dream triggers use entropy and churn per Constitution v6.0.0:
//! "Topics emerge from multi-space clustering, no manual goal setting"
//! Dream trigger conditions: entropy > 0.7 AND churn > 0.5

use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::dream::{ExtendedTriggerReason, TriggerManager};
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};

/// Callback type for dream consolidation signaling.
///
/// Invoked when TriggerManager determines a dream cycle should start.
/// The callback receives the trigger reason (e.g., HighEntropy, GpuOverload).
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use context_graph_core::gwt::listeners::DreamConsolidationCallback;
///
/// let callback: DreamConsolidationCallback = Arc::new(|reason| {
///     println!("Dream triggered: {:?}", reason);
/// });
/// ```
pub type DreamConsolidationCallback = Arc<dyn Fn(ExtendedTriggerReason) + Send + Sync>;

/// Listener that queues exiting memories for dream replay.
///
/// # Constitution Compliance (v6.0.0)
///
/// Per Constitution v6.0.0, dreams are triggered by entropy and churn conditions.
/// Topic-based coherence governs workspace state transitions.
///
/// Dream trigger conditions (per Constitution):
/// - entropy > 0.7 for 5+ min
/// - entropy > 0.7 AND churn > 0.5
///
/// # Usage
///
/// ```ignore
/// use std::sync::Arc;
/// use parking_lot::Mutex;
/// use tokio::sync::RwLock;
/// use context_graph_core::dream::TriggerManager;
/// use context_graph_core::gwt::listeners::DreamEventListener;
///
/// let queue = Arc::new(RwLock::new(Vec::new()));
/// let trigger_manager = Arc::new(Mutex::new(TriggerManager::new()));
/// let callback = Arc::new(|reason| println!("Dream: {:?}", reason));
///
/// let listener = DreamEventListener::new(queue, trigger_manager, callback);
/// ```
pub struct DreamEventListener {
    /// Queue for memories exiting workspace (for dream replay)
    dream_queue: Arc<RwLock<Vec<Uuid>>>,

    /// TriggerManager for entropy/churn-based dream triggering
    trigger_manager: Arc<Mutex<TriggerManager>>,

    /// Callback for dream consolidation signaling
    /// Called when TriggerManager returns Some(reason)
    consolidation_callback: DreamConsolidationCallback,
}

impl DreamEventListener {
    /// Create a new dream event listener with all required components.
    ///
    /// # Arguments
    ///
    /// * `dream_queue` - Queue for memories exiting workspace (for dream replay)
    /// * `trigger_manager` - TriggerManager for entropy/churn-based dream triggering
    /// * `consolidation_callback` - Callback invoked when dream cycle should start
    ///
    /// # Example
    ///
    /// ```ignore
    /// let queue = Arc::new(RwLock::new(Vec::new()));
    /// let manager = Arc::new(Mutex::new(TriggerManager::new()));
    /// let callback = Arc::new(|reason| println!("Dream: {:?}", reason));
    ///
    /// let listener = DreamEventListener::new(queue, manager, callback);
    /// ```
    pub fn new(
        dream_queue: Arc<RwLock<Vec<Uuid>>>,
        trigger_manager: Arc<Mutex<TriggerManager>>,
        consolidation_callback: DreamConsolidationCallback,
    ) -> Self {
        Self {
            dream_queue,
            trigger_manager,
            consolidation_callback,
        }
    }

    /// Create a DreamEventListener for testing.
    ///
    /// Provides a no-op callback for tests that don't test dream triggering.
    ///
    /// # Example
    ///
    /// ```ignore
    /// #[cfg(test)]
    /// let listener = DreamEventListener::new_for_testing(queue);
    /// ```
    #[cfg(test)]
    pub fn new_for_testing(dream_queue: Arc<RwLock<Vec<Uuid>>>) -> Self {
        Self {
            dream_queue,
            trigger_manager: Arc::new(Mutex::new(TriggerManager::new())),
            consolidation_callback: Arc::new(|_reason| {
                // No-op callback for testing
            }),
        }
    }

    /// Get a clone of the dream queue arc for external access.
    ///
    /// Use this to inspect or drain the queue after events are processed.
    pub fn queue(&self) -> Arc<RwLock<Vec<Uuid>>> {
        Arc::clone(&self.dream_queue)
    }

    /// Check dream triggers based on entropy and churn.
    ///
    /// Per Constitution v6.0.0, dreams are triggered by:
    /// - entropy > 0.7 for 5+ min
    /// - entropy > 0.7 AND churn > 0.5
    pub fn check_and_trigger_dream(&self) {
        let manager = self.trigger_manager.lock();

        if let Some(trigger_reason) = manager.check_triggers() {
            tracing::info!("Dream trigger activated: {:?}", trigger_reason);
            drop(manager);

            // Re-acquire to mark triggered
            let mut manager = self.trigger_manager.lock();
            manager.mark_triggered(trigger_reason);

            (self.consolidation_callback)(trigger_reason);
        }
    }
}

impl WorkspaceEventListener for DreamEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match event {
            WorkspaceEvent::MemoryExits {
                id,
                order_parameter,
                timestamp: _,
            } => {
                // Queue memory for dream replay - non-blocking acquire
                match self.dream_queue.try_write() {
                    Ok(mut queue) => {
                        queue.push(*id);
                        tracing::debug!(
                            "Queued memory {:?} for dream replay (r={:.3})",
                            id,
                            order_parameter
                        );
                    }
                    Err(e) => {
                        tracing::error!("CRITICAL: Failed to acquire dream_queue lock: {:?}", e);
                        panic!("DreamEventListener: Lock poisoned or deadlocked - cannot queue memory {:?}", id);
                    }
                }
            }
            // No-op for other events
            WorkspaceEvent::MemoryEnters { .. } => {}
            WorkspaceEvent::WorkspaceConflict { .. } => {}
            WorkspaceEvent::WorkspaceEmpty { .. } => {}
        }
    }
}

impl std::fmt::Debug for DreamEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DreamEventListener")
            .field("has_trigger_manager", &true)
            .field("has_callback", &true)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // ============================================================
    // FSV Tests for DreamEventListener
    // ============================================================

    #[tokio::test]
    async fn test_fsv_dream_listener_memory_exits() {
        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new_for_testing(dream_queue.clone());
        let memory_id = Uuid::new_v4();

        let before_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        assert_eq!(before_len, 0, "Queue must start empty");

        let event = WorkspaceEvent::MemoryExits {
            id: memory_id,
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let after_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        let queued_id = {
            let queue = dream_queue.read().await;
            queue.first().cloned()
        };

        assert_eq!(after_len, 1, "Queue must have exactly 1 item");
        assert_eq!(queued_id, Some(memory_id), "Queued ID must match");
    }

    #[tokio::test]
    async fn test_dream_listener_ignores_other_events() {
        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new_for_testing(dream_queue.clone());

        // Send MemoryEnters - should be ignored
        let event = WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
            fingerprint: None,
        };
        listener.on_event(&event);

        // Send WorkspaceEmpty - should be ignored
        let event = WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 1000,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let queue_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };

        assert_eq!(
            queue_len, 0,
            "Queue should remain empty for non-MemoryExits events"
        );
    }

    #[test]
    fn test_queue_functionality_preserved() {
        let queue = Arc::new(RwLock::new(Vec::new()));
        let manager = Arc::new(Mutex::new(TriggerManager::new()));
        let callback: DreamConsolidationCallback = Arc::new(|_| {});
        let listener = DreamEventListener::new(Arc::clone(&queue), manager, callback);

        let memory_id = Uuid::new_v4();

        let before_len = queue.blocking_read().len();
        assert_eq!(before_len, 0, "Queue must start empty");

        let event = WorkspaceEvent::MemoryExits {
            id: memory_id,
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let q = queue.blocking_read();
        assert_eq!(q.len(), 1, "Queue should have 1 memory");
        assert_eq!(q[0], memory_id, "Queued memory should match");
    }

    #[test]
    fn test_debug_impl() {
        let queue = Arc::new(RwLock::new(Vec::new()));
        let manager = Arc::new(Mutex::new(TriggerManager::new()));
        let callback: DreamConsolidationCallback = Arc::new(|_| {});
        let listener = DreamEventListener::new(queue, manager, callback);
        let debug_str = format!("{:?}", listener);

        assert!(debug_str.contains("has_trigger_manager: true"));
        assert!(debug_str.contains("has_callback: true"));
    }
}
