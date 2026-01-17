//! Dream Event Listener
//!
//! Queues exiting memories for dream replay consolidation and triggers
//! dream cycles on identity crisis events.
//!
//! # Constitution Compliance
//!
//! - AP-26: "IC<0.5 MUST trigger dream - no silent failures"
//! - AP-38: "IC<0.5 MUST auto-trigger dream"
//! - IDENTITY-007: "IC < 0.5 → auto-trigger dream"

use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::dream::{ExtendedTriggerReason, TriggerManager};
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};

/// Callback type for dream consolidation signaling.
///
/// Invoked when TriggerManager determines a dream cycle should start.
/// The callback receives the trigger reason (e.g., IdentityCritical with IC value).
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

/// Listener that queues exiting memories for dream replay and triggers
/// dream consolidation on identity crisis.
///
/// # Constitution Compliance
///
/// Per AP-26, AP-38, IDENTITY-007, and GWT-003: When IC < 0.5, this listener
/// MUST trigger dream consolidation. There are NO optional components - the
/// TriggerManager and consolidation callback are REQUIRED to ensure fail-fast
/// behavior on identity crisis events.
///
/// ## AP-26: "IC<0.5 MUST trigger dream - no silent failures"
/// ## IDENTITY-007: "IC < 0.5 -> auto-trigger dream"
/// ## GWT-003: "Workspace events propagate to all registered listeners"
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
/// // ALL fields are REQUIRED - no optional wiring
/// let listener = DreamEventListener::new(queue, trigger_manager, callback);
/// ```
///
/// # Breaking Change (AP-26 Enforcement)
///
/// This struct no longer supports optional TriggerManager or callback.
/// Code that previously used `None` will fail to compile. This is intentional
/// per AP-26 fail-fast requirements - IC crisis events MUST be handled.
///
/// For tests that don't test IC handling, use `new_for_testing()` which
/// provides a fail-fast callback that panics on IC events.
pub struct DreamEventListener {
    /// Queue for memories exiting workspace (for dream replay)
    dream_queue: Arc<RwLock<Vec<Uuid>>>,

    /// TriggerManager for IC-based dream triggering (REQUIRED per AP-26)
    trigger_manager: Arc<Mutex<TriggerManager>>,

    /// Callback for dream consolidation signaling (REQUIRED per AP-26)
    /// Called when TriggerManager returns Some(reason)
    consolidation_callback: DreamConsolidationCallback,
}

impl DreamEventListener {
    /// Create a new dream event listener with all required components.
    ///
    /// # Constitution Compliance
    ///
    /// Per AP-26, AP-38, IDENTITY-007, and GWT-003: All parameters are REQUIRED.
    /// There is no "backwards-compatible" mode - IC crisis events MUST be handled.
    ///
    /// # Arguments
    ///
    /// * `dream_queue` - Queue for memories exiting workspace (for dream replay)
    /// * `trigger_manager` - TriggerManager for IC-based dream triggering
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

    /// Create a DreamEventListener for testing (panics on IdentityCritical events).
    ///
    /// # Constitution Compliance
    ///
    /// Per AP-26: This constructor enforces fail-fast behavior for tests that
    /// don't explicitly test IC crisis handling. If an IC event occurs, the
    /// test will panic with a clear message.
    ///
    /// # WARNING
    ///
    /// This constructor is ONLY for unit tests that don't test IC crisis handling.
    /// Production code MUST use `new()` with proper TriggerManager and callback.
    ///
    /// # Example
    ///
    /// ```ignore
    /// #[cfg(test)]
    /// let listener = DreamEventListener::new_for_testing(queue);
    /// // Will panic if any IdentityCritical event triggers a dream
    /// ```
    #[cfg(test)]
    pub fn new_for_testing(dream_queue: Arc<RwLock<Vec<Uuid>>>) -> Self {
        Self {
            dream_queue,
            trigger_manager: Arc::new(Mutex::new(TriggerManager::new())),
            consolidation_callback: Arc::new(|reason| {
                panic!(
                    "FAIL FAST [AP-26, IDENTITY-007]: Dream consolidation callback invoked \
                     on test-only listener! Reason: {:?}. \n\
                     \n\
                     If you are testing IC crisis handling, use DreamEventListener::new() \
                     with a mock callback instead of new_for_testing(). \n\
                     \n\
                     new_for_testing() is only for tests that do NOT trigger IC < 0.5.",
                    reason
                );
            }),
        }
    }

    /// Get a clone of the dream queue arc for external access.
    ///
    /// Use this to inspect or drain the queue after events are processed.
    pub fn queue(&self) -> Arc<RwLock<Vec<Uuid>>> {
        Arc::clone(&self.dream_queue)
    }

    /// Handle identity critical event - updates TriggerManager and checks triggers.
    ///
    /// # Constitution Compliance
    ///
    /// Per AP-26, AP-38, IDENTITY-007, GWT-003:
    /// - IC < threshold MUST trigger dream consolidation
    /// - Lock failures are fatal (indicates deadlock/poison)
    /// - TriggerManager and callback are REQUIRED (not optional)
    ///
    /// # Arguments
    ///
    /// * `identity_coherence` - Current IC value [0.0, 1.0]
    /// * `previous_status` - Status before transition
    /// * `current_status` - Current status
    /// * `reason` - Human-readable reason for crisis
    ///
    /// # Panics
    ///
    /// Panics if lock acquisition fails (AP-26: no silent failures).
    fn handle_identity_critical(
        &self,
        identity_coherence: f32,
        previous_status: &str,
        current_status: &str,
        reason: &str,
    ) {
        // Always log the IC event
        tracing::warn!(
            "Identity critical (IC={:.3}): {} (transition: {} -> {})",
            identity_coherence,
            reason,
            previous_status,
            current_status,
        );

        // Use parking_lot Mutex - never poisons, blocking is fine in sync context
        // AP-26: TriggerManager is REQUIRED, no Option check needed
        let mut manager = self.trigger_manager.lock();

        // Update IC in TriggerManager
        manager.update_identity_coherence(identity_coherence);

        // Check if any trigger fires (IC, entropy, GPU, manual)
        if let Some(trigger_reason) = manager.check_triggers() {
            tracing::info!(
                "Dream trigger activated: {:?} (IC={:.3})",
                trigger_reason,
                identity_coherence
            );

            // Mark as triggered to start cooldown
            manager.mark_triggered(trigger_reason);

            // AP-26: Callback is REQUIRED, invoke directly (no Option check)
            (self.consolidation_callback)(trigger_reason);
        } else {
            tracing::debug!(
                "No dream trigger (IC={:.3}, cooldown or above threshold)",
                identity_coherence
            );
        }
    }

    /// Handle identity critical event from IdentityContinuityMonitor.
    ///
    /// This is the entry point for the IC crisis -> dream consolidation chain.
    /// Called by the callback created via `IdentityContinuityMonitor::create_dream_callback()`.
    ///
    /// # Constitution Compliance
    ///
    /// This method is part of the AP-26, IDENTITY-007 enforcement chain:
    /// 1. IdentityContinuityMonitor detects IC < 0.5
    /// 2. IC callback invokes this method
    /// 3. This method invokes TriggerManager
    /// 4. TriggerManager checks triggers and invokes consolidation callback
    ///
    /// # Arguments
    ///
    /// * `identity_coherence` - Current IC value (should be < 0.5)
    /// * `previous_status` - String representation of previous status
    /// * `current_status` - String representation of current status (should be "Critical")
    /// * `reason` - Human-readable reason for the crisis
    pub fn handle_identity_critical_from_monitor(
        &self,
        identity_coherence: f32,
        previous_status: &str,
        current_status: &str,
        reason: &str,
    ) {
        // Delegate to existing handle_identity_critical
        // This method provides the translation layer from monitor callback
        self.handle_identity_critical(identity_coherence, previous_status, current_status, reason);
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
                        // AP-26: Lock failure is fatal
                        tracing::error!("CRITICAL: Failed to acquire dream_queue lock: {:?}", e);
                        panic!("DreamEventListener: Lock poisoned or deadlocked - cannot queue memory {:?}", id);
                    }
                }
            }
            WorkspaceEvent::IdentityCritical {
                identity_coherence,
                previous_status,
                current_status,
                reason,
                timestamp: _,
            } => {
                // Delegate to handler which manages TriggerManager integration
                self.handle_identity_critical(
                    *identity_coherence,
                    previous_status,
                    current_status,
                    reason,
                );
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
        // AP-26: Both fields are now REQUIRED, show presence indicators
        f.debug_struct("DreamEventListener")
            .field("has_trigger_manager", &true) // Always true per AP-26
            .field("has_callback", &true) // Always true per AP-26
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;

    use crate::dream::TriggerConfig;

    // ============================================================
    // FSV Tests for DreamEventListener
    // ============================================================

    #[tokio::test]
    async fn test_fsv_dream_listener_memory_exits() {
        println!("=== FSV: DreamEventListener - MemoryExits ===");

        // SETUP - use new_for_testing since this test doesn't test IC handling
        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new_for_testing(dream_queue.clone());
        let memory_id = Uuid::new_v4();

        // BEFORE
        let before_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        println!("BEFORE: queue.len() = {}", before_len);
        assert_eq!(before_len, 0, "Queue must start empty");

        // EXECUTE
        let event = WorkspaceEvent::MemoryExits {
            id: memory_id,
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER - SEPARATE READ
        let after_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        let queued_id = {
            let queue = dream_queue.read().await;
            queue.first().cloned()
        };
        println!("AFTER: queue.len() = {}", after_len);

        // VERIFY
        assert_eq!(after_len, 1, "Queue must have exactly 1 item");
        assert_eq!(queued_id, Some(memory_id), "Queued ID must match");

        // EVIDENCE
        println!(
            "EVIDENCE: Memory {:?} correctly queued for dream replay",
            memory_id
        );
    }

    #[tokio::test]
    async fn test_dream_listener_ignores_other_events() {
        println!("=== TEST: DreamEventListener ignores non-MemoryExits ===");

        // Use new_for_testing since this test doesn't test IC handling
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
        println!("EVIDENCE: DreamEventListener correctly ignores non-MemoryExits events");
    }

    // ============================================================
    // TASK-24: TriggerManager Integration Tests
    // ============================================================

    #[test]
    fn test_ic_crisis_triggers_dream_consolidation() {
        println!("=== FSV: IC crisis triggers dream consolidation ===");

        // SETUP: TriggerManager with IC threshold 0.5 (constitution default)
        let config = TriggerConfig::default().with_cooldown(Duration::from_millis(1)); // Short cooldown for test
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        // Track callback invocation
        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_ic = Arc::new(AtomicU32::new(0));
        let cb_called = Arc::clone(&callback_called);
        let cb_ic = Arc::clone(&callback_ic);

        let callback: DreamConsolidationCallback = Arc::new(move |reason| {
            cb_called.store(true, Ordering::SeqCst);
            if let ExtendedTriggerReason::IdentityCritical { ic_value } = reason {
                cb_ic.store(ic_value.to_bits(), Ordering::SeqCst);
            }
        });

        // Create listener with required TriggerManager and callback (AP-26)
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue, manager, callback);

        // BEFORE
        println!(
            "BEFORE: callback_called = {}",
            callback_called.load(Ordering::SeqCst)
        );
        assert!(
            !callback_called.load(Ordering::SeqCst),
            "Callback should not be called yet"
        );

        // EXECUTE: Emit IC crisis event (IC=0.3 < threshold 0.5)
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.3,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Test IC crisis".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER: VERIFY callback was invoked
        println!(
            "AFTER: callback_called = {}",
            callback_called.load(Ordering::SeqCst)
        );
        assert!(
            callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST be called when IC < threshold"
        );

        // VERIFY: Correct IC value passed
        let stored_ic = f32::from_bits(callback_ic.load(Ordering::SeqCst));
        println!("EVIDENCE: Callback received IC value: {:.3}", stored_ic);
        assert!(
            (stored_ic - 0.3).abs() < 0.001,
            "Callback MUST receive correct IC value, got {}",
            stored_ic
        );
    }

    #[test]
    fn test_ic_above_threshold_no_trigger() {
        println!("=== FSV: IC above threshold does NOT trigger ===");

        let config = TriggerConfig::default().with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let callback_called = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&callback_called);
        let callback: DreamConsolidationCallback = Arc::new(move |_| {
            cb.store(true, Ordering::SeqCst);
        });

        // Create listener with required TriggerManager and callback (AP-26)
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue, manager, callback);

        // BEFORE
        println!(
            "BEFORE: callback_called = {}",
            callback_called.load(Ordering::SeqCst)
        );

        // IC=0.7 > threshold 0.5, should NOT trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.7,
            previous_status: "Stable".to_string(),
            current_status: "Warning".to_string(),
            reason: "Test warning".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER
        println!(
            "AFTER: callback_called = {}",
            callback_called.load(Ordering::SeqCst)
        );
        assert!(
            !callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST NOT be called when IC >= threshold"
        );

        println!("EVIDENCE: No dream trigger for IC=0.7 (above threshold 0.5)");
    }

    #[test]
    fn test_ic_at_threshold_no_trigger() {
        println!("=== FSV: IC exactly at threshold does NOT trigger ===");

        let config = TriggerConfig::default().with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let callback_called = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&callback_called);
        let callback: DreamConsolidationCallback = Arc::new(move |_| {
            cb.store(true, Ordering::SeqCst);
        });

        // Create listener with required TriggerManager and callback (AP-26)
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue, manager, callback);

        // IC=0.5 = threshold 0.5 (not < threshold), should NOT trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.5,
            previous_status: "Stable".to_string(),
            current_status: "Warning".to_string(),
            reason: "Test at threshold".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        assert!(
            !callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST NOT be called when IC = threshold (need IC < threshold)"
        );

        println!("EVIDENCE: No dream trigger for IC=0.5 (at threshold, not below)");
    }

    #[test]
    fn test_ic_just_below_threshold_triggers() {
        println!("=== FSV: IC just below threshold DOES trigger ===");

        let config = TriggerConfig::default().with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let callback_called = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&callback_called);
        let callback: DreamConsolidationCallback = Arc::new(move |_| {
            cb.store(true, Ordering::SeqCst);
        });

        // Create listener with required TriggerManager and callback (AP-26)
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue, manager, callback);

        // IC=0.4999 < threshold 0.5, should trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.4999,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Test just below".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        assert!(
            callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST be called when IC < threshold (even just below)"
        );

        println!("EVIDENCE: Dream trigger for IC=0.4999 (just below threshold 0.5)");
    }

    #[test]
    fn test_ic_zero_triggers() {
        println!("=== FSV: IC=0.0 (minimum) triggers dream ===");

        let config = TriggerConfig::default().with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_ic = Arc::new(AtomicU32::new(u32::MAX));
        let cb = Arc::clone(&callback_called);
        let cb_ic = Arc::clone(&callback_ic);
        let callback: DreamConsolidationCallback = Arc::new(move |reason| {
            cb.store(true, Ordering::SeqCst);
            if let ExtendedTriggerReason::IdentityCritical { ic_value } = reason {
                cb_ic.store(ic_value.to_bits(), Ordering::SeqCst);
            }
        });

        // Create listener with required TriggerManager and callback (AP-26)
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue, manager, callback);

        // IC=0.0 (minimum possible) should definitely trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.0,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Complete identity loss".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        assert!(
            callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST be called for IC=0.0"
        );

        let stored_ic = f32::from_bits(callback_ic.load(Ordering::SeqCst));
        assert!(
            stored_ic.abs() < 0.001,
            "Callback should receive IC=0.0, got {}",
            stored_ic
        );

        println!("EVIDENCE: Dream trigger for IC=0.0 (minimum, complete identity loss)");
    }

    #[test]
    fn test_queue_functionality_preserved() {
        println!("=== FSV: Existing queue functionality preserved ===");

        // Create listener with required TriggerManager and callback (AP-26)
        let queue = Arc::new(RwLock::new(Vec::new()));
        let manager = Arc::new(Mutex::new(TriggerManager::new()));
        let callback: DreamConsolidationCallback = Arc::new(|_| {});
        let listener = DreamEventListener::new(Arc::clone(&queue), manager, callback);

        let memory_id = Uuid::new_v4();

        // BEFORE
        let before_len = queue.blocking_read().len();
        println!("BEFORE: queue.len() = {}", before_len);

        // EXECUTE
        let event = WorkspaceEvent::MemoryExits {
            id: memory_id,
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER
        let q = queue.blocking_read();
        println!("AFTER: queue.len() = {}", q.len());

        assert_eq!(q.len(), 1, "Queue should have 1 memory");
        assert_eq!(q[0], memory_id, "Queued memory should match");

        println!("EVIDENCE: Queue functionality preserved with TriggerManager wired");
    }

    #[test]
    fn test_cooldown_prevents_rapid_triggers() {
        println!("=== FSV: Cooldown prevents rapid IC triggers ===");

        // TriggerManager with 100ms cooldown
        let config = TriggerConfig::default().with_cooldown(Duration::from_millis(100));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let trigger_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tc = Arc::clone(&trigger_count);
        let callback: DreamConsolidationCallback = Arc::new(move |_| {
            tc.fetch_add(1, Ordering::SeqCst);
        });

        // Create listener with required TriggerManager and callback (AP-26)
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue, manager, callback);

        // First IC crisis - should trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.3,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "First crisis".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // Second IC crisis immediately - should NOT trigger (cooldown)
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.2,
            previous_status: "Critical".to_string(),
            current_status: "Critical".to_string(),
            reason: "Second crisis".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let count = trigger_count.load(Ordering::SeqCst);
        println!(
            "EVIDENCE: Trigger count = {} (expected 1 due to cooldown)",
            count
        );

        assert_eq!(count, 1, "Only first trigger should fire due to cooldown");
    }

    #[test]
    fn test_new_for_testing_panics_on_ic_trigger() {
        println!("=== FSV: new_for_testing panics on IC crisis (AP-26 fail-fast) ===");

        // new_for_testing provides fail-fast behavior for tests
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new_for_testing(queue);

        // IC above threshold (0.7 > 0.5) should NOT trigger panic
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.7,
            previous_status: "Stable".to_string(),
            current_status: "Warning".to_string(),
            reason: "Above threshold".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event); // Should not panic

        println!("EVIDENCE: new_for_testing handles IC above threshold without panic");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_new_for_testing_panics_on_ic_below_threshold() {
        println!("=== FSV: new_for_testing panics on IC < threshold (AP-26 fail-fast) ===");

        // new_for_testing provides fail-fast behavior for tests
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new_for_testing(queue);

        // IC below threshold (0.3 < 0.5) SHOULD trigger panic per AP-26
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.3,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Below threshold - should panic".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event); // Should panic with "FAIL FAST"
    }

    #[test]
    fn test_debug_impl() {
        // AP-26: Both fields are now REQUIRED, debug always shows true
        let queue = Arc::new(RwLock::new(Vec::new()));
        let manager = Arc::new(Mutex::new(TriggerManager::new()));
        let callback: DreamConsolidationCallback = Arc::new(|_| {});
        let listener = DreamEventListener::new(queue, manager, callback);
        let debug_str = format!("{:?}", listener);

        // Both should always be true per AP-26 (required fields)
        assert!(debug_str.contains("has_trigger_manager: true"));
        assert!(debug_str.contains("has_callback: true"));

        println!("EVIDENCE: Debug impl shows trigger_manager and callback always present (AP-26)");
    }
}
