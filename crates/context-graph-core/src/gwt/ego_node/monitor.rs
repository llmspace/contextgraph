//! IdentityContinuityMonitor - Continuous IC tracking wrapper
//!
//! Wraps PurposeVectorHistory to provide real-time identity continuity
//! monitoring and status classification.
//!
//! # TASK-IDENTITY-P0-004: Crisis Detection
//!
//! This module implements crisis detection with:
//! - `CrisisDetectionResult`: Captures all transition information
//! - Status transition tracking: Tracks previous vs current status
//! - Cooldown mechanism: Prevents event spam during IC fluctuations
//! - Helper methods: For downstream consumers (P0-005, P0-006)

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::cosine::cosine_similarity_13d;
use super::identity_continuity::IdentityContinuity;
use super::purpose_vector_history::{PurposeVectorHistory, PurposeVectorHistoryProvider};
use super::types::{IdentityStatus, CRISIS_EVENT_COOLDOWN, IC_CRITICAL_THRESHOLD};
use crate::gwt::listeners::DreamEventListener;
use std::sync::Arc;

/// Callback type for IC crisis notifications.
///
/// This callback is invoked when Identity Continuity (IC) drops below
/// the critical threshold (0.5). The callback receives a reference to
/// the `CrisisDetectionResult` containing all transition details.
///
/// # Constitution Compliance
///
/// Per AP-26, IDENTITY-007: This callback MUST be invoked when IC < 0.5.
/// The callback is responsible for triggering dream consolidation via
/// `DreamEventListener::handle_identity_critical()`.
///
/// # Implementation Notes
///
/// The callback:
/// - MUST be `Send + Sync` for thread-safe sharing
/// - MUST NOT block for extended periods (< 10ms recommended)
/// - SHOULD delegate heavy work to async tasks
/// - MUST NOT panic (panics will propagate to caller)
///
/// # Example Usage in Struct
///
/// ```rust,ignore
/// struct CrisisHandler {
///     callback: IcCrisisCallback,
/// }
/// ```
pub type IcCrisisCallback = Arc<dyn Fn(&CrisisDetectionResult) + Send + Sync>;

/// Optional crisis callback for deserialized monitors.
///
/// After deserializing an `IdentityContinuityMonitor`, use
/// `set_crisis_callback()` to re-wire the callback before
/// calling `detect_crisis()`.
///
/// # Example Usage in Struct
///
/// ```rust,ignore
/// struct DeserializedMonitor {
///     crisis_callback: OptionalIcCrisisCallback,
/// }
/// ```
pub type OptionalIcCrisisCallback = Option<IcCrisisCallback>;

/// Result of crisis detection analysis
///
/// Contains all information needed by CrisisProtocol (P0-005)
/// to decide what actions to take.
///
/// # TASK-IDENTITY-P0-004: Crisis Detection
///
/// # Fields
/// - `identity_coherence`: Current IC value (0.0-1.0)
/// - `previous_status`: Status before this detection
/// - `current_status`: Status after this detection
/// - `status_changed`: True if status transitioned
/// - `entering_crisis`: True if transitioned FROM Healthy to any lower state
/// - `entering_critical`: True if transitioned TO Critical from any other state
/// - `recovering`: True if status improved (lower ordinal -> higher ordinal)
/// - `time_since_last_event`: Time since last crisis event was emitted
/// - `can_emit_event`: True if cooldown allows new event emission
#[derive(Debug, Clone, PartialEq)]
pub struct CrisisDetectionResult {
    /// Current IC value
    pub identity_coherence: f32,
    /// Previous status (before this computation)
    pub previous_status: IdentityStatus,
    /// Current status (after this computation)
    pub current_status: IdentityStatus,
    /// Whether status changed
    pub status_changed: bool,
    /// Whether entering crisis (transition from Healthy to Warning/Degraded/Critical)
    pub entering_crisis: bool,
    /// Whether entering critical (transition to Critical specifically)
    pub entering_critical: bool,
    /// Whether recovering (transition from lower to higher status)
    pub recovering: bool,
    /// Time since last crisis event emission
    pub time_since_last_event: Option<Duration>,
    /// Whether cooldown allows event emission
    pub can_emit_event: bool,
}

/// Convert status to ordinal for comparison
/// Higher ordinal = healthier state
/// Critical=0, Degraded=1, Warning=2, Healthy=3
#[inline]
fn status_ordinal(status: IdentityStatus) -> u8 {
    match status {
        IdentityStatus::Critical => 0,
        IdentityStatus::Degraded => 1,
        IdentityStatus::Warning => 2,
        IdentityStatus::Healthy => 3,
    }
}

/// Default status for deserialization
fn default_healthy_status() -> IdentityStatus {
    IdentityStatus::Healthy
}

/// Identity Continuity Monitor - Continuous IC tracking wrapper
///
/// Wraps `PurposeVectorHistory` to provide real-time identity continuity
/// monitoring and status classification.
///
/// # Constitution Reference
/// From constitution.yaml lines 365-392:
/// - IC = cos(PV_t, PV_{t-1}) x r(t)
/// - Thresholds: healthy>0.9, warning<0.7, dream<0.5
/// - self_ego_node.identity_trajectory: max 1000 snapshots
///
/// # TECH-IDENTITY-001: Crisis Callback
///
/// This monitor now requires a crisis callback in production (AP-26).
/// Use `new(callback)` for production and `new_for_testing()` for tests.
#[derive(Clone, Serialize, Deserialize)]
pub struct IdentityContinuityMonitor {
    /// Purpose vector history buffer (delegates to PurposeVectorHistory)
    history: PurposeVectorHistory,
    /// Cached last computation result (None until first compute_continuity call)
    last_result: Option<IdentityContinuity>,
    /// Configurable crisis threshold (default: IC_CRITICAL_THRESHOLD = 0.5)
    crisis_threshold: f32,

    // === TASK-IDENTITY-P0-004: Crisis Detection Fields ===

    /// Previous status for transition detection (default: Healthy)
    #[serde(default = "default_healthy_status")]
    previous_status: IdentityStatus,
    /// Last time a crisis event was emitted (not serialized - transient state)
    #[serde(skip)]
    last_event_time: Option<Instant>,

    // === TASK-IDENTITY-P0-007: MCP Tool Exposure Fields ===

    /// Cached last crisis detection result for MCP tool exposure.
    /// Not serialized - transient state reconstructed on detect_crisis() calls.
    #[serde(skip)]
    last_detection: Option<CrisisDetectionResult>,

    // === TECH-IDENTITY-001: Crisis Callback Field ===

    /// Callback invoked when IC drops to CRITICAL (< 0.5).
    ///
    /// Not serialized - must be re-wired on deserialization via `set_crisis_callback()`.
    /// REQUIRED for production per AP-26; None only allowed in tests via `new_for_testing()`.
    #[serde(skip)]
    crisis_callback: Option<IcCrisisCallback>,
}

impl IdentityContinuityMonitor {
    /// Create new monitor with required crisis callback.
    ///
    /// # Constitution Compliance
    ///
    /// Per AP-26, IDENTITY-007: The crisis callback is REQUIRED in production.
    /// When IC < 0.5, the callback MUST be invoked to trigger dream consolidation.
    ///
    /// # Arguments
    ///
    /// * `crisis_callback` - Callback invoked when IC drops to CRITICAL (< 0.5)
    ///
    /// # Defaults
    ///
    /// - history capacity: MAX_PV_HISTORY_SIZE (1000)
    /// - crisis_threshold: IC_CRITICAL_THRESHOLD (0.5)
    /// - previous_status: Healthy
    ///
    /// # Example
    ///
    /// ```ignore
    /// let callback: IcCrisisCallback = Arc::new(|result| {
    ///     dream_listener.handle_identity_critical_from_monitor(
    ///         result.identity_coherence,
    ///         &format!("{:?}", result.previous_status),
    ///         &format!("{:?}", result.current_status),
    ///         "IC dropped below critical threshold",
    ///     );
    /// });
    /// let monitor = IdentityContinuityMonitor::new(callback);
    /// ```
    pub fn new(crisis_callback: IcCrisisCallback) -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: IC_CRITICAL_THRESHOLD,
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None,
            crisis_callback: Some(crisis_callback),
        }
    }

    /// Test-only constructor without crisis callback.
    ///
    /// # Warning
    ///
    /// This constructor creates a monitor without a crisis callback.
    /// Use only for testing non-crisis scenarios.
    ///
    /// # Constitution Reference
    ///
    /// Production code MUST use `new()` with all dependencies per AP-26.
    /// This constructor exists only for test isolation.
    #[cfg(test)]
    pub fn new_for_testing() -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: IC_CRITICAL_THRESHOLD,
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None,
            crisis_callback: None,
        }
    }

    /// Test-only constructor with custom threshold.
    ///
    /// # Warning
    ///
    /// Creates a monitor without a crisis callback.
    /// Use only for testing threshold behavior.
    ///
    /// # Arguments
    /// * `threshold` - Custom crisis threshold (clamped to [0, 1])
    #[cfg(test)]
    pub fn with_threshold_for_testing(threshold: f32) -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: threshold.clamp(0.0, 1.0),
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None,
            crisis_callback: None,
        }
    }

    /// Test-only constructor with custom capacity.
    ///
    /// # Warning
    ///
    /// Creates a monitor without a crisis callback.
    /// Use only for testing history capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum history entries (0 = unlimited)
    #[cfg(test)]
    pub fn with_capacity_for_testing(capacity: usize) -> Self {
        Self {
            history: PurposeVectorHistory::with_max_size(capacity),
            last_result: None,
            crisis_threshold: IC_CRITICAL_THRESHOLD,
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None,
            crisis_callback: None,
        }
    }

    /// Set the crisis callback for this monitor.
    ///
    /// Used when deserializing or re-wiring the monitor. The callback
    /// is not serialized, so this must be called after deserialization.
    ///
    /// # Constitution Reference
    ///
    /// Per AP-26: Crisis callback is REQUIRED in production.
    pub fn set_crisis_callback(&mut self, callback: IcCrisisCallback) {
        self.crisis_callback = Some(callback);
    }

    /// Check if crisis callback is configured.
    ///
    /// Returns false if monitor was deserialized without re-wiring
    /// or created with `new_for_testing()`.
    #[inline]
    pub fn has_crisis_callback(&self) -> bool {
        self.crisis_callback.is_some()
    }

    /// Get a reference to the crisis callback, if configured.
    ///
    /// Used internally by detect_crisis() to invoke the callback
    /// when IC < 0.5.
    #[inline]
    pub fn crisis_callback(&self) -> Option<&IcCrisisCallback> {
        self.crisis_callback.as_ref()
    }

    /// Create monitor with custom crisis threshold.
    ///
    /// # Arguments
    /// * `threshold` - Custom crisis threshold (clamped to [0, 1])
    /// * `crisis_callback` - Callback invoked when IC drops below threshold
    pub fn with_threshold(threshold: f32, crisis_callback: IcCrisisCallback) -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: threshold.clamp(0.0, 1.0),
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None,
            crisis_callback: Some(crisis_callback),
        }
    }

    /// Create monitor with custom history capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum history entries (0 = unlimited)
    /// * `crisis_callback` - Callback invoked when IC drops below threshold
    pub fn with_capacity(capacity: usize, crisis_callback: IcCrisisCallback) -> Self {
        Self {
            history: PurposeVectorHistory::with_max_size(capacity),
            last_result: None,
            crisis_threshold: IC_CRITICAL_THRESHOLD,
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None,
            crisis_callback: Some(crisis_callback),
        }
    }

    /// Create crisis callback that sends to DreamEventListener.
    ///
    /// This is the standard production wiring per AP-26, IDENTITY-007.
    /// The callback translates IC crisis detection into dream trigger.
    ///
    /// # Arguments
    ///
    /// * `dream_listener` - Arc reference to the wired DreamEventListener
    ///
    /// # Returns
    ///
    /// `IcCrisisCallback` that invokes `handle_identity_critical_from_monitor()`
    /// on the provided listener.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let dream_listener = Arc::new(DreamEventListener::new(queue, tm, cb));
    /// let ic_callback = IdentityContinuityMonitor::create_dream_callback(
    ///     dream_listener.clone()
    /// );
    /// let monitor = IdentityContinuityMonitor::new(ic_callback);
    /// ```
    pub fn create_dream_callback(
        dream_listener: Arc<DreamEventListener>,
    ) -> IcCrisisCallback {
        Arc::new(move |result: &CrisisDetectionResult| {
            // Delegate to DreamEventListener for actual trigger
            dream_listener.handle_identity_critical_from_monitor(
                result.identity_coherence,
                &format!("{:?}", result.previous_status),
                &format!("{:?}", result.current_status),
                &format!(
                    "IC crisis: IC={:.4} dropped below critical threshold (0.5)",
                    result.identity_coherence
                ),
            );
        })
    }

    /// Compute identity continuity from new purpose vector and Kuramoto r.
    ///
    /// # Algorithm
    /// 1. Push new PV to history, get previous PV
    /// 2. If first vector: return IdentityContinuity::first_vector()
    /// 3. Compute cos(PV_t, PV_{t-1}) using cosine_similarity_13d
    /// 4. Create IdentityContinuity::new(cosine, kuramoto_r)
    /// 5. Cache and return result
    ///
    /// # Arguments
    /// * `purpose_vector` - Current 13D purpose alignment vector (PV_t)
    /// * `kuramoto_r` - Current Kuramoto order parameter r(t) in [0, 1]
    /// * `context` - Description for history snapshot
    ///
    /// # Returns
    /// * `IdentityContinuity` with computed IC and status
    pub fn compute_continuity(
        &mut self,
        purpose_vector: &[f32; 13],
        kuramoto_r: f32,
        context: impl Into<String>,
    ) -> IdentityContinuity {
        // Push current PV and get previous (if any)
        let previous = self.history.push(*purpose_vector, context);

        // Compute result based on whether this is first vector
        let result = match previous {
            None => {
                // First vector: per EC-IDENTITY-01, default to healthy
                IdentityContinuity::first_vector()
            }
            Some(prev_pv) => {
                // Compute cosine similarity between consecutive PVs
                let cosine = cosine_similarity_13d(purpose_vector, &prev_pv);

                // Create IdentityContinuity with IC = cos x r
                IdentityContinuity::new(cosine, kuramoto_r)
            }
        };

        // Cache result for subsequent getters
        self.last_result = Some(result.clone());

        result
    }

    /// Get the last computed IdentityContinuity result.
    #[inline]
    pub fn last_result(&self) -> Option<&IdentityContinuity> {
        self.last_result.as_ref()
    }

    /// Get current identity coherence value (IC).
    #[inline]
    pub fn identity_coherence(&self) -> Option<f32> {
        self.last_result.as_ref().map(|r| r.identity_coherence)
    }

    /// Get current identity status classification.
    #[inline]
    pub fn current_status(&self) -> Option<IdentityStatus> {
        self.last_result.as_ref().map(|r| r.status)
    }

    /// Check if identity is in crisis (IC < crisis_threshold).
    #[inline]
    pub fn is_in_crisis(&self) -> bool {
        self.last_result
            .as_ref()
            .map(|r| r.identity_coherence < self.crisis_threshold)
            .unwrap_or(false)
    }

    /// Get the number of snapshots in history.
    #[inline]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get the configured crisis threshold.
    #[inline]
    pub fn crisis_threshold(&self) -> f32 {
        self.crisis_threshold
    }

    /// Get read-only access to underlying history.
    pub fn history(&self) -> &PurposeVectorHistory {
        &self.history
    }

    /// Check if history is empty (no vectors recorded).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Check if this is the first vector (exactly one entry).
    #[inline]
    pub fn is_first_vector(&self) -> bool {
        self.history.is_first_vector()
    }

    // === TASK-IDENTITY-P0-004: Crisis Detection Methods ===

    /// Detect crisis state and track transitions.
    ///
    /// # Algorithm
    /// 1. Get current status from last_result
    /// 2. Compare with previous_status to detect transitions
    /// 3. Compute entering_crisis (from Healthy to lower)
    /// 4. Compute entering_critical (to Critical from any other)
    /// 5. Compute recovering (status improvement)
    /// 6. Check cooldown for event emission
    /// 7. Update previous_status for next call
    ///
    /// # Returns
    /// `CrisisDetectionResult` with all transition information
    ///
    /// # Panics
    /// Never panics. Returns default result if no computation has occurred.
    pub fn detect_crisis(&mut self) -> CrisisDetectionResult {
        // Get current values (default to Healthy if no computation yet)
        let current_status = self.current_status().unwrap_or(IdentityStatus::Healthy);
        let ic = self.identity_coherence().unwrap_or(1.0);
        let prev_status = self.previous_status;

        // Detect transitions
        let status_changed = current_status != prev_status;

        // Entering crisis = transitioning FROM Healthy to any lower state
        let entering_crisis = status_changed
            && prev_status == IdentityStatus::Healthy
            && current_status != IdentityStatus::Healthy;

        // Entering critical = transitioning TO Critical from any other state
        let entering_critical = status_changed
            && current_status == IdentityStatus::Critical
            && prev_status != IdentityStatus::Critical;

        // Recovering = improving status (lower ordinal to higher ordinal)
        let recovering =
            status_changed && status_ordinal(current_status) > status_ordinal(prev_status);

        // Cooldown check
        let time_since_last_event = self.last_event_time.map(|t| t.elapsed());
        let can_emit_event = match time_since_last_event {
            None => true, // No previous event, can emit
            Some(elapsed) => elapsed >= CRISIS_EVENT_COOLDOWN,
        };

        // Update previous_status for next detection
        self.previous_status = current_status;

        let result = CrisisDetectionResult {
            identity_coherence: ic,
            previous_status: prev_status,
            current_status,
            status_changed,
            entering_crisis,
            entering_critical,
            recovering,
            time_since_last_event,
            can_emit_event,
        };

        // TASK-IDENTITY-P0-007: Cache result for MCP tool exposure
        self.last_detection = Some(result.clone());

        // TECH-IDENTITY-001: Invoke crisis callback when entering CRITICAL
        // Per AP-26, IDENTITY-007: IC < 0.5 MUST trigger dream consolidation
        if entering_critical && can_emit_event {
            match &self.crisis_callback {
                Some(callback) => {
                    // Log at ERROR level per REQ-IC-015
                    tracing::error!(
                        target: "identity_crisis",
                        ic = %ic,
                        previous_status = ?prev_status,
                        current_status = ?current_status,
                        "IDENTITY CRISIS [AP-26, IDENTITY-007]: IC={:.4} < 0.5, \
                         triggering dream consolidation",
                        ic
                    );

                    // Invoke callback - this should trigger DreamEventListener
                    // The callback is responsible for actual dream trigger
                    callback(&result);

                    // Mark event emitted to start cooldown
                    // Prevents spam if IC oscillates around threshold
                    self.mark_event_emitted();

                    tracing::info!(
                        target: "identity_crisis",
                        ic = %ic,
                        "Crisis callback invoked successfully, cooldown started"
                    );
                }
                None => {
                    // FAIL FAST per AP-26: No silent failures
                    // This should only happen in test builds or misconfigured production
                    panic!(
                        "FAIL FAST [AP-26, IDENTITY-007]: IC crisis detected (IC={:.4}) \
                         but no crisis_callback configured! \n\
                         \n\
                         Production code MUST use IdentityContinuityMonitor::new(callback) \
                         with proper callback wiring. \n\
                         \n\
                         This violates constitution rules: \n\
                         - AP-26: IC<0.5 MUST trigger dream - no silent failures \n\
                         - IDENTITY-007: IC < 0.5 -> auto-trigger dream \n\
                         - GWT-003: IC < 0.5 -> dream consolidation \n\
                         \n\
                         If you are in a test, use new_for_testing() and ensure your test \
                         does not trigger IC < 0.5. If testing crisis behavior, provide a \
                         mock callback via new().",
                        ic
                    );
                }
            }
        }

        result
    }

    /// Get the previous status (before last detection).
    #[inline]
    pub fn previous_status(&self) -> IdentityStatus {
        self.previous_status
    }

    /// Check if status changed compared to previous detection.
    ///
    /// Note: This compares current computed status with previously recorded status,
    /// NOT the result of the last detect_crisis call.
    #[inline]
    pub fn status_changed(&self) -> bool {
        self.current_status()
            .map(|curr| curr != self.previous_status)
            .unwrap_or(false)
    }

    /// Check if currently entering critical state.
    ///
    /// Returns true if current status is Critical and previous status was not Critical.
    #[inline]
    pub fn entering_critical(&self) -> bool {
        self.current_status()
            .map(|curr| {
                curr == IdentityStatus::Critical
                    && self.previous_status != IdentityStatus::Critical
            })
            .unwrap_or(false)
    }

    /// Mark that a crisis event was emitted (resets cooldown timer).
    #[inline]
    pub fn mark_event_emitted(&mut self) {
        self.last_event_time = Some(Instant::now());
    }

    /// Get time since last event emission.
    #[inline]
    pub fn time_since_last_event(&self) -> Option<Duration> {
        self.last_event_time.map(|t| t.elapsed())
    }

    // === TASK-IDENTITY-P0-007: MCP Tool Exposure Methods ===

    /// Get the last crisis detection result.
    ///
    /// Returns `None` if `detect_crisis()` has never been called.
    /// The cached result allows MCP tools to access crisis state
    /// without triggering a new detection cycle.
    ///
    /// # TASK-IDENTITY-P0-007
    #[inline]
    pub fn last_detection(&self) -> Option<CrisisDetectionResult> {
        self.last_detection.clone()
    }
}

impl std::fmt::Debug for IdentityContinuityMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdentityContinuityMonitor")
            .field("history_len", &self.history.len())
            .field("last_result", &self.last_result)
            .field("crisis_threshold", &self.crisis_threshold)
            .field("previous_status", &self.previous_status)
            .field("has_crisis_callback", &self.crisis_callback.is_some())
            .finish()
    }
}

/// Default impl requires callback, so only available in tests via new_for_testing()
#[cfg(test)]
impl Default for IdentityContinuityMonitor {
    fn default() -> Self {
        Self::new_for_testing()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Test TASK-FOUNDATION-004: IcCrisisCallback type alias works correctly.
    ///
    /// Verifies:
    /// 1. IcCrisisCallback can be created from a closure
    /// 2. The callback receives CrisisDetectionResult by reference
    /// 3. The callback is Send + Sync (can be shared across threads)
    #[test]
    fn test_ic_crisis_callback_type_alias() {
        // Create a flag to verify callback was invoked
        let was_called = Arc::new(AtomicBool::new(false));
        let was_called_clone = Arc::clone(&was_called);

        // Create an IcCrisisCallback
        let callback: IcCrisisCallback = Arc::new(move |result: &CrisisDetectionResult| {
            // Verify we can access fields from the result
            assert!(result.identity_coherence >= 0.0);
            assert!(result.identity_coherence <= 1.0);
            was_called_clone.store(true, Ordering::SeqCst);
        });

        // Verify callback is Send + Sync by requiring these bounds
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        assert_send_sync(&callback);

        // Create a mock CrisisDetectionResult
        let result = CrisisDetectionResult {
            identity_coherence: 0.45,
            previous_status: IdentityStatus::Warning,
            current_status: IdentityStatus::Critical,
            status_changed: true,
            entering_crisis: false,
            entering_critical: true,
            recovering: false,
            time_since_last_event: None,
            can_emit_event: true,
        };

        // Invoke the callback
        callback(&result);

        // Verify callback was called
        assert!(was_called.load(Ordering::SeqCst));
    }

    /// Test OptionalIcCrisisCallback type alias.
    #[test]
    fn test_optional_ic_crisis_callback() {
        // Test None case
        let no_callback: OptionalIcCrisisCallback = None;
        assert!(no_callback.is_none());

        // Test Some case
        let callback: IcCrisisCallback = Arc::new(|_result| {
            // Do nothing
        });
        let some_callback: OptionalIcCrisisCallback = Some(callback);
        assert!(some_callback.is_some());

        // Verify we can invoke it
        let result = CrisisDetectionResult {
            identity_coherence: 0.8,
            previous_status: IdentityStatus::Healthy,
            current_status: IdentityStatus::Healthy,
            status_changed: false,
            entering_crisis: false,
            entering_critical: false,
            recovering: false,
            time_since_last_event: None,
            can_emit_event: true,
        };

        if let Some(cb) = some_callback {
            cb(&result);
        }
    }

    /// Test TASK-LOGIC-004: Callback is invoked when entering critical state.
    ///
    /// Verifies that detect_crisis() invokes the callback when:
    /// - entering_critical is true (IC < 0.5 from higher state)
    /// - can_emit_event is true (cooldown elapsed)
    #[test]
    fn test_detect_crisis_invokes_callback_on_entering_critical() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let was_called = Arc::new(AtomicBool::new(false));
        let was_called_clone = Arc::clone(&was_called);

        // Create callback that records invocation
        let callback: IcCrisisCallback = Arc::new(move |result| {
            assert!(result.identity_coherence < 0.5);
            assert_eq!(result.current_status, IdentityStatus::Critical);
            was_called_clone.store(true, Ordering::SeqCst);
        });

        // Create monitor with callback
        let mut monitor = IdentityContinuityMonitor::new(callback);

        // Start with healthy state
        let pv = [0.8; 13];
        monitor.compute_continuity(&pv, 0.9, "Healthy");
        let _ = monitor.detect_crisis(); // Establish baseline

        // Trigger critical state (IC < 0.5)
        monitor.compute_continuity(&pv, 0.3, "Crisis"); // IC = 1.0 * 0.3 = 0.3
        let result = monitor.detect_crisis();

        // Verify callback was invoked
        assert!(
            was_called.load(Ordering::SeqCst),
            "Callback should be invoked when entering critical"
        );
        assert!(result.entering_critical);
    }

    /// Test TASK-LOGIC-004: Panic on missing callback when entering critical.
    ///
    /// Verifies that detect_crisis() panics with FAIL FAST message when:
    /// - entering_critical is true
    /// - crisis_callback is None
    #[test]
    #[should_panic(expected = "FAIL FAST [AP-26, IDENTITY-007]")]
    fn test_detect_crisis_panics_on_missing_callback() {
        // Create monitor WITHOUT callback (test-only constructor)
        let mut monitor = IdentityContinuityMonitor::new_for_testing();

        // Start with healthy state
        let pv = [0.8; 13];
        monitor.compute_continuity(&pv, 0.9, "Healthy");
        let _ = monitor.detect_crisis(); // Establish baseline

        // Trigger critical state - this should panic
        monitor.compute_continuity(&pv, 0.3, "Crisis"); // IC = 1.0 * 0.3 = 0.3
        let _ = monitor.detect_crisis(); // Should panic!
    }

    /// Test TASK-LOGIC-004: Panic message contains required constitution references.
    ///
    /// Verifies panic message includes AP-26, IDENTITY-007, and GWT-003.
    #[test]
    fn test_detect_crisis_panic_message_contains_required_strings() {
        // Create monitor WITHOUT callback
        let mut monitor = IdentityContinuityMonitor::new_for_testing();

        // Start with healthy state
        let pv = [0.8; 13];
        monitor.compute_continuity(&pv, 0.9, "Healthy");
        let _ = monitor.detect_crisis();

        // Trigger critical state
        monitor.compute_continuity(&pv, 0.3, "Crisis");

        // Catch the panic and verify message content
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = monitor.detect_crisis();
        }));

        assert!(result.is_err(), "Should panic on missing callback");

        let panic_payload = result.unwrap_err();
        let panic_msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
            s.as_str()
        } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
            *s
        } else {
            ""
        };

        // Verify all required strings are present
        assert!(
            panic_msg.contains("AP-26"),
            "Panic message must contain 'AP-26': {}",
            panic_msg
        );
        assert!(
            panic_msg.contains("IDENTITY-007"),
            "Panic message must contain 'IDENTITY-007': {}",
            panic_msg
        );
        assert!(
            panic_msg.contains("GWT-003"),
            "Panic message must contain 'GWT-003': {}",
            panic_msg
        );
        assert!(
            panic_msg.contains("FAIL FAST"),
            "Panic message must contain 'FAIL FAST': {}",
            panic_msg
        );
    }

    /// Test TASK-LOGIC-004: Callback is NOT invoked during cooldown.
    ///
    /// Verifies that detect_crisis() does not invoke callback when
    /// can_emit_event is false (within cooldown period).
    #[test]
    fn test_detect_crisis_respects_cooldown() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let callback: IcCrisisCallback = Arc::new(move |_result| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let mut monitor = IdentityContinuityMonitor::new(callback);

        // Start with healthy state
        let pv = [0.8; 13];
        monitor.compute_continuity(&pv, 0.9, "Healthy");
        let _ = monitor.detect_crisis();

        // First crisis - callback should be invoked
        monitor.compute_continuity(&pv, 0.3, "Crisis1");
        let _ = monitor.detect_crisis();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Second crisis immediately after - should be blocked by cooldown
        // (we need to go back to non-critical first, then back to critical)
        monitor.compute_continuity(&pv, 0.9, "Recovery");
        let _ = monitor.detect_crisis();

        // Try to enter critical again - should be blocked by cooldown
        monitor.compute_continuity(&pv, 0.3, "Crisis2");
        let _ = monitor.detect_crisis();

        // Callback should NOT have been called again due to cooldown
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "Callback should not be invoked during cooldown"
        );
    }
}
