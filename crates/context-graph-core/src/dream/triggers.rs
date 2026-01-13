//! Dream Trigger Implementations
//!
//! Implements trigger mechanisms beyond idle timeout:
//! - High entropy trigger (>0.7 for 5 minutes) - Constitution Section dream.trigger
//! - GPU overload trigger (approaching 30% budget) - Constitution Section dream.constraints
//! - Manual trigger (highest priority)
//!
//! Constitution Reference: docs2/constitution.yaml lines 255-256, 274
//! - entropy: ">0.7 for 5min" (line 255)
//! - gpu: "<30%" during dream (line 274)

use std::time::{Duration, Instant};

use tracing::{debug, info};

use super::types::{EntropyWindow, ExtendedTriggerReason, GpuTriggerState};

/// Configuration for trigger manager.
///
/// Holds thresholds for dream trigger conditions.
///
/// # Constitution Compliance
///
/// - `ic_threshold`: default 0.5 per `gwt.self_ego_node.thresholds.critical`
/// - `entropy_threshold`: default 0.7 per `dream.trigger.entropy`
/// - `cooldown`: default 60s to prevent trigger spam
///
/// # Example
///
/// ```
/// use context_graph_core::dream::TriggerConfig;
///
/// let config = TriggerConfig::default();
/// assert_eq!(config.ic_threshold, 0.5);
///
/// // Custom configuration
/// let custom = TriggerConfig::default()
///     .with_ic_threshold(0.4)
///     .with_entropy_threshold(0.8);
/// custom.validate(); // Panics if invalid
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerConfig {
    /// IC threshold for identity crisis (default: 0.5)
    /// Constitution: `gwt.self_ego_node.thresholds.critical = 0.5`
    /// When IC drops below this, triggers `ExtendedTriggerReason::IdentityCritical`
    pub ic_threshold: f32,

    /// Entropy threshold for high entropy trigger (default: 0.7)
    /// Constitution: `dream.trigger.entropy > 0.7 for 5min`
    pub entropy_threshold: f32,

    /// Cooldown between triggers (default: 60 seconds)
    /// Prevents rapid re-triggering
    pub cooldown: Duration,
}

impl Default for TriggerConfig {
    /// Create config with constitution-mandated defaults.
    fn default() -> Self {
        Self {
            ic_threshold: 0.5,        // Constitution: gwt.self_ego_node.thresholds.critical
            entropy_threshold: 0.7,    // Constitution: dream.trigger.entropy
            cooldown: Duration::from_secs(60),
        }
    }
}

impl TriggerConfig {
    /// Validate configuration against constitution bounds.
    ///
    /// # Panics
    ///
    /// Panics with detailed error message if any value is out of bounds.
    /// Per AP-26: fail-fast on invalid configuration.
    ///
    /// # Constitution Bounds
    ///
    /// - `ic_threshold`: MUST be in [0.0, 1.0]
    /// - `entropy_threshold`: MUST be in [0.0, 1.0]
    /// - `cooldown`: No explicit bound, but Duration::ZERO is unusual
    #[track_caller]
    pub fn validate(&self) {
        assert!(
            (0.0..=1.0).contains(&self.ic_threshold),
            "TriggerConfig: ic_threshold must be in [0.0, 1.0], got {}. \
             Constitution: gwt.self_ego_node.thresholds.critical = 0.5",
            self.ic_threshold
        );
        assert!(
            (0.0..=1.0).contains(&self.entropy_threshold),
            "TriggerConfig: entropy_threshold must be in [0.0, 1.0], got {}. \
             Constitution: dream.trigger.entropy threshold",
            self.entropy_threshold
        );
    }

    /// Create a validated config, panicking if invalid.
    ///
    /// Use this in constructors to fail-fast per AP-26.
    #[track_caller]
    pub fn validated(self) -> Self {
        self.validate();
        self
    }

    /// Builder: set IC threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - IC threshold [0.0, 1.0]. Values < 0.5 are more sensitive.
    pub fn with_ic_threshold(mut self, threshold: f32) -> Self {
        self.ic_threshold = threshold;
        self
    }

    /// Builder: set entropy threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Entropy threshold [0.0, 1.0]. Higher = less sensitive.
    pub fn with_entropy_threshold(mut self, threshold: f32) -> Self {
        self.entropy_threshold = threshold;
        self
    }

    /// Builder: set cooldown duration.
    ///
    /// # Arguments
    ///
    /// * `cooldown` - Duration between allowed triggers.
    pub fn with_cooldown(mut self, cooldown: Duration) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Check if IC value indicates identity crisis.
    ///
    /// # Arguments
    ///
    /// * `ic_value` - Current Identity Continuity value [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// `true` if `ic_value < ic_threshold` (crisis state)
    #[inline]
    pub fn is_identity_critical(&self, ic_value: f32) -> bool {
        ic_value < self.ic_threshold
    }

    /// Check if entropy value exceeds threshold.
    ///
    /// # Arguments
    ///
    /// * `entropy` - Current entropy value [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// `true` if `entropy > entropy_threshold`
    #[inline]
    pub fn is_high_entropy(&self, entropy: f32) -> bool {
        entropy > self.entropy_threshold
    }
}

/// Unified trigger manager for dream cycles.
///
/// Combines all trigger mechanisms into a single interface:
/// - High entropy (>0.7 sustained for 5 minutes)
/// - GPU overload (approaching 30% usage)
/// - Identity Critical (IC < 0.5)
/// - Manual trigger
///
/// # Constitution Compliance
///
/// - IC threshold: 0.5 (Constitution gwt.self_ego_node.thresholds.critical)
/// - Entropy threshold: 0.7 (Constitution dream.trigger)
/// - Entropy window: 5 minutes (Constitution dream.trigger)
/// - GPU threshold: 0.30 (Constitution dream.constraints.gpu)
/// - Cooldown: 60 seconds (prevents trigger spam)
///
/// # Priority Order (highest to lowest)
///
/// 1. Manual - User-initiated, bypasses cooldown
/// 2. IdentityCritical - IC < 0.5 (AP-26, AP-38, IDENTITY-007)
/// 3. GpuOverload - GPU approaching 30% budget
/// 4. HighEntropy - Entropy > 0.7 for 5 minutes
#[derive(Debug)]
pub struct TriggerManager {
    /// Configuration with thresholds (TASK-21)
    config: TriggerConfig,

    /// Current Identity Continuity value (TASK-21)
    /// None = not yet measured, Some(x) = current IC
    current_ic: Option<f32>,

    /// Entropy tracking window
    entropy_window: EntropyWindow,

    /// GPU utilization state
    gpu_state: GpuTriggerState,

    /// Whether manual trigger was requested
    manual_trigger: bool,

    /// Last trigger reason (for reporting)
    last_trigger_reason: Option<ExtendedTriggerReason>,

    /// Cooldown after trigger (to prevent rapid re-triggering)
    trigger_cooldown: Duration,

    /// Last trigger time
    last_trigger_time: Option<Instant>,

    /// Whether triggers are enabled
    enabled: bool,
}

impl TriggerManager {
    /// Create a new trigger manager with constitution defaults.
    ///
    /// # Constitution Values Applied
    /// - IC threshold: 0.5 (gwt.self_ego_node.thresholds.critical)
    /// - Entropy threshold: 0.7 (dream.trigger.entropy)
    /// - Entropy window: 5 minutes (dream.trigger)
    /// - GPU threshold: 0.30 (30%) (dream.constraints.gpu)
    /// - Cooldown: 60 seconds
    pub fn new() -> Self {
        let config = TriggerConfig::default();
        Self {
            trigger_cooldown: config.cooldown, // Use config cooldown
            config,
            current_ic: None,
            entropy_window: EntropyWindow::new(), // Uses Constitution defaults
            gpu_state: GpuTriggerState::new(),    // Uses Constitution defaults
            manual_trigger: false,
            last_trigger_reason: None,
            last_trigger_time: None,
            enabled: true,
        }
    }

    /// Create with custom config.
    ///
    /// # Arguments
    /// * `config` - Custom TriggerConfig with thresholds
    ///
    /// # Panics
    /// Panics if config validation fails (per AP-26: fail-fast on invalid config).
    #[track_caller]
    pub fn with_config(config: TriggerConfig) -> Self {
        config.validate(); // Fail-fast per AP-26
        Self {
            trigger_cooldown: config.cooldown,
            config,
            current_ic: None,
            entropy_window: EntropyWindow::new(),
            gpu_state: GpuTriggerState::new(),
            manual_trigger: false,
            last_trigger_reason: None,
            last_trigger_time: None,
            enabled: true,
        }
    }

    /// Create with custom cooldown (for testing with REAL time).
    ///
    /// # Arguments
    /// * `cooldown` - Duration before trigger can fire again
    ///
    /// # Note
    /// Tests MUST use real durations, not mocked time.
    pub fn with_cooldown(cooldown: Duration) -> Self {
        let mut manager = Self::new();
        manager.trigger_cooldown = cooldown;
        manager
    }

    /// Update entropy reading.
    ///
    /// Called periodically (e.g., every second) with system entropy value.
    /// High entropy (>0.7) indicates system stress/confusion.
    ///
    /// # Arguments
    /// * `entropy` - Current system entropy [0.0, 1.0]
    ///
    /// # Constitution Reference
    /// Trigger fires when entropy > 0.7 for 5 minutes continuously.
    pub fn update_entropy(&mut self, entropy: f32) {
        if !self.enabled {
            return;
        }

        self.entropy_window.push(entropy);

        if self.entropy_window.should_trigger() {
            debug!(
                "Entropy trigger condition met: avg={:.3}, threshold=0.7",
                self.entropy_window.average()
            );
        }
    }

    /// Update GPU utilization reading.
    ///
    /// Called periodically with GPU usage percentage.
    /// High GPU usage approaching budget indicates need for consolidation.
    ///
    /// # Arguments
    /// * `usage` - Current GPU usage [0.0, 1.0]
    ///
    /// # Constitution Reference
    /// GPU budget during dream is <30%. Trigger fires when approaching this limit.
    pub fn update_gpu_usage(&mut self, usage: f32) {
        if !self.enabled {
            return;
        }

        self.gpu_state.update(usage);

        if self.gpu_state.should_trigger() {
            debug!("GPU trigger condition met: usage={:.1}%", usage * 100.0);
        }
    }

    /// Update the current Identity Continuity value.
    ///
    /// # Arguments
    ///
    /// * `ic` - Current IC value, expected in [0.0, 1.0]
    ///
    /// # Clamping Behavior
    ///
    /// - NaN → clamped to 0.0 (worst case) with warning
    /// - Infinity → clamped to 1.0 (best case) with warning
    /// - Out of range → clamped to [0.0, 1.0] with warning
    ///
    /// # Constitution
    ///
    /// Per AP-10: No NaN/Infinity in UTL values.
    /// Per IDENTITY-007: IC < 0.5 → auto-trigger dream.
    pub fn update_identity_coherence(&mut self, ic: f32) {
        if !self.enabled {
            return;
        }

        let ic = if ic.is_nan() {
            tracing::warn!("Invalid IC value NaN, clamping to 0.0 per AP-10");
            0.0
        } else if ic.is_infinite() {
            tracing::warn!("Invalid IC value Infinity, clamping to 1.0 per AP-10");
            1.0
        } else if !(0.0..=1.0).contains(&ic) {
            tracing::warn!("IC value {} out of range, clamping to [0.0, 1.0]", ic);
            ic.clamp(0.0, 1.0)
        } else {
            ic
        };

        self.current_ic = Some(ic);

        if self.config.is_identity_critical(ic) {
            debug!(
                "IC {} < threshold {} - identity critical state",
                ic, self.config.ic_threshold
            );
        }
    }

    /// Check if identity continuity is in crisis state.
    ///
    /// # Returns
    ///
    /// `true` if `current_ic < config.ic_threshold`
    ///
    /// # Constitution
    ///
    /// Per gwt.self_ego_node.thresholds.critical: IC < 0.5 is critical.
    #[inline]
    pub fn check_identity_continuity(&self) -> bool {
        match self.current_ic {
            Some(ic) => self.config.is_identity_critical(ic),
            None => false, // No IC measured yet, cannot be critical
        }
    }

    /// Request a manual dream trigger.
    ///
    /// Manual triggers have highest priority and bypass cooldown.
    pub fn request_manual_trigger(&mut self) {
        info!("Manual dream trigger requested");
        self.manual_trigger = true;
    }

    /// Clear manual trigger flag.
    pub fn clear_manual_trigger(&mut self) {
        self.manual_trigger = false;
    }

    /// Check all trigger conditions and return highest priority trigger.
    ///
    /// # Priority Order (highest first)
    ///
    /// 1. Manual - User-initiated, bypasses cooldown
    /// 2. IdentityCritical - IC < 0.5 (AP-26, AP-38, IDENTITY-007)
    /// 3. GpuOverload - GPU approaching 30% budget
    /// 4. HighEntropy - Entropy > 0.7 for 5 minutes
    ///
    /// # Returns
    ///
    /// * `Some(reason)` - If trigger condition met
    /// * `None` - If no trigger condition met or in cooldown
    ///
    /// # Constitution Compliance
    ///
    /// - Manual bypasses cooldown (highest priority)
    /// - IdentityCritical MUST trigger when IC < 0.5 (AP-26, AP-38)
    /// - GpuOverload when GPU > 30% (Constitution dream.constraints.gpu)
    /// - HighEntropy when entropy > 0.7 for 5min (Constitution dream.trigger.entropy)
    pub fn check_triggers(&self) -> Option<ExtendedTriggerReason> {
        if !self.enabled {
            return None;
        }

        // Check cooldown (manual trigger bypasses cooldown)
        if !self.manual_trigger {
            if let Some(last_time) = self.last_trigger_time {
                if last_time.elapsed() < self.trigger_cooldown {
                    return None;
                }
            }
        }

        // Priority 1: Manual (highest)
        if self.manual_trigger {
            return Some(ExtendedTriggerReason::Manual);
        }

        // Priority 2: IdentityCritical (CONSTITUTION CRITICAL - AP-26, AP-38)
        if let Some(ic) = self.current_ic {
            if self.config.is_identity_critical(ic) {
                return Some(ExtendedTriggerReason::IdentityCritical { ic_value: ic });
            }
        }

        // Priority 3: GpuOverload
        if self.gpu_state.should_trigger() {
            return Some(ExtendedTriggerReason::GpuOverload);
        }

        // Priority 4: HighEntropy
        if self.entropy_window.should_trigger() {
            return Some(ExtendedTriggerReason::HighEntropy);
        }

        None
    }

    /// Check if dream should be triggered (simple boolean).
    #[inline]
    pub fn should_trigger(&self) -> bool {
        self.check_triggers().is_some()
    }

    /// Mark that a trigger fired (starts cooldown).
    ///
    /// Call this AFTER starting a dream cycle.
    pub fn mark_triggered(&mut self, reason: ExtendedTriggerReason) {
        info!("Dream triggered: {:?}", reason);

        self.last_trigger_reason = Some(reason);
        self.last_trigger_time = Some(Instant::now());

        // Reset states
        self.manual_trigger = false;
        self.gpu_state.mark_triggered();
        self.entropy_window.clear();
    }

    /// Reset after dream completion.
    ///
    /// Call this AFTER dream cycle completes.
    pub fn reset(&mut self) {
        debug!("Trigger manager reset");

        self.gpu_state.reset();
        self.entropy_window.clear();
        self.manual_trigger = false;
    }

    /// Get time remaining in cooldown, if any.
    pub fn cooldown_remaining(&self) -> Option<Duration> {
        self.last_trigger_time.and_then(|last| {
            let elapsed = last.elapsed();
            if elapsed < self.trigger_cooldown {
                Some(self.trigger_cooldown - elapsed)
            } else {
                None
            }
        })
    }

    /// Get current entropy average.
    #[inline]
    pub fn current_entropy(&self) -> f32 {
        self.entropy_window.average()
    }

    /// Get current GPU usage.
    #[inline]
    pub fn current_gpu_usage(&self) -> f32 {
        self.gpu_state.current_usage
    }

    /// Get current Identity Continuity value.
    ///
    /// # Returns
    ///
    /// * `Some(ic)` - Current IC value [0.0, 1.0] if measured
    /// * `None` - If no IC has been set yet
    #[inline]
    pub fn current_ic(&self) -> Option<f32> {
        self.current_ic
    }

    /// Get current IC threshold from config.
    ///
    /// # Returns
    ///
    /// The IC threshold below which triggers IdentityCritical.
    /// Default: 0.5 per Constitution gwt.self_ego_node.thresholds.critical.
    #[inline]
    pub fn ic_threshold(&self) -> f32 {
        self.config.ic_threshold
    }

    /// Get last trigger reason.
    pub fn last_trigger_reason(&self) -> Option<ExtendedTriggerReason> {
        self.last_trigger_reason
    }

    /// Enable or disable triggers.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if enabled {
            info!("Dream triggers enabled");
        } else {
            info!("Dream triggers disabled");
        }
    }

    /// Check if triggers are enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Access entropy window for inspection/testing.
    pub fn entropy_window(&self) -> &EntropyWindow {
        &self.entropy_window
    }

    /// Access entropy window mutably for testing with custom parameters.
    ///
    /// # Note
    /// This method is primarily for testing purposes to configure
    /// shorter window durations with REAL time (not mocked).
    pub fn entropy_window_mut(&mut self) -> &mut EntropyWindow {
        &mut self.entropy_window
    }

    /// Access GPU state for inspection/testing.
    pub fn gpu_state(&self) -> &GpuTriggerState {
        &self.gpu_state
    }
}

impl Default for TriggerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU utilization monitor stub.
///
/// Provides a placeholder for actual GPU monitoring.
/// Real implementation would use NVML (NVIDIA) or ROCm (AMD).
///
/// # Note
/// This is a STUB. Production requires actual GPU monitoring integration.
#[derive(Debug, Clone)]
pub struct GpuMonitor {
    /// Simulated GPU usage (for testing)
    simulated_usage: f32,

    /// Whether to use simulated values
    use_simulated: bool,
}

impl GpuMonitor {
    /// Create a new GPU monitor.
    pub fn new() -> Self {
        Self {
            simulated_usage: 0.0,
            use_simulated: true, // Default to simulated until real impl
        }
    }

    /// Get current GPU utilization.
    ///
    /// Returns value in [0.0, 1.0].
    pub fn get_usage(&self) -> f32 {
        if self.use_simulated {
            self.simulated_usage
        } else {
            // TODO(FUTURE): Implement real GPU monitoring via NVML
            // For now, return 0.0 (no GPU usage)
            0.0
        }
    }

    /// Set simulated GPU usage (for testing).
    pub fn set_simulated_usage(&mut self, usage: f32) {
        self.simulated_usage = usage.clamp(0.0, 1.0);
    }

    /// Check if GPU is available.
    pub fn is_available(&self) -> bool {
        // TODO(FUTURE): Check for actual GPU
        false
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy calculator from system state.
///
/// Computes system entropy based on query rate variance.
/// High variance = high entropy = system confusion.
///
/// # Algorithm
/// Uses coefficient of variation (CV = std/mean) of query inter-arrival times.
/// CV > 1 indicates high variability (chaotic) = high entropy.
/// CV < 1 indicates regularity (predictable) = low entropy.
#[derive(Debug, Clone)]
pub struct EntropyCalculator {
    /// Recent query timestamps for rate calculation
    query_times: Vec<Instant>,

    /// Maximum queries to track
    max_queries: usize,

    /// Time window for entropy calculation
    window: Duration,
}

impl EntropyCalculator {
    /// Create a new entropy calculator.
    pub fn new() -> Self {
        Self {
            query_times: Vec::with_capacity(100),
            max_queries: 100,
            window: Duration::from_secs(60), // 1 minute window
        }
    }

    /// Record a query event.
    pub fn record_query(&mut self) {
        let now = Instant::now();

        self.query_times.push(now);

        // Trim old queries outside window
        self.query_times
            .retain(|&t| now.duration_since(t) < self.window);

        // Cap size
        while self.query_times.len() > self.max_queries {
            self.query_times.remove(0);
        }
    }

    /// Calculate current entropy based on query patterns.
    ///
    /// Returns value in [0.0, 1.0] where:
    /// - 0.0 = no activity or regular pattern (low entropy)
    /// - 1.0 = high chaotic activity (high entropy)
    pub fn calculate(&self) -> f32 {
        if self.query_times.len() < 2 {
            return 0.0;
        }

        // Calculate inter-arrival times
        let mut intervals: Vec<f32> = Vec::new();
        for i in 1..self.query_times.len() {
            let interval = self.query_times[i]
                .duration_since(self.query_times[i - 1])
                .as_secs_f32();
            intervals.push(interval);
        }

        if intervals.is_empty() {
            return 0.0;
        }

        // Calculate coefficient of variation (CV = std / mean)
        let mean: f32 = intervals.iter().sum::<f32>() / intervals.len() as f32;
        if mean < 1e-6 {
            return 1.0; // Extremely rapid queries = high entropy
        }

        let variance: f32 = intervals
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;
        let std = variance.sqrt();
        let cv = std / mean;

        // Normalize CV to [0, 1]
        // CV > 1 indicates high variability (high entropy)
        // CV < 1 indicates regularity (low entropy)
        (cv / 2.0).min(1.0)
    }

    /// Clear query history.
    pub fn clear(&mut self) {
        self.query_times.clear();
    }

    /// Get the number of queries in window.
    pub fn query_count(&self) -> usize {
        self.query_times.len()
    }
}

impl Default for EntropyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS - NO MOCK DATA, REAL TIME ONLY
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // ============ Constitution Compliance Tests ============

    #[test]
    fn test_trigger_manager_constitution_defaults() {
        let manager = TriggerManager::new();

        assert!(manager.is_enabled());
        assert!(!manager.should_trigger());

        // Verify entropy window uses Constitution defaults
        assert_eq!(
            manager.entropy_window.threshold, 0.7,
            "Entropy threshold must be 0.7 per Constitution"
        );
        assert_eq!(
            manager.entropy_window.window_duration,
            Duration::from_secs(300),
            "Entropy window must be 5 minutes per Constitution"
        );

        // Verify GPU state uses Constitution defaults
        assert_eq!(
            manager.gpu_state.threshold, 0.30,
            "GPU threshold must be 0.30 (30%) per Constitution, NOT 80%"
        );
    }

    // ============ Manual Trigger Tests ============

    #[test]
    fn test_trigger_manager_manual_highest_priority() {
        let mut manager = TriggerManager::new();

        assert!(!manager.should_trigger());

        manager.request_manual_trigger();

        assert!(manager.should_trigger());
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::Manual)
        );
    }

    #[test]
    fn test_trigger_manager_manual_bypasses_cooldown() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_secs(3600)); // 1 hour

        // Trigger and start cooldown
        manager.request_manual_trigger();
        manager.mark_triggered(ExtendedTriggerReason::Manual);

        // Another manual trigger should bypass cooldown
        manager.request_manual_trigger();
        assert!(
            manager.should_trigger(),
            "Manual trigger should bypass cooldown"
        );
    }

    // ============ GPU Trigger Tests ============

    #[test]
    fn test_trigger_manager_gpu_threshold_30_percent() {
        let mut manager = TriggerManager::new();

        // Below threshold
        manager.update_gpu_usage(0.25);
        assert!(!manager.should_trigger(), "25% < 30% should not trigger");

        // Above threshold
        manager.update_gpu_usage(0.35);
        assert!(manager.should_trigger(), "35% > 30% should trigger");
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::GpuOverload)
        );
    }

    #[test]
    fn test_trigger_manager_gpu_uses_smoothed_average() {
        let mut manager = TriggerManager::new();

        // Push multiple samples - average should be used
        manager.update_gpu_usage(0.20);
        manager.update_gpu_usage(0.22);
        manager.update_gpu_usage(0.25);

        assert!(
            !manager.should_trigger(),
            "Average ~22% should not trigger"
        );
    }

    // ============ Entropy Trigger Tests ============

    #[test]
    fn test_trigger_manager_entropy_requires_sustained_high() {
        // Use short window for testing with REAL time
        let mut manager = TriggerManager::new();
        manager.entropy_window = EntropyWindow::with_params(Duration::from_millis(50), 0.7);
        manager.trigger_cooldown = Duration::from_millis(1);

        // Push high entropy - starts tracking
        manager.update_entropy(0.8);
        assert!(!manager.should_trigger(), "Should not trigger immediately");

        // Wait for window duration plus margin
        thread::sleep(Duration::from_millis(60));
        manager.update_entropy(0.85);

        assert!(
            manager.should_trigger(),
            "Should trigger after sustained high entropy"
        );
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::HighEntropy)
        );
    }

    #[test]
    fn test_trigger_manager_entropy_resets_on_low() {
        let mut manager = TriggerManager::new();
        manager.entropy_window = EntropyWindow::with_params(Duration::from_millis(50), 0.7);

        // Start high
        manager.update_entropy(0.9);
        thread::sleep(Duration::from_millis(30));

        // Drop below threshold - resets tracking
        manager.update_entropy(0.5);

        // Wait and check - should NOT trigger because tracking reset
        thread::sleep(Duration::from_millis(30));
        manager.update_entropy(0.5);

        assert!(
            !manager.should_trigger(),
            "Low entropy should reset tracking"
        );
    }

    // ============ Cooldown Tests ============

    #[test]
    fn test_trigger_manager_cooldown_prevents_rapid_retrigger() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_millis(100));

        // Trigger with GPU
        manager.update_gpu_usage(0.35);
        assert!(manager.should_trigger());

        manager.mark_triggered(ExtendedTriggerReason::GpuOverload);

        // Try to trigger again immediately (within cooldown)
        // Even after reset(), cooldown should prevent re-trigger
        manager.reset(); // Simulate dream cycle completion
        manager.update_gpu_usage(0.40);
        assert!(
            !manager.should_trigger(),
            "Cooldown should prevent re-trigger even after reset"
        );

        // Wait for cooldown to expire
        thread::sleep(Duration::from_millis(150));

        // Now with cooldown expired, new GPU trigger should fire
        manager.update_gpu_usage(0.40);
        assert!(
            manager.should_trigger(),
            "Should trigger after cooldown expires"
        );
    }

    #[test]
    fn test_trigger_manager_cooldown_remaining() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_millis(100));

        assert!(
            manager.cooldown_remaining().is_none(),
            "No cooldown initially"
        );

        manager.request_manual_trigger();
        manager.mark_triggered(ExtendedTriggerReason::Manual);

        let remaining = manager.cooldown_remaining();
        assert!(remaining.is_some(), "Should have cooldown after trigger");
        assert!(remaining.unwrap() <= Duration::from_millis(100));
    }

    // ============ Disabled Trigger Tests ============

    #[test]
    fn test_trigger_manager_disabled_blocks_all() {
        let mut manager = TriggerManager::new();

        manager.set_enabled(false);

        // None of these should trigger
        manager.request_manual_trigger();
        manager.update_gpu_usage(0.95);
        manager.update_entropy(0.99);

        assert!(
            !manager.should_trigger(),
            "Disabled manager should not trigger"
        );
    }

    // ============ Reset Tests ============

    #[test]
    fn test_trigger_manager_reset_clears_state() {
        let mut manager = TriggerManager::new();

        manager.update_gpu_usage(0.35);
        manager.update_entropy(0.9);
        manager.mark_triggered(ExtendedTriggerReason::GpuOverload);

        manager.reset();

        // GPU state should be reset
        assert_eq!(manager.current_gpu_usage(), 0.0);
        // Entropy window should be cleared
        assert!(manager.entropy_window.is_empty());
    }

    // ============ GpuMonitor Tests ============

    #[test]
    fn test_gpu_monitor_simulated() {
        let mut monitor = GpuMonitor::new();

        assert_eq!(monitor.get_usage(), 0.0);

        monitor.set_simulated_usage(0.75);
        assert_eq!(monitor.get_usage(), 0.75);

        // Test clamping
        monitor.set_simulated_usage(1.5);
        assert_eq!(monitor.get_usage(), 1.0);

        monitor.set_simulated_usage(-0.5);
        assert_eq!(monitor.get_usage(), 0.0);
    }

    // ============ EntropyCalculator Tests ============

    #[test]
    fn test_entropy_calculator_empty_returns_zero() {
        let calc = EntropyCalculator::new();
        assert_eq!(calc.calculate(), 0.0);
    }

    #[test]
    fn test_entropy_calculator_single_query_returns_zero() {
        let mut calc = EntropyCalculator::new();
        calc.record_query();
        assert_eq!(calc.calculate(), 0.0);
    }

    #[test]
    fn test_entropy_calculator_regular_queries_low_entropy() {
        let mut calc = EntropyCalculator::new();

        // Simulate regular queries at fixed intervals
        for _ in 0..5 {
            calc.record_query();
            thread::sleep(Duration::from_millis(10));
        }

        let entropy = calc.calculate();
        // Regular intervals = low entropy
        assert!(
            entropy < 0.5,
            "Regular queries should have low entropy: {}",
            entropy
        );
    }

    #[test]
    fn test_entropy_calculator_irregular_queries_high_entropy() {
        let mut calc = EntropyCalculator::new();

        // Simulate irregular queries
        calc.record_query();
        thread::sleep(Duration::from_millis(5));
        calc.record_query();
        thread::sleep(Duration::from_millis(50));
        calc.record_query();
        thread::sleep(Duration::from_millis(10));
        calc.record_query();
        thread::sleep(Duration::from_millis(100));
        calc.record_query();

        let entropy = calc.calculate();
        // Irregular intervals = higher entropy
        assert!(
            entropy > 0.3,
            "Irregular queries should have higher entropy: {}",
            entropy
        );
    }

    // ============ Priority Order Tests ============

    #[test]
    fn test_trigger_manager_priority_manual_over_gpu_over_entropy() {
        let mut manager = TriggerManager::new();

        // Setup GPU trigger
        manager.update_gpu_usage(0.35);
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::GpuOverload)
        );

        // Add manual - should take priority
        manager.request_manual_trigger();
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::Manual)
        );
    }

    // ============ Identity Continuity Trigger Tests ============

    #[test]
    fn test_trigger_manager_ic_check_triggers_below_threshold() {
        let mut manager = TriggerManager::new();

        // IC = 0.49 < 0.5 threshold → should trigger IdentityCritical
        manager.update_identity_coherence(0.49);

        let trigger = manager.check_triggers();
        assert!(trigger.is_some(), "IC below threshold should trigger");

        match trigger.unwrap() {
            ExtendedTriggerReason::IdentityCritical { ic_value } => {
                assert!(
                    (ic_value - 0.49).abs() < 0.001,
                    "IC value should be preserved: got {}",
                    ic_value
                );
            }
            other => panic!("Expected IdentityCritical, got {:?}", other),
        }
    }

    #[test]
    fn test_trigger_manager_ic_at_threshold_no_trigger() {
        let mut manager = TriggerManager::new();

        // IC = 0.5 (exactly at threshold) → should NOT trigger
        // Constitution: IC < 0.5 is critical (strict less than)
        manager.update_identity_coherence(0.5);

        assert!(
            !manager.check_identity_continuity(),
            "IC at threshold should not be critical"
        );
    }

    #[test]
    fn test_trigger_manager_ic_above_threshold_no_trigger() {
        let mut manager = TriggerManager::new();

        // IC = 0.9 (healthy) → should not trigger
        manager.update_identity_coherence(0.9);

        assert!(
            !manager.check_identity_continuity(),
            "IC above threshold should not be critical"
        );
        assert!(
            manager.check_triggers().is_none(),
            "No trigger expected for healthy IC"
        );
    }

    #[test]
    fn test_trigger_manager_ic_priority_over_gpu() {
        let mut manager = TriggerManager::new();

        // Set up BOTH IC crisis AND GPU overload
        manager.update_identity_coherence(0.3);
        manager.update_gpu_usage(0.35);

        // IdentityCritical should have higher priority than GpuOverload
        let trigger = manager.check_triggers();
        match trigger {
            Some(ExtendedTriggerReason::IdentityCritical { .. }) => {} // Expected
            other => panic!("Expected IdentityCritical to have priority, got {:?}", other),
        }
    }

    #[test]
    fn test_trigger_manager_manual_priority_over_ic() {
        let mut manager = TriggerManager::new();

        // Set up IC crisis
        manager.update_identity_coherence(0.3);

        // Request manual trigger
        manager.request_manual_trigger();

        // Manual should have highest priority
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::Manual)
        );
    }

    #[test]
    fn test_trigger_manager_ic_nan_handling() {
        let mut manager = TriggerManager::new();

        // NaN should be clamped to 0.0 per AP-10
        manager.update_identity_coherence(f32::NAN);

        // Should trigger (0.0 < 0.5)
        let trigger = manager.check_triggers();
        match trigger {
            Some(ExtendedTriggerReason::IdentityCritical { ic_value }) => {
                assert_eq!(ic_value, 0.0, "NaN should clamp to 0.0");
            }
            other => panic!("Expected IdentityCritical, got {:?}", other),
        }
    }

    #[test]
    fn test_trigger_manager_ic_infinity_handling() {
        let mut manager = TriggerManager::new();

        // Infinity should be clamped to 1.0 per AP-10
        manager.update_identity_coherence(f32::INFINITY);

        // Should NOT trigger (1.0 >= 0.5)
        assert!(!manager.check_identity_continuity());
    }

    #[test]
    fn test_trigger_manager_with_custom_config() {
        let config = TriggerConfig::default().with_ic_threshold(0.6); // Higher threshold for more sensitive detection

        let mut manager = TriggerManager::with_config(config);

        // IC = 0.55 < 0.6 (custom threshold) → should trigger
        manager.update_identity_coherence(0.55);

        assert!(manager.check_identity_continuity());

        match manager.check_triggers() {
            Some(ExtendedTriggerReason::IdentityCritical { ic_value }) => {
                assert!((ic_value - 0.55).abs() < 0.001);
            }
            other => panic!("Expected IdentityCritical, got {:?}", other),
        }
    }

    #[test]
    fn test_trigger_manager_no_ic_measured_no_trigger() {
        let manager = TriggerManager::new();

        // No IC has been set → should not be critical
        assert!(!manager.check_identity_continuity());
        assert!(manager.current_ic().is_none());
    }

    #[test]
    fn test_trigger_manager_ic_accessors() {
        let mut manager = TriggerManager::new();

        // Initially no IC
        assert!(manager.current_ic().is_none());
        assert_eq!(manager.ic_threshold(), 0.5); // Default threshold

        // Set IC
        manager.update_identity_coherence(0.42);

        assert_eq!(manager.current_ic(), Some(0.42));
        assert_eq!(manager.ic_threshold(), 0.5);
    }

    #[test]
    fn test_trigger_manager_ic_negative_clamping() {
        let mut manager = TriggerManager::new();

        // Negative value should be clamped to 0.0
        manager.update_identity_coherence(-0.5);

        assert_eq!(manager.current_ic(), Some(0.0));
        assert!(manager.check_identity_continuity()); // 0.0 < 0.5 = critical
    }

    #[test]
    fn test_trigger_manager_ic_over_one_clamping() {
        let mut manager = TriggerManager::new();

        // Value > 1.0 should be clamped to 1.0
        manager.update_identity_coherence(1.5);

        assert_eq!(manager.current_ic(), Some(1.0));
        assert!(!manager.check_identity_continuity()); // 1.0 >= 0.5 = not critical
    }

    #[test]
    fn test_trigger_manager_ic_minimum_value() {
        let mut manager = TriggerManager::new();

        // IC = 0.0 (minimum) → should trigger
        manager.update_identity_coherence(0.0);

        assert!(manager.check_identity_continuity());
        match manager.check_triggers() {
            Some(ExtendedTriggerReason::IdentityCritical { ic_value }) => {
                assert_eq!(ic_value, 0.0);
            }
            other => panic!("Expected IdentityCritical, got {:?}", other),
        }
    }

    #[test]
    fn test_trigger_manager_ic_disabled_no_update() {
        let mut manager = TriggerManager::new();

        // Disable triggers
        manager.set_enabled(false);

        // Update IC while disabled
        manager.update_identity_coherence(0.3);

        // IC should not have been updated
        assert!(
            manager.current_ic().is_none(),
            "IC should not update when disabled"
        );
    }

    // ============ TriggerConfig Tests ============

    #[test]
    fn test_trigger_config_constitution_defaults() {
        let config = TriggerConfig::default();

        assert_eq!(
            config.ic_threshold, 0.5,
            "ic_threshold must be 0.5 per Constitution gwt.self_ego_node.thresholds.critical"
        );
        assert_eq!(
            config.entropy_threshold, 0.7,
            "entropy_threshold must be 0.7 per Constitution dream.trigger.entropy"
        );
        assert_eq!(
            config.cooldown,
            Duration::from_secs(60),
            "cooldown default is 60 seconds"
        );
    }

    #[test]
    fn test_trigger_config_validate_passes_valid() {
        let config = TriggerConfig::default();
        config.validate(); // Should not panic
    }

    #[test]
    #[should_panic(expected = "ic_threshold must be in [0.0, 1.0]")]
    fn test_trigger_config_validate_panics_negative_ic() {
        let config = TriggerConfig {
            ic_threshold: -0.1,
            ..Default::default()
        };
        config.validate();
    }

    #[test]
    #[should_panic(expected = "ic_threshold must be in [0.0, 1.0]")]
    fn test_trigger_config_validate_panics_ic_over_one() {
        let config = TriggerConfig {
            ic_threshold: 1.5,
            ..Default::default()
        };
        config.validate();
    }

    #[test]
    #[should_panic(expected = "entropy_threshold must be in [0.0, 1.0]")]
    fn test_trigger_config_validate_panics_negative_entropy() {
        let config = TriggerConfig {
            entropy_threshold: -0.1,
            ..Default::default()
        };
        config.validate();
    }

    #[test]
    fn test_trigger_config_builder_pattern() {
        let config = TriggerConfig::default()
            .with_ic_threshold(0.4)
            .with_entropy_threshold(0.8)
            .with_cooldown(Duration::from_secs(30));

        assert_eq!(config.ic_threshold, 0.4);
        assert_eq!(config.entropy_threshold, 0.8);
        assert_eq!(config.cooldown, Duration::from_secs(30));
    }

    #[test]
    fn test_trigger_config_validated_returns_self() {
        let config = TriggerConfig::default().validated();
        assert_eq!(config.ic_threshold, 0.5);
    }

    #[test]
    #[should_panic(expected = "ic_threshold must be in [0.0, 1.0]")]
    fn test_trigger_config_validated_panics_invalid() {
        TriggerConfig::default()
            .with_ic_threshold(-1.0)
            .validated();
    }

    #[test]
    fn test_trigger_config_is_identity_critical() {
        let config = TriggerConfig::default(); // ic_threshold = 0.5

        // Below threshold = crisis
        assert!(config.is_identity_critical(0.49), "0.49 < 0.5 should be critical");
        assert!(config.is_identity_critical(0.0), "0.0 < 0.5 should be critical");

        // At or above threshold = not crisis
        assert!(!config.is_identity_critical(0.5), "0.5 >= 0.5 should NOT be critical");
        assert!(!config.is_identity_critical(0.51), "0.51 > 0.5 should NOT be critical");
        assert!(!config.is_identity_critical(1.0), "1.0 > 0.5 should NOT be critical");
    }

    #[test]
    fn test_trigger_config_is_high_entropy() {
        let config = TriggerConfig::default(); // entropy_threshold = 0.7

        // Above threshold = high entropy
        assert!(config.is_high_entropy(0.71), "0.71 > 0.7 should be high entropy");
        assert!(config.is_high_entropy(1.0), "1.0 > 0.7 should be high entropy");

        // At or below threshold = not high entropy
        assert!(!config.is_high_entropy(0.7), "0.7 <= 0.7 should NOT be high entropy");
        assert!(!config.is_high_entropy(0.69), "0.69 < 0.7 should NOT be high entropy");
        assert!(!config.is_high_entropy(0.0), "0.0 < 0.7 should NOT be high entropy");
    }

    #[test]
    fn test_trigger_config_edge_case_boundary_values() {
        // Test exact boundary values
        let config = TriggerConfig {
            ic_threshold: 0.0,
            entropy_threshold: 1.0,
            cooldown: Duration::ZERO,
        };
        config.validate(); // Should pass - 0.0 and 1.0 are valid

        let config_max = TriggerConfig {
            ic_threshold: 1.0,
            entropy_threshold: 0.0,
            cooldown: Duration::from_secs(86400), // 24 hours
        };
        config_max.validate(); // Should pass
    }

    #[test]
    fn test_trigger_config_serialization_roundtrip() {
        // TriggerConfig does not derive Serialize/Deserialize by default
        // but if it did, this test would verify roundtrip
        let config = TriggerConfig::default()
            .with_ic_threshold(0.45)
            .with_entropy_threshold(0.75);

        // Verify config fields survive clone (basic roundtrip)
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }
}
