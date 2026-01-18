//! Global Workspace State and Coherence State Machine
//!
//! Implements Global Workspace Theory (GWT) state management.
//!
//! # Constitution Compliance (v6.0.0)
//!
//! Per Constitution v6.0.0 Section 14, topic-based coherence scoring is used.
//! This module provides CoherenceState for workspace state management.

use serde::{Deserialize, Serialize};

#[allow(deprecated)]
use super::constants::{FRAGMENTATION_THRESHOLD, HYPERSYNC_THRESHOLD};
use super::thresholds::GwtThresholds;

/// Global Workspace state for GWT implementation.
///
/// The Global Workspace represents the currently coherent content
/// that is broadcast to all subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspace {
    /// Whether the workspace is active (ignited)
    pub active: bool,
    /// Current ignition level (coherence from per-space clustering)
    pub ignition_level: f32,
    /// Broadcast content when ignited
    pub broadcast_content: Option<serde_json::Value>,
    /// Current coherence state
    pub state: CoherenceState,
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        Self {
            active: false,
            ignition_level: 0.0,
            broadcast_content: None,
            state: CoherenceState::Dormant,
        }
    }
}

/// Coherence state from GWT state machine.
///
/// # Constitution Compliance (v6.0.0)
///
/// Per v6.0.0, CoherenceState represents topic-based coherence states.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CoherenceState {
    /// r < 0.3, no active workspace
    Dormant,
    /// 0.3 <= r < 0.5, partial sync
    Fragmented,
    /// 0.5 <= r < 0.8, approaching coherence
    Emerging,
    /// r >= 0.8, coherent workspace active
    Stable,
    /// r > 0.95, possibly pathological
    Hypersync,
}

impl CoherenceState {
    /// Determine state from order parameter r using provided thresholds.
    ///
    /// This method uses domain-aware thresholds from the ATC system to classify
    /// coherence states. Different domains may have different thresholds
    /// based on their strictness requirements.
    ///
    /// # State Classification
    ///
    /// - `Hypersync`: r > thresholds.hypersync (pathological over-synchronization)
    /// - `Stable`: r >= thresholds.gate (coherent workspace active)
    /// - `Emerging`: r >= thresholds.fragmentation (approaching coherence)
    /// - `Fragmented`: r >= 0.3 (partial synchronization)
    /// - `Dormant`: r < 0.3 (no active workspace)
    ///
    /// # Arguments
    ///
    /// * `r` - Coherence order parameter [0, 1]
    /// * `thresholds` - GWT thresholds for state boundaries
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use context_graph_core::layers::coherence::{CoherenceState, GwtThresholds};
    ///
    /// let thresholds = GwtThresholds::default_general();
    /// let state = CoherenceState::from_order_parameter_with_thresholds(0.75, &thresholds);
    /// assert_eq!(state, CoherenceState::Stable);
    /// ```
    pub fn from_order_parameter_with_thresholds(r: f32, thresholds: &GwtThresholds) -> Self {
        if r > thresholds.hypersync {
            Self::Hypersync
        } else if r >= thresholds.gate {
            Self::Stable
        } else if r >= thresholds.fragmentation {
            Self::Emerging
        } else if r >= 0.3 {
            Self::Fragmented
        } else {
            Self::Dormant
        }
    }

    /// Determine state from order parameter r using legacy default thresholds.
    ///
    /// # Deprecation Notice
    ///
    /// This method is deprecated. Use [`from_order_parameter_with_thresholds`](Self::from_order_parameter_with_thresholds)
    /// with explicit [`GwtThresholds`] for domain-aware behavior.
    ///
    /// # Legacy Thresholds Used
    ///
    /// - hypersync: 0.95 (HYPERSYNC_THRESHOLD)
    /// - gate: 0.80 (hardcoded)
    /// - fragmentation: 0.50 (FRAGMENTATION_THRESHOLD)
    #[deprecated(
        since = "0.5.0",
        note = "Use from_order_parameter_with_thresholds with GwtThresholds instead"
    )]
    #[allow(deprecated)]
    pub fn from_order_parameter(r: f32) -> Self {
        // Note: This uses 0.8 for gate (hardcoded in original) not 0.7 (GW_THRESHOLD)
        // This is the original behavior preserved for backwards compatibility
        if r > HYPERSYNC_THRESHOLD {
            Self::Hypersync
        } else if r >= 0.8 {
            Self::Stable
        } else if r >= FRAGMENTATION_THRESHOLD {
            Self::Emerging
        } else if r >= 0.3 {
            Self::Fragmented
        } else {
            Self::Dormant
        }
    }

    /// Check if this is a healthy state (not Dormant or Hypersync).
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Fragmented | Self::Emerging | Self::Stable)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================
    // NEW API TESTS
    // ========================================================

    #[test]
    fn test_coherence_state_with_default_thresholds() {
        let t = GwtThresholds::default_general();

        // Dormant: r < 0.3
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.1, &t),
            CoherenceState::Dormant
        );
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.29, &t),
            CoherenceState::Dormant
        );

        // Fragmented: 0.3 <= r < fragmentation(0.50)
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.30, &t),
            CoherenceState::Fragmented
        );
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.49, &t),
            CoherenceState::Fragmented
        );

        // Emerging: fragmentation(0.50) <= r < gate(0.70)
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.50, &t),
            CoherenceState::Emerging
        );
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.69, &t),
            CoherenceState::Emerging
        );

        // Stable: gate(0.70) <= r <= hypersync(0.95)
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.70, &t),
            CoherenceState::Stable
        );
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.85, &t),
            CoherenceState::Stable
        );
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.95, &t),
            CoherenceState::Stable
        );

        // Hypersync: r > hypersync(0.95)
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.96, &t),
            CoherenceState::Hypersync
        );
        assert_eq!(
            CoherenceState::from_order_parameter_with_thresholds(0.98, &t),
            CoherenceState::Hypersync
        );

        println!("[VERIFIED] Coherence state classification with GwtThresholds");
    }

    #[test]
    fn test_coherence_state_domain_variation() {
        use crate::atc::{AdaptiveThresholdCalibration, Domain};

        let atc = AdaptiveThresholdCalibration::new();

        // Code domain is stricter (higher gate)
        let code_t = GwtThresholds::from_atc(&atc, Domain::Code).unwrap();
        // Creative domain is looser (lower gate)
        let creative_t = GwtThresholds::from_atc(&atc, Domain::Creative).unwrap();

        // Test with r=0.80 - should be Stable in Creative but Emerging in Code
        let r = 0.80;
        let code_state = CoherenceState::from_order_parameter_with_thresholds(r, &code_t);
        let creative_state =
            CoherenceState::from_order_parameter_with_thresholds(r, &creative_t);

        println!("r=0.80:");
        println!("  Code domain (gate={:.3}): {:?}", code_t.gate, code_state);
        println!(
            "  Creative domain (gate={:.3}): {:?}",
            creative_t.gate, creative_state
        );

        // Code has higher gate, so 0.80 might still be Emerging
        // Creative has lower gate, so 0.80 should definitely be Stable
        assert_eq!(
            creative_state,
            CoherenceState::Stable,
            "r=0.80 should be Stable in Creative domain (gate={:.3})",
            creative_t.gate
        );

        println!("[VERIFIED] Domain-aware coherence classification");
    }

    // ========================================================
    // LEGACY API TESTS (with deprecation suppression)
    // ========================================================

    #[test]
    #[allow(deprecated)]
    fn test_coherence_state_from_r_legacy() {
        // Test legacy API still works (with deprecation warnings suppressed)
        assert_eq!(
            CoherenceState::from_order_parameter(0.1),
            CoherenceState::Dormant
        );
        assert_eq!(
            CoherenceState::from_order_parameter(0.35),
            CoherenceState::Fragmented
        );
        assert_eq!(
            CoherenceState::from_order_parameter(0.6),
            CoherenceState::Emerging
        );
        assert_eq!(
            CoherenceState::from_order_parameter(0.85),
            CoherenceState::Stable
        );
        assert_eq!(
            CoherenceState::from_order_parameter(0.98),
            CoherenceState::Hypersync
        );
        println!("[VERIFIED] Legacy from_order_parameter() still works");
    }

    #[test]
    fn test_coherence_state_health() {
        assert!(!CoherenceState::Dormant.is_healthy());
        assert!(CoherenceState::Fragmented.is_healthy());
        assert!(CoherenceState::Emerging.is_healthy());
        assert!(CoherenceState::Stable.is_healthy());
        assert!(!CoherenceState::Hypersync.is_healthy());
        println!("[VERIFIED] Coherence state health check");
    }

    // ========================================================
    // FSV TEST
    // ========================================================

    #[test]
    fn test_fsv_coherence_state_classification() {
        println!("\n=== FSV: CoherenceState Classification ===\n");

        let t = GwtThresholds::default_general();
        println!(
            "Using default thresholds: gate={}, hypersync={}, frag={}",
            t.gate, t.hypersync, t.fragmentation
        );
        println!();

        let test_cases: [(f32, CoherenceState); 10] = [
            (0.10, CoherenceState::Dormant),
            (0.29, CoherenceState::Dormant),
            (0.30, CoherenceState::Fragmented),
            (0.49, CoherenceState::Fragmented),
            (0.50, CoherenceState::Emerging),
            (0.69, CoherenceState::Emerging),
            (0.70, CoherenceState::Stable),
            (0.95, CoherenceState::Stable),
            (0.96, CoherenceState::Hypersync),
            (1.00, CoherenceState::Hypersync),
        ];

        println!("State Boundaries:");
        for (r, expected) in &test_cases {
            let actual = CoherenceState::from_order_parameter_with_thresholds(*r, &t);
            println!("  r={:.2} => {:?} (expected: {:?})", r, actual, expected);
            assert_eq!(actual, *expected, "Failed for r={}", r);
        }

        println!("\n[VERIFIED] All state classifications correct");
        println!("\n=== FSV COMPLETE ===\n");
    }
}
