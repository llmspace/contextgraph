//! State Machine Types
//!
//! Defines the state types for coherence levels.
//!
//! # Constitution Compliance (v6.0.0)
//!
//! Per Constitution v6.0.0 Section 14, this module implements topic-based
//! coherence scoring. CoherenceState represents order parameter levels.

use chrono::{DateTime, Utc};

/// Coherence state levels based on order parameter r.
///
/// # Constitution Compliance (v6.0.0)
///
/// States represent coherence levels aligned with topic stability:
/// - Dormant: r < 0.3, no active workspace
/// - Fragmented: 0.3 <= r < 0.5, partial synchronization
/// - Emerging: 0.5 <= r < 0.8, approaching coherence
/// - Stable: r >= 0.8, stable topic coherence (churn < 0.3)
/// - Hypersync: r > 0.95, pathological over-synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceState {
    /// r < 0.3, no active workspace
    Dormant,
    /// 0.3 <= r < 0.5, partial synchronization
    Fragmented,
    /// 0.5 <= r < 0.8, approaching coherence
    Emerging,
    /// r >= 0.8, stable topic coherence (churn < 0.3)
    Stable,
    /// r > 0.95, pathological over-synchronization (warning state)
    Hypersync,
}

impl CoherenceState {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dormant => "DORMANT",
            Self::Fragmented => "FRAGMENTED",
            Self::Emerging => "EMERGING",
            Self::Stable => "STABLE",
            Self::Hypersync => "HYPERSYNC",
        }
    }

    /// Get 3-character code for minimal token output.
    ///
    /// Used by PreToolUse hook which has ~20 token budget.
    /// Format: "[S:XXX r=Y.YY]"
    ///
    /// # Returns
    /// - "DOR" for Dormant (r < 0.3)
    /// - "FRG" for Fragmented (0.3 <= r < 0.5)
    /// - "EMG" for Emerging (0.5 <= r < 0.8)
    /// - "STB" for Stable (0.8 <= r < 0.95)
    /// - "HYP" for Hypersync (r > 0.95)
    #[inline]
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Dormant => "DOR",
            Self::Fragmented => "FRG",
            Self::Emerging => "EMG",
            Self::Stable => "STB",
            Self::Hypersync => "HYP",
        }
    }

    /// Determine state from coherence level (order parameter r).
    pub fn from_level(level: f32) -> Self {
        match level {
            l if l > 0.95 => Self::Hypersync,
            l if l >= 0.8 => Self::Stable,
            l if l >= 0.5 => Self::Emerging,
            l if l >= 0.3 => Self::Fragmented,
            _ => Self::Dormant,
        }
    }

    /// Check if this is a healthy state (not Dormant or Hypersync).
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Fragmented | Self::Emerging | Self::Stable)
    }
}

/// State transition with timestamp and context.
///
/// # Constitution Compliance (v6.0.0)
///
/// The `level` field represents the order parameter level (0.0-1.0).
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: CoherenceState,
    pub to: CoherenceState,
    pub timestamp: DateTime<Utc>,
    /// Order parameter level (0.0-1.0) at time of transition.
    pub level: f32,
}

/// Detailed transition analysis.
///
/// # Constitution Compliance (v6.0.0)
///
/// The `level_delta` field represents delta in coherence level.
#[derive(Debug, Clone)]
pub struct TransitionAnalysis {
    pub from_state: CoherenceState,
    pub to_state: CoherenceState,
    /// Delta in order parameter level.
    pub level_delta: f32,
    pub coherence_was_increasing: bool,
    pub is_recovery: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TC-SESSION-04: All State Short Name Mappings
    // Source of Truth: CoherenceState enum
    // =========================================================================
    #[test]
    fn test_coherence_state_short_name() {
        println!("\n=== TC-SESSION-04: CoherenceState.short_name() ===");
        println!("SOURCE OF TRUTH: CoherenceState enum variants");

        // Verify each variant returns correct 3-char code
        println!("\nBEFORE: Testing all 5 variants");

        let test_cases = [
            (CoherenceState::Dormant, "DOR", "Dormant"),
            (CoherenceState::Fragmented, "FRG", "Fragmented"),
            (CoherenceState::Emerging, "EMG", "Emerging"),
            (CoherenceState::Stable, "STB", "Stable"),
            (CoherenceState::Hypersync, "HYP", "Hypersync"),
        ];

        for (state, expected_code, full_name) in test_cases {
            let actual = state.short_name();
            println!(
                "  {:?} -> '{}' (expected '{}')",
                state, actual, expected_code
            );
            assert_eq!(
                actual, expected_code,
                "{} must return '{}', got '{}'",
                full_name, expected_code, actual
            );
        }

        println!("\nAFTER: All assertions passed");
        println!("RESULT: PASS - All 5 variants return correct 3-char codes");
    }

    // =========================================================================
    // TC-SESSION-04a: Verify All Codes Are Exactly 3 Characters
    // =========================================================================
    #[test]
    fn test_short_name_length_exactly_3() {
        println!("\n=== TC-SESSION-04a: short_name() Length Check ===");

        let all_states = [
            CoherenceState::Dormant,
            CoherenceState::Fragmented,
            CoherenceState::Emerging,
            CoherenceState::Stable,
            CoherenceState::Hypersync,
        ];

        println!("BEFORE: Checking length of each short_name()");

        for state in all_states {
            let code = state.short_name();
            println!(
                "  {:?}.short_name() = '{}' (len={})",
                state,
                code,
                code.len()
            );
            assert_eq!(
                code.len(),
                3,
                "{:?}.short_name() must be exactly 3 chars, got {} chars: '{}'",
                state,
                code.len(),
                code
            );
        }

        println!("AFTER: All codes verified to be 3 characters");
        println!("RESULT: PASS - All short_name() outputs are exactly 3 characters");
    }

    // =========================================================================
    // TC-SESSION-04b: Verify short_name() Returns Static Str (No Allocation)
    // =========================================================================
    #[test]
    fn test_short_name_is_static_str() {
        // This test verifies the return type at compile time
        // If this compiles, short_name() returns &'static str
        fn assert_static(_: &'static str) {}

        assert_static(CoherenceState::Dormant.short_name());
        assert_static(CoherenceState::Fragmented.short_name());
        assert_static(CoherenceState::Emerging.short_name());
        assert_static(CoherenceState::Stable.short_name());
        assert_static(CoherenceState::Hypersync.short_name());

        println!("RESULT: PASS - short_name() returns &'static str (no allocation)");
    }

    // =========================================================================
    // TC-SESSION-04c: Verify name() Works Correctly
    // =========================================================================
    #[test]
    fn test_name_method() {
        println!("\n=== TC-SESSION-04c: name() Test ===");

        assert_eq!(CoherenceState::Dormant.name(), "DORMANT");
        assert_eq!(CoherenceState::Fragmented.name(), "FRAGMENTED");
        assert_eq!(CoherenceState::Emerging.name(), "EMERGING");
        assert_eq!(CoherenceState::Stable.name(), "STABLE");
        assert_eq!(CoherenceState::Hypersync.name(), "HYPERSYNC");

        println!("RESULT: PASS - name() method works correctly");
    }

    // =========================================================================
    // TC-SESSION-04d: Verify from_level() Works Correctly
    // =========================================================================
    #[test]
    fn test_from_level() {
        println!("\n=== TC-SESSION-04d: from_level() Test ===");

        // Test boundary values
        let test_cases = [
            (0.0, CoherenceState::Dormant),
            (0.29, CoherenceState::Dormant),
            (0.3, CoherenceState::Fragmented),
            (0.49, CoherenceState::Fragmented),
            (0.5, CoherenceState::Emerging),
            (0.79, CoherenceState::Emerging),
            (0.8, CoherenceState::Stable),
            (0.95, CoherenceState::Stable),
            (0.96, CoherenceState::Hypersync),
            (1.0, CoherenceState::Hypersync),
        ];

        for (level, expected) in test_cases {
            let actual = CoherenceState::from_level(level);
            println!(
                "  from_level({:.2}) = {:?} (expected {:?})",
                level, actual, expected
            );
            assert_eq!(
                actual, expected,
                "from_level({}) must return {:?}",
                level, expected
            );
        }

        println!("RESULT: PASS - from_level() works correctly");
    }

    // =========================================================================
    // TC-SESSION-04e: Verify is_healthy() Works Correctly
    // =========================================================================
    #[test]
    fn test_is_healthy() {
        println!("\n=== TC-SESSION-04e: is_healthy() Test ===");

        assert!(!CoherenceState::Dormant.is_healthy());
        assert!(CoherenceState::Fragmented.is_healthy());
        assert!(CoherenceState::Emerging.is_healthy());
        assert!(CoherenceState::Stable.is_healthy());
        assert!(!CoherenceState::Hypersync.is_healthy());

        println!("RESULT: PASS - is_healthy() works correctly");
    }
}
