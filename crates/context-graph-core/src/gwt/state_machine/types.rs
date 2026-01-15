//! Consciousness State Machine Types
//!
//! Defines the state types for consciousness levels as specified in
//! Constitution v4.0.0 Section gwt.state_machine (lines 394-408).

use chrono::{DateTime, Utc};

/// Consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessState {
    Dormant,
    Fragmented,
    Emerging,
    Conscious,
    Hypersync,
}

impl ConsciousnessState {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dormant => "DORMANT",
            Self::Fragmented => "FRAGMENTED",
            Self::Emerging => "EMERGING",
            Self::Conscious => "CONSCIOUS",
            Self::Hypersync => "HYPERSYNC",
        }
    }

    /// Get 3-character code for minimal token output.
    ///
    /// Used by PreToolUse hook which has ~20 token budget.
    /// Format: "[C:XXX r=Y.YY IC=Z.ZZ]"
    ///
    /// # Returns
    /// - "DOR" for Dormant (C < 0.3)
    /// - "FRG" for Fragmented (0.3 <= C < 0.5)
    /// - "EMG" for Emerging (0.5 <= C < 0.8)
    /// - "CON" for Conscious (0.8 <= C < 0.95)
    /// - "HYP" for Hypersync (C > 0.95)
    #[inline]
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Dormant => "DOR",
            Self::Fragmented => "FRG",
            Self::Emerging => "EMG",
            Self::Conscious => "CON",
            Self::Hypersync => "HYP",
        }
    }

    /// Determine state from consciousness level
    pub fn from_level(level: f32) -> Self {
        match level {
            l if l > 0.95 => Self::Hypersync,
            l if l >= 0.8 => Self::Conscious,
            l if l >= 0.5 => Self::Emerging,
            l if l >= 0.3 => Self::Fragmented,
            _ => Self::Dormant,
        }
    }
}

/// State transition with timestamp and context
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: ConsciousnessState,
    pub to: ConsciousnessState,
    pub timestamp: DateTime<Utc>,
    pub consciousness_level: f32,
}

/// Detailed transition analysis
#[derive(Debug, Clone)]
pub struct TransitionAnalysis {
    pub from_state: ConsciousnessState,
    pub to_state: ConsciousnessState,
    pub consciousness_delta: f32,
    pub coherence_was_increasing: bool,
    pub is_recovery: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TC-SESSION-04: All State Short Name Mappings
    // Source of Truth: ConsciousnessState enum
    // =========================================================================
    #[test]
    fn test_consciousness_state_short_name() {
        println!("\n=== TC-SESSION-04: ConsciousnessState.short_name() ===");
        println!("SOURCE OF TRUTH: ConsciousnessState enum variants");

        // Verify each variant returns correct 3-char code
        println!("\nBEFORE: Testing all 5 variants");

        let test_cases = [
            (ConsciousnessState::Dormant, "DOR", "Dormant"),
            (ConsciousnessState::Fragmented, "FRG", "Fragmented"),
            (ConsciousnessState::Emerging, "EMG", "Emerging"),
            (ConsciousnessState::Conscious, "CON", "Conscious"),
            (ConsciousnessState::Hypersync, "HYP", "Hypersync"),
        ];

        for (state, expected_code, full_name) in test_cases {
            let actual = state.short_name();
            println!("  {:?} -> '{}' (expected '{}')", state, actual, expected_code);
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
            ConsciousnessState::Dormant,
            ConsciousnessState::Fragmented,
            ConsciousnessState::Emerging,
            ConsciousnessState::Conscious,
            ConsciousnessState::Hypersync,
        ];

        println!("BEFORE: Checking length of each short_name()");

        for state in all_states {
            let code = state.short_name();
            println!("  {:?}.short_name() = '{}' (len={})", state, code, code.len());
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

        assert_static(ConsciousnessState::Dormant.short_name());
        assert_static(ConsciousnessState::Fragmented.short_name());
        assert_static(ConsciousnessState::Emerging.short_name());
        assert_static(ConsciousnessState::Conscious.short_name());
        assert_static(ConsciousnessState::Hypersync.short_name());

        println!("RESULT: PASS - short_name() returns &'static str (no allocation)");
    }

    // =========================================================================
    // TC-SESSION-04c: Verify name() Still Works (Regression Test)
    // =========================================================================
    #[test]
    fn test_name_method_unchanged() {
        println!("\n=== TC-SESSION-04c: name() Regression Test ===");

        assert_eq!(ConsciousnessState::Dormant.name(), "DORMANT");
        assert_eq!(ConsciousnessState::Fragmented.name(), "FRAGMENTED");
        assert_eq!(ConsciousnessState::Emerging.name(), "EMERGING");
        assert_eq!(ConsciousnessState::Conscious.name(), "CONSCIOUS");
        assert_eq!(ConsciousnessState::Hypersync.name(), "HYPERSYNC");

        println!("RESULT: PASS - name() method unchanged");
    }

    // =========================================================================
    // EDGE CASE: Verify from_level() Still Works (Regression Test)
    // =========================================================================
    #[test]
    fn test_from_level_unchanged() {
        println!("\n=== EDGE CASE: from_level() Regression Test ===");

        // Test boundary values
        let test_cases = [
            (0.0, ConsciousnessState::Dormant),
            (0.29, ConsciousnessState::Dormant),
            (0.3, ConsciousnessState::Fragmented),
            (0.49, ConsciousnessState::Fragmented),
            (0.5, ConsciousnessState::Emerging),
            (0.79, ConsciousnessState::Emerging),
            (0.8, ConsciousnessState::Conscious),
            (0.95, ConsciousnessState::Conscious),
            (0.96, ConsciousnessState::Hypersync),
            (1.0, ConsciousnessState::Hypersync),
        ];

        for (level, expected) in test_cases {
            let actual = ConsciousnessState::from_level(level);
            println!("  from_level({:.2}) = {:?} (expected {:?})", level, actual, expected);
            assert_eq!(actual, expected, "from_level({}) must return {:?}", level, expected);
        }

        println!("RESULT: PASS - from_level() unchanged");
    }
}
