//! Synthetic test data with known inputs and expected outputs.
//!
//! These constants are provided as a reference library for test fixtures.
//! Not all constants are used in every test suite.

#![allow(dead_code)]

/// Content strings for testing with known semantic properties.
pub mod content {
    pub const SIMPLE_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
    pub const TECHNICAL_CODE: &str = "fn main() { println!(\"Hello, world!\"); }";
    pub const EMPTY: &str = "";
    pub const VERY_LONG: &str = "This is a very long content string that exceeds typical sizes. \
        It contains multiple sentences and paragraphs to test how the system handles larger inputs. \
        The system should process this without issues and return appropriate results. \
        Memory systems need to handle content of varying sizes gracefully.";
    pub const UNICODE: &str = "こんにちは世界 🌍 Привет мир 你好世界";
    pub const SPECIAL_CHARS: &str = "Content with <html> tags & \"quotes\" and 'apostrophes'";
}

/// Importance values for testing boundary conditions.
pub mod importance {
    pub const MIN: f64 = 0.0;
    pub const MAX: f64 = 1.0;
    pub const MID: f64 = 0.5;
    pub const HIGH: f64 = 0.85;
    pub const LOW: f64 = 0.15;
    pub const INVALID_NEGATIVE: f64 = -0.1;
    pub const INVALID_ABOVE_MAX: f64 = 1.1;
}

/// Test UUIDs for causal/workspace operations.
pub mod uuids {
    pub const VALID_SOURCE: &str = "550e8400-e29b-41d4-a716-446655440000";
    pub const VALID_TARGET: &str = "550e8400-e29b-41d4-a716-446655440001";
    pub const INVALID_FORMAT: &str = "not-a-valid-uuid";
    pub const NON_EXISTENT: &str = "00000000-0000-0000-0000-000000000000";
}


/// Dream system states (accepts both cases from different implementations).
pub mod dream {
    pub const STATE_AWAKE: &str = "Awake";
    pub const STATE_NREM: &str = "NREM";
    pub const STATE_REM: &str = "REM";
    pub const STATE_WAKING: &str = "Waking";
    // Include lowercase variants returned by some handlers
    pub const VALID_STATES: [&str; 8] = [
        "Awake", "awake", "NREM", "nrem", "REM", "rem", "Waking", "waking",
    ];
}

/// Neuromodulator expected ranges (from constitution.yaml).
pub mod neuromod {
    pub const DOPAMINE_MIN: f64 = 1.0;
    pub const DOPAMINE_MAX: f64 = 5.0;
    pub const SEROTONIN_MIN: f64 = 0.0;
    pub const SEROTONIN_MAX: f64 = 1.0;
    pub const NORADRENALINE_MIN: f64 = 0.5;
    pub const NORADRENALINE_MAX: f64 = 2.0;
    pub const ACETYLCHOLINE_MIN: f64 = 0.001;
    pub const ACETYLCHOLINE_MAX: f64 = 0.002;
}

/// ATC levels.
pub mod atc {
    pub const LEVEL_EWMA: i32 = 1;
    pub const LEVEL_TEMPERATURE: i32 = 2;
    pub const LEVEL_BANDIT: i32 = 3;
    pub const LEVEL_BAYESIAN: i32 = 4;
    pub const LEVEL_MIN: i32 = 1;
    pub const LEVEL_MAX: i32 = 4;
}

/// UTL lifecycle phases.
pub mod lifecycle {
    pub const INFANCY: &str = "Infancy";
    pub const CHILDHOOD: &str = "Childhood";
    pub const ADOLESCENCE: &str = "Adolescence";
    pub const ADULTHOOD: &str = "Adulthood";
}

/// Consolidation phases.
pub mod consolidation {
    pub const WAKE: &str = "Wake";
    pub const NREM: &str = "NREM";
    pub const REM: &str = "REM";
    pub const VALID_PHASES: [&str; 3] = [WAKE, NREM, REM];
}
