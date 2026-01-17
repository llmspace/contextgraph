//! Goal hierarchy level definition.
//!
//! TASK-P0-001: Removed NorthStar level per ARCH-03 (autonomous operation).
//! Goals now emerge autonomously from data patterns, not manual definition.
//! Strategic is now the top level (value 0).

use serde::{Deserialize, Serialize};

/// Goal hierarchy level.
///
/// Defines the position of a goal in the hierarchical tree structure.
/// Each level has a different propagation weight for alignment computation.
///
/// From constitution.yaml (after TASK-P0-001):
/// - Strategic: 1.0 weight (top-level, emergent from data)
/// - Tactical: 0.6 weight (short-term)
/// - Immediate: 0.3 weight (per-operation)
///
/// # ARCH-03 Compliance
/// Goals emerge autonomously from teleological fingerprints.
/// Manual goal setting (set_north_star, define_goal) is forbidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum GoalLevel {
    // REMOVED: NorthStar per TASK-P0-001 (ARCH-03)
    // North Star was a manual goal concept - goals now emerge autonomously
    /// Top-level strategic objectives (emergent from data patterns).
    /// Highest level in hierarchy.
    Strategic = 0,

    /// Short-term tactical goals.
    /// Children of Strategic goals.
    Tactical = 1,

    /// Immediate context goals.
    /// Lowest level, most specific.
    Immediate = 2,
}

impl GoalLevel {
    /// Weight factor for hierarchical propagation.
    ///
    /// From constitution.yaml (after TASK-P0-001):
    /// - Strategic: 1.0 (top-level, emergent)
    /// - Tactical: 0.6
    /// - Immediate: 0.3
    #[inline]
    pub fn propagation_weight(&self) -> f32 {
        match self {
            // REMOVED: NorthStar per TASK-P0-001
            GoalLevel::Strategic => 1.0, // Now top-level
            GoalLevel::Tactical => 0.6,
            GoalLevel::Immediate => 0.3,
        }
    }

    /// Get numeric depth (0 = Strategic, 2 = Immediate).
    #[inline]
    pub fn depth(&self) -> u8 {
        *self as u8
    }

    /// Check if this is the top level (Strategic).
    ///
    /// Replaces the old `is_north_star()` check.
    #[inline]
    pub fn is_top_level(&self) -> bool {
        *self == GoalLevel::Strategic
    }
}
