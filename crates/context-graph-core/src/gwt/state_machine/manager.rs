//! State Machine Manager
//!
//! Manages coherence state transitions and events.
//!
//! # Constitution Compliance (v6.0.0)
//!
//! Per Constitution v6.0.0 Section 14, this manager handles topic-based coherence
//! state transitions.

use crate::error::CoreResult;
use chrono::{DateTime, Duration, Utc};

use super::types::{CoherenceState, StateTransition};

/// Manages coherence state transitions and events.
///
/// # Constitution Compliance (v6.0.0)
///
/// Handles order parameter level transitions for coherence states.
#[derive(Debug)]
pub struct StateMachineManager {
    /// Current coherence state
    current_state: CoherenceState,
    /// Last state transition
    last_transition: Option<StateTransition>,
    /// Time entered current state
    entered_state_at: DateTime<Utc>,
    /// Inactivity timeout (dormant after this duration)
    inactivity_timeout_secs: u64,
    /// Last activity timestamp
    last_activity: DateTime<Utc>,
}

impl StateMachineManager {
    /// Create a new state machine manager
    pub fn new() -> Self {
        Self {
            current_state: CoherenceState::Dormant,
            last_transition: None,
            entered_state_at: Utc::now(),
            inactivity_timeout_secs: 600, // 10 minutes
            last_activity: Utc::now(),
        }
    }

    /// Get current coherence state
    pub fn current_state(&self) -> CoherenceState {
        self.current_state
    }

    /// Get time spent in current state
    pub fn time_in_state(&self) -> Duration {
        Utc::now() - self.entered_state_at
    }

    /// Update state based on order parameter level.
    ///
    /// # Arguments
    ///
    /// * `level` - Order parameter level (0.0-1.0)
    pub async fn update(&mut self, level: f32) -> CoreResult<CoherenceState> {
        // Update activity timestamp
        self.last_activity = Utc::now();

        // Determine new state
        let new_state = CoherenceState::from_level(level);

        // Check for inactivity-driven transition to dormant
        if self.time_in_state().num_seconds() > self.inactivity_timeout_secs as i64
            && self.current_state != CoherenceState::Dormant
        {
            self.transition_to(CoherenceState::Dormant, level, "inactivity_timeout")
                .await?;
            return Ok(self.current_state);
        }

        // Check if state changed
        if new_state != self.current_state {
            self.transition_to(new_state, level, "level_change")
                .await?;
        }

        Ok(self.current_state)
    }

    /// Execute transition with logging.
    async fn transition_to(
        &mut self,
        new_state: CoherenceState,
        level: f32,
        reason: &str,
    ) -> CoreResult<()> {
        let old_state = self.current_state;
        self.current_state = new_state;
        self.entered_state_at = Utc::now();

        let transition = StateTransition {
            from: old_state,
            to: new_state,
            timestamp: Utc::now(),
            level,
        };

        self.last_transition = Some(transition.clone());

        // Log transition
        tracing::info!(
            "State transition: {} -> {} (level={:.3}, reason={})",
            old_state.name(),
            new_state.name(),
            level,
            reason
        );

        Ok(())
    }

    /// Get the last transition
    pub fn last_transition(&self) -> Option<&StateTransition> {
        self.last_transition.as_ref()
    }

    /// Check if system just became stable (coherent).
    ///
    /// Returns true if the system transitioned to Stable state within the last 1000ms.
    pub fn just_became_stable(&self) -> bool {
        if let Some(trans) = &self.last_transition {
            trans.to == CoherenceState::Stable
                && (Utc::now() - trans.timestamp).num_milliseconds() < 1000
        } else {
            false
        }
    }

    /// Check if system is in a stable (coherent) state.
    ///
    /// Returns true for Stable or Hypersync states.
    /// Hypersync is considered a form of stability (albeit pathological).
    pub fn is_stable(&self) -> bool {
        matches!(
            self.current_state,
            CoherenceState::Stable | CoherenceState::Hypersync
        )
    }

    /// Check if system is hypersynchronized (warning state)
    pub fn is_hypersync(&self) -> bool {
        self.current_state == CoherenceState::Hypersync
    }

    /// Check if coherence is fragmented
    pub fn is_fragmented(&self) -> bool {
        self.current_state == CoherenceState::Fragmented
    }

    /// Check if system is dormant
    pub fn is_dormant(&self) -> bool {
        self.current_state == CoherenceState::Dormant
    }

    /// Set inactivity timeout
    pub fn set_inactivity_timeout(&mut self, seconds: u64) {
        self.inactivity_timeout_secs = seconds;
    }
}

impl Default for StateMachineManager {
    fn default() -> Self {
        Self::new()
    }
}
