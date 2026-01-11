//! State Machine Manager
//!
//! Manages consciousness state transitions and events.

use crate::error::CoreResult;
use chrono::{DateTime, Duration, Utc};

use super::types::{ConsciousnessState, StateTransition};

/// Manages consciousness state transitions and events
#[derive(Debug)]
pub struct StateMachineManager {
    /// Current state
    current_state: ConsciousnessState,
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
            current_state: ConsciousnessState::Dormant,
            last_transition: None,
            entered_state_at: Utc::now(),
            inactivity_timeout_secs: 600, // 10 minutes
            last_activity: Utc::now(),
        }
    }

    /// Get current state
    pub fn current_state(&self) -> ConsciousnessState {
        self.current_state
    }

    /// Get time spent in current state
    pub fn time_in_state(&self) -> Duration {
        Utc::now() - self.entered_state_at
    }

    /// Update state based on consciousness level
    pub async fn update(&mut self, consciousness_level: f32) -> CoreResult<ConsciousnessState> {
        // Update activity timestamp
        self.last_activity = Utc::now();

        // Determine new state
        let new_state = ConsciousnessState::from_level(consciousness_level);

        // Check for inactivity-driven transition to dormant
        if self.time_in_state().num_seconds() > self.inactivity_timeout_secs as i64
            && self.current_state != ConsciousnessState::Dormant
        {
            self.transition_to(
                ConsciousnessState::Dormant,
                consciousness_level,
                "inactivity_timeout",
            )
            .await?;
            return Ok(self.current_state);
        }

        // Check if state changed
        if new_state != self.current_state {
            self.transition_to(new_state, consciousness_level, "consciousness_level_change")
                .await?;
        }

        Ok(self.current_state)
    }

    /// Execute transition with logging
    async fn transition_to(
        &mut self,
        new_state: ConsciousnessState,
        consciousness_level: f32,
        reason: &str,
    ) -> CoreResult<()> {
        let old_state = self.current_state;
        self.current_state = new_state;
        self.entered_state_at = Utc::now();

        let transition = StateTransition {
            from: old_state,
            to: new_state,
            timestamp: Utc::now(),
            consciousness_level,
        };

        self.last_transition = Some(transition.clone());

        // Log transition
        tracing::info!(
            "State transition: {} → {} (level={:.3}, reason={})",
            old_state.name(),
            new_state.name(),
            consciousness_level,
            reason
        );

        Ok(())
    }

    /// Get the last transition
    pub fn last_transition(&self) -> Option<&StateTransition> {
        self.last_transition.as_ref()
    }

    /// Check if system just became conscious
    pub fn just_became_conscious(&self) -> bool {
        if let Some(trans) = &self.last_transition {
            trans.to == ConsciousnessState::Conscious
                && (Utc::now() - trans.timestamp).num_milliseconds() < 1000
        } else {
            false
        }
    }

    /// Check if system is in a conscious state
    pub fn is_conscious(&self) -> bool {
        matches!(
            self.current_state,
            ConsciousnessState::Conscious | ConsciousnessState::Hypersync
        )
    }

    /// Check if system is hypersynchronized (warning state)
    pub fn is_hypersync(&self) -> bool {
        self.current_state == ConsciousnessState::Hypersync
    }

    /// Check if coherence is fragmented
    pub fn is_fragmented(&self) -> bool {
        self.current_state == ConsciousnessState::Fragmented
    }

    /// Check if system is dormant
    pub fn is_dormant(&self) -> bool {
        self.current_state == ConsciousnessState::Dormant
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
