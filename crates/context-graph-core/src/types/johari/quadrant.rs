//! JohariQuadrant enum and related implementations.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use super::transition::TransitionTrigger;
use super::JohariTransition;

/// Represents the four quadrants of the Johari Window model for memory classification.
///
/// # UTL Integration
/// From constitution.yaml, Johari quadrants map to UTL states:
/// - **Open**: ΔS<0.5, ΔC>0.5 → direct recall
/// - **Blind**: ΔS>0.5, ΔC<0.5 → discovery (epistemic_action/dream)
/// - **Hidden**: ΔS<0.5, ΔC<0.5 → private (get_neighborhood)
/// - **Unknown**: ΔS>0.5, ΔC>0.5 → frontier
///
/// # Performance
/// All methods are O(1) with no allocations per constitution.yaml requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    /// Known to self and others - direct recall
    Open,
    /// Known to self, hidden from others - private knowledge
    Hidden,
    /// Unknown to self, known to others - discovered patterns
    Blind,
    /// Unknown to both - frontier knowledge
    Unknown,
}

impl JohariQuadrant {
    /// Returns true if this quadrant represents self-aware knowledge.
    /// Open and Hidden quadrants are self-aware.
    ///
    /// # Returns
    /// - `true` for Open, Hidden
    /// - `false` for Blind, Unknown
    #[inline]
    pub fn is_self_aware(&self) -> bool {
        matches!(self, Self::Open | Self::Hidden)
    }

    /// Returns true if this quadrant represents knowledge visible to others.
    /// Open and Blind quadrants are other-aware.
    ///
    /// # Returns
    /// - `true` for Open, Blind
    /// - `false` for Hidden, Unknown
    #[inline]
    pub fn is_other_aware(&self) -> bool {
        matches!(self, Self::Open | Self::Blind)
    }

    /// Returns the default retrieval weight for this quadrant.
    ///
    /// # Returns
    /// - Open: 1.0 (full weight, always retrieve)
    /// - Hidden: 0.3 (reduced weight, private)
    /// - Blind: 0.7 (high weight, discovery)
    /// - Unknown: 0.5 (medium weight, frontier)
    ///
    /// # Constraint
    /// All values in range [0.0, 1.0]
    #[inline]
    pub fn default_retrieval_weight(&self) -> f32 {
        match self {
            Self::Open => 1.0,
            Self::Hidden => 0.3,
            Self::Blind => 0.7,
            Self::Unknown => 0.5,
        }
    }

    /// Returns whether this quadrant should be included in default context retrieval.
    ///
    /// # Returns
    /// - `true` for Open, Blind, Unknown
    /// - `false` for Hidden (private knowledge requires explicit request)
    #[inline]
    pub fn include_in_default_context(&self) -> bool {
        matches!(self, Self::Open | Self::Blind | Self::Unknown)
    }

    /// Returns a human-readable description of this quadrant.
    ///
    /// # Returns
    /// Static string describing quadrant semantics.
    #[inline]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Open => "Known to self and others - direct recall",
            Self::Hidden => "Known to self, hidden from others - private knowledge",
            Self::Blind => "Unknown to self, known to others - discovered patterns",
            Self::Unknown => "Unknown to both - frontier knowledge",
        }
    }

    /// Returns the RocksDB column family name for this quadrant.
    ///
    /// # Column Family Names
    /// - "johari_open"
    /// - "johari_hidden"
    /// - "johari_blind"
    /// - "johari_unknown"
    #[inline]
    pub fn column_family(&self) -> &'static str {
        match self {
            Self::Open => "johari_open",
            Self::Hidden => "johari_hidden",
            Self::Blind => "johari_blind",
            Self::Unknown => "johari_unknown",
        }
    }

    /// Returns all quadrant variants as a fixed-size array.
    ///
    /// # Returns
    /// Array containing [Open, Hidden, Blind, Unknown] in canonical order.
    #[inline]
    pub fn all() -> [JohariQuadrant; 4] {
        [Self::Open, Self::Hidden, Self::Blind, Self::Unknown]
    }

    /// Get all valid transitions from this quadrant.
    ///
    /// Returns a static slice of (target_quadrant, trigger) pairs representing
    /// all legal state transitions from the current quadrant.
    ///
    /// # Transition Rules (from constitution.yaml)
    /// - Open → Hidden (Privatize)
    /// - Hidden → Open (ExplicitShare)
    /// - Blind → Open (SelfRecognition), Hidden (SelfRecognition)
    /// - Unknown → Open (DreamConsolidation, PatternDiscovery), Hidden (DreamConsolidation), Blind (ExternalObservation)
    pub fn valid_transitions(&self) -> &'static [(JohariQuadrant, TransitionTrigger)] {
        use TransitionTrigger::*;
        static OPEN_TRANSITIONS: [(JohariQuadrant, TransitionTrigger); 1] =
            [(JohariQuadrant::Hidden, Privatize)];
        static HIDDEN_TRANSITIONS: [(JohariQuadrant, TransitionTrigger); 1] =
            [(JohariQuadrant::Open, ExplicitShare)];
        static BLIND_TRANSITIONS: [(JohariQuadrant, TransitionTrigger); 2] = [
            (JohariQuadrant::Open, SelfRecognition),
            (JohariQuadrant::Hidden, SelfRecognition),
        ];
        static UNKNOWN_TRANSITIONS: [(JohariQuadrant, TransitionTrigger); 4] = [
            (JohariQuadrant::Open, DreamConsolidation),
            (JohariQuadrant::Open, PatternDiscovery),
            (JohariQuadrant::Hidden, DreamConsolidation),
            (JohariQuadrant::Blind, ExternalObservation),
        ];

        match self {
            Self::Open => &OPEN_TRANSITIONS,
            Self::Hidden => &HIDDEN_TRANSITIONS,
            Self::Blind => &BLIND_TRANSITIONS,
            Self::Unknown => &UNKNOWN_TRANSITIONS,
        }
    }

    /// Check if a transition to the target quadrant is valid.
    ///
    /// Returns false for self-transitions (from == to).
    pub fn can_transition_to(&self, target: JohariQuadrant) -> bool {
        if *self == target {
            return false; // No self-transitions allowed
        }
        self.valid_transitions().iter().any(|(t, _)| *t == target)
    }

    /// Attempt to transition to a target quadrant with the given trigger.
    ///
    /// # Returns
    /// - `Ok(JohariTransition)` if the transition is valid for this trigger
    /// - `Err(String)` with descriptive message if transition is invalid
    ///
    /// # Errors
    /// - Self-transitions (from == to)
    /// - Invalid target quadrant for this source
    /// - Wrong trigger for the source→target pair
    pub fn transition_to(
        &self,
        target: JohariQuadrant,
        trigger: TransitionTrigger,
    ) -> Result<JohariTransition, String> {
        if *self == target {
            return Err(format!("Cannot transition to same quadrant: {:?}", self));
        }

        let is_valid = self
            .valid_transitions()
            .iter()
            .any(|(t, tr)| *t == target && *tr == trigger);

        if is_valid {
            Ok(JohariTransition::new(*self, target, trigger))
        } else {
            Err(format!(
                "Invalid transition: {:?} -> {:?} via {:?}. Valid transitions from {:?}: {:?}",
                self,
                target,
                trigger,
                self,
                self.valid_transitions()
            ))
        }
    }
}

impl Default for JohariQuadrant {
    /// Default quadrant is Open (most accessible).
    fn default() -> Self {
        Self::Open
    }
}

impl fmt::Display for JohariQuadrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "Open"),
            Self::Hidden => write!(f, "Hidden"),
            Self::Blind => write!(f, "Blind"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl FromStr for JohariQuadrant {
    type Err = String;

    /// Parses a string into a JohariQuadrant (case-insensitive).
    ///
    /// # Accepted Values
    /// "open", "OPEN", "Open", etc. for each variant
    ///
    /// # Errors
    /// Returns error string if input doesn't match any variant.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "open" => Ok(Self::Open),
            "hidden" => Ok(Self::Hidden),
            "blind" => Ok(Self::Blind),
            "unknown" => Ok(Self::Unknown),
            _ => Err(format!(
                "Invalid JohariQuadrant: '{}'. Valid values: open, hidden, blind, unknown",
                s
            )),
        }
    }
}
