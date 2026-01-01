//! Knowledge domain classification for context-aware neurotransmitter weighting.
//!
//! # Constitution Reference
//! - edge_model.nt_weights.domain: Code|Legal|Medical|Creative|Research|General

use serde::{Deserialize, Serialize};
use std::fmt;

/// Knowledge domain for context-aware neurotransmitter weighting.
///
/// Different domains have different optimal retrieval characteristics:
/// - Code: High precision, structured relationships
/// - Legal: High inhibition, careful reasoning
/// - Medical: High causal awareness, evidence-based
/// - Creative: High exploration, associative connections
/// - Research: Balanced exploration and precision
/// - General: Default balanced profile
///
/// # Constitution Compliance
/// - Naming: PascalCase enum per constitution.yaml
/// - Serde: snake_case serialization per JSON naming rules
///
/// # Example
/// ```rust
/// use context_graph_core::marblestone::Domain;
///
/// let domain = Domain::Code;
/// assert_eq!(domain.to_string(), "code");
/// assert!(domain.description().contains("precision"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    /// Programming and software development context.
    /// Characteristics: High precision, structured relationships, strong type awareness.
    Code,
    /// Legal documents and reasoning context.
    /// Characteristics: High inhibition, careful reasoning, precedent-based.
    Legal,
    /// Medical and healthcare context.
    /// Characteristics: High causal awareness, evidence-based, risk-conscious.
    Medical,
    /// Creative writing and artistic context.
    /// Characteristics: High exploration, associative connections, novelty-seeking.
    Creative,
    /// Academic research context.
    /// Characteristics: Balanced exploration and precision, citation-aware.
    Research,
    /// General purpose context.
    /// Characteristics: Default balanced profile for mixed contexts.
    General,
}

impl Domain {
    /// Returns a human-readable description of this domain's characteristics.
    ///
    /// # Returns
    /// Static string describing the domain's retrieval behavior.
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::Domain;
    ///
    /// let desc = Domain::Medical.description();
    /// assert!(desc.contains("causal"));
    /// ```
    #[inline]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Code => "High precision, structured relationships, strong type awareness",
            Self::Legal => "High inhibition, careful reasoning, precedent-based",
            Self::Medical => "High causal awareness, evidence-based, risk-conscious",
            Self::Creative => "High exploration, associative connections, novelty-seeking",
            Self::Research => "Balanced exploration and precision, citation-aware",
            Self::General => "Default balanced profile for mixed contexts",
        }
    }

    /// Returns all domain variants as an array.
    ///
    /// # Returns
    /// Array containing all 6 Domain variants in definition order.
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::marblestone::Domain;
    ///
    /// let all = Domain::all();
    /// assert_eq!(all.len(), 6);
    /// assert_eq!(all[0], Domain::Code);
    /// assert_eq!(all[5], Domain::General);
    /// ```
    #[inline]
    pub fn all() -> [Domain; 6] {
        [
            Self::Code,
            Self::Legal,
            Self::Medical,
            Self::Creative,
            Self::Research,
            Self::General,
        ]
    }
}

impl Default for Domain {
    /// Returns `Domain::General` as the default.
    ///
    /// General is the most balanced profile, suitable for mixed contexts.
    #[inline]
    fn default() -> Self {
        Self::General
    }
}

impl fmt::Display for Domain {
    /// Formats the domain as a lowercase string.
    ///
    /// # Output
    /// - Code → "code"
    /// - Legal → "legal"
    /// - Medical → "medical"
    /// - Creative → "creative"
    /// - Research → "research"
    /// - General → "general"
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Code => "code",
            Self::Legal => "legal",
            Self::Medical => "medical",
            Self::Creative => "creative",
            Self::Research => "research",
            Self::General => "general",
        };
        write!(f, "{}", s)
    }
}
