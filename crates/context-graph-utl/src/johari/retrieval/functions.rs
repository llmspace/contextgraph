//! Standalone retrieval functions for Johari quadrant operations.
//!
//! These functions provide simple mappings between quadrants and their
//! corresponding actions and weights.

use context_graph_core::types::JohariQuadrant;

use super::action::SuggestedAction;

/// Returns the suggested action for a given Johari quadrant.
///
/// # Constitution Compliance (constitution.yaml utl.johari lines 154-157)
/// - Open (ΔS<0.5, ΔC>0.5) → DirectRecall
/// - Hidden (ΔS<0.5, ΔC<0.5) → GetNeighborhood
/// - Blind (ΔS>0.5, ΔC<0.5) → TriggerDream (ISS-011 FIX)
/// - Unknown (ΔS>0.5, ΔC>0.5) → EpistemicAction (ISS-011 FIX)
///
/// # Arguments
///
/// * `quadrant` - The Johari quadrant to get the action for
///
/// # Returns
///
/// The appropriate `SuggestedAction` for the quadrant.
///
/// # Example
///
/// ```
/// use context_graph_utl::johari::{get_suggested_action, SuggestedAction, JohariQuadrant};
///
/// assert_eq!(get_suggested_action(JohariQuadrant::Open), SuggestedAction::DirectRecall);
/// assert_eq!(get_suggested_action(JohariQuadrant::Blind), SuggestedAction::TriggerDream);
/// assert_eq!(get_suggested_action(JohariQuadrant::Hidden), SuggestedAction::GetNeighborhood);
/// assert_eq!(get_suggested_action(JohariQuadrant::Unknown), SuggestedAction::EpistemicAction);
/// ```
#[inline]
pub fn get_suggested_action(quadrant: JohariQuadrant) -> SuggestedAction {
    match quadrant {
        // Low surprise, high confidence → Direct retrieval works
        JohariQuadrant::Open => SuggestedAction::DirectRecall,

        // Low surprise, low confidence → Explore neighborhood for context
        JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,

        // High surprise, low confidence → Need dream consolidation to integrate
        // FIXED ISS-011: Was incorrectly EpistemicAction
        JohariQuadrant::Blind => SuggestedAction::TriggerDream,

        // High surprise, high confidence → Epistemic action to update beliefs
        // FIXED ISS-011: Was incorrectly TriggerDream
        JohariQuadrant::Unknown => SuggestedAction::EpistemicAction,
    }
}

/// Returns the retrieval weight for a given Johari quadrant.
///
/// This function returns the default retrieval weight as defined by the
/// `JohariQuadrant::default_retrieval_weight()` method in context-graph-core.
///
/// # Arguments
///
/// * `quadrant` - The Johari quadrant to get the weight for
///
/// # Returns
///
/// A weight value in range [0.0, 1.0]:
/// - Open: 1.0 (full weight)
/// - Blind: 0.7 (high discovery weight)
/// - Hidden: 0.3 (reduced private weight)
/// - Unknown: 0.5 (medium frontier weight)
///
/// # Example
///
/// ```
/// use context_graph_utl::johari::{get_retrieval_weight, JohariQuadrant};
///
/// assert_eq!(get_retrieval_weight(JohariQuadrant::Open), 1.0);
/// assert_eq!(get_retrieval_weight(JohariQuadrant::Hidden), 0.3);
/// ```
#[inline]
pub fn get_retrieval_weight(quadrant: JohariQuadrant) -> f32 {
    quadrant.default_retrieval_weight()
}
