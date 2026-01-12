//! Conversion helpers for Teleological types.
//!
//! Provides conversions between JSON representations and core types,
//! as well as parsing utilities for scope and feedback types.

use context_graph_core::teleological::{
    services::feedback_learner::FeedbackType, ComparisonScope, ComponentWeights, GroupType,
    TeleologicalVector,
};
use uuid::Uuid;

use super::types::{ComponentWeightsJson, TeleologicalVectorJson};

// ============================================================================
// CONVERSION IMPLEMENTATIONS
// ============================================================================

impl TeleologicalVectorJson {
    /// Convert to core TeleologicalVector.
    /// Reserved for future use when clients need to submit pre-computed vectors.
    #[allow(dead_code)]
    pub fn to_core(&self) -> TeleologicalVector {
        use context_graph_core::teleological::groups::GroupAlignments;
        use context_graph_core::types::fingerprint::PurposeVector;

        // Create PurposeVector from alignments
        let purpose_vector = PurposeVector::new(self.purpose_vector);

        // Create GroupAlignments from 6D array
        let group_alignments = GroupAlignments::new(
            self.group_alignments[0], // factual
            self.group_alignments[1], // temporal
            self.group_alignments[2], // causal
            self.group_alignments[3], // relational
            self.group_alignments[4], // qualitative
            self.group_alignments[5], // implementation
        );

        // Use with_all for complete construction
        TeleologicalVector::with_all(
            purpose_vector,
            self.cross_correlations.clone(),
            group_alignments,
            self.confidence,
        )
    }

    /// Create from core TeleologicalVector.
    pub fn from_core(tv: &TeleologicalVector, id: Option<Uuid>) -> Self {
        Self {
            purpose_vector: tv.purpose_vector.alignments,
            cross_correlations: tv.cross_correlations.clone(),
            group_alignments: tv.group_alignments.as_array(),
            confidence: tv.confidence,
            id: id.map(|u| u.to_string()),
        }
    }
}

impl ComponentWeightsJson {
    /// Convert to core ComponentWeights.
    /// Reserved for future use when custom component weights are supported in API.
    #[allow(dead_code)]
    pub fn to_core(&self) -> ComponentWeights {
        ComponentWeights {
            purpose_vector: self.purpose_vector,
            cross_correlations: self.cross_correlations,
            group_alignments: self.group_alignments,
            confidence: self.confidence,
        }
    }
}

// ============================================================================
// PARSING UTILITIES
// ============================================================================

/// Parse scope string into ComparisonScope enum.
///
/// ISSUE-1 FIX: Updated to match tool definition - removed specific_pairs (not in schema).
#[allow(dead_code)]
pub fn parse_scope(
    s: &str,
    specific_groups: Option<Vec<String>>,
    specific_embedder: Option<usize>,
) -> ComparisonScope {
    match s.to_lowercase().as_str() {
        "full" => ComparisonScope::Full,
        "purpose_vector_only" | "purpose" => ComparisonScope::PurposeVectorOnly,
        "cross_correlations_only" | "correlations" => ComparisonScope::CrossCorrelationsOnly,
        "group_alignments_only" | "groups" => ComparisonScope::GroupAlignmentsOnly,
        "specific_groups" => {
            let groups: Vec<GroupType> = specific_groups
                .unwrap_or_default()
                .iter()
                .filter_map(|s| match s.to_lowercase().as_str() {
                    "factual" => Some(GroupType::Factual),
                    "temporal" => Some(GroupType::Temporal),
                    "causal" => Some(GroupType::Causal),
                    "relational" => Some(GroupType::Relational),
                    "qualitative" => Some(GroupType::Qualitative),
                    "implementation" => Some(GroupType::Implementation),
                    _ => None,
                })
                .collect();
            ComparisonScope::SpecificGroups(groups)
        }
        "single_embedder" | "embedder" => {
            ComparisonScope::SingleEmbedderPattern(specific_embedder.unwrap_or(0))
        }
        _ => ComparisonScope::Full,
    }
}

/// Parse feedback type string into FeedbackType enum.
pub fn parse_feedback_type(s: &str) -> FeedbackType {
    match s.to_lowercase().as_str() {
        "positive_retrieval" | "positive" | "accept" | "success" => {
            FeedbackType::Positive { magnitude: 1.0 }
        }
        "negative_retrieval" | "negative" | "reject" | "failure" => {
            FeedbackType::Negative { magnitude: 1.0 }
        }
        "neutral" | "none" => FeedbackType::Neutral,
        _ => FeedbackType::Positive { magnitude: 1.0 },
    }
}
