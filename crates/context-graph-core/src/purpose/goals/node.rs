//! Goal node structure and implementation.

use super::discovery::GoalDiscoveryMetadata;
use super::error::GoalNodeError;
use super::level::GoalLevel;
use crate::types::fingerprint::TeleologicalArray;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A goal node in the purpose hierarchy.
///
/// Goals are discovered AUTONOMOUSLY from memory patterns.
/// They represent emergent purpose from stored teleological fingerprints.
///
/// # Architectural Rules (constitution.yaml)
///
/// - ARCH-02: Goals use TeleologicalArray for apples-to-apples comparison
/// - ARCH-03: Goals are discovered, not manually created
/// - ARCH-05: All 13 embedders must be present in teleological_array
///
/// # Example
///
/// ```ignore
/// use context_graph_core::purpose::{GoalNode, GoalLevel, DiscoveryMethod, GoalDiscoveryMetadata};
/// use context_graph_core::types::fingerprint::SemanticFingerprint;
///
/// // Goals are created from clustering analysis
/// let discovery = GoalDiscoveryMetadata::new(
///     DiscoveryMethod::Clustering,
///     0.85,  // confidence
///     42,    // cluster_size
///     0.78,  // coherence
/// ).unwrap();
///
/// let centroid_fingerprint = SemanticFingerprint::zeroed();
/// let goal = GoalNode::autonomous_goal(
///     "Emergent ML mastery goal".to_string(),
///     GoalLevel::Strategic,
///     centroid_fingerprint,
///     discovery,
/// ).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalNode {
    /// Unique identifier (UUID).
    pub id: Uuid,

    /// Human-readable description.
    pub description: String,

    /// Hierarchical level.
    pub level: GoalLevel,

    /// The teleological array representing this goal.
    ///
    /// This is a SemanticFingerprint containing all 13 embeddings.
    /// Goals can be compared apples-to-apples with memories:
    /// - Goal.E1 vs Memory.E1 (semantic)
    /// - Goal.E5 vs Memory.E5 (causal)
    /// - Goal.E7 vs Memory.E7 (code)
    ///   etc.
    pub teleological_array: TeleologicalArray,

    /// Parent goal (None for Strategic goals).
    pub parent_id: Option<Uuid>,

    /// Child goal IDs.
    pub child_ids: Vec<Uuid>,

    /// How this goal was discovered.
    pub discovery: GoalDiscoveryMetadata,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl GoalNode {
    /// Create a new autonomously discovered goal.
    ///
    /// This is the ONLY way to create goals. Manual goal creation is forbidden
    /// per ARCH-03.
    ///
    /// # Arguments
    ///
    /// * `description` - Human-readable goal description
    /// * `level` - Position in goal hierarchy
    /// * `teleological_array` - The 13-embedder fingerprint from clustering centroid
    /// * `discovery` - Metadata about how this goal was discovered
    ///
    /// # Errors
    ///
    /// Returns `GoalNodeError::InvalidArray` if:
    /// - The teleological array fails validation (incomplete embeddings, wrong dimensions)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let goal = GoalNode::autonomous_goal(
    ///     "Strategic code quality goal".into(),
    ///     GoalLevel::Strategic,
    ///     centroid_fingerprint,
    ///     discovery,
    /// )?;
    /// ```
    pub fn autonomous_goal(
        description: String,
        level: GoalLevel,
        teleological_array: TeleologicalArray,
        discovery: GoalDiscoveryMetadata,
    ) -> Result<Self, GoalNodeError> {
        // Fail fast if array is invalid
        teleological_array.validate_strict()?;

        Ok(Self {
            id: Uuid::new_v4(),
            description,
            level,
            teleological_array,
            parent_id: None,
            child_ids: Vec::new(),
            discovery,
            created_at: Utc::now(),
        })
    }

    /// Create a child goal with a parent reference.
    ///
    /// Used when decomposing a parent goal into sub-goals.
    ///
    /// # Panics
    ///
    /// Panics if `level` is `GoalLevel::Strategic` (Strategic goals have no parent).
    ///
    /// # Errors
    ///
    /// Returns `GoalNodeError::InvalidArray` if the teleological array fails validation.
    pub fn child_goal(
        description: String,
        level: GoalLevel,
        parent_id: Uuid,
        teleological_array: TeleologicalArray,
        discovery: GoalDiscoveryMetadata,
    ) -> Result<Self, GoalNodeError> {
        assert!(
            level != GoalLevel::Strategic,
            "Strategic goals cannot have a parent - they are top-level"
        );

        teleological_array.validate_strict()?;

        Ok(Self {
            id: Uuid::new_v4(),
            description,
            level,
            teleological_array,
            parent_id: Some(parent_id),
            child_ids: Vec::new(),
            discovery,
            created_at: Utc::now(),
        })
    }

    /// Get the teleological array for comparison.
    #[inline]
    pub fn array(&self) -> &TeleologicalArray {
        &self.teleological_array
    }

    /// Check if this is a top-level (Strategic) goal.
    #[inline]
    pub fn is_top_level(&self) -> bool {
        self.level == GoalLevel::Strategic
    }

    /// Check if this goal has the given ancestor.
    ///
    /// Note: This only checks the immediate parent. For full ancestry check,
    /// use `GoalHierarchy::path_to_root()`.
    #[inline]
    pub fn has_parent(&self, ancestor_id: Uuid) -> bool {
        self.parent_id == Some(ancestor_id)
    }

    /// Add a child goal ID.
    pub fn add_child(&mut self, child_id: Uuid) {
        if !self.child_ids.contains(&child_id) {
            self.child_ids.push(child_id);
        }
    }

    /// Remove a child goal ID.
    pub fn remove_child(&mut self, child_id: Uuid) {
        self.child_ids.retain(|id| *id != child_id);
    }
}
