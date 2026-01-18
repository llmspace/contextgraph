//! Goal hierarchy tree structure.
//!
//! Goals emerge autonomously from data patterns.
//! Strategic level is the top level.

use super::error::GoalHierarchyError;
use super::level::GoalLevel;
use super::node::GoalNode;
use std::collections::HashMap;
use uuid::Uuid;

/// Goal hierarchy tree structure.
///
/// Manages a tree of goals with Strategic goals at the root level.
/// Used for hierarchical alignment propagation in purpose vector computation.
///
/// # Invariants
///
/// - All child goals have valid parent references
/// - No cycles in the hierarchy
/// - Strategic goals have no parent
///
/// # ARCH-03 Compliance
/// Goals emerge autonomously from teleological fingerprints.
#[derive(Clone, Debug, Default)]
pub struct GoalHierarchy {
    pub(crate) nodes: HashMap<Uuid, GoalNode>,
}

impl GoalHierarchy {
    /// Create a new empty goal hierarchy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a goal to the hierarchy.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Child goal's parent doesn't exist
    ///
    /// # Example
    ///
    /// ```ignore
    /// use context_graph_core::purpose::{GoalHierarchy, GoalNode, GoalLevel, GoalDiscoveryMetadata};
    /// use context_graph_core::types::fingerprint::SemanticFingerprint;
    ///
    /// let mut hierarchy = GoalHierarchy::new();
    ///
    /// let discovery = GoalDiscoveryMetadata::bootstrap();
    /// let fp = SemanticFingerprint::zeroed();
    /// let goal = GoalNode::autonomous_goal(
    ///     "Strategic Goal".into(),
    ///     GoalLevel::Strategic,
    ///     fp,
    ///     discovery,
    /// ).unwrap();
    /// hierarchy.add_goal(goal).unwrap();
    /// ```
    pub fn add_goal(&mut self, goal: GoalNode) -> Result<(), GoalHierarchyError> {
        // Validate parent exists (except for Strategic level which is root)
        if let Some(ref parent_id) = goal.parent_id {
            if !self.nodes.contains_key(parent_id) {
                return Err(GoalHierarchyError::ParentNotFound(*parent_id));
            }
        }

        // Update parent's child list
        if let Some(parent_id) = goal.parent_id {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.add_child(goal.id);
            }
        }

        self.nodes.insert(goal.id, goal);
        Ok(())
    }

    /// Get all top-level (Strategic) goals.
    ///
    /// Multiple Strategic goals can exist (they emerge autonomously).
    pub fn top_level_goals(&self) -> Vec<&GoalNode> {
        self.at_level(GoalLevel::Strategic)
    }

    /// Check if any top-level goals exist.
    #[inline]
    pub fn has_top_level_goals(&self) -> bool {
        !self.top_level_goals().is_empty()
    }

    /// Get a goal by ID.
    pub fn get(&self, id: &Uuid) -> Option<&GoalNode> {
        self.nodes.get(id)
    }

    /// Get direct children of a goal.
    pub fn children(&self, parent_id: &Uuid) -> Vec<&GoalNode> {
        self.get(parent_id)
            .map(|parent| {
                parent
                    .child_ids
                    .iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all goals at a specific level.
    pub fn at_level(&self, level: GoalLevel) -> Vec<&GoalNode> {
        self.nodes.values().filter(|n| n.level == level).collect()
    }

    /// Total number of goals in the hierarchy.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the hierarchy is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all goals.
    pub fn iter(&self) -> impl Iterator<Item = &GoalNode> {
        self.nodes.values()
    }

    /// Get path from a goal to the root (top-level goal).
    ///
    /// Returns the sequence of goal IDs from the given goal up to (and including)
    /// the top-level parent. Returns empty vec if goal not found.
    pub fn path_to_root(&self, goal_id: &Uuid) -> Vec<Uuid> {
        let mut path = Vec::new();
        let mut current = self.nodes.get(goal_id);

        while let Some(node) = current {
            path.push(node.id);
            current = node.parent_id.and_then(|pid| self.nodes.get(&pid));
        }

        path
    }

    /// Validate hierarchy integrity.
    ///
    /// Checks:
    /// - All parent references are valid
    ///
    /// Empty hierarchies are valid. Multiple Strategic goals are allowed.
    pub fn validate(&self) -> Result<(), GoalHierarchyError> {
        // Check all parents exist
        for node in self.nodes.values() {
            if let Some(ref parent_id) = node.parent_id {
                if !self.nodes.contains_key(parent_id) {
                    return Err(GoalHierarchyError::ParentNotFound(*parent_id));
                }
            }
        }

        Ok(())
    }
}
