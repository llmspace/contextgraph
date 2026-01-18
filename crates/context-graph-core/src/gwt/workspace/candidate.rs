//! Workspace Candidate - Memory candidate competing for workspace entry
//!
//! Implements the data structure for memories competing in the
//! winner-take-all workspace selection algorithm per Constitution v4.0.0
//! Section gwt.global_workspace (lines 352-369).

use crate::error::{CoreError, CoreResult};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// A memory candidate competing for workspace entry
#[derive(Debug, Clone)]
pub struct WorkspaceCandidate {
    /// Memory unique identifier
    pub id: Uuid,
    /// Coherence order parameter r
    pub order_parameter: f32,
    /// Memory importance score [0,1]
    pub importance: f32,
    /// Strategic goal alignment score [0,1]
    pub alignment: f32,
    /// Computed competition score
    pub score: f32,
    /// Entry timestamp
    pub timestamp: DateTime<Utc>,
}

impl WorkspaceCandidate {
    /// Create a new workspace candidate
    pub fn new(
        id: Uuid,
        order_parameter: f32,
        importance: f32,
        alignment: f32,
    ) -> CoreResult<Self> {
        if !(0.0..=1.0).contains(&order_parameter) {
            return Err(CoreError::ValidationError {
                field: "order_parameter".to_string(),
                message: format!("out of [0,1]: {}", order_parameter),
            });
        }
        if !(0.0..=1.0).contains(&importance) {
            return Err(CoreError::ValidationError {
                field: "importance".to_string(),
                message: format!("out of [0,1]: {}", importance),
            });
        }
        if !(0.0..=1.0).contains(&alignment) {
            return Err(CoreError::ValidationError {
                field: "alignment".to_string(),
                message: format!("out of [0,1]: {}", alignment),
            });
        }

        let score = order_parameter * importance * alignment;

        Ok(Self {
            id,
            order_parameter,
            importance,
            alignment,
            score,
            timestamp: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_candidate_creation() {
        let id = Uuid::new_v4();
        let candidate = WorkspaceCandidate::new(id, 0.85, 0.9, 0.88).unwrap();

        assert_eq!(candidate.id, id);
        assert_eq!(candidate.order_parameter, 0.85);
        assert!(candidate.score > 0.65);
    }

    #[test]
    fn test_workspace_candidate_invalid_bounds() {
        let id = Uuid::new_v4();

        // Test invalid order parameter
        assert!(WorkspaceCandidate::new(id, 1.5, 0.9, 0.88).is_err());

        // Test invalid importance
        assert!(WorkspaceCandidate::new(id, 0.85, 1.5, 0.88).is_err());

        // Test invalid alignment
        assert!(WorkspaceCandidate::new(id, 0.85, 0.9, 1.5).is_err());
    }
}
