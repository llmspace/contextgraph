//! Steering tool definitions.
//! TASK-STEERING-001: Gardener, Curator, Assessor feedback.

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns Steering tool definitions (1 tool).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // get_steering_feedback - Get steering feedback from Gardener, Curator, Assessor
        ToolDefinition::new(
            "get_steering_feedback",
            "Get steering feedback from the Gardener (graph health), Curator (memory quality), \
             and Assessor (performance) components. Returns a SteeringReward in [-1, 1] \
             with detailed component scores and recommendations.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),
    ]
}
