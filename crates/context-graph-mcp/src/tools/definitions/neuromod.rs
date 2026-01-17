//! Neuromodulation tool definitions.
//! TASK-NEUROMOD-MCP: Neuromodulator state and adjustments.

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns Neuromodulation tool definitions (2 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // get_neuromodulation_state - Get all 4 neuromodulator levels
        ToolDefinition::new(
            "get_neuromodulation_state",
            "Get current neuromodulation state including all 4 modulators: \
             Dopamine (hopfield.beta [1,5]), Serotonin (space_weights [0,1]), \
             Noradrenaline (attention.temp [0.5,2]), Acetylcholine (utl.lr [0.001,0.002]).",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),
        // adjust_neuromodulator - Adjust a specific modulator
        ToolDefinition::new(
            "adjust_neuromodulator",
            "Adjust a specific neuromodulator level. ACh is read-only (managed by GWT). \
             Changes are clamped to constitution-mandated ranges.",
            json!({
                "type": "object",
                "properties": {
                    "modulator": {
                        "type": "string",
                        "enum": ["dopamine", "serotonin", "noradrenaline"],
                        "description": "Which modulator to adjust (ACh is read-only)"
                    },
                    "delta": {
                        "type": "number",
                        "description": "Amount to add (positive) or subtract (negative) from current level"
                    }
                },
                "required": ["modulator", "delta"]
            }),
        ),
    ]
}
