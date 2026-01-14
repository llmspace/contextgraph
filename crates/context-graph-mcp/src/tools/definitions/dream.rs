//! Dream consolidation tool definitions.
//! TASK-DREAM-MCP: Dream triggers, status, abort, amortized shortcuts.

use serde_json::json;
use crate::tools::types::ToolDefinition;

/// Returns Dream tool definitions (4 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // trigger_dream - Manually trigger a dream consolidation cycle
        // TASK-35: Updated to require rationale and use TriggerManager
        ToolDefinition::new(
            "trigger_dream",
            "Manually trigger a dream consolidation cycle via TriggerManager. \
             Requires rationale for audit logging. GPU must be < 80% eligible. \
             Manual triggers have highest priority and bypass cooldown. \
             Returns immediately; actual dream executes in background scheduler.",
            json!({
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1024,
                        "description": "REQUIRED: Reason for manual trigger (for audit logging)"
                    },
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force trigger even if GPU busy or in cooldown (not recommended)"
                    }
                },
                "required": ["rationale"]
            }),
        ),

        // get_dream_status - Get current dream system status
        ToolDefinition::new(
            "get_dream_status",
            "Get current dream system status including state (Awake/NREM/REM/Waking), \
             GPU usage, activity level, and time since last dream cycle.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // abort_dream - Abort current dream cycle
        ToolDefinition::new(
            "abort_dream",
            "Abort the current dream cycle. Must complete wake within 100ms (constitution mandate). \
             Returns wake latency and partial dream report.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // get_amortized_shortcuts - Get shortcut candidates from amortized learning
        ToolDefinition::new(
            "get_amortized_shortcuts",
            "Get shortcut candidates from amortized learning. Returns paths traversed 5+ times \
             with 3+ hops that qualify for direct edge creation.",
            json!({
                "type": "object",
                "properties": {
                    "min_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                        "description": "Minimum confidence threshold for shortcuts"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "description": "Maximum shortcuts to return"
                    }
                },
                "required": []
            }),
        ),
    ]
}
