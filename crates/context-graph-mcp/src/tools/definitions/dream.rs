//! Dream consolidation tool definitions.
//! TASK-DREAM-MCP: Dream triggers, status, abort, amortized shortcuts.
//! TASK-37: Added get_gpu_status for GPU utilization monitoring.
//! TASK-S01: Added trigger_mental_check for entropy-based triggering.

use serde_json::json;
use crate::tools::types::ToolDefinition;

/// Returns Dream tool definitions (6 tools).
/// TASK-37: Added get_gpu_status tool.
/// TASK-S01: Added trigger_mental_check tool.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // trigger_dream - Manually trigger a dream consolidation cycle
        // TASK-35: Updated to require rationale and use TriggerManager
        // TASK-DREAM-PH-002: Added phase parameter per PRD Section 5.2
        ToolDefinition::new(
            "trigger_dream",
            "Manually trigger a dream consolidation cycle via TriggerManager. \
             Requires rationale for audit logging. GPU must be < 80% eligible. \
             Manual triggers have highest priority and bypass cooldown. \
             Returns immediately; actual dream executes in background scheduler. \
             Phase controls which dream phases run (nrem=consolidation, rem=discovery, full_cycle=both).",
            json!({
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1024,
                        "description": "REQUIRED: Reason for manual trigger (for audit logging)"
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["nrem", "rem", "full_cycle"],
                        "default": "full_cycle",
                        "description": "Dream phase to execute: 'nrem' (Hebbian consolidation), 'rem' (hyperbolic walk), or 'full_cycle' (both)"
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

        // get_gpu_status - Get GPU utilization and dream eligibility
        // TASK-37: Exposes GpuMonitor trait from TASK-23
        ToolDefinition::new(
            "get_gpu_status",
            "Get GPU utilization and dream eligibility status. Returns current GPU usage (0.0-1.0), \
             dream eligibility (GPU < 80%), and whether dream should abort (GPU > 30%). \
             Per constitution: dream.trigger.gpu = '<80%', dream.constraints.gpu = '<30%'. \
             FAILS FAST with explicit error if GpuMonitor not initialized (AP-26).",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // trigger_mental_check - Trigger mental_check workflow based on entropy
        // TASK-S01: Per SPEC-TRIGGER-MCP-001
        ToolDefinition::new(
            "trigger_mental_check",
            "Trigger a mental_check workflow based on entropy threshold. \
             Entropy values in [0.0, 1.0] are validated. Triggers fire when entropy > threshold (default 0.7). \
             Returns status (initiated/queued/skipped), trigger reason, and workflow_id if initiated. \
             Use force=true to trigger even if entropy is below threshold. \
             FAILS FAST if TriggerManager not initialized (AP-26).",
            json!({
                "type": "object",
                "properties": {
                    "entropy": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "REQUIRED: Current entropy value [0.0, 1.0]. Trigger fires when > threshold (0.7)."
                    },
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force trigger even if entropy below threshold (for testing/debugging)."
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["nrem", "rem", "full_cycle"],
                        "default": "full_cycle",
                        "description": "Dream phase to execute: 'nrem' (consolidation), 'rem' (discovery), 'full_cycle' (both)."
                    }
                },
                "required": ["entropy"]
            }),
        ),

        // get_trigger_config - Get current trigger configuration
        // TASK-S02: Per SPEC-TRIGGER-MCP-001 REQ-CONFIG-01
        ToolDefinition::new(
            "get_trigger_config",
            "Get current trigger configuration including thresholds, cooldowns, and trigger count. \
             Returns entropy_threshold (default 0.7), ic_threshold (default 0.3), cooldown_ms, \
             last_trigger_timestamp, trigger_count, and enabled status. \
             FAILS FAST if TriggerManager not initialized (AP-26).",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // get_trigger_history - Get trigger history
        // TASK-S03: Per SPEC-TRIGGER-MCP-001 REQ-HISTORY-01
        ToolDefinition::new(
            "get_trigger_history",
            "Get recent trigger history showing when and why triggers fired. \
             Returns a list of trigger events with timestamp, entropy value, reason, and workflow status. \
             Useful for debugging and observability of dream cycle triggers. \
             Limited to last 100 entries. FAILS FAST if TriggerManager not initialized (AP-26).",
            json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "description": "Maximum number of history entries to return (default 20, max 100)"
                    }
                },
                "required": []
            }),
        ),
    ]
}
