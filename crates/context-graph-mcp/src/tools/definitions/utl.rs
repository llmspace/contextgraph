//! UTL (Unified Theory of Learning) tool definitions.
//! TASK-UTL-P1-001: Delta-S/Delta-C computation.

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns UTL tool definitions (1 tool).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // gwt/compute_delta_sc - Compute per-embedder delta-S and aggregate delta-C
        ToolDefinition::new(
            "gwt/compute_delta_sc",
            "Compute per-embedder entropy (delta-S) and aggregate coherence (delta-C) for GWT workspace \
             evaluation. Returns 13 delta-S values (one per embedder), aggregate delta-S, delta-C, Johari \
             quadrant classifications, and UTL learning potential. Used by GWT workspace for \
             winner-take-all selection and consciousness broadcasting. \
             Constitution ref: delta_sc.delta-S_methods (lines 792-802).",
            json!({
                "type": "object",
                "properties": {
                    "vertex_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "Memory vertex ID (optional, for context tracking)"
                    },
                    "old_fingerprint": {
                        "type": "object",
                        "description": "Previous TeleologicalFingerprint (serialized)"
                    },
                    "new_fingerprint": {
                        "type": "object",
                        "description": "Current TeleologicalFingerprint (serialized)"
                    },
                    "johari_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "description": "Threshold for Johari quadrant classification"
                    },
                    "include_diagnostics": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include per-embedder diagnostic details"
                    }
                },
                "required": ["old_fingerprint", "new_fingerprint"]
            }),
        ),
    ]
}
