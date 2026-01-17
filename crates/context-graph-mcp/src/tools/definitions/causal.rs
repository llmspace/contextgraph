//! Causal inference tool definitions.
//! TASK-CAUSAL-001: Omni-directional causal inference.

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns Causal tool definitions (1 tool).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // omni_infer - Perform omni-directional causal inference
        ToolDefinition::new(
            "omni_infer",
            "Perform omni-directional causal inference. Supports 5 directions: \
             forward (A->B effect), backward (B->A cause), bidirectional (A<->B mutual), \
             bridge (cross-domain), and abduction (best hypothesis). \
             Constitution ref: line 539.",
            json!({
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "format": "uuid",
                        "description": "Source node UUID for inference"
                    },
                    "target": {
                        "type": "string",
                        "format": "uuid",
                        "description": "Target node UUID (required for forward/backward/bidirectional)"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "bidirectional", "bridge", "abduction"],
                        "default": "forward",
                        "description": "Inference direction: forward (A->B), backward (B->A), \
                                       bidirectional (A<->B), bridge (cross-domain), abduction (best hypothesis)"
                    }
                },
                "required": ["source"]
            }),
        ),
    ]
}
