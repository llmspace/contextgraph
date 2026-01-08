//! MCP tool definitions following the MCP 2024-11-05 protocol specification.
//!
//! This module defines the tools available through the MCP server's `tools/list`
//! and `tools/call` endpoints.

use serde::{Deserialize, Serialize};
use serde_json::json;

/// MCP tool definition following the protocol specification.
///
/// Each tool has a name, description, and JSON Schema for input validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool name
    pub name: String,

    /// Human-readable description of what the tool does
    pub description: String,

    /// JSON Schema defining the tool's input parameters
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Get all tool definitions for the `tools/list` response.
///
/// Returns the complete list of MCP tools exposed by the Context Graph server.
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // inject_context - primary context injection tool
        ToolDefinition::new(
            "inject_context",
            "Inject context into the knowledge graph with UTL processing. \
             Analyzes content for learning potential and stores with computed metrics.",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to inject into the knowledge graph"
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this context is relevant and should be stored"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "default": "text",
                        "description": "The type/modality of the content"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "description": "Importance score for the memory [0.0, 1.0]"
                    }
                },
                "required": ["content", "rationale"]
            }),
        ),

        // store_memory - store a memory node directly
        ToolDefinition::new(
            "store_memory",
            "Store a memory node directly in the knowledge graph without UTL processing.",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "description": "Importance score for the memory [0.0, 1.0]"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "default": "text",
                        "description": "The type/modality of the content"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional tags for categorization"
                    }
                },
                "required": ["content"]
            }),
        ),

        // get_memetic_status - get UTL metrics and system state
        ToolDefinition::new(
            "get_memetic_status",
            "Get current system status with LIVE UTL metrics from the UtlProcessor: \
             entropy (novelty), coherence (understanding), learning score (magnitude), \
             Johari quadrant classification, consolidation phase, and suggested action. \
             Also returns node count and 5-layer bio-nervous system status.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // get_graph_manifest - describe the 5-layer architecture
        ToolDefinition::new(
            "get_graph_manifest",
            "Get the 5-layer bio-nervous system architecture description and current layer statuses.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // search_graph - semantic search
        ToolDefinition::new(
            "search_graph",
            "Search the knowledge graph using semantic similarity. \
             Returns nodes matching the query with relevance scores.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "topK": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum number of results to return"
                    },
                    "minSimilarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.0,
                        "description": "Minimum similarity threshold [0.0, 1.0]"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "description": "Filter results by modality"
                    }
                },
                "required": ["query"]
            }),
        ),
        // utl_status - query UTL system state
        ToolDefinition::new(
            "utl_status",
            "Query current UTL (Unified Theory of Learning) system state including lifecycle phase, \
             entropy, coherence, learning score, Johari quadrant, and consolidation phase.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // get_consciousness_state - GWT consciousness state (TASK-GWT-001)
        ToolDefinition::new(
            "get_consciousness_state",
            "Get current consciousness state including Kuramoto sync (r), consciousness level (C), \
             meta-cognitive score, differentiation, workspace status, and identity coherence. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for consciousness tracking (optional, uses default if not provided)"
                    }
                },
                "required": []
            }),
        ),

        // get_kuramoto_sync - Kuramoto oscillator network synchronization (TASK-GWT-001)
        ToolDefinition::new(
            "get_kuramoto_sync",
            "Get Kuramoto oscillator network synchronization state including order parameter (r), \
             mean phase (psi), all 13 oscillator phases, natural frequencies, and coupling strength. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional, uses default if not provided)"
                    }
                },
                "required": []
            }),
        ),

        // get_workspace_status - Global Workspace status (TASK-GWT-001)
        ToolDefinition::new(
            "get_workspace_status",
            "Get Global Workspace status including active memory, competing candidates, \
             broadcast state, and coherence threshold. Returns WTA selection details. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional, uses default if not provided)"
                    }
                },
                "required": []
            }),
        ),

        // get_ego_state - Self-Ego Node state (TASK-GWT-001)
        ToolDefinition::new(
            "get_ego_state",
            "Get Self-Ego Node state including purpose vector (13D), identity continuity, \
             coherence with actions, and trajectory length. Used for identity monitoring. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional, uses default if not provided)"
                    }
                },
                "required": []
            }),
        ),

        // trigger_workspace_broadcast - Trigger WTA selection (TASK-GWT-001)
        ToolDefinition::new(
            "trigger_workspace_broadcast",
            "Trigger winner-take-all workspace broadcast with a specific memory. \
             Forces memory into workspace competition. Requires write lock on workspace. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of memory to broadcast into workspace"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "description": "Importance score for the memory [0.0, 1.0]"
                    },
                    "alignment": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "description": "North star alignment score [0.0, 1.0]"
                    },
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force broadcast even if below coherence threshold"
                    }
                },
                "required": ["memory_id"]
            }),
        ),

        // adjust_coupling - Adjust Kuramoto coupling strength (TASK-GWT-001)
        ToolDefinition::new(
            "adjust_coupling",
            "Adjust Kuramoto oscillator network coupling strength K. \
             Higher K leads to faster synchronization. K is clamped to [0, 10]. \
             Returns old and new K values plus predicted order parameter r. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "new_K": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                        "description": "New coupling strength K (clamped to [0, 10])"
                    }
                },
                "required": ["new_K"]
            }),
        ),

        // ========== ADAPTIVE THRESHOLD CALIBRATION (ATC) TOOLS ==========

        // get_threshold_status - Get current threshold status (TASK-ATC-001)
        ToolDefinition::new(
            "get_threshold_status",
            "Get current ATC threshold status including all thresholds, calibration state, \
             and adaptation metrics. Returns per-embedder temperatures, drift scores, and \
             bandit exploration stats. Requires ATC provider to be initialized.",
            json!({
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "enum": ["Code", "Medical", "Legal", "Creative", "Research", "General"],
                        "default": "General",
                        "description": "Domain for threshold context (affects priors)"
                    },
                    "embedder_id": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 13,
                        "description": "Optional: specific embedder (1-13) for detailed temperature info"
                    }
                },
                "required": []
            }),
        ),

        // get_calibration_metrics - Get calibration quality metrics (TASK-ATC-001)
        ToolDefinition::new(
            "get_calibration_metrics",
            "Get calibration quality metrics: ECE (Expected Calibration Error), \
             MCE (Maximum Calibration Error), Brier Score, drift scores per threshold, \
             and calibration status. Targets: ECE < 0.05 (excellent), < 0.10 (good).",
            json!({
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "enum": ["1h", "24h", "7d", "30d"],
                        "default": "24h",
                        "description": "Timeframe for metrics aggregation"
                    }
                },
                "required": []
            }),
        ),

        // trigger_recalibration - Manually trigger recalibration (TASK-ATC-001)
        ToolDefinition::new(
            "trigger_recalibration",
            "Manually trigger recalibration at a specific ATC level. \
             Level 1: EWMA drift adjustment. Level 2: Temperature scaling. \
             Level 3: Thompson Sampling exploration. Level 4: Bayesian meta-optimization. \
             Returns new thresholds and number of observations used.",
            json!({
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                        "description": "ATC level to trigger (1=EWMA, 2=Temperature, 3=Bandit, 4=Bayesian)"
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["Code", "Medical", "Legal", "Creative", "Research", "General"],
                        "default": "General",
                        "description": "Domain context for recalibration"
                    }
                },
                "required": ["level"]
            }),
        ),

        // ========== DREAM TOOLS (TASK-DREAM-MCP) ==========

        // trigger_dream - Manually trigger a dream consolidation cycle
        ToolDefinition::new(
            "trigger_dream",
            "Manually trigger a dream consolidation cycle. System must be idle (activity < 0.15). \
             Executes NREM (3 min) + REM (2 min) phases. Returns DreamReport with metrics. \
             Aborts automatically on external query (wake latency < 100ms).",
            json!({
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force dream even if activity is above threshold (not recommended)"
                    }
                },
                "required": []
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

        // ========== NEUROMODULATION TOOLS (TASK-NEUROMOD-MCP) ==========

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

        // ========== STEERING TOOLS (TASK-STEERING-001) ==========

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

        // ========== CAUSAL INFERENCE TOOLS (TASK-CAUSAL-001) ==========

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

/// Tool names as constants for dispatch matching.
pub mod tool_names {
    pub const INJECT_CONTEXT: &str = "inject_context";
    pub const STORE_MEMORY: &str = "store_memory";
    pub const GET_MEMETIC_STATUS: &str = "get_memetic_status";
    pub const GET_GRAPH_MANIFEST: &str = "get_graph_manifest";
    pub const SEARCH_GRAPH: &str = "search_graph";
    pub const UTL_STATUS: &str = "utl_status";
    /// TASK-GWT-001: Get consciousness state from GWT/Kuramoto system
    pub const GET_CONSCIOUSNESS_STATE: &str = "get_consciousness_state";
    /// TASK-GWT-001: Get Kuramoto oscillator network synchronization state
    pub const GET_KURAMOTO_SYNC: &str = "get_kuramoto_sync";
    /// TASK-GWT-001: Get Global Workspace status (active memory, competing, broadcast)
    pub const GET_WORKSPACE_STATUS: &str = "get_workspace_status";
    /// TASK-GWT-001: Get Self-Ego Node state (purpose vector, identity continuity)
    pub const GET_EGO_STATE: &str = "get_ego_state";
    /// TASK-GWT-001: Trigger workspace broadcast with a memory
    pub const TRIGGER_WORKSPACE_BROADCAST: &str = "trigger_workspace_broadcast";
    /// TASK-GWT-001: Adjust Kuramoto coupling strength K
    pub const ADJUST_COUPLING: &str = "adjust_coupling";

    // ========== ADAPTIVE THRESHOLD CALIBRATION (ATC) TOOLS (TASK-ATC-001) ==========

    /// TASK-ATC-001: Get current ATC threshold status
    pub const GET_THRESHOLD_STATUS: &str = "get_threshold_status";
    /// TASK-ATC-001: Get calibration quality metrics (ECE, MCE, Brier)
    pub const GET_CALIBRATION_METRICS: &str = "get_calibration_metrics";
    /// TASK-ATC-001: Manually trigger recalibration at a specific level
    pub const TRIGGER_RECALIBRATION: &str = "trigger_recalibration";

    // ========== DREAM TOOLS (TASK-DREAM-MCP) ==========

    /// TASK-DREAM-MCP: Manually trigger a dream consolidation cycle
    pub const TRIGGER_DREAM: &str = "trigger_dream";
    /// TASK-DREAM-MCP: Get current dream system status
    pub const GET_DREAM_STATUS: &str = "get_dream_status";
    /// TASK-DREAM-MCP: Abort current dream cycle
    pub const ABORT_DREAM: &str = "abort_dream";
    /// TASK-DREAM-MCP: Get shortcut candidates from amortized learning
    pub const GET_AMORTIZED_SHORTCUTS: &str = "get_amortized_shortcuts";

    // ========== NEUROMODULATION TOOLS (TASK-NEUROMOD-MCP) ==========

    /// TASK-NEUROMOD-MCP: Get all 4 neuromodulator levels
    pub const GET_NEUROMODULATION_STATE: &str = "get_neuromodulation_state";
    /// TASK-NEUROMOD-MCP: Adjust a specific modulator
    pub const ADJUST_NEUROMODULATOR: &str = "adjust_neuromodulator";

    // ========== STEERING TOOLS (TASK-STEERING-001) ==========

    /// TASK-STEERING-001: Get steering feedback from Gardener, Curator, Assessor
    pub const GET_STEERING_FEEDBACK: &str = "get_steering_feedback";

    // ========== CAUSAL INFERENCE TOOLS (TASK-CAUSAL-001) ==========

    /// TASK-CAUSAL-001: Perform omni-directional causal inference
    pub const OMNI_INFER: &str = "omni_infer";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tool_definitions() {
        let tools = get_tool_definitions();
        // 6 original + 6 GWT tools + 3 ATC tools + 4 Dream tools + 2 Neuromod tools + 1 Steering + 1 Causal = 23 total
        assert_eq!(tools.len(), 23);

        let tool_names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        // Original 6 tools
        assert!(tool_names.contains(&"inject_context"));
        assert!(tool_names.contains(&"store_memory"));
        assert!(tool_names.contains(&"get_memetic_status"));
        assert!(tool_names.contains(&"get_graph_manifest"));
        assert!(tool_names.contains(&"search_graph"));
        assert!(tool_names.contains(&"utl_status"));
        // GWT tools (TASK-GWT-001)
        assert!(tool_names.contains(&"get_consciousness_state"));
        assert!(tool_names.contains(&"get_kuramoto_sync"));
        assert!(tool_names.contains(&"get_workspace_status"));
        assert!(tool_names.contains(&"get_ego_state"));
        assert!(tool_names.contains(&"trigger_workspace_broadcast"));
        assert!(tool_names.contains(&"adjust_coupling"));
        // ATC tools (TASK-ATC-001)
        assert!(tool_names.contains(&"get_threshold_status"));
        assert!(tool_names.contains(&"get_calibration_metrics"));
        assert!(tool_names.contains(&"trigger_recalibration"));
        // Dream tools (TASK-DREAM-MCP)
        assert!(tool_names.contains(&"trigger_dream"));
        assert!(tool_names.contains(&"get_dream_status"));
        assert!(tool_names.contains(&"abort_dream"));
        assert!(tool_names.contains(&"get_amortized_shortcuts"));
        // Neuromod tools (TASK-NEUROMOD-MCP)
        assert!(tool_names.contains(&"get_neuromodulation_state"));
        assert!(tool_names.contains(&"adjust_neuromodulator"));
        // Steering tools (TASK-STEERING-001)
        assert!(tool_names.contains(&"get_steering_feedback"));
        // Causal tools (TASK-CAUSAL-001)
        assert!(tool_names.contains(&"omni_infer"));
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tools = get_tool_definitions();
        let json = serde_json::to_string(&tools).unwrap();
        assert!(json.contains("inject_context"));
        assert!(json.contains("inputSchema"));
    }

    #[test]
    fn test_inject_context_schema() {
        let tools = get_tool_definitions();
        let inject = tools.iter().find(|t| t.name == "inject_context").unwrap();

        let schema = &inject.input_schema;
        let required = schema.get("required").unwrap().as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("content")));
        assert!(required.iter().any(|v| v.as_str() == Some("rationale")));
    }
}
