//! Core tool definitions: inject_context, store_memory, get_memetic_status,
//! get_graph_manifest, search_graph, utl_status.

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns core tool definitions (6 tools).
pub fn definitions() -> Vec<ToolDefinition> {
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
    ]
}
