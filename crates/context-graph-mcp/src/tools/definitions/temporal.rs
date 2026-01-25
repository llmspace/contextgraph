//! Temporal tool definitions for E2 V_freshness recency search.
//!
//! Per Constitution v6.5: E2 (V_freshness) finds recency patterns.
//! Temporal embedders are POST-RETRIEVAL only per ARCH-25.
//!
//! Tools:
//! - search_recent: Search with E2 temporal boost applied

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns temporal tool definitions.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // search_recent - temporal search with E2 freshness boost
        ToolDefinition::new(
            "search_recent",
            "Search for recent memories with E2 temporal boost applied. \
             Automatically applies freshness decay to prioritize recent results. \
             Use for queries like 'what did we discuss recently', 'latest updates', \
             'yesterday's conversation', or any time-sensitive retrieval. \
             Per ARCH-25: Temporal boost is applied POST-retrieval, not in similarity fusion.",
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
                    "temporalWeight": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.3,
                        "description": "Temporal boost weight [0.1, 1.0]. Higher = more recency preference. Default: 0.3"
                    },
                    "decayFunction": {
                        "type": "string",
                        "enum": ["linear", "exponential", "step"],
                        "default": "exponential",
                        "description": "Decay function for freshness. 'exponential' (default) = natural forgetting curve, 'linear' = simple decay, 'step' = time bucket based"
                    },
                    "temporalScale": {
                        "type": "string",
                        "enum": ["micro", "meso", "macro", "long"],
                        "default": "meso",
                        "description": "Time scale for decay. 'micro' = 1 hour horizon, 'meso' = 1 day (default), 'macro' = 1 week, 'long' = 1 month"
                    },
                    "includeContent": {
                        "type": "boolean",
                        "default": true,
                        "description": "Include content text in results (default: true)"
                    },
                    "minSimilarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.1,
                        "description": "Minimum semantic similarity threshold before temporal boost"
                    }
                },
                "required": ["query"]
            }),
        ),
    ]
}
