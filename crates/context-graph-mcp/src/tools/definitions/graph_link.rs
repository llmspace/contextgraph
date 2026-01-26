//! Graph linking tool definitions.
//!
//! Defines MCP tools for K-NN graph navigation and typed edge queries:
//! - `get_memory_neighbors`: Get K nearest neighbors in specific embedder space
//! - `get_typed_edges`: Get typed edges from a memory
//! - `traverse_graph`: Multi-hop graph traversal

use serde_json::json;

use super::super::types::ToolDefinition;

/// Get all graph linking tool definitions.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        get_memory_neighbors_definition(),
        get_typed_edges_definition(),
        traverse_graph_definition(),
        get_unified_neighbors_definition(),
    ]
}

fn get_memory_neighbors_definition() -> ToolDefinition {
    ToolDefinition::new(
        "get_memory_neighbors",
        "Get K nearest neighbors of a memory in a specific embedder space. Returns neighbors \
         sorted by similarity. Use to find related memories according to different perspectives: \
         E1 (semantic), E5 (causal), E7 (code), E8 (graph), E10 (intent), E11 (entity).",
        json!({
            "type": "object",
            "required": ["memory_id"],
            "properties": {
                "memory_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "UUID of the memory to find neighbors for"
                },
                "embedder_id": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 12,
                    "default": 0,
                    "description": "Embedder space to search (0=E1 semantic, 4=E5 causal, 6=E7 code, 7=E8 graph, 9=E10 intent, 10=E11 entity)"
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Number of neighbors to return (default: 10)"
                },
                "min_similarity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Minimum similarity threshold (default: 0.0)"
                },
                "include_content": {
                    "type": "boolean",
                    "default": false,
                    "description": "Include memory content in results (default: false)"
                }
            },
            "additionalProperties": false
        }),
    )
}

fn get_typed_edges_definition() -> ToolDefinition {
    ToolDefinition::new(
        "get_typed_edges",
        "Get typed edges from a memory. Typed edges represent relationships derived from \
         embedder agreement patterns: semantic_similar, code_related, entity_shared, \
         causal_chain, graph_connected, intent_aligned, keyword_overlap, multi_agreement.",
        json!({
            "type": "object",
            "required": ["memory_id"],
            "properties": {
                "memory_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "UUID of the memory to get edges from"
                },
                "edge_type": {
                    "type": "string",
                    "enum": [
                        "semantic_similar",
                        "code_related",
                        "entity_shared",
                        "causal_chain",
                        "graph_connected",
                        "intent_aligned",
                        "keyword_overlap",
                        "multi_agreement"
                    ],
                    "description": "Filter by edge type (optional, returns all types if not specified)"
                },
                "direction": {
                    "type": "string",
                    "enum": ["outgoing", "incoming", "both"],
                    "default": "outgoing",
                    "description": "Edge direction: outgoing (from memory), incoming (to memory), both"
                },
                "min_weight": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Minimum edge weight threshold (default: 0.0)"
                },
                "include_content": {
                    "type": "boolean",
                    "default": false,
                    "description": "Include memory content in results (default: false)"
                }
            },
            "additionalProperties": false
        }),
    )
}

fn traverse_graph_definition() -> ToolDefinition {
    ToolDefinition::new(
        "traverse_graph",
        "Multi-hop graph traversal starting from a memory. Explores the knowledge graph \
         following typed edges up to a maximum depth. Useful for discovering connected \
         memories, causal chains, or code dependencies.",
        json!({
            "type": "object",
            "required": ["start_memory_id"],
            "properties": {
                "start_memory_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "UUID of the starting memory"
                },
                "max_hops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 2,
                    "description": "Maximum traversal depth (default: 2, max: 5)"
                },
                "edge_type": {
                    "type": "string",
                    "enum": [
                        "semantic_similar",
                        "code_related",
                        "entity_shared",
                        "causal_chain",
                        "graph_connected",
                        "intent_aligned",
                        "keyword_overlap",
                        "multi_agreement"
                    ],
                    "description": "Filter traversal by edge type (optional)"
                },
                "min_weight": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3,
                    "description": "Minimum edge weight to follow (default: 0.3)"
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                    "description": "Maximum paths to return (default: 20)"
                },
                "include_content": {
                    "type": "boolean",
                    "default": false,
                    "description": "Include memory content in results (default: false)"
                }
            },
            "additionalProperties": false
        }),
    )
}

fn get_unified_neighbors_definition() -> ToolDefinition {
    ToolDefinition::new(
        "get_unified_neighbors",
        "Find neighbors using Weighted RRF fusion across all 13 embedders, providing a unified \
         view where neighbors are ranked by how consistently multiple embedders agree they are \
         related. Unlike get_memory_neighbors (single embedder), this shows what ALL embedders \
         agree on. Per ARCH-21: Uses Weighted RRF, not weighted sum. Per AP-60: Temporal \
         embedders (E2-E4) are excluded from semantic fusion.",
        json!({
            "type": "object",
            "required": ["memory_id"],
            "properties": {
                "memory_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "UUID of the memory to find unified neighbors for"
                },
                "weight_profile": {
                    "type": "string",
                    "enum": [
                        "semantic_search",
                        "code_search",
                        "causal_reasoning",
                        "fact_checking",
                        "intent_search",
                        "intent_enhanced",
                        "graph_reasoning",
                        "category_weighted"
                    ],
                    "default": "semantic_search",
                    "description": "Weight profile for RRF fusion (default: semantic_search)"
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Number of neighbors to return (default: 10)"
                },
                "min_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Minimum RRF score threshold (default: 0.0)"
                },
                "include_content": {
                    "type": "boolean",
                    "default": false,
                    "description": "Include memory content in results (default: false)"
                },
                "include_embedder_breakdown": {
                    "type": "boolean",
                    "default": true,
                    "description": "Include per-embedder scores and ranks in results (default: true)"
                }
            },
            "additionalProperties": false
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definitions_count() {
        let defs = definitions();
        assert_eq!(defs.len(), 4);
    }

    #[test]
    fn test_definition_names() {
        let defs = definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"get_memory_neighbors"));
        assert!(names.contains(&"get_typed_edges"));
        assert!(names.contains(&"traverse_graph"));
        assert!(names.contains(&"get_unified_neighbors"));
    }

    #[test]
    fn test_schemas_valid_json() {
        for def in definitions() {
            // Ensure schema is valid JSON
            assert!(def.input_schema.is_object());
            assert!(def.input_schema.get("type").is_some());
            assert!(def.input_schema.get("properties").is_some());
        }
    }

    #[test]
    fn test_unified_neighbors_schema() {
        let defs = definitions();
        let unified = defs.iter().find(|d| d.name == "get_unified_neighbors").unwrap();
        let props = unified.input_schema.get("properties").unwrap();

        // Verify required fields
        assert!(props.get("memory_id").is_some());
        assert!(props.get("weight_profile").is_some());
        assert!(props.get("top_k").is_some());
        assert!(props.get("include_embedder_breakdown").is_some());

        // Verify description mentions RRF and ARCH-21
        assert!(unified.description.contains("RRF"));
        assert!(unified.description.contains("ARCH-21"));
    }
}
