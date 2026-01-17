//! Teleological (13-embedder fusion) tool definitions.
//! TELEO-007 through TELEO-011: Search, compute, fuse, update synergy, manage profiles.

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns Teleological tool definitions (5 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // search_teleological - Cross-correlation search across all 13 embedders
        ToolDefinition::new(
            "search_teleological",
            "Perform teleological matrix search across all 13 embedder dimensions. \
             Computes cross-correlation similarity at multiple levels: full matrix (78 pairs), \
             purpose vector (13D), group alignments (6D), and single embedder patterns.",
            json!({
                "type": "object",
                "properties": {
                    "query_content": {
                        "type": "string",
                        "description": "Content to search for (will be embedded)"
                    },
                    "query_vector_id": {
                        "type": "string",
                        "description": "Alternative: ID of existing teleological vector to use as query"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["cosine", "euclidean", "synergy_weighted", "group_hierarchical", "cross_correlation_dominant", "tucker_compressed", "adaptive"],
                        "default": "adaptive",
                        "description": "Search strategy for comparing teleological vectors"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["full", "purpose_vector_only", "cross_correlations_only", "group_alignments_only"],
                        "default": "full",
                        "description": "Which components to compare"
                    },
                    "specific_groups": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["factual", "temporal", "causal", "relational", "qualitative", "implementation"]
                        },
                        "description": "Compare only specific embedding groups"
                    },
                    "specific_embedder": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 12,
                        "description": "Compare single embedder pattern (0=Semantic, 5=Code, 12=Sparse, etc.)"
                    },
                    "weight_purpose": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.4,
                        "description": "Weight for purpose vector similarity"
                    },
                    "weight_correlations": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.35,
                        "description": "Weight for cross-correlation similarity"
                    },
                    "weight_groups": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.15,
                        "description": "Weight for group alignments similarity"
                    },
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3,
                        "description": "Minimum similarity threshold for results"
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 20,
                        "description": "Maximum number of results to return"
                    },
                    "include_breakdown": {
                        "type": "boolean",
                        "default": true,
                        "description": "Include per-component similarity breakdown in results"
                    }
                },
                "required": []
            }),
        ),
        // compute_teleological_vector - Compute full 13-embedder teleological vector
        ToolDefinition::new(
            "compute_teleological_vector",
            "Compute a complete teleological vector from content using all 13 embedders. \
             Returns purpose vector (13D), cross-correlations (78D), group alignments (6D), \
             and optional Tucker core decomposition.",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to compute teleological vector for"
                    },
                    "profile_id": {
                        "type": "string",
                        "description": "Optional profile ID for task-specific weighting"
                    },
                    "compute_tucker": {
                        "type": "boolean",
                        "default": false,
                        "description": "Compute Tucker decomposition for compressed representation"
                    },
                    "tucker_ranks": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "minItems": 3,
                        "maxItems": 3,
                        "default": [4, 4, 128],
                        "description": "Tucker decomposition ranks [r1, r2, r3]"
                    },
                    "include_per_embedder": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include raw per-embedder outputs (large)"
                    }
                },
                "required": ["content"]
            }),
        ),
        // fuse_embeddings - Fuse multiple embeddings using synergy matrix
        ToolDefinition::new(
            "fuse_embeddings",
            "Fuse embedding outputs using the synergy matrix and optional profile weights. \
             Supports multiple fusion methods: linear, attention-weighted, gated, hierarchical.",
            json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of memory to fuse embeddings for"
                    },
                    "fusion_method": {
                        "type": "string",
                        "enum": ["linear", "attention", "gated", "hierarchical", "tucker"],
                        "default": "hierarchical",
                        "description": "Fusion method to use"
                    },
                    "profile_id": {
                        "type": "string",
                        "description": "Profile ID for task-specific fusion weights"
                    },
                    "custom_weights": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 13,
                        "maxItems": 13,
                        "description": "Custom per-embedder weights [E1..E13]"
                    },
                    "apply_synergy": {
                        "type": "boolean",
                        "default": true,
                        "description": "Apply synergy matrix weighting to cross-correlations"
                    },
                    "store_result": {
                        "type": "boolean",
                        "default": true,
                        "description": "Store fused teleological vector in database"
                    }
                },
                "required": ["memory_id"]
            }),
        ),
        // update_synergy_matrix - Adaptively update synergy matrix from feedback
        ToolDefinition::new(
            "update_synergy_matrix",
            "Update the synergy matrix based on feedback from retrieval success/failure. \
             Implements online learning to adapt cross-embedding relationships.",
            json!({
                "type": "object",
                "properties": {
                    "query_vector_id": {
                        "type": "string",
                        "description": "ID of query teleological vector"
                    },
                    "result_vector_id": {
                        "type": "string",
                        "description": "ID of retrieved result vector"
                    },
                    "feedback": {
                        "type": "string",
                        "enum": ["relevant", "not_relevant", "partially_relevant"],
                        "description": "User feedback on retrieval quality"
                    },
                    "relevance_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Fine-grained relevance score [0.0, 1.0]"
                    },
                    "learning_rate": {
                        "type": "number",
                        "minimum": 0.001,
                        "maximum": 0.5,
                        "default": 0.01,
                        "description": "Learning rate for synergy update"
                    },
                    "update_scope": {
                        "type": "string",
                        "enum": ["all_pairs", "high_synergy_only", "contributing_pairs"],
                        "default": "contributing_pairs",
                        "description": "Which synergy pairs to update"
                    }
                },
                "required": ["query_vector_id", "result_vector_id", "feedback"]
            }),
        ),
        // manage_teleological_profile - CRUD for task-specific profiles
        ToolDefinition::new(
            "manage_teleological_profile",
            "Manage teleological profiles for task-specific embedding fusion. \
             Profiles define per-embedder weights, fusion strategy, and group priorities.",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "read", "update", "delete", "list"],
                        "description": "CRUD action to perform"
                    },
                    "profile_id": {
                        "type": "string",
                        "description": "Profile ID (required for read/update/delete)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable profile name"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["code_implementation", "research", "creative", "analysis", "debugging", "documentation", "custom"],
                        "description": "Predefined task type for default weights"
                    },
                    "embedder_weights": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 13,
                        "maxItems": 13,
                        "description": "Per-embedder weights [E1..E13]"
                    },
                    "group_priorities": {
                        "type": "object",
                        "properties": {
                            "factual": { "type": "number" },
                            "temporal": { "type": "number" },
                            "causal": { "type": "number" },
                            "relational": { "type": "number" },
                            "qualitative": { "type": "number" },
                            "implementation": { "type": "number" }
                        },
                        "description": "Priority weights for each embedding group"
                    },
                    "fusion_strategy": {
                        "type": "string",
                        "enum": ["linear", "attention", "gated", "hierarchical"],
                        "default": "hierarchical",
                        "description": "Default fusion strategy for this profile"
                    }
                },
                "required": ["action"]
            }),
        ),
    ]
}
