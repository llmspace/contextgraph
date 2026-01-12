//! Type definitions for Teleological MCP Tool Handlers.
//!
//! Contains parameter structs and JSON representations for the 5 teleological tools.

use context_graph_core::teleological::types::NUM_EMBEDDERS;
use serde::{Deserialize, Serialize};

// ============================================================================
// DEFAULT VALUE FUNCTIONS
// ============================================================================

pub(super) fn default_strategy() -> String {
    "adaptive".to_string()
}

pub(super) fn default_scope() -> String {
    "full".to_string()
}

pub(super) fn default_weight_purpose() -> f32 {
    0.4
}

pub(super) fn default_weight_correlations() -> f32 {
    0.35
}

pub(super) fn default_weight_groups() -> f32 {
    0.15
}

pub(super) fn default_min_similarity() -> f32 {
    0.3
}

pub(super) fn default_max_results() -> usize {
    20
}

pub(super) fn default_include_breakdown() -> bool {
    true
}

pub(super) fn default_confidence() -> f32 {
    1.0
}

pub(super) fn default_pv_weight() -> f32 {
    0.4
}

pub(super) fn default_cc_weight() -> f32 {
    0.35
}

pub(super) fn default_ga_weight() -> f32 {
    0.15
}

pub(super) fn default_conf_weight() -> f32 {
    0.1
}

pub(super) fn default_fusion_method() -> String {
    "weighted_average".to_string()
}

// ============================================================================
// PARAMETER STRUCTS
// ============================================================================

/// Parameters for search_teleological tool.
///
/// ISSUE-1 FIX: Updated to match tool definition in tools.rs (lines 558-644).
/// Accepts `query_content` (string to embed) OR `query_vector_id` (existing vector ID).
/// Searches against stored vectors in TeleologicalMemoryStore, NOT a candidates array.
#[derive(Debug, Deserialize)]
pub struct SearchTeleologicalParams {
    /// Content to search for (will be embedded using all 13 embedders).
    /// Mutually exclusive with `query_vector_id`.
    pub query_content: Option<String>,

    /// ID of existing teleological vector to use as query.
    /// Mutually exclusive with `query_content`.
    pub query_vector_id: Option<String>,

    /// Search strategy (default: "adaptive")
    #[serde(default = "default_strategy")]
    pub strategy: String,

    /// Comparison scope (default: "full")
    #[serde(default = "default_scope")]
    pub scope: String,

    /// Specific groups for SpecificGroups scope.
    /// Reserved for future use with parse_scope when SpecificGroups scope is enabled.
    #[allow(dead_code)]
    pub specific_groups: Option<Vec<String>>,

    /// Specific embedder index for SingleEmbedderPattern scope (0=Semantic, 5=Code, 12=Sparse).
    /// Reserved for future use with parse_scope when SingleEmbedderPattern scope is enabled.
    #[allow(dead_code)]
    pub specific_embedder: Option<usize>,

    /// Weight for purpose vector similarity (default: 0.4).
    /// Reserved for future use with custom component weighting.
    #[allow(dead_code)]
    #[serde(default = "default_weight_purpose")]
    pub weight_purpose: f32,

    /// Weight for cross-correlation similarity (default: 0.35).
    /// Reserved for future use with custom component weighting.
    #[allow(dead_code)]
    #[serde(default = "default_weight_correlations")]
    pub weight_correlations: f32,

    /// Weight for group alignments similarity (default: 0.15).
    /// Reserved for future use with custom component weighting.
    #[allow(dead_code)]
    #[serde(default = "default_weight_groups")]
    pub weight_groups: f32,

    /// Minimum similarity threshold (default: 0.3)
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,

    /// Maximum number of results to return (default: 20)
    #[serde(default = "default_max_results")]
    pub max_results: usize,

    /// Include per-component similarity breakdown in results (default: true)
    #[serde(default = "default_include_breakdown")]
    pub include_breakdown: bool,

    /// Include original content text in results (default: false).
    /// TASK-CONTENT-011: When true, content is hydrated from CF_CONTENT.
    #[serde(default)]
    pub include_content: bool,
}

/// JSON representation of TeleologicalVector.
/// Uses Vec<f32> for cross_correlations due to serde's 32-element array limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalVectorJson {
    /// 13D purpose vector alignments
    pub purpose_vector: [f32; NUM_EMBEDDERS],
    /// 78 cross-correlations (Vec for serde compatibility)
    pub cross_correlations: Vec<f32>,
    /// 6D group alignments [factual, temporal, causal, relational, qualitative, implementation]
    pub group_alignments: [f32; 6],
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    pub id: Option<String>,
}

/// JSON representation of ComponentWeights.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentWeightsJson {
    #[serde(default = "default_pv_weight")]
    pub purpose_vector: f32,
    #[serde(default = "default_cc_weight")]
    pub cross_correlations: f32,
    #[serde(default = "default_ga_weight")]
    pub group_alignments: f32,
    #[serde(default = "default_conf_weight")]
    pub confidence: f32,
}

/// Parameters for compute_teleological_vector tool.
#[derive(Debug, Deserialize)]
pub struct ComputeTeleologicalVectorParams {
    /// Content to embed
    pub content: String,
    /// Whether to compute Tucker compression (default: false)
    #[serde(default)]
    pub compute_tucker: bool,
    /// Profile ID for weighted computation (optional)
    #[allow(dead_code)]
    pub profile_id: Option<String>,
}

/// Parameters for fuse_embeddings tool.
#[derive(Debug, Deserialize)]
pub struct FuseEmbeddingsParams {
    /// 13 embedding vectors to fuse
    pub embeddings: Vec<Vec<f32>>,
    /// 13D alignment scores for purpose vector
    pub alignments: Option<[f32; NUM_EMBEDDERS]>,
    /// Profile ID for fusion weights (optional)
    #[allow(dead_code)]
    pub profile_id: Option<String>,
    /// Custom synergy matrix (optional, uses default if omitted)
    #[allow(dead_code)]
    pub synergy_matrix: Option<SynergyMatrixJson>,
    /// Fusion method (default: "weighted_average")
    #[serde(default = "default_fusion_method")]
    pub fusion_method: String,
}

/// JSON representation of SynergyMatrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynergyMatrixJson {
    pub weights: [[f32; NUM_EMBEDDERS]; NUM_EMBEDDERS],
}

/// Parameters for update_synergy_matrix tool.
#[derive(Debug, Deserialize)]
pub struct UpdateSynergyMatrixParams {
    /// Vector ID that was retrieved
    pub vector_id: String,
    /// Type of feedback
    pub feedback_type: String,
    /// Per-embedder contributions (optional)
    pub contributions: Option<[f32; NUM_EMBEDDERS]>,
    /// Context string (optional)
    pub context: Option<String>,
}

/// Parameters for manage_teleological_profile tool.
#[derive(Debug, Deserialize)]
pub struct ManageTeleologicalProfileParams {
    /// Action to perform
    pub action: String,
    /// Profile ID (required for create/get/update/delete)
    pub profile_id: Option<String>,
    /// Weights (required for create/update)
    pub weights: Option<[f32; NUM_EMBEDDERS]>,
    /// Context string for find_best action
    pub context: Option<String>,
}
