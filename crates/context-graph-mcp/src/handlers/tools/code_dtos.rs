//! DTOs for E7 code search MCP tools.
//!
//! Per PRD v6 and CLAUDE.md, E7 (V_correctness) provides:
//! - Code patterns and function signatures via 1536D dense embeddings
//! - Code-specific understanding that E1 misses by treating code as natural language
//!
//! # Constitution Compliance
//!
//! - ARCH-12: E1 is the semantic foundation, E7 enhances with code understanding
//! - E7 finds: "Code patterns, function signatures" that E1 misses by "Treating code as NL"
//! - Use E7 for: "Code queries (implementations, functions)"
//! - FAIL FAST: All errors propagate immediately with logging

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default topK for code search results.
pub const DEFAULT_CODE_SEARCH_TOP_K: usize = 10;

/// Maximum topK for code search results.
pub const MAX_CODE_SEARCH_TOP_K: usize = 50;

/// Default minimum score threshold for code search results.
pub const DEFAULT_MIN_CODE_SCORE: f32 = 0.2;

/// Default blend weight for E7 vs E1 semantic.
/// 0.4 means 60% E1 semantic + 40% E7 code.
/// E7 needs significant weight for code-specific queries.
pub const DEFAULT_CODE_BLEND: f32 = 0.4;

// ============================================================================
// DETECTED LANGUAGE
// ============================================================================

/// Detected programming language in query.
#[derive(Debug, Clone, Serialize)]
pub struct DetectedLanguageInfo {
    /// Primary language detected (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_language: Option<String>,

    /// Confidence score for language detection (0-1).
    pub confidence: f32,

    /// Indicators that led to detection.
    pub indicators: Vec<String>,
}

// ============================================================================
// REQUEST DTOs
// ============================================================================

/// Request parameters for search_code tool.
///
/// # Example JSON
/// ```json
/// {
///   "query": "async function that handles HTTP requests",
///   "topK": 10,
///   "minScore": 0.2,
///   "blendWithSemantic": 0.4,
///   "includeContent": true
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct SearchCodeRequest {
    /// The code query to search for (required).
    /// Can describe functionality, patterns, or specific code constructs.
    pub query: String,

    /// Maximum number of results to return (1-50, default: 10).
    #[serde(rename = "topK", default = "default_top_k")]
    pub top_k: usize,

    /// Minimum score threshold (0-1, default: 0.2).
    #[serde(rename = "minScore", default = "default_min_score")]
    pub min_score: f32,

    /// Blend weight for E7 code vs E1 semantic (0-1, default: 0.4).
    /// 0.0 = pure E1 semantic, 1.0 = pure E7 code.
    #[serde(rename = "blendWithSemantic", default = "default_blend")]
    pub blend_with_semantic: f32,

    /// Whether to include full content text in results (default: false).
    #[serde(rename = "includeContent", default)]
    pub include_content: bool,
}

fn default_top_k() -> usize {
    DEFAULT_CODE_SEARCH_TOP_K
}

fn default_min_score() -> f32 {
    DEFAULT_MIN_CODE_SCORE
}

fn default_blend() -> f32 {
    DEFAULT_CODE_BLEND
}

impl Default for SearchCodeRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            top_k: DEFAULT_CODE_SEARCH_TOP_K,
            min_score: DEFAULT_MIN_CODE_SCORE,
            blend_with_semantic: DEFAULT_CODE_BLEND,
            include_content: false,
        }
    }
}

impl SearchCodeRequest {
    /// Validate the request parameters.
    ///
    /// # Errors
    /// Returns an error message if:
    /// - query is empty
    /// - topK is outside [1, 50]
    /// - minScore is outside [0, 1] or NaN/infinite
    /// - blendWithSemantic is outside [0, 1] or NaN/infinite
    pub fn validate(&self) -> Result<(), String> {
        if self.query.is_empty() {
            return Err("query is required and cannot be empty".to_string());
        }

        if self.top_k < 1 || self.top_k > MAX_CODE_SEARCH_TOP_K {
            return Err(format!(
                "topK must be between 1 and {}, got {}",
                MAX_CODE_SEARCH_TOP_K, self.top_k
            ));
        }

        if self.min_score.is_nan() || self.min_score.is_infinite() {
            return Err("minScore must be a finite number".to_string());
        }

        if self.min_score < 0.0 || self.min_score > 1.0 {
            return Err(format!(
                "minScore must be between 0.0 and 1.0, got {}",
                self.min_score
            ));
        }

        if self.blend_with_semantic.is_nan() || self.blend_with_semantic.is_infinite() {
            return Err("blendWithSemantic must be a finite number".to_string());
        }

        if self.blend_with_semantic < 0.0 || self.blend_with_semantic > 1.0 {
            return Err(format!(
                "blendWithSemantic must be between 0.0 and 1.0, got {}",
                self.blend_with_semantic
            ));
        }

        Ok(())
    }
}

// ============================================================================
// RESPONSE DTOs
// ============================================================================

/// A single search result for code search.
#[derive(Debug, Clone, Serialize)]
pub struct CodeSearchResult {
    /// UUID of the matched memory.
    #[serde(rename = "memoryId")]
    pub memory_id: Uuid,

    /// Blended score (E1 semantic + E7 code).
    pub score: f32,

    /// Raw E1 semantic similarity (before blending).
    #[serde(rename = "e1Similarity")]
    pub e1_similarity: f32,

    /// Raw E7 code similarity (before blending).
    #[serde(rename = "e7CodeScore")]
    pub e7_code_score: f32,

    /// Full content text (if includeContent=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Source provenance information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<CodeSourceInfo>,
}

/// Source provenance information.
#[derive(Debug, Clone, Serialize)]
pub struct CodeSourceInfo {
    /// Type of source (HookDescription, ClaudeResponse, MDFileChunk).
    #[serde(rename = "sourceType")]
    pub source_type: String,

    /// File path if from file source.
    #[serde(skip_serializing_if = "Option::is_none", rename = "filePath")]
    pub file_path: Option<String>,

    /// Hook type if from hook source.
    #[serde(skip_serializing_if = "Option::is_none", rename = "hookType")]
    pub hook_type: Option<String>,

    /// Tool name if from tool use.
    #[serde(skip_serializing_if = "Option::is_none", rename = "toolName")]
    pub tool_name: Option<String>,
}

/// Response metadata for code search.
#[derive(Debug, Clone, Serialize)]
pub struct CodeSearchMetadata {
    /// Number of candidates evaluated before filtering.
    #[serde(rename = "candidatesEvaluated")]
    pub candidates_evaluated: usize,

    /// Number of results filtered by score threshold.
    #[serde(rename = "filteredByScore")]
    pub filtered_by_score: usize,

    /// E7 blend weight used.
    #[serde(rename = "e7BlendWeight")]
    pub e7_blend_weight: f32,

    /// E1 weight (1.0 - e7BlendWeight).
    #[serde(rename = "e1Weight")]
    pub e1_weight: f32,

    /// Detected language info from query.
    #[serde(rename = "detectedLanguage")]
    pub detected_language: DetectedLanguageInfo,
}

/// Response for search_code tool.
#[derive(Debug, Clone, Serialize)]
pub struct SearchCodeResponse {
    /// Original query.
    pub query: String,

    /// Matched results with blended scores.
    pub results: Vec<CodeSearchResult>,

    /// Number of results returned.
    pub count: usize,

    /// Metadata about the search.
    pub metadata: CodeSearchMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_code_validation_success() {
        let req = SearchCodeRequest {
            query: "async function HTTP handler".to_string(),
            ..Default::default()
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_search_code_empty_query() {
        let req = SearchCodeRequest::default();
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_search_code_invalid_blend() {
        let req = SearchCodeRequest {
            query: "test".to_string(),
            blend_with_semantic: 1.5,
            ..Default::default()
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_search_code_invalid_top_k() {
        let req = SearchCodeRequest {
            query: "test".to_string(),
            top_k: 100,
            ..Default::default()
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_default_blend_weight() {
        // E7 needs significant weight (0.4) for code queries
        assert!((DEFAULT_CODE_BLEND - 0.4).abs() < 0.001);
    }
}
