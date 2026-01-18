//! Type definitions for the BootstrapService
//!
//! Contains configuration, candidate, and result types used in the
//! bootstrap process for Strategic goal extraction.

use std::path::PathBuf;

use crate::autonomous::{BootstrapConfig, GoalId};

/// Configuration for the BootstrapService
#[derive(Clone, Debug)]
pub struct BootstrapServiceConfig {
    /// Directory containing documents to scan
    pub doc_dir: PathBuf,
    /// File extensions to include (e.g., ["md", "txt", "yaml"])
    pub file_extensions: Vec<String>,
    /// Maximum number of documents to process
    pub max_docs: usize,
    /// Bootstrap configuration from parent module
    pub bootstrap_config: BootstrapConfig,
}

impl Default for BootstrapServiceConfig {
    fn default() -> Self {
        Self {
            doc_dir: PathBuf::from("."),
            file_extensions: vec!["md".into(), "txt".into(), "yaml".into(), "json".into()],
            max_docs: 100,
            bootstrap_config: BootstrapConfig::default(),
        }
    }
}

/// A candidate goal extracted from a document
#[derive(Clone, Debug)]
pub struct GoalCandidate {
    /// The extracted goal text
    pub text: String,
    /// Source file where this candidate was found
    pub source: String,
    /// Position in source (0.0 = start, 1.0 = end)
    pub position: f32,
    /// Semantic density score (keyword concentration)
    pub density: f32,
    /// Number of goal-related keywords found
    pub keyword_count: usize,
    /// Line number where candidate was found
    pub line_number: usize,
}

/// Result of a successful bootstrap operation (service-specific)
#[derive(Clone, Debug)]
pub struct BootstrapResult {
    /// Unique identifier for the bootstrapped goal
    pub goal_id: GoalId,
    /// The extracted goal text
    pub goal_text: String,
    /// Confidence score for this goal (0.0 to 1.0)
    pub confidence: f32,
    /// Source file from which the goal was extracted
    pub extracted_from: String,
}
