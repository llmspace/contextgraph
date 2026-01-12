//! Main BootstrapService implementation
//!
//! NORTH-008: Service for initializing North Star goal from documentation sources.
//! Extracts goal candidates from project documents and selects the best
//! goal to serve as the North Star for the context graph system.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::autonomous::GoalId;

use super::extraction::extract_goal_candidates;
use super::scoring::CandidateScoring;
use super::types::{BootstrapResult, BootstrapServiceConfig, GoalCandidate};

/// NORTH-008: Service for bootstrapping North Star goals from documents
#[derive(Debug)]
pub struct BootstrapService {
    config: BootstrapServiceConfig,
    results_cache: HashMap<PathBuf, Vec<GoalCandidate>>,
}

impl CandidateScoring for BootstrapService {
    fn config(&self) -> &BootstrapServiceConfig {
        &self.config
    }
}

impl BootstrapService {
    /// Create a new BootstrapService with default configuration
    pub fn new() -> Self {
        Self {
            config: BootstrapServiceConfig::default(),
            results_cache: HashMap::new(),
        }
    }

    /// Create a new BootstrapService with custom configuration
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `config.max_docs` is 0
    /// - `config.file_extensions` is empty
    pub fn with_config(config: BootstrapServiceConfig) -> Self {
        assert!(config.max_docs > 0, "max_docs must be greater than 0");
        assert!(
            !config.file_extensions.is_empty(),
            "file_extensions cannot be empty"
        );

        Self {
            config,
            results_cache: HashMap::new(),
        }
    }

    /// Bootstrap goals from documents in the specified directory
    ///
    /// # Arguments
    ///
    /// * `doc_dir` - Directory containing documents to scan
    ///
    /// # Returns
    ///
    /// Vector of bootstrap results, one for each successfully extracted goal
    ///
    /// # Panics
    ///
    /// Panics if `doc_dir` does not exist or is not a directory
    pub fn bootstrap_from_documents(&mut self, doc_dir: &Path) -> Vec<BootstrapResult> {
        assert!(
            doc_dir.exists(),
            "Document directory does not exist: {:?}",
            doc_dir
        );
        assert!(doc_dir.is_dir(), "Path is not a directory: {:?}", doc_dir);

        let mut all_candidates: Vec<GoalCandidate> = Vec::new();

        // Collect matching files
        let files = self.collect_files(doc_dir);

        // Process each file up to max_docs limit
        for file_path in files.into_iter().take(self.config.max_docs) {
            if let Ok(content) = fs::read_to_string(&file_path) {
                let source = file_path.to_string_lossy().to_string();
                let candidates = self.extract_goal_candidates(&content, &source);

                // Cache results for potential reuse
                self.results_cache
                    .insert(file_path.clone(), candidates.clone());

                all_candidates.extend(candidates);
            }
        }

        // Select best goal and return as single-element vec if found
        let mut results = Vec::new();
        if let Some(best_result) = self.select_best_goal(&all_candidates) {
            results.push(best_result);
        }

        results
    }

    /// Extract goal candidates from document content
    ///
    /// # Arguments
    ///
    /// * `content` - The document text content
    /// * `source` - The source file path or identifier
    ///
    /// # Returns
    ///
    /// Vector of goal candidates found in the document
    ///
    /// # Panics
    ///
    /// Panics if `source` is empty
    pub fn extract_goal_candidates(&self, content: &str, source: &str) -> Vec<GoalCandidate> {
        extract_goal_candidates(content, source)
    }

    /// Select the best goal from a set of candidates
    ///
    /// # Arguments
    ///
    /// * `candidates` - Slice of goal candidates to evaluate
    ///
    /// # Returns
    ///
    /// The best bootstrap result if any candidates meet the minimum threshold
    pub fn select_best_goal(&self, candidates: &[GoalCandidate]) -> Option<BootstrapResult> {
        if candidates.is_empty() {
            return None;
        }

        // Score all candidates
        let mut scored: Vec<(&GoalCandidate, f32)> = candidates
            .iter()
            .map(|c| (c, self.score_candidate(c)))
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get the best candidate
        let (best_candidate, confidence) = scored.into_iter().next()?;

        // Apply minimum confidence threshold from config
        let min_confidence = self.config.bootstrap_config.min_confidence;
        if confidence < min_confidence {
            return None;
        }

        Some(BootstrapResult {
            goal_id: GoalId::new(),
            goal_text: best_candidate.text.clone(),
            confidence,
            extracted_from: best_candidate.source.clone(),
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &BootstrapServiceConfig {
        &self.config
    }

    /// Get cached results
    pub fn results_cache(&self) -> &HashMap<PathBuf, Vec<GoalCandidate>> {
        &self.results_cache
    }

    /// Collect all files matching the configured extensions
    fn collect_files(&self, dir: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext_str = ext.to_string_lossy().to_lowercase();
                        if self
                            .config
                            .file_extensions
                            .iter()
                            .any(|e| e.to_lowercase() == ext_str)
                        {
                            files.push(path);
                        }
                    }
                } else if path.is_dir() {
                    // Recursively scan subdirectories
                    files.extend(self.collect_files(&path));
                }
            }
        }

        files
    }
}

impl Default for BootstrapService {
    fn default() -> Self {
        Self::new()
    }
}
