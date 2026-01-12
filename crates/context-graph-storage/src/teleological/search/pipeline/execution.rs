//! Main pipeline execution logic.
//!
//! This module contains the `RetrievalPipeline` struct and the core
//! execution logic for the 5-stage retrieval pipeline.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use super::super::super::indexes::{EmbedderIndex, EmbedderIndexRegistry};
use super::super::error::SearchError;
use super::super::multi::MultiEmbedderSearch;
use super::super::single::SingleEmbedderSearch;
use super::stages::StageExecutor;
use super::traits::{InMemorySpladeIndex, InMemoryTokenStorage, SpladeIndex, TokenStorage};
use super::types::{
    PipelineCandidate, PipelineConfig, PipelineError, PipelineResult, PipelineStage, StageResult,
};

// ============================================================================
// RETRIEVAL PIPELINE
// ============================================================================

/// The 5-stage retrieval pipeline.
pub struct RetrievalPipeline {
    /// Single embedder search (for Stages 2, 4).
    single_search: SingleEmbedderSearch,
    /// Multi embedder search (for Stage 3).
    /// Currently unused - reserved for enhanced RRF with multiple embedders.
    #[allow(dead_code)]
    multi_search: MultiEmbedderSearch,
    /// SPLADE inverted index (for Stage 1).
    splade_index: Arc<dyn SpladeIndex>,
    /// Token storage (for Stage 5 MaxSim).
    token_storage: Arc<dyn TokenStorage>,
    /// Pipeline configuration.
    pub(crate) config: PipelineConfig,
}

impl RetrievalPipeline {
    /// Create a new pipeline with registry.
    ///
    /// # Arguments
    /// * `registry` - Embedder index registry
    /// * `splade_index` - Optional SPLADE index (creates empty in-memory if None)
    /// * `token_storage` - Optional token storage (creates empty in-memory if None)
    pub fn new(
        registry: Arc<EmbedderIndexRegistry>,
        splade_index: Option<Arc<dyn SpladeIndex>>,
        token_storage: Option<Arc<dyn TokenStorage>>,
    ) -> Self {
        Self {
            single_search: SingleEmbedderSearch::new(Arc::clone(&registry)),
            multi_search: MultiEmbedderSearch::new(registry),
            splade_index: splade_index
                .unwrap_or_else(|| Arc::new(InMemorySpladeIndex::new())),
            token_storage: token_storage
                .unwrap_or_else(|| Arc::new(InMemoryTokenStorage::new())),
            config: PipelineConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        registry: Arc<EmbedderIndexRegistry>,
        config: PipelineConfig,
        splade_index: Option<Arc<dyn SpladeIndex>>,
        token_storage: Option<Arc<dyn TokenStorage>>,
    ) -> Self {
        Self {
            single_search: SingleEmbedderSearch::new(Arc::clone(&registry)),
            multi_search: MultiEmbedderSearch::new(registry),
            splade_index: splade_index
                .unwrap_or_else(|| Arc::new(InMemorySpladeIndex::new())),
            token_storage: token_storage
                .unwrap_or_else(|| Arc::new(InMemoryTokenStorage::new())),
            config,
        }
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Execute full 5-stage pipeline.
    ///
    /// # Arguments
    /// * `query_splade` - Sparse vector for Stage 1 as (term_id, weight) pairs
    /// * `query_matryoshka` - 128D vector for Stage 2
    /// * `query_semantic` - 1024D vector for Stage 3 RRF
    /// * `query_tokens` - Token embeddings for Stage 5 MaxSim (each 128D)
    ///
    /// # FAIL FAST Errors
    /// - `SearchError::InvalidVector` if query embeddings are invalid
    /// - `SearchError::DimensionMismatch` if query dimensions wrong
    /// - `PipelineError::Timeout` if any stage exceeds max_latency_ms
    /// - `PipelineError::MissingPurposeVector` if Stage 4 enabled but no purpose vector
    pub fn execute(
        &self,
        query_splade: &[(usize, f32)],
        query_matryoshka: &[f32],
        query_semantic: &[f32],
        query_tokens: &[Vec<f32>],
    ) -> Result<PipelineResult, PipelineError> {
        self.execute_stages(
            query_splade,
            query_matryoshka,
            query_semantic,
            query_tokens,
            &PipelineStage::all(),
        )
    }

    /// Execute with stage selection.
    pub fn execute_stages(
        &self,
        query_splade: &[(usize, f32)],
        query_matryoshka: &[f32],
        query_semantic: &[f32],
        query_tokens: &[Vec<f32>],
        stages: &[PipelineStage],
    ) -> Result<PipelineResult, PipelineError> {
        let pipeline_start = Instant::now();
        let mut stage_results = Vec::with_capacity(5);
        let mut stages_executed = Vec::with_capacity(5);
        let mut candidates: Vec<PipelineCandidate> = Vec::new();
        let mut alignment_verified = false;

        // Validate queries upfront - FAIL FAST
        self.validate_queries(query_matryoshka, query_semantic, query_tokens, stages)?;

        // Create stage set for O(1) lookup
        let stage_set: HashSet<_> = stages.iter().copied().collect();

        // Create stage executor
        let executor = StageExecutor {
            single_search: &self.single_search,
            splade_index: &self.splade_index,
            token_storage: &self.token_storage,
            config: &self.config,
        };

        // Stage 1: SPLADE Filter
        if stage_set.contains(&PipelineStage::SpladeFilter)
            && self.config.stages[0].enabled
        {
            let result = executor.stage_splade_filter(query_splade, &self.config.stages[0])?;
            // Extract Copy fields before moving Vec
            let latency_us = result.latency_us;
            let candidates_in = result.candidates_in;
            let candidates_out = result.candidates_out;
            let stage = result.stage;
            candidates = result.candidates;
            let stage_result = StageResult {
                candidates: Vec::new(), // Don't store candidates in stage result
                latency_us,
                candidates_in,
                candidates_out,
                stage,
            };
            stage_results.push(stage_result);
            stages_executed.push(PipelineStage::SpladeFilter);
        }

        // Stage 2: Matryoshka ANN
        if stage_set.contains(&PipelineStage::MatryoshkaAnn)
            && self.config.stages[1].enabled
        {
            let result = executor.stage_matryoshka_ann(
                query_matryoshka,
                candidates,
                &self.config.stages[1],
            )?;
            // Extract Copy fields before moving Vec
            let latency_us = result.latency_us;
            let candidates_in = result.candidates_in;
            let candidates_out = result.candidates_out;
            let stage = result.stage;
            candidates = result.candidates;
            let stage_result = StageResult {
                candidates: Vec::new(),
                latency_us,
                candidates_in,
                candidates_out,
                stage,
            };
            stage_results.push(stage_result);
            stages_executed.push(PipelineStage::MatryoshkaAnn);
        }

        // Stage 3: RRF Rerank
        if stage_set.contains(&PipelineStage::RrfRerank)
            && self.config.stages[2].enabled
        {
            let result = executor.stage_rrf_rerank(
                query_semantic,
                candidates,
                &self.config.stages[2],
            )?;
            // Extract Copy fields before moving Vec
            let latency_us = result.latency_us;
            let candidates_in = result.candidates_in;
            let candidates_out = result.candidates_out;
            let stage = result.stage;
            candidates = result.candidates;
            let stage_result = StageResult {
                candidates: Vec::new(),
                latency_us,
                candidates_in,
                candidates_out,
                stage,
            };
            stage_results.push(stage_result);
            stages_executed.push(PipelineStage::RrfRerank);
        }

        // Stage 4: Alignment Filter
        if stage_set.contains(&PipelineStage::AlignmentFilter)
            && self.config.stages[3].enabled
        {
            let result = executor.stage_alignment_filter(
                candidates,
                &self.config.stages[3],
            )?;
            // Extract Copy fields before moving Vec
            let latency_us = result.latency_us;
            let candidates_in = result.candidates_in;
            let candidates_out = result.candidates_out;
            let stage = result.stage;
            candidates = result.candidates;
            let stage_result = StageResult {
                candidates: Vec::new(),
                latency_us,
                candidates_in,
                candidates_out,
                stage,
            };
            stage_results.push(stage_result);
            stages_executed.push(PipelineStage::AlignmentFilter);
            alignment_verified = true;
        }

        // Stage 5: MaxSim Rerank
        if stage_set.contains(&PipelineStage::MaxSimRerank)
            && self.config.stages[4].enabled
        {
            let result = executor.stage_maxsim_rerank(
                query_tokens,
                candidates,
                &self.config.stages[4],
            )?;
            // Extract Copy fields before moving Vec
            let latency_us = result.latency_us;
            let candidates_in = result.candidates_in;
            let candidates_out = result.candidates_out;
            let stage = result.stage;
            candidates = result.candidates;
            let stage_result = StageResult {
                candidates: Vec::new(),
                latency_us,
                candidates_in,
                candidates_out,
                stage,
            };
            stage_results.push(stage_result);
            stages_executed.push(PipelineStage::MaxSimRerank);
        }

        // Final truncation to k
        candidates.truncate(self.config.k);

        let total_latency_us = pipeline_start.elapsed().as_micros() as u64;

        Ok(PipelineResult {
            results: candidates,
            stage_results,
            total_latency_us,
            stages_executed,
            alignment_verified,
        })
    }

    /// Validate query vectors upfront - FAIL FAST.
    fn validate_queries(
        &self,
        query_matryoshka: &[f32],
        query_semantic: &[f32],
        query_tokens: &[Vec<f32>],
        stages: &[PipelineStage],
    ) -> Result<(), PipelineError> {
        let stage_set: HashSet<_> = stages.iter().copied().collect();

        // Validate Matryoshka dimension (Stage 2)
        if stage_set.contains(&PipelineStage::MatryoshkaAnn) && self.config.stages[1].enabled {
            if query_matryoshka.len() != 128 {
                return Err(SearchError::DimensionMismatch {
                    embedder: EmbedderIndex::E1Matryoshka128,
                    expected: 128,
                    actual: query_matryoshka.len(),
                }
                .into());
            }
            self.validate_vector(query_matryoshka, EmbedderIndex::E1Matryoshka128)?;
        }

        // Validate semantic dimension (Stage 3)
        if stage_set.contains(&PipelineStage::RrfRerank) && self.config.stages[2].enabled {
            if query_semantic.len() != 1024 {
                return Err(SearchError::DimensionMismatch {
                    embedder: EmbedderIndex::E1Semantic,
                    expected: 1024,
                    actual: query_semantic.len(),
                }
                .into());
            }
            self.validate_vector(query_semantic, EmbedderIndex::E1Semantic)?;
        }

        // Validate token dimensions (Stage 5)
        if stage_set.contains(&PipelineStage::MaxSimRerank) && self.config.stages[4].enabled {
            for (i, token) in query_tokens.iter().enumerate() {
                if token.len() != 128 {
                    return Err(SearchError::InvalidVector {
                        embedder: EmbedderIndex::E12LateInteraction,
                        message: format!(
                            "Token {} has dimension {}, expected 128",
                            i,
                            token.len()
                        ),
                    }
                    .into());
                }
                self.validate_vector(token, EmbedderIndex::E12LateInteraction)?;
            }
        }

        // Validate purpose vector for Stage 4
        if stage_set.contains(&PipelineStage::AlignmentFilter)
            && self.config.stages[3].enabled
            && self.config.purpose_vector.is_none()
        {
            return Err(PipelineError::MissingPurposeVector);
        }

        Ok(())
    }

    /// Validate a single vector for NaN/Inf - FAIL FAST.
    fn validate_vector(&self, vector: &[f32], embedder: EmbedderIndex) -> Result<(), PipelineError> {
        for (i, &v) in vector.iter().enumerate() {
            if v.is_nan() {
                return Err(SearchError::InvalidVector {
                    embedder,
                    message: format!("NaN at index {}", i),
                }
                .into());
            }
            if v.is_infinite() {
                return Err(SearchError::InvalidVector {
                    embedder,
                    message: format!("Inf at index {}", i),
                }
                .into());
            }
        }
        Ok(())
    }
}
