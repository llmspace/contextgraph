//! UTL processor implementation.
//!
//! Contains the main `StubUtlProcessor` struct and its `UtlProcessor` trait implementation.

use async_trait::async_trait;

use crate::error::CoreResult;
use crate::traits::UtlProcessor;
use crate::types::{MemoryNode, UtlContext, UtlMetrics};

use super::math::{
    compute_delta_c_from_embeddings, compute_delta_s_from_embeddings, cosine_similarity, sigmoid,
};

/// Real UTL processor implementing constitution-specified computation.
///
/// Computes ΔS (surprise) using KNN distance from reference embeddings.
/// Computes ΔC (coherence) using connectivity to existing memories.
/// Applies sigmoid activation per the multi-embedding formula.
#[derive(Debug, Clone)]
pub struct StubUtlProcessor {
    /// Threshold for memory consolidation
    consolidation_threshold: f32,
    /// Default edge similarity threshold (θ_edge = 0.7 per constitution)
    default_edge_threshold: f32,
    /// Default max edges for connectivity normalization
    default_max_edges: usize,
    /// Default k for KNN computation
    default_k: usize,
}

impl Default for StubUtlProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StubUtlProcessor {
    /// Create a new UTL processor with default parameters.
    pub fn new() -> Self {
        Self {
            consolidation_threshold: 0.7,
            default_edge_threshold: 0.7, // θ_edge prior from constitution
            default_max_edges: 10,       // max_edges from constitution
            default_k: 5,                // k for KNN
        }
    }

    /// Create with custom consolidation threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            consolidation_threshold: threshold,
            ..Self::new()
        }
    }
}

#[async_trait]
impl UtlProcessor for StubUtlProcessor {
    /// Compute the full UTL learning score.
    ///
    /// Per constitution multi-embedding formula:
    /// L_multi = sigmoid(2.0 · ΔS · ΔC · wₑ · cos φ)
    ///
    /// When embeddings are available, uses real KNN-based computation.
    /// Falls back to sigmoid(2.0 * prior_entropy * current_coherence * wₑ * cos φ) otherwise.
    async fn compute_learning_score(&self, input: &str, context: &UtlContext) -> CoreResult<f32> {
        let surprise = self.compute_surprise(input, context).await?;
        let coherence_change = self.compute_coherence_change(input, context).await?;
        let emotional_weight = self.compute_emotional_weight(input, context).await?;
        let alignment = self.compute_alignment(input, context).await?;

        // Per constitution: L_multi = sigmoid(2.0 · ΔS · ΔC · wₑ · cos φ)
        // The 2.0 scaling factor ensures sigmoid output spans meaningful range
        let raw_score = 2.0 * surprise * coherence_change * emotional_weight * alignment;
        let score = sigmoid(raw_score);

        Ok(score.clamp(0.0, 1.0))
    }

    /// Compute surprise (ΔS) using KNN distance.
    ///
    /// Per constitution: ΔS_knn = σ((d_k - μ_corpus) / σ_corpus)
    ///
    /// When input_embedding and reference_embeddings are provided in context,
    /// computes real KNN-based surprise. Otherwise falls back to prior_entropy.
    async fn compute_surprise(&self, _input: &str, context: &UtlContext) -> CoreResult<f32> {
        // Check if we have embeddings for real computation
        if let (Some(input_emb), Some(ref_embs)) =
            (&context.input_embedding, &context.reference_embeddings)
        {
            // Get corpus statistics (use defaults if not provided)
            let stats = context.corpus_stats.clone().unwrap_or_default();

            // Real ΔS computation using KNN distance
            let delta_s = compute_delta_s_from_embeddings(
                input_emb,
                ref_embs,
                stats.mean_knn_distance,
                stats.std_knn_distance,
                stats.k,
            );

            return Ok(delta_s.clamp(0.0, 1.0));
        }

        // Fallback: use prior_entropy as a proxy for surprise
        // High prior entropy suggests high novelty potential
        Ok(context.prior_entropy.clamp(0.0, 1.0))
    }

    /// Compute coherence change (ΔC) using connectivity measure.
    ///
    /// Per constitution: ΔC = |{neighbors: sim(e, n) > θ_edge}| / max_edges
    ///
    /// When embeddings are available, computes real connectivity.
    /// Otherwise falls back to current_coherence from context.
    async fn compute_coherence_change(
        &self,
        _input: &str,
        context: &UtlContext,
    ) -> CoreResult<f32> {
        // Check if we have embeddings for real computation
        if let (Some(input_emb), Some(ref_embs)) =
            (&context.input_embedding, &context.reference_embeddings)
        {
            let edge_threshold = context
                .edge_similarity_threshold
                .unwrap_or(self.default_edge_threshold);
            let max_edges = context.max_edges.unwrap_or(self.default_max_edges);

            // Real ΔC computation using connectivity
            let delta_c =
                compute_delta_c_from_embeddings(input_emb, ref_embs, edge_threshold, max_edges);

            return Ok(delta_c.clamp(0.0, 1.0));
        }

        // Fallback: use current_coherence as a proxy
        Ok(context.current_coherence.clamp(0.0, 1.0))
    }

    /// Compute emotional weight (wₑ).
    ///
    /// Per constitution: wₑ ∈ [0.5, 1.5]
    ///
    /// Applies emotional state modifier to base weight of 1.0.
    async fn compute_emotional_weight(
        &self,
        _input: &str,
        context: &UtlContext,
    ) -> CoreResult<f32> {
        // Base weight is 1.0, modified by emotional state
        let weight = context.emotional_state.weight_modifier();
        Ok(weight.clamp(0.5, 1.5))
    }

    /// Compute goal alignment (cos φ).
    ///
    /// Per constitution: cos φ ∈ [-1, 1]
    ///
    /// When goal_vector and input_embedding are available, computes real cosine similarity.
    /// Otherwise defaults to 1.0 (full alignment).
    async fn compute_alignment(&self, _input: &str, context: &UtlContext) -> CoreResult<f32> {
        // Check if we have vectors for real alignment computation
        if let (Some(input_emb), Some(goal_vec)) = (&context.input_embedding, &context.goal_vector)
        {
            // Real alignment: cosine similarity to goal/Strategic goal vector
            let alignment = cosine_similarity(input_emb, goal_vec);
            return Ok(alignment.clamp(-1.0, 1.0));
        }

        // Default to full alignment (cos φ = 1.0)
        // Per constitution, this is the default for wₑ=1.0 and cos(φ)=1.0
        Ok(1.0)
    }

    /// Determine if a node should be consolidated to long-term memory.
    async fn should_consolidate(&self, node: &MemoryNode) -> CoreResult<bool> {
        Ok(node.importance >= self.consolidation_threshold)
    }

    /// Get full UTL metrics for input.
    async fn compute_metrics(&self, input: &str, context: &UtlContext) -> CoreResult<UtlMetrics> {
        let surprise = self.compute_surprise(input, context).await?;
        let coherence_change = self.compute_coherence_change(input, context).await?;
        let emotional_weight = self.compute_emotional_weight(input, context).await?;
        let alignment = self.compute_alignment(input, context).await?;
        let learning_score = self.compute_learning_score(input, context).await?;

        Ok(UtlMetrics {
            entropy: context.prior_entropy,
            coherence: context.current_coherence,
            learning_score,
            surprise,
            coherence_change,
            emotional_weight,
            alignment,
        })
    }

    /// Get current UTL system status as JSON.
    fn get_status(&self) -> serde_json::Value {
        serde_json::json!({
            "lifecycle_phase": "Infancy",
            "interaction_count": 0,
            "entropy": 0.0,
            "coherence": 0.0,
            "learning_score": 0.0,
            "classification": "Hidden",
            "consolidation_phase": "Wake",
            "phase_angle": 0.0,
            "computation_mode": "real",  // Indicates real UTL, not stub
            "formula": "L = sigmoid(2.0 * ΔS * ΔC * wₑ * cos φ)",
            "thresholds": {
                "entropy_trigger": 0.9,
                "coherence_trigger": 0.2,
                "min_importance_store": 0.1,
                "consolidation_threshold": self.consolidation_threshold,
                "edge_similarity": self.default_edge_threshold,
                "max_edges": self.default_max_edges,
                "knn_k": self.default_k
            }
        })
    }
}
