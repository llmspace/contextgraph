//! Real Embedding-Based UTL Computation Tests.

use crate::stubs::utl_stub::StubUtlProcessor;
use crate::traits::UtlProcessor;
use crate::types::{CorpusStats, EmotionalState, UtlContext};

fn create_test_embedding(dim: usize, base_val: f32) -> Vec<f32> {
    (0..dim).map(|i| base_val + (i as f32 * 0.01)).collect()
}

fn create_reference_embeddings(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count).map(|i| create_test_embedding(dim, i as f32 * 0.1)).collect()
}

#[tokio::test]
async fn test_real_surprise_computation_with_embeddings() {
    let processor = StubUtlProcessor::new();
    let dim = 8;
    let mut references = Vec::new();
    for i in 0..5 {
        let mut emb = vec![0.0; dim];
        emb[0] = i as f32 * 0.1;
        references.push(emb);
    }

    let close_input = vec![0.0; dim];
    let mut far_input = vec![0.0; dim];
    far_input[0] = 2.0;

    let corpus_stats = CorpusStats { mean_knn_distance: 0.5, std_knn_distance: 0.5, k: 1 };

    let close_context = UtlContext {
        input_embedding: Some(close_input),
        reference_embeddings: Some(references.clone()),
        corpus_stats: Some(corpus_stats.clone()),
        ..Default::default()
    };

    let far_context = UtlContext {
        input_embedding: Some(far_input),
        reference_embeddings: Some(references),
        corpus_stats: Some(corpus_stats),
        ..Default::default()
    };

    let close_surprise = processor.compute_surprise("", &close_context).await.unwrap();
    let far_surprise = processor.compute_surprise("", &far_context).await.unwrap();

    assert!(far_surprise > close_surprise, "Far > close surprise");
    assert!((0.0..=1.0).contains(&close_surprise));
    assert!((0.0..=1.0).contains(&far_surprise));
    assert!(close_surprise < 0.5, "Close input should have surprise < 0.5");
    assert!(far_surprise > 0.5, "Far input should have surprise > 0.5");
}

#[tokio::test]
async fn test_real_coherence_computation_with_embeddings() {
    let processor = StubUtlProcessor::new();
    let dim = 128;

    let mut references = Vec::new();
    for i in 0..5 {
        let mut emb = vec![0.0; dim];
        emb[0] = 1.0;
        emb[1] = i as f32 * 0.1;
        references.push(emb);
    }

    let mut similar_input = vec![0.0; dim];
    similar_input[0] = 1.0;
    similar_input[1] = 0.25;

    let mut different_input = vec![0.0; dim];
    different_input[dim - 1] = 1.0;

    let similar_context = UtlContext {
        input_embedding: Some(similar_input),
        reference_embeddings: Some(references.clone()),
        edge_similarity_threshold: Some(0.5),
        max_edges: Some(5),
        ..Default::default()
    };

    let different_context = UtlContext {
        input_embedding: Some(different_input),
        reference_embeddings: Some(references),
        edge_similarity_threshold: Some(0.5),
        max_edges: Some(5),
        ..Default::default()
    };

    let similar_coherence = processor.compute_coherence_change("", &similar_context).await.unwrap();
    let different_coherence = processor.compute_coherence_change("", &different_context).await.unwrap();

    assert!(similar_coherence > different_coherence, "Similar > different coherence");
    assert!((0.0..=1.0).contains(&similar_coherence));
    assert!((0.0..=1.0).contains(&different_coherence));
}

#[tokio::test]
async fn test_real_alignment_computation_with_embeddings() {
    let processor = StubUtlProcessor::new();
    let dim = 128;

    let mut goal = vec![0.0; dim];
    goal[0] = 1.0;

    let mut aligned_input = vec![0.0; dim];
    aligned_input[0] = 1.0;

    let mut orthogonal_input = vec![0.0; dim];
    orthogonal_input[1] = 1.0;

    let mut opposite_input = vec![0.0; dim];
    opposite_input[0] = -1.0;

    let aligned_context = UtlContext { input_embedding: Some(aligned_input), goal_vector: Some(goal.clone()), ..Default::default() };
    let orthogonal_context = UtlContext { input_embedding: Some(orthogonal_input), goal_vector: Some(goal.clone()), ..Default::default() };
    let opposite_context = UtlContext { input_embedding: Some(opposite_input), goal_vector: Some(goal), ..Default::default() };

    let aligned = processor.compute_alignment("", &aligned_context).await.unwrap();
    let orthogonal = processor.compute_alignment("", &orthogonal_context).await.unwrap();
    let opposite = processor.compute_alignment("", &opposite_context).await.unwrap();

    assert!((aligned - 1.0).abs() < 0.001, "Aligned ~1.0");
    assert!(orthogonal.abs() < 0.001, "Orthogonal ~0.0");
    assert!((opposite - (-1.0)).abs() < 0.001, "Opposite ~-1.0");
}

#[tokio::test]
async fn test_full_utl_with_real_embeddings() {
    let processor = StubUtlProcessor::new();
    let dim = 128;
    let references = create_reference_embeddings(10, dim);

    let mut goal = vec![0.0; dim];
    goal[0] = 1.0;

    let mut input = vec![0.0; dim];
    input[0] = 0.8;
    input[1] = 0.6;

    let context = UtlContext {
        input_embedding: Some(input),
        reference_embeddings: Some(references),
        goal_vector: Some(goal),
        corpus_stats: Some(CorpusStats { mean_knn_distance: 0.5, std_knn_distance: 0.2, k: 3 }),
        edge_similarity_threshold: Some(0.5),
        max_edges: Some(10),
        emotional_state: EmotionalState::Engaged,
        ..Default::default()
    };

    let metrics = processor.compute_metrics("test", &context).await.unwrap();

    assert!((0.0..=1.0).contains(&metrics.surprise), "Surprise in range");
    assert!((0.0..=1.0).contains(&metrics.coherence_change), "Coherence in range");
    assert!((0.5..=1.5).contains(&metrics.emotional_weight), "Weight in range");
    assert!((-1.0..=1.0).contains(&metrics.alignment), "Alignment in range");
    assert!((0.0..=1.0).contains(&metrics.learning_score), "Score in range");

    let raw = 2.0 * metrics.surprise * metrics.coherence_change * metrics.emotional_weight * metrics.alignment;
    let expected = 1.0 / (1.0 + (-raw).exp());
    assert!((metrics.learning_score - expected).abs() < 0.0001, "Sigmoid formula");
}
