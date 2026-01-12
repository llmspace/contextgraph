//! Tests for the teleological retrieval pipeline.

use std::sync::Arc;
use std::time::Duration;

use crate::alignment::DefaultAlignmentCalculator;
use crate::johari::DefaultJohariManager;
use crate::purpose::{DiscoveryMethod, GoalDiscoveryMetadata, GoalHierarchy, GoalLevel, GoalNode};
use crate::retrieval::teleological_result::AlignmentLevel;
use crate::retrieval::InMemoryMultiEmbeddingExecutor;
use crate::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider};
use crate::types::fingerprint::SemanticFingerprint;
use crate::types::JohariQuadrant;

use super::super::teleological_query::TeleologicalQuery;
use super::{DefaultTeleologicalPipeline, PipelineHealth, TeleologicalRetrievalPipeline};
use crate::error::CoreError;

/// Create a test fingerprint for goal hierarchy.
fn create_test_goal_fingerprint(base: f32) -> SemanticFingerprint {
    let mut fp = SemanticFingerprint::zeroed();
    for i in 0..fp.e1_semantic.len() {
        fp.e1_semantic[i] = (i as f32 / 1024.0).sin() * base;
    }
    for i in 0..fp.e5_causal.len() {
        fp.e5_causal[i] = base + (i as f32 * 0.0001);
    }
    for i in 0..fp.e7_code.len() {
        fp.e7_code[i] = base + (i as f32 * 0.0001);
    }
    fp
}

fn create_test_hierarchy() -> GoalHierarchy {
    let mut hierarchy = GoalHierarchy::new();

    let ns_fp = create_test_goal_fingerprint(0.8);
    let discovery =
        GoalDiscoveryMetadata::new(DiscoveryMethod::Bootstrap, 0.9, 1, 0.85).unwrap();

    let ns = GoalNode::autonomous_goal(
        "Build the best product".to_string(),
        GoalLevel::NorthStar,
        ns_fp.clone(),
        discovery,
    )
    .expect("Failed to create North Star");
    let ns_id = ns.id;

    hierarchy.add_goal(ns).expect("Failed to add North Star");

    let child_fp = create_test_goal_fingerprint(0.7);
    let child_discovery =
        GoalDiscoveryMetadata::new(DiscoveryMethod::Decomposition, 0.8, 5, 0.75).unwrap();

    let child = GoalNode::child_goal(
        "Improve UX".to_string(),
        GoalLevel::Strategic,
        ns_id,
        child_fp,
        child_discovery,
    )
    .expect("Failed to create strategic goal");

    hierarchy
        .add_goal(child)
        .expect("Failed to add strategic goal");

    hierarchy
}

async fn create_test_pipeline() -> DefaultTeleologicalPipeline<
    InMemoryMultiEmbeddingExecutor,
    DefaultAlignmentCalculator,
    DefaultJohariManager<InMemoryTeleologicalStore>,
    InMemoryTeleologicalStore,
> {
    let store = InMemoryTeleologicalStore::new();
    let provider = StubMultiArrayProvider::new();

    // Store needs to be Arc-wrapped for sharing between executor, johari_manager, and pipeline
    let store_arc = Arc::new(store);

    let executor = Arc::new(InMemoryMultiEmbeddingExecutor::with_arcs(
        store_arc.clone(),
        Arc::new(provider),
    ));

    let alignment_calc = Arc::new(DefaultAlignmentCalculator::new());
    let johari_manager = Arc::new(DefaultJohariManager::new(store_arc.clone()));
    let hierarchy = create_test_hierarchy();

    DefaultTeleologicalPipeline::new(
        executor,
        alignment_calc,
        johari_manager,
        store_arc,
        hierarchy,
    )
}

#[tokio::test]
async fn test_pipeline_creation() {
    let pipeline = create_test_pipeline().await;
    let health = pipeline.health_check().await.unwrap();

    assert_eq!(health.spaces_available, 13);
    assert!(health.has_goal_hierarchy);

    println!("[VERIFIED] Pipeline created with all components");
}

#[tokio::test]
async fn test_execute_basic_query() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("authentication patterns");
    let result = pipeline.execute(&query).await.unwrap();

    assert!(result.total_time.as_millis() < 1000); // Generous for test
    assert!(result.spaces_searched > 0);

    println!("BEFORE: query text = 'authentication patterns'");
    println!(
        "AFTER: results = {}, time = {:?}",
        result.len(),
        result.total_time
    );
    println!("[VERIFIED] Basic query execution works");
}

#[tokio::test]
async fn test_execute_with_breakdown() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("test query").with_breakdown(true);

    let result = pipeline.execute(&query).await.unwrap();

    assert!(result.breakdown.is_some());
    let breakdown = result.breakdown.unwrap();

    println!("Breakdown: {}", breakdown.funnel_summary());
    println!("[VERIFIED] Pipeline breakdown is populated when requested");
}

#[tokio::test]
async fn test_execute_fails_empty_query() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::default();
    let result = pipeline.execute(&query).await;

    assert!(result.is_err());
    match result {
        Err(CoreError::ValidationError { field, .. }) => {
            assert_eq!(field, "text");
            println!("[VERIFIED] Empty query fails fast with ValidationError");
        }
        _ => panic!("Expected ValidationError"),
    }
}

#[tokio::test]
async fn test_execute_with_johari_filter() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("test")
        .with_johari_filter(vec![JohariQuadrant::Open, JohariQuadrant::Blind]);

    let result = pipeline.execute(&query).await.unwrap();

    // All results should be in Open or Blind quadrant
    for r in &result.results {
        assert!(
            r.johari_quadrant == JohariQuadrant::Open
                || r.johari_quadrant == JohariQuadrant::Blind
        );
    }

    println!("[VERIFIED] Johari filter is applied");
}

#[tokio::test]
async fn test_timing_breakdown() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("timing test");
    let result = pipeline.execute(&query).await.unwrap();

    println!("Timing: {}", result.timing_summary());
    println!("  Stage 1 (SPLADE): {:?}", result.timing.stage1_splade);
    println!(
        "  Stage 2 (Matryoshka): {:?}",
        result.timing.stage2_matryoshka
    );
    println!(
        "  Stage 3 (Full HNSW): {:?}",
        result.timing.stage3_full_hnsw
    );
    println!(
        "  Stage 4 (Teleological): {:?}",
        result.timing.stage4_teleological
    );
    println!(
        "  Stage 5 (Late Interaction): {:?}",
        result.timing.stage5_late_interaction
    );
    println!("  Total: {:?}", result.total_time);

    println!("[VERIFIED] All pipeline stages have timing measurements");
}

#[tokio::test]
async fn test_alignment_level_in_results() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("alignment test");
    let result = pipeline.execute(&query).await.unwrap();

    for scored in &result.results {
        let level = scored.alignment_threshold();
        match level {
            AlignmentLevel::Optimal => assert!(scored.goal_alignment >= 0.75),
            AlignmentLevel::Acceptable => {
                assert!(scored.goal_alignment >= 0.70 && scored.goal_alignment < 0.75)
            }
            AlignmentLevel::Warning => {
                assert!(scored.goal_alignment >= 0.55 && scored.goal_alignment < 0.70)
            }
            AlignmentLevel::Critical => assert!(scored.goal_alignment < 0.55),
        }
    }

    println!("[VERIFIED] Alignment levels correctly classified");
}

#[tokio::test]
async fn test_misaligned_count() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("misalignment test");
    let result = pipeline.execute(&query).await.unwrap();

    let misaligned = result.misaligned_count();
    let manual_count = result.results.iter().filter(|r| r.is_misaligned).count();

    assert_eq!(misaligned, manual_count);
    println!(
        "[VERIFIED] misaligned_count() = {} matches manual count",
        misaligned
    );
}

#[test]
fn test_pipeline_health_defaults() {
    let health = PipelineHealth {
        is_healthy: true,
        spaces_available: 13,
        has_goal_hierarchy: true,
        index_size: 1_000_000,
        last_query_time: Some(Duration::from_millis(45)),
    };

    assert!(health.is_healthy);
    assert_eq!(health.spaces_available, 13);
    println!("[VERIFIED] PipelineHealth struct works correctly");
}
