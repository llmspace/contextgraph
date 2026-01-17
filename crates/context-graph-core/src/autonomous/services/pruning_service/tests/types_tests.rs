//! Tests for PruningService types

use chrono::Utc;

use crate::autonomous::curation::MemoryId;
use crate::autonomous::services::pruning_service::types::{
    ExtendedPruningConfig, MemoryMetadata, PruneReason, PruningCandidate, PruningReport,
};

use super::helpers::{make_metadata, make_metadata_with_access};

// ============================================================================
// PruneReason tests
// ============================================================================

#[test]
fn test_prune_reason_description() {
    println!("[PASS] PruneReason::LowAlignment description");
    assert_eq!(
        PruneReason::LowAlignment.description(),
        "Alignment score below threshold"
    );

    println!("[PASS] PruneReason::Stale description");
    assert_eq!(
        PruneReason::Stale.description(),
        "Not accessed for extended period"
    );

    println!("[PASS] PruneReason::Orphaned description");
    assert_eq!(
        PruneReason::Orphaned.description(),
        "No connections to other memories"
    );

    println!("[PASS] PruneReason::Redundant description");
    assert_eq!(
        PruneReason::Redundant.description(),
        "Duplicate or near-duplicate content"
    );

    println!("[PASS] PruneReason::LowQuality description");
    assert_eq!(
        PruneReason::LowQuality.description(),
        "Quality metrics below threshold"
    );
}

#[test]
fn test_prune_reason_equality() {
    println!("[PASS] PruneReason equality");
    assert_eq!(PruneReason::LowAlignment, PruneReason::LowAlignment);
    assert_ne!(PruneReason::LowAlignment, PruneReason::Stale);
    assert_ne!(PruneReason::Stale, PruneReason::Orphaned);
    assert_ne!(PruneReason::Orphaned, PruneReason::Redundant);
    assert_ne!(PruneReason::Redundant, PruneReason::LowQuality);
}

#[test]
fn test_prune_reason_serialization() {
    let reasons = [
        PruneReason::LowAlignment,
        PruneReason::Stale,
        PruneReason::Orphaned,
        PruneReason::Redundant,
        PruneReason::LowQuality,
    ];

    for reason in reasons {
        let json = serde_json::to_string(&reason).expect("serialize");
        let deserialized: PruneReason = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, reason);
    }
    println!("[PASS] PruneReason serialization round-trip");
}

// ============================================================================
// MemoryMetadata tests
// ============================================================================

#[test]
fn test_memory_metadata_new() {
    let id = MemoryId::new();
    let created = Utc::now();
    let meta = MemoryMetadata::new(id.clone(), created, 0.75);

    assert_eq!(meta.id, id);
    assert!((meta.alignment - 0.75).abs() < f32::EPSILON);
    assert_eq!(meta.connection_count, 0);
    assert_eq!(meta.byte_size, 0);
    assert!(meta.last_accessed.is_none());
    assert!(meta.quality_score.is_none());
    assert!(meta.content_hash.is_none());
    println!("[PASS] MemoryMetadata::new");
}

#[test]
fn test_memory_metadata_age_days() {
    let meta = make_metadata(0.75, 45, 0, 0);
    let age = meta.age_days();
    assert!((44..=46).contains(&age)); // Allow for timing variance
    println!("[PASS] MemoryMetadata::age_days = {}", age);
}

#[test]
fn test_memory_metadata_days_since_access() {
    let meta = make_metadata_with_access(0.75, 100, 0, 30);
    let days = meta.days_since_access();
    assert!(days.is_some());
    let d = days.unwrap();
    assert!((29..=31).contains(&d));
    println!("[PASS] MemoryMetadata::days_since_access = {}", d);
}

#[test]
fn test_memory_metadata_never_accessed() {
    let meta = make_metadata(0.75, 100, 0, 0);
    assert!(meta.days_since_access().is_none());
    println!("[PASS] MemoryMetadata::days_since_access returns None for never-accessed");
}

// ============================================================================
// PruningCandidate tests
// ============================================================================

#[test]
fn test_pruning_candidate_new() {
    let id = MemoryId::new();
    let candidate = PruningCandidate::new(id.clone(), 60, 0.30, 2, PruneReason::LowAlignment, 1024);

    assert_eq!(candidate.memory_id, id);
    assert_eq!(candidate.age_days, 60);
    assert!((candidate.alignment - 0.30).abs() < f32::EPSILON);
    assert_eq!(candidate.connections, 2);
    assert_eq!(candidate.reason, PruneReason::LowAlignment);
    assert_eq!(candidate.byte_size, 1024);
    assert!(candidate.priority_score > 0.0);
    println!(
        "[PASS] PruningCandidate::new with priority_score = {:.3}",
        candidate.priority_score
    );
}

#[test]
fn test_pruning_candidate_priority_ordering() {
    // Lower alignment = higher priority
    let low_align =
        PruningCandidate::new(MemoryId::new(), 30, 0.10, 0, PruneReason::LowAlignment, 100);
    let high_align =
        PruningCandidate::new(MemoryId::new(), 30, 0.80, 0, PruneReason::LowAlignment, 100);
    assert!(low_align.priority_score > high_align.priority_score);
    println!("[PASS] Lower alignment has higher priority");

    // Older age = higher priority
    let old = PruningCandidate::new(MemoryId::new(), 365, 0.50, 0, PruneReason::Stale, 100);
    let young = PruningCandidate::new(MemoryId::new(), 30, 0.50, 0, PruneReason::Stale, 100);
    assert!(old.priority_score > young.priority_score);
    println!("[PASS] Older memories have higher priority");

    // Fewer connections = higher priority
    let isolated = PruningCandidate::new(MemoryId::new(), 60, 0.50, 0, PruneReason::Orphaned, 100);
    let connected =
        PruningCandidate::new(MemoryId::new(), 60, 0.50, 10, PruneReason::Orphaned, 100);
    assert!(isolated.priority_score > connected.priority_score);
    println!("[PASS] Isolated memories have higher priority");
}

// ============================================================================
// PruningReport tests
// ============================================================================

#[test]
fn test_pruning_report_default() {
    let report = PruningReport::default();
    assert_eq!(report.candidates_evaluated, 0);
    assert_eq!(report.pruned_count, 0);
    assert_eq!(report.bytes_freed, 0);
    assert!(report.reasons_breakdown.is_empty());
    assert_eq!(report.preserved_count, 0);
    assert!(!report.daily_limit_reached);
    println!("[PASS] PruningReport::default");
}

#[test]
fn test_pruning_report_record_prune() {
    let mut report = PruningReport::new();
    let candidate = PruningCandidate::new(
        MemoryId::new(),
        60,
        0.30,
        0,
        PruneReason::LowAlignment,
        2048,
    );

    report.record_prune(&candidate);
    assert_eq!(report.pruned_count, 1);
    assert_eq!(report.bytes_freed, 2048);
    assert_eq!(
        *report
            .reasons_breakdown
            .get(&PruneReason::LowAlignment)
            .unwrap(),
        1
    );
    println!("[PASS] PruningReport::record_prune");
}

#[test]
fn test_pruning_report_multiple_reasons() {
    let mut report = PruningReport::new();

    report.record_prune(&PruningCandidate::new(
        MemoryId::new(),
        60,
        0.30,
        0,
        PruneReason::LowAlignment,
        100,
    ));
    report.record_prune(&PruningCandidate::new(
        MemoryId::new(),
        60,
        0.30,
        0,
        PruneReason::LowAlignment,
        100,
    ));
    report.record_prune(&PruningCandidate::new(
        MemoryId::new(),
        60,
        0.50,
        0,
        PruneReason::Orphaned,
        200,
    ));

    assert_eq!(report.pruned_count, 3);
    assert_eq!(report.bytes_freed, 400);
    assert_eq!(
        *report
            .reasons_breakdown
            .get(&PruneReason::LowAlignment)
            .unwrap(),
        2
    );
    assert_eq!(
        *report
            .reasons_breakdown
            .get(&PruneReason::Orphaned)
            .unwrap(),
        1
    );
    println!("[PASS] PruningReport reasons breakdown");
}

// ============================================================================
// ExtendedPruningConfig tests
// ============================================================================

#[test]
fn test_extended_pruning_config_default() {
    let config = ExtendedPruningConfig::default();
    assert!(config.base.enabled);
    assert_eq!(config.base.min_age_days, 30);
    assert!((config.base.min_alignment - 0.40).abs() < f32::EPSILON);
    assert_eq!(config.base.min_connections, 3);
    assert_eq!(config.max_daily_prunes, 100);
    assert_eq!(config.stale_days, 90);
    assert!((config.min_quality - 0.30).abs() < f32::EPSILON);
    println!("[PASS] ExtendedPruningConfig::default");
}

#[test]
fn test_extended_pruning_config_serialization() {
    let config = ExtendedPruningConfig::default();
    let json = serde_json::to_string(&config).expect("serialize");
    let deserialized: ExtendedPruningConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.max_daily_prunes, config.max_daily_prunes);
    assert_eq!(deserialized.stale_days, config.stale_days);
    println!("[PASS] ExtendedPruningConfig serialization");
}
