//! Tests for ConsolidationService construction and configuration

use crate::autonomous::curation::{ConsolidationConfig, MemoryId};
use crate::autonomous::services::consolidation_service::{
    ConsolidationService, ServiceConsolidationCandidate,
};

#[test]
fn test_service_new() {
    let service = ConsolidationService::new();
    let config = service.config();

    assert!(config.enabled);
    assert!((config.similarity_threshold - 0.92).abs() < f32::EPSILON);
    assert_eq!(config.max_daily_merges, 50);
    assert!((config.theta_diff_threshold - 0.05).abs() < f32::EPSILON);
    assert_eq!(service.daily_merges(), 0);

    println!("[PASS] test_service_new");
}

#[test]
fn test_service_default() {
    let service = ConsolidationService::default();
    assert!(service.config().enabled);
    assert_eq!(service.daily_merges(), 0);

    println!("[PASS] test_service_default");
}

#[test]
fn test_service_with_config() {
    let config = ConsolidationConfig {
        enabled: false,
        similarity_threshold: 0.85,
        max_daily_merges: 100,
        theta_diff_threshold: 0.10,
    };

    let service = ConsolidationService::with_config(config);

    assert!(!service.config().enabled);
    assert!((service.config().similarity_threshold - 0.85).abs() < f32::EPSILON);
    assert_eq!(service.config().max_daily_merges, 100);

    println!("[PASS] test_service_with_config");
}

#[test]
fn test_service_reset_daily_counter() {
    let config = ConsolidationConfig::default();
    let mut service = ConsolidationService::with_config(config);

    // Create and consolidate some candidates
    let id1 = MemoryId::new();
    let id2 = MemoryId::new();
    let candidates = vec![ServiceConsolidationCandidate::new(
        vec![id1, id2],
        MemoryId::new(),
        0.95,
        0.8,
    )];

    service.consolidate(&candidates);
    assert_eq!(service.daily_merges(), 1);

    service.reset_daily_counter();
    assert_eq!(service.daily_merges(), 0);

    println!("[PASS] test_service_reset_daily_counter");
}
