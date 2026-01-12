//! Tests for consolidate method

use crate::autonomous::curation::{ConsolidationConfig, MemoryId};
use crate::autonomous::services::consolidation_service::{
    ConsolidationService, ServiceConsolidationCandidate,
};

#[test]
fn test_consolidate_empty() {
    let mut service = ConsolidationService::new();
    let report = service.consolidate(&[]);

    assert_eq!(report.candidates_found, 0);
    assert_eq!(report.merged, 0);
    assert_eq!(report.skipped, 0);
    assert!(!report.daily_limit_reached);

    println!("[PASS] test_consolidate_empty");
}

#[test]
fn test_consolidate_disabled() {
    let config = ConsolidationConfig {
        enabled: false,
        ..Default::default()
    };
    let mut service = ConsolidationService::with_config(config);

    let candidates = vec![ServiceConsolidationCandidate::new(
        vec![MemoryId::new(), MemoryId::new()],
        MemoryId::new(),
        0.95,
        0.8,
    )];

    let report = service.consolidate(&candidates);
    assert_eq!(report.merged, 0);
    assert_eq!(report.skipped, 1);

    println!("[PASS] test_consolidate_disabled");
}

#[test]
fn test_consolidate_merges() {
    let mut service = ConsolidationService::new();

    let candidates = vec![
        ServiceConsolidationCandidate::new(
            vec![MemoryId::new(), MemoryId::new()],
            MemoryId::new(),
            0.95,
            0.8,
        ),
        ServiceConsolidationCandidate::new(
            vec![MemoryId::new(), MemoryId::new()],
            MemoryId::new(),
            0.93,
            0.85,
        ),
    ];

    let report = service.consolidate(&candidates);
    assert_eq!(report.candidates_found, 2);
    assert_eq!(report.merged, 2);
    assert_eq!(report.skipped, 0);
    assert_eq!(service.daily_merges(), 2);

    println!("[PASS] test_consolidate_merges");
}

#[test]
fn test_consolidate_daily_limit() {
    let config = ConsolidationConfig {
        enabled: true,
        similarity_threshold: 0.92,
        max_daily_merges: 2,
        theta_diff_threshold: 0.05,
    };
    let mut service = ConsolidationService::with_config(config);

    let candidates: Vec<_> = (0..5)
        .map(|_| {
            ServiceConsolidationCandidate::new(
                vec![MemoryId::new(), MemoryId::new()],
                MemoryId::new(),
                0.95,
                0.8,
            )
        })
        .collect();

    let report = service.consolidate(&candidates);

    assert_eq!(report.candidates_found, 5);
    assert_eq!(report.merged, 2);
    assert_eq!(report.skipped, 3);
    assert!(report.daily_limit_reached);

    println!("[PASS] test_consolidate_daily_limit");
}
