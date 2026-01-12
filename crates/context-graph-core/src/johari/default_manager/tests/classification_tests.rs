//! Classification tests for DefaultJohariManager.

use crate::johari::default_manager::DefaultJohariManager;
use crate::johari::manager::{ClassificationContext, JohariTransitionManager, QuadrantPattern};
use crate::stubs::InMemoryTeleologicalStore;
use crate::types::fingerprint::{JohariFingerprint, SemanticFingerprint, NUM_EMBEDDERS};
use crate::types::JohariQuadrant;
use std::sync::Arc;

use super::super::helpers::matches_pattern;

fn create_test_store() -> Arc<InMemoryTeleologicalStore> {
    Arc::new(InMemoryTeleologicalStore::new())
}

#[tokio::test]
async fn test_classify_from_utl_state() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store);
    let semantic = SemanticFingerprint::zeroed();

    let context = ClassificationContext {
        delta_s: [0.3; NUM_EMBEDDERS],
        delta_c: [0.7; NUM_EMBEDDERS],
        disclosure_intent: [true; NUM_EMBEDDERS],
        access_counts: [0; NUM_EMBEDDERS],
    };

    let result = manager.classify(&semantic, &context).await.unwrap();

    for i in 0..NUM_EMBEDDERS {
        assert_eq!(
            result.dominant_quadrant(i),
            JohariQuadrant::Open,
            "Embedder {} should be Open",
            i
        );
    }
}

#[tokio::test]
async fn test_classify_all_quadrants() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store);
    let semantic = SemanticFingerprint::zeroed();

    let test_cases = [
        (0.3, 0.7, JohariQuadrant::Open),
        (0.3, 0.3, JohariQuadrant::Hidden),
        (0.7, 0.3, JohariQuadrant::Blind),
        (0.7, 0.7, JohariQuadrant::Unknown),
    ];

    for (delta_s, delta_c, expected) in test_cases {
        let context = ClassificationContext::uniform(delta_s, delta_c);
        let result = manager.classify(&semantic, &context).await.unwrap();

        for i in 0..NUM_EMBEDDERS {
            assert_eq!(result.dominant_quadrant(i), expected);
        }
    }
}

#[tokio::test]
async fn test_disclosure_intent_overrides_open() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store);
    let semantic = SemanticFingerprint::zeroed();

    let context = ClassificationContext {
        delta_s: [0.3; NUM_EMBEDDERS],
        delta_c: [0.7; NUM_EMBEDDERS],
        disclosure_intent: [false; NUM_EMBEDDERS],
        access_counts: [0; NUM_EMBEDDERS],
    };

    let result = manager.classify(&semantic, &context).await.unwrap();

    for i in 0..NUM_EMBEDDERS {
        assert_eq!(result.dominant_quadrant(i), JohariQuadrant::Hidden);
    }
}

#[test]
fn test_matches_pattern_all_in() {
    let mut johari = JohariFingerprint::zeroed();
    for i in 0..NUM_EMBEDDERS {
        johari.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0);
    }

    assert!(matches_pattern(
        &johari,
        &QuadrantPattern::AllIn(JohariQuadrant::Open)
    ));
    assert!(!matches_pattern(
        &johari,
        &QuadrantPattern::AllIn(JohariQuadrant::Hidden)
    ));
}

#[test]
fn test_matches_pattern_at_least() {
    let mut johari = JohariFingerprint::zeroed();
    for i in 0..5 {
        johari.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0);
    }

    assert!(matches_pattern(
        &johari,
        &QuadrantPattern::AtLeast {
            quadrant: JohariQuadrant::Open,
            count: 5
        }
    ));
    assert!(!matches_pattern(
        &johari,
        &QuadrantPattern::AtLeast {
            quadrant: JohariQuadrant::Open,
            count: 6
        }
    ));
}
