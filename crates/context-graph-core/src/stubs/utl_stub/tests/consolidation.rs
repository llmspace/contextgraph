//! Consolidation threshold tests for UTL processor.

use crate::stubs::utl_stub::StubUtlProcessor;
use crate::traits::UtlProcessor;

#[tokio::test]
async fn test_utl_consolidation_threshold() {
    // TC-GHOST-001: Consolidation decision must respect threshold
    let processor = StubUtlProcessor::with_threshold(0.7);
    let embedding = vec![0.5; 1536];

    // Node below threshold
    let mut low_importance = crate::types::MemoryNode::new("low".to_string(), embedding.clone());
    low_importance.importance = 0.5;

    // Node at threshold
    let mut at_threshold = crate::types::MemoryNode::new("at".to_string(), embedding.clone());
    at_threshold.importance = 0.7;

    // Node above threshold
    let mut high_importance = crate::types::MemoryNode::new("high".to_string(), embedding);
    high_importance.importance = 0.9;

    assert!(
        !processor.should_consolidate(&low_importance).await.unwrap(),
        "Node below threshold should not consolidate"
    );
    assert!(
        processor.should_consolidate(&at_threshold).await.unwrap(),
        "Node at threshold should consolidate"
    );
    assert!(
        processor
            .should_consolidate(&high_importance)
            .await
            .unwrap(),
        "Node above threshold should consolidate"
    );
}
