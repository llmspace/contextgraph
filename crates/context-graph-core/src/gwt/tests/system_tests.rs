//! Basic GwtSystem creation and structure tests

use crate::gwt::GwtSystem;
use std::sync::Arc;

#[tokio::test]
async fn test_gwt_system_creation() {
    let gwt = GwtSystem::new().await.expect("Failed to create GWT system");
    // Verify system has the required components
    assert!(Arc::strong_count(&gwt.consciousness_calc) > 0);
    assert!(Arc::strong_count(&gwt.workspace) > 0);
    assert!(Arc::strong_count(&gwt.self_ego_node) > 0);
}
