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

/// TECH-GWT-IC-001: Verify TriggerManager is wired to GwtSystem
///
/// This test verifies that:
/// 1. GwtSystem has a TriggerManager field
/// 2. The TriggerManager is accessible via trigger_manager() accessor
/// 3. The TriggerManager has constitution-mandated defaults (IC threshold 0.5)
/// 4. Triggers are enabled by default
#[tokio::test]
async fn test_gwt_system_has_trigger_manager_wired() {
    let system = GwtSystem::new().await.expect("GwtSystem creation failed");
    let tm = system.trigger_manager();
    let manager = tm.lock();
    assert_eq!(
        manager.ic_threshold(),
        0.5,
        "IC threshold must be 0.5 per Constitution"
    );
    assert!(manager.is_enabled(), "Triggers must be enabled by default");
}
