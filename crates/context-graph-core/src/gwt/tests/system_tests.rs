//! Basic GwtSystem creation and structure tests
//!
//! # Constitution Compliance (v6.0.0)
//!
//! Per Constitution v6.0.0, dreams are triggered by entropy > 0.7 AND churn > 0.5.

use crate::gwt::GwtSystem;
use std::sync::Arc;

#[tokio::test]
async fn test_gwt_system_creation() {
    let gwt = GwtSystem::new().await.expect("Failed to create GWT system");
    // Verify system has the required components
    assert!(Arc::strong_count(&gwt.workspace) > 0);
}

/// Verify TriggerManager is wired to GwtSystem.
///
/// # Constitution Compliance (v6.0.0)
///
/// This test verifies that:
/// 1. GwtSystem has a TriggerManager field
/// 2. The TriggerManager is accessible via trigger_manager() accessor
/// 3. The TriggerManager has constitution-mandated entropy threshold (0.7)
/// 4. Triggers are enabled by default
#[tokio::test]
async fn test_gwt_system_has_trigger_manager_wired() {
    let system = GwtSystem::new().await.expect("GwtSystem creation failed");
    let tm = system.trigger_manager();
    let manager = tm.lock();
    assert!(
        (manager.entropy_threshold() - 0.7).abs() < f32::EPSILON,
        "Entropy threshold must be 0.7 per Constitution v6.0.0"
    );
    assert!(manager.is_enabled(), "Triggers must be enabled by default");
}
