//! Self-Healing Manager Service (NORTH-020)
//!
//! This module implements the system health monitoring and automatic recovery
//! service for the autonomous North Star system. It detects health issues,
//! diagnoses problems, and applies healing actions.
//!
//! # Module Structure
//!
//! - `types` - Core type definitions (config, health state, issues, actions)
//! - `manager` - The `SelfHealingManager` implementation
//!
//! # Example
//!
//! ```
//! use context_graph_core::autonomous::services::self_healing_manager::{
//!     SelfHealingManager, SelfHealingConfig, SystemHealthState,
//! };
//!
//! let manager = SelfHealingManager::new();
//! let health = manager.check_health();
//! let issues = manager.diagnose_issues(&health);
//! ```

mod manager;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use manager::SelfHealingManager;
pub use types::{
    HealingAction, HealingResult, HealthIssue, IssueSeverity, SelfHealingConfig, SystemHealthState,
    SystemOperationalStatus,
};
