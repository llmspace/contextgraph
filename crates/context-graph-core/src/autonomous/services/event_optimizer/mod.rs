//! NORTH-019: Event-driven optimization service
//!
//! This module provides the EventOptimizer service that performs event-driven
//! optimization based on system events such as high drift, low performance,
//! memory pressure, and scheduled maintenance.

mod action;
mod config;
mod metrics;
mod optimizer;
mod plan;
mod record;
mod trigger;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use action::OptimizationAction;
pub use config::EventOptimizerConfig;
pub use metrics::SystemMetrics;
pub use optimizer::EventOptimizer;
pub use plan::OptimizationPlan;
pub use record::OptimizationEventRecord;
pub use trigger::OptimizationTrigger;
