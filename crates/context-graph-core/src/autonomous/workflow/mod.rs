//! Autonomous workflow orchestration types
//!
//! This module defines the complete configuration and status types for the
//! autonomous system, including scheduling, event handling, and
//! health monitoring.

mod config;
mod events;
mod health;
mod schedule;
mod status;
#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::config::AutonomousConfig;
pub use self::events::OptimizationEvent;
pub use self::health::AutonomousHealth;
pub use self::schedule::{DailySchedule, ScheduleValidationError, ScheduledCheckType};
pub use self::status::AutonomousStatus;
