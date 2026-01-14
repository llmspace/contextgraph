//! Dream and trigger MCP handlers.
//!
//! Per SPEC-TRIGGER-MCP-001.

mod handlers;
pub mod types;

// Re-export handlers from inner module for backward compatibility
pub use handlers::*;

// Re-export types for external use
pub use types::{
    TriggerConfigDto, TriggerHistoryEntry, TriggerMcpError, TriggerReasonDto, TriggerStatus,
};
