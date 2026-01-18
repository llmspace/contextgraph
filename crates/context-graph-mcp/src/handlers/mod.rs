//! Request handlers for MCP methods.
//!
//! This module provides the main `Handlers` struct that dispatches JSON-RPC
//! requests to their appropriate handler methods.
//!
//! # Module Organization
//!
//! - `core`: Core Handlers struct and dispatch logic
//! - `lifecycle`: MCP lifecycle handlers (initialize, shutdown)
//! - `tools`: MCP tool call handlers
//! - `memory`: Legacy memory operation handlers
//! - `search`: Multi-embedding weighted search handlers (TASK-S002)
//! - `purpose`: Purpose and goal alignment handlers (TASK-S003)
//! - `utl`: UTL computation handlers
//! - `system`: System status and health handlers
//! - `gwt_traits`: GWT provider traits for state management (TASK-GWT-001)
//! - `gwt_providers`: Real GWT provider implementations wrapping actual components (TASK-GWT-001)
//! - `atc`: Adaptive Threshold Calibration handlers (TASK-ATC-001)
//! - `steering`: Steering subsystem handlers (TASK-STEERING-001)
//! - `causal`: Causal inference handlers (TASK-CAUSAL-001)
//! - `teleological`: Teleological search, fusion, and profile handlers (TELEO-H1 to TELEO-H5)
//! - `autonomous`: NORTH autonomous system handlers (TASK-AUTONOMOUS-MCP)
//! - `epistemic`: Epistemic action handlers for GWT workspace (TASK-MCP-002)
//! - `merge`: Merge concepts handler for node consolidation (TASK-MCP-004)
//! - `session`: Session lifecycle management for MCP hooks (TASK-012)

mod atc;
mod autonomous;
mod causal;
mod core;
mod dream;
mod epistemic;
mod lifecycle;
mod memory;
mod merge;
mod neuromod;
pub mod gwt_providers;
pub mod gwt_traits;
mod purpose;
mod search;
// TASK-012: Session lifecycle management
pub mod session;
mod steering;
mod system;
mod teleological;
mod tools;
mod utl;

// TASK-METAUTL-P0-005: Meta-learning handlers for self-correction tools
pub mod meta_learning;

#[cfg(test)]
mod tests;

// Re-export the main Handlers struct for backward compatibility
pub use self::core::Handlers;

// Re-export MetaUtlTracker for server initialization
pub use self::core::MetaUtlTracker;

// Re-export GWT traits for external use (TASK-GWT-001)
// Note: These are public API re-exports - unused within this crate but available to consumers
#[allow(unused_imports)]
pub use self::gwt_traits::{GwtSystemProvider, MetaCognitiveProvider, WorkspaceProvider};

// Re-export GWT provider implementations for wiring (TASK-GWT-001)
// Note: These are public API re-exports - unused within this crate but available to consumers
#[allow(unused_imports)]
pub use self::gwt_providers::{
    GwtSystemProviderImpl, MetaCognitiveProviderImpl, WorkspaceProviderImpl,
};

// ============================================================================
// Session Management Singleton (TASK-014)
// ============================================================================

use once_cell::sync::Lazy;

/// Global session manager for MCP lifecycle hooks.
///
/// Per ARCH-07: Hooks control memory lifecycle - SessionStart/PreToolUse/PostToolUse/SessionEnd.
/// This singleton manages all MCP sessions across the server lifetime.
///
/// # TASK-014
pub static SESSION_MANAGER: Lazy<session::SessionManager> = Lazy::new(session::SessionManager::new);
