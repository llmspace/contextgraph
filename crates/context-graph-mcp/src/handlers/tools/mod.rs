//! MCP tool call handlers.
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore.
//!
//! All tool responses include `_cognitive_pulse` with live UTL metrics.
//! This provides real-time cognitive state in every MCP response.
//!
//! # Constitution Reference
//!
//! Suggested action mapping based on entropy/coherence:
//! - Low entropy, high coherence -> DirectRecall
//! - High entropy, low coherence -> TriggerDream
//! - Low entropy, low coherence -> GetNeighborhood
//! - High entropy, high coherence -> EpistemicAction
//!
//! # Module Organization
//!
//! - `dispatch` - Tool dispatch logic (handle_tools_list, handle_tools_call)
//! - `helpers` - MCP result helpers with CognitivePulse injection
//! - `memory_tools` - Memory operations (inject_context, store_memory, search_graph)
//! - `status_tools` - Status queries (get_memetic_status, get_graph_manifest, utl_status)
//! - `gwt_workspace` - GWT workspace tools (get_workspace_status, trigger_workspace_broadcast, adjust_coupling)

mod dispatch;
mod gwt_workspace;
mod helpers;
mod memory_tools;
mod meta_learning_tools;
// TASK-015: Session lifecycle hook tools per ARCH-07
mod session_tools;
mod status_tools;

// All implementations are on the Handlers struct, so no re-exports needed.
// The modules add impl blocks to Handlers which are automatically available.
