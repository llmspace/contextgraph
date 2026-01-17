//! MCP Transport layer.
//!
//! TASK-32: SSE transport types for real-time streaming.
//! TASK-33: SSE handler implementation.
//!
//! PRD Section 5.1: "JSON-RPC 2.0, stdio/SSE"

pub mod sse;

pub use sse::{
    create_sse_router,
    next_event_id,
    sse_handler,
    // TASK-32: Types
    McpSseEvent,
    // TASK-33: Handler
    SseAppState,
    SseConfig,
    SseConfigError,
    SseHandlerError,
};
