//! Session tool handlers for MCP lifecycle hooks.
//!
//! TASK-015: Implements Handlers methods that wrap session::handlers functions.
//!
//! # Constitution Compliance
//!
//! - ARCH-07: Hooks control memory lifecycle - SessionStart/PreToolUse/PostToolUse/SessionEnd

use serde_json::Value;

use crate::handlers::session::handlers::{
    call_post_tool_use, call_pre_tool_use, call_session_end, call_session_start, PostToolUseParams,
    PreToolUseParams, SessionEndParams, SessionStartParams,
};
use crate::handlers::SESSION_MANAGER;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

impl Handlers {
    /// Handle session_start tool call.
    ///
    /// Per ARCH-07: SessionStart hook initializes memory lifecycle.
    /// Uses the global SESSION_MANAGER singleton.
    ///
    /// # TASK-015
    pub(crate) async fn call_session_start(
        &self,
        id: Option<JsonRpcId>,
        arguments: Value,
    ) -> JsonRpcResponse {
        let params: SessionStartParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid session_start parameters: {}", e),
                );
            }
        };

        match call_session_start(&SESSION_MANAGER, params) {
            Ok(result) => JsonRpcResponse::success(id, result),
            Err((code, message)) => JsonRpcResponse::error(id, code, message),
        }
    }

    /// Handle session_end tool call.
    ///
    /// Per ARCH-07: SessionEnd hook finalizes memory lifecycle.
    /// Uses the global SESSION_MANAGER singleton.
    ///
    /// # TASK-015
    pub(crate) async fn call_session_end(
        &self,
        id: Option<JsonRpcId>,
        arguments: Value,
    ) -> JsonRpcResponse {
        let params: SessionEndParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid session_end parameters: {}", e),
                );
            }
        };

        match call_session_end(&SESSION_MANAGER, params) {
            Ok(result) => JsonRpcResponse::success(id, result),
            Err((code, message)) => JsonRpcResponse::error(id, code, message),
        }
    }

    /// Handle pre_tool_use hook.
    ///
    /// Per ARCH-07: PreToolUse hook prepares memory context.
    /// Uses the global SESSION_MANAGER singleton.
    ///
    /// # TASK-015
    pub(crate) async fn call_pre_tool_use(
        &self,
        id: Option<JsonRpcId>,
        arguments: Value,
    ) -> JsonRpcResponse {
        let params: PreToolUseParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid pre_tool_use parameters: {}", e),
                );
            }
        };

        match call_pre_tool_use(&SESSION_MANAGER, params) {
            Ok(result) => JsonRpcResponse::success(id, result),
            Err((code, message)) => JsonRpcResponse::error(id, code, message),
        }
    }

    /// Handle post_tool_use hook.
    ///
    /// Per ARCH-07: PostToolUse hook records tool outcome.
    /// Uses the global SESSION_MANAGER singleton.
    ///
    /// # TASK-015
    pub(crate) async fn call_post_tool_use(
        &self,
        id: Option<JsonRpcId>,
        arguments: Value,
    ) -> JsonRpcResponse {
        let params: PostToolUseParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid post_tool_use parameters: {}", e),
                );
            }
        };

        match call_post_tool_use(&SESSION_MANAGER, params) {
            Ok(result) => JsonRpcResponse::success(id, result),
            Err((code, message)) => JsonRpcResponse::error(id, code, message),
        }
    }
}
