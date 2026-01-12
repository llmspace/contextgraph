//! MCP tool result helpers with CognitivePulse injection.

use serde_json::json;
use tracing::{error, warn};

use crate::middleware::CognitivePulse;
use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

impl Handlers {
    /// MCP-compliant tool result helper WITH CognitivePulse injection.
    ///
    /// Wraps tool output in the required MCP format with live UTL metrics:
    /// ```json
    /// {
    ///   "content": [{"type": "text", "text": "..."}],
    ///   "isError": false,
    ///   "_cognitive_pulse": {
    ///     "entropy": 0.42,
    ///     "coherence": 0.78,
    ///     "learning_score": 0.55,
    ///     "quadrant": "Open",
    ///     "suggested_action": "DirectRecall"
    ///   }
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// CognitivePulse computation targets < 1ms. Warning logged if exceeded.
    ///
    /// # Error Handling
    ///
    /// FAIL FAST: If CognitivePulse computation fails, the ENTIRE tool call
    /// fails with a detailed error. NO fallbacks, NO default values.
    pub(crate) fn tool_result_with_pulse(
        &self,
        id: Option<JsonRpcId>,
        data: serde_json::Value,
    ) -> JsonRpcResponse {
        // Compute CognitivePulse - FAIL FAST if unavailable
        let pulse = match CognitivePulse::from_processor(self.utl_processor.as_ref()) {
            Ok(p) => p,
            Err(e) => {
                // FAIL FAST - no fallbacks
                error!(
                    error = %e,
                    "CognitivePulse computation FAILED - tool call rejected"
                );
                return JsonRpcResponse::success(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": format!("UTL pulse computation failed: {}", e)
                        }],
                        "isError": true
                    }),
                );
            }
        };

        JsonRpcResponse::success(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())
                }],
                "isError": false,
                "_cognitive_pulse": pulse
            }),
        )
    }

    /// MCP-compliant tool error helper WITH CognitivePulse injection.
    ///
    /// Even error responses include the cognitive pulse to maintain
    /// consistent system state visibility.
    ///
    /// # Error Handling
    ///
    /// If pulse computation fails during error response, logs warning
    /// but still returns the original error (pulse failure is secondary).
    pub(crate) fn tool_error_with_pulse(
        &self,
        id: Option<JsonRpcId>,
        message: &str,
    ) -> JsonRpcResponse {
        // Try to compute pulse, but don't fail the error response if it fails
        let pulse_result = CognitivePulse::from_processor(self.utl_processor.as_ref());

        match pulse_result {
            Ok(pulse) => JsonRpcResponse::success(
                id,
                json!({
                    "content": [{
                        "type": "text",
                        "text": message
                    }],
                    "isError": true,
                    "_cognitive_pulse": pulse
                }),
            ),
            Err(e) => {
                warn!(
                    error = %e,
                    original_error = message,
                    "CognitivePulse computation failed for error response"
                );
                // Still return the original error, just without pulse
                JsonRpcResponse::success(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": format!("{} (pulse unavailable: {})", message, e)
                        }],
                        "isError": true
                    }),
                )
            }
        }
    }
}
