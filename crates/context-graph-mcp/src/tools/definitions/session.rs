//! Session tool definitions per ARCH-07.
//!
//! TASK-013: These tools control memory lifecycle for MCP sessions.
//!
//! # Constitution Compliance
//!
//! - ARCH-07: Hooks control memory lifecycle - SessionStart/PreToolUse/PostToolUse/SessionEnd

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns Session tool definitions (4 tools).
/// Per ARCH-07: Hooks control memory lifecycle.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        session_start(),
        session_end(),
        pre_tool_use(),
        post_tool_use(),
    ]
}

fn session_start() -> ToolDefinition {
    ToolDefinition::new(
        "session_start",
        "Initialize a new MCP session. Per ARCH-07: SessionStart hook controls memory lifecycle. \
         Must be called before any other tool use. Returns session_id for subsequent calls.",
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Unique session identifier. If not provided, a UUID will be generated.",
                    "pattern": "^[a-zA-Z0-9_-]{1,64}$"
                },
                "ttl_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1440,
                    "default": 30,
                    "description": "Session time-to-live in minutes (default: 30)"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to attach to session"
                }
            },
            "required": []
        }),
    )
}

fn session_end() -> ToolDefinition {
    ToolDefinition::new(
        "session_end",
        "Terminate an MCP session and perform cleanup. Per ARCH-07: SessionEnd hook finalizes \
         memory lifecycle. Returns session summary with tool count and duration.",
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier to terminate"
                }
            },
            "required": ["session_id"]
        }),
    )
}

fn pre_tool_use() -> ToolDefinition {
    ToolDefinition::new(
        "pre_tool_use",
        "Hook called before tool execution. Per ARCH-07: PreToolUse hook prepares memory context. \
         Records tool name and prepares session state for tool execution.",
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Active session identifier"
                },
                "tool_name": {
                    "type": "string",
                    "description": "Name of tool about to execute"
                },
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments (for context logging)"
                }
            },
            "required": ["session_id", "tool_name"]
        }),
    )
}

fn post_tool_use() -> ToolDefinition {
    ToolDefinition::new(
        "post_tool_use",
        "Hook called after tool execution. Per ARCH-07: PostToolUse hook records tool outcome. \
         Updates session statistics and triggers memory consolidation if needed.",
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Active session identifier"
                },
                "tool_name": {
                    "type": "string",
                    "description": "Name of tool that executed"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether tool execution succeeded"
                },
                "duration_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Execution duration in milliseconds"
                },
                "result_summary": {
                    "type": "string",
                    "description": "Brief summary of tool result"
                }
            },
            "required": ["session_id", "tool_name", "success"]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definitions_count() {
        let defs = definitions();
        assert_eq!(defs.len(), 4, "Should have 4 session tools");
    }

    #[test]
    fn test_tool_names() {
        let defs = definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"session_start"));
        assert!(names.contains(&"session_end"));
        assert!(names.contains(&"pre_tool_use"));
        assert!(names.contains(&"post_tool_use"));
    }

    #[test]
    fn test_session_start_schema() {
        let defs = definitions();
        let tool = defs.iter().find(|d| d.name == "session_start").unwrap();
        let props = tool
            .input_schema
            .get("properties")
            .expect("should have properties");

        // Check session_id property
        assert!(props.get("session_id").is_some());
        // Check ttl_minutes property
        assert!(props.get("ttl_minutes").is_some());
        // Check metadata property
        assert!(props.get("metadata").is_some());

        // session_start has no required fields
        let required = tool
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(required.is_empty());
    }

    #[test]
    fn test_session_end_schema() {
        let defs = definitions();
        let tool = defs.iter().find(|d| d.name == "session_end").unwrap();

        // session_end requires session_id
        let required = tool
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        let required_fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required_fields.contains(&"session_id"));
    }

    #[test]
    fn test_pre_tool_use_schema() {
        let defs = definitions();
        let tool = defs.iter().find(|d| d.name == "pre_tool_use").unwrap();

        // pre_tool_use requires session_id and tool_name
        let required = tool
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        let required_fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required_fields.contains(&"session_id"));
        assert!(required_fields.contains(&"tool_name"));
    }

    #[test]
    fn test_post_tool_use_schema() {
        let defs = definitions();
        let tool = defs.iter().find(|d| d.name == "post_tool_use").unwrap();

        // post_tool_use requires session_id, tool_name, success
        let required = tool
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        let required_fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required_fields.contains(&"session_id"));
        assert!(required_fields.contains(&"tool_name"));
        assert!(required_fields.contains(&"success"));
    }

    #[test]
    fn test_descriptions_reference_arch07() {
        let defs = definitions();
        for tool in defs {
            assert!(
                tool.description.contains("ARCH-07"),
                "Tool {} should reference ARCH-07",
                tool.name
            );
        }
    }
}
