//! Timestamp parsing utilities for the Temporal-Positional model.

use chrono::{DateTime, Utc};

use crate::types::ModelInput;

/// Extract timestamp from ModelInput.
///
/// Attempts to parse timestamp from the instruction field:
/// - ISO 8601 format: "timestamp:2024-01-15T10:30:00Z"
/// - Unix epoch: "epoch:1705315800"
///
/// Falls back to current time if no valid timestamp found.
pub fn extract_timestamp(input: &ModelInput) -> DateTime<Utc> {
    match input {
        ModelInput::Text { instruction, .. } => instruction
            .as_ref()
            .and_then(|inst| parse_timestamp(inst))
            .unwrap_or_else(Utc::now),
        // For non-text inputs, use current time
        _ => Utc::now(),
    }
}

/// Parse timestamp from instruction string.
///
/// Supports formats:
/// - ISO 8601: "timestamp:2024-01-15T10:30:00Z"
/// - Unix epoch: "epoch:1705315800"
pub fn parse_timestamp(instruction: &str) -> Option<DateTime<Utc>> {
    // Try ISO 8601 format: "timestamp:2024-01-15T10:30:00Z"
    if let Some(ts_str) = instruction.strip_prefix("timestamp:") {
        if let Ok(dt) = DateTime::parse_from_rfc3339(ts_str.trim()) {
            return Some(dt.with_timezone(&Utc));
        }
    }

    // Try Unix epoch: "epoch:1705315800"
    if let Some(epoch_str) = instruction.strip_prefix("epoch:") {
        if let Ok(secs) = epoch_str.trim().parse::<i64>() {
            return DateTime::from_timestamp(secs, 0);
        }
    }

    None
}
